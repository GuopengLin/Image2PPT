#!/usr/bin/env python
"""Closed-loop font-size calibration from rendered PPT previews.

Like position calibration, this builds a temporary text-only deck where
each text element is painted with a unique colour. The rendered ink bbox
is compared to the source image's text ink bbox, then the element font
size (and any run-level font sizes) are scaled before position calibration
runs.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import shutil
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from calibrate_text_positions import (  # noqa: E402
    SCRIPTS_ROOT,
    _box_to_preview,
    _calibration_layout,
    _detect_colour_bbox,
    _preview_for_slide,
    _source_for_slide,
    _source_to_preview_transform,
    _is_dense_small_text,
    _target_ink_for_element,
    _text_elements,
)
from image_sources import supported_image_formats  # noqa: E402
from inventory_to_layout import (  # noqa: E402
    _fontconfig_font_path,
    _render_text_metrics,
    _sample_background_for_bbox,
    _target_text_metrics,
)

# In-process render+build to skip per-iteration subprocess startup.
sys.path.insert(0, str(SCRIPTS_ROOT / "deck"))
sys.path.insert(0, str(SCRIPTS_ROOT / "verify"))
import build_pptx_from_layout as _builder  # noqa: E402
import render_preview as _render  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--layout", required=True, help="Combined layout JSON.")
    p.add_argument("--source-dir", required=True,
                   help=f"Directory with page_NN source images "
                        f"({supported_image_formats()}).")
    p.add_argument("--work-dir", required=True,
                   help="Working directory for calibration artifacts.")
    p.add_argument("--assets-root", default=None,
                   help="Assets root for build_pptx_from_layout.")
    p.add_argument("--out-layout", default=None,
                   help="Output layout path. Default: update --layout in place.")
    p.add_argument("--iterations", type=int, default=1,
                   help="Font-size calibration iterations (default: 1).")
    p.add_argument("--dpi", type=int, default=100,
                   help="Calibration preview DPI (default: 100).")
    p.add_argument("--min-size", type=float, default=4.0,
                   help="Minimum font size in points.")
    p.add_argument("--max-size", type=float, default=72.0,
                   help="Maximum font size in points.")
    p.add_argument("--max-scale-step", type=float, default=0.06,
                   help="Maximum proportional size change per iteration.")
    p.add_argument("--min-ratio-change", type=float, default=0.012,
                   help="Ignore smaller proportional changes.")
    return p.parse_args()


def _build_pptx(layout_path: Path, out_path: Path,
                assets_root: Path | None) -> None:
    _builder.run(
        layout=str(layout_path),
        out=str(out_path),
        assets_root=str(assets_root) if assets_root else None,
    )


def _render_pptx(pptx_path: Path, out_dir: Path, dpi: int) -> None:
    _render.run(
        pptx=str(pptx_path), out_dir=str(out_dir),
        dpi=dpi, verbose=False,
    )


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def _clean_size(value: float) -> int:
    return int(round(float(value)))


def _snap_size(value: float, min_size: float, max_size: float) -> int:
    lo = int(math.ceil(float(min_size)))
    hi = int(math.floor(float(max_size)))
    if hi < lo:
        hi = lo
    return max(lo, min(hi, int(round(float(value)))))


def _size_ratio(el: dict[str, Any],
                target: list[int],
                rendered: tuple[float, float, float, float],
                max_scale_step: float,
                min_ratio_change: float) -> float:
    tx1, ty1, tx2, ty2 = (float(v) for v in target)
    rx1, ry1, rx2, ry2 = rendered
    target_w = max(1.0, tx2 - tx1)
    target_h = max(1.0, ty2 - ty1)
    render_w = max(1.0, rx2 - rx1)
    render_h = max(1.0, ry2 - ry1)

    height_ratio = _clamp(target_h / render_h, 0.35, 2.50)
    width_ratio = _clamp(target_w / render_w, 0.35, 2.50)

    text = str(el.get("text", "") or "").strip()
    visible_chars = sum(1 for c in text if not c.isspace())
    # Height usually best tracks font size. For very short labels and
    # digits, width carries more useful signal because the ink height can
    # be dominated by a single glyph's shape.
    width_weight = 0.12
    if visible_chars <= 4:
        width_weight = 0.24
    elif visible_chars <= 8:
        width_weight = 0.18
    if 0.92 <= width_ratio <= 1.08:
        width_weight *= 0.55
    if target_h <= 8 or render_h <= 8:
        width_weight = max(width_weight, 0.40)

    ratio = math.exp(
        (1.0 - width_weight) * math.log(height_ratio)
        + width_weight * math.log(width_ratio)
    )
    step = max(0.01, min(0.60, float(max_scale_step)))
    ratio = _clamp(ratio, 1.0 - step, 1.0 + step)
    if abs(ratio - 1.0) < float(min_ratio_change):
        return 1.0
    return ratio


def _probe_char(ch: str) -> bool:
    if not ch or ch.isspace():
        return False
    code = ord(ch)
    return ch.isalnum() or 0x4E00 <= code <= 0x9FFF


def _refined_mask_dims(mask: np.ndarray) -> tuple[int, int, int] | None:
    mask_u8 = mask.astype(np.uint8)
    if int(mask_u8.sum()) < 3:
        return None
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask_u8, 8)
    if count <= 1:
        ys, xs = np.where(mask_u8 > 0)
    else:
        areas = stats[1:, cv2.CC_STAT_AREA]
        if areas.size == 0:
            return None
        largest = int(areas.max())
        keep = np.zeros_like(mask_u8, dtype=bool)
        for label in range(1, count):
            area = int(stats[label, cv2.CC_STAT_AREA])
            if area >= max(2, int(round(largest * 0.08))):
                keep |= labels == label
        ys, xs = np.where(keep)
    if xs.size < 2 or ys.size < 2:
        return None
    return int(xs.max() - xs.min() + 1), int(ys.max() - ys.min() + 1), int(xs.size)


def _char_box_ink_sample(
    source_bgr: np.ndarray,
    box: list[int] | tuple[int, int, int, int],
    line_bg: np.ndarray,
) -> dict[str, Any] | None:
    h_img, w_img = source_bgr.shape[:2]
    x1, y1, x2, y2 = (int(v) for v in box)
    x1 = max(0, min(w_img, x1)); x2 = max(0, min(w_img, x2))
    y1 = max(0, min(h_img, y1)); y2 = max(0, min(h_img, y2))
    if x2 <= x1 or y2 <= y1:
        return None
    sub = source_bgr[y1:y2, x1:x2]
    if sub.size == 0:
        return None

    hsv = cv2.cvtColor(sub, cv2.COLOR_BGR2HSV)
    sat = hsv[:, :, 1] > 50
    local_bg = line_bg
    if int(sat.sum()) > 0.45 * sat.size:
        local_bg = np.median(sub[sat], axis=0)
    diff = np.abs(sub.astype(int) - local_bg.astype(int)).max(axis=2)
    mask = diff > 35

    if int(sat.sum()) > 0.45 * sat.size:
        bg_hsv = cv2.cvtColor(
            np.array([[local_bg]], dtype=np.uint8),
            cv2.COLOR_BGR2HSV,
        )[0, 0]
        light = (
            (hsv[:, :, 1] < 130)
            & (hsv[:, :, 2].astype(int) > int(bg_hsv[2]) + 25)
        )
        if int(light.sum()) >= 2:
            mask = light

    if int(mask.sum()) < 2:
        return None
    row_has_ink = mask.sum(axis=1) >= 1
    col_has_ink = mask.sum(axis=0) >= 1
    rows = np.where(row_has_ink)[0]
    cols = np.where(col_has_ink)[0]
    if rows.size < 2 or cols.size < 1:
        return None

    ix1 = int(cols.min())
    ix2 = int(cols.max()) + 1
    iy1 = int(rows.min())
    iy2 = int(rows.max()) + 1
    return {
        "bbox": [x1 + ix1, y1 + iy1, x1 + ix2, y1 + iy2],
        "width": int(ix2 - ix1),
        "height": int(iy2 - iy1),
        "area": int(mask[iy1:iy2, ix1:ix2].sum()),
    }


def _char_ink_samples(source_bgr: np.ndarray,
                      el: dict[str, Any]) -> dict[str, Any] | None:
    boxes = el.get("source_char_boxes") or el.get("char_boxes")
    if not boxes:
        return _connected_component_ink_samples(source_bgr, el)
    text = str(el.get("text", "") or "")
    chars = el.get("source_chars") or list(text)
    if len(chars) != len(boxes):
        return _connected_component_ink_samples(source_bgr, el)

    line_bbox = el.get("source_bbox")
    if not line_bbox or len(line_bbox) != 4:
        line_bbox = [
            min(int(b[0]) for b in boxes if len(b) == 4),
            min(int(b[1]) for b in boxes if len(b) == 4),
            max(int(b[2]) for b in boxes if len(b) == 4),
            max(int(b[3]) for b in boxes if len(b) == 4),
        ]
    line_bg = _sample_background_for_bbox(
        source_bgr, tuple(int(v) for v in line_bbox))

    samples: list[dict[str, Any]] = []
    inks: list[list[int]] = []
    for ch, box in zip(chars, boxes):
        if len(box) != 4:
            continue
        metrics = _char_box_ink_sample(source_bgr, box, line_bg)
        if metrics is None:
            continue
        width = int(metrics["width"])
        height = int(metrics["height"])
        area = int(metrics["area"])
        inks.append([int(v) for v in metrics["bbox"]])
        if _probe_char(str(ch)) and height >= 3 and width >= 2:
            samples.append({
                "char": str(ch),
                "width": width,
                "height": height,
                "area": area,
            })
    if not samples or not inks:
        return _connected_component_ink_samples(source_bgr, el)
    heights = np.array([s["height"] for s in samples], dtype=np.float32)
    widths = np.array([s["width"] for s in samples], dtype=np.float32)
    # Trim one-character outliers such as dots or OCR boxes that clipped a
    # glyph. Median is the sizing signal; the union remains available for
    # position calibration and reporting.
    med_h = float(np.median(heights))
    med_w = float(np.median(widths))
    full_ink = [
        min(b[0] for b in inks),
        min(b[1] for b in inks),
        max(b[2] for b in inks),
        max(b[3] for b in inks),
    ]
    return {
        "count": len(samples),
        "basis": "char_boxes",
        "median_h": med_h,
        "median_w": med_w,
        "ink_bbox": [int(round(float(v))) for v in full_ink[:4]],
    }


def _connected_component_ink_samples(source_bgr: np.ndarray,
                                     el: dict[str, Any]) -> dict[str, Any] | None:
    bbox = el.get("source_bbox")
    if not bbox or len(bbox) != 4:
        return None
    metrics = _target_text_metrics(source_bgr, tuple(int(v) for v in bbox))
    if metrics is None:
        return None
    mask = metrics["mask"].astype(np.uint8)
    if int(mask.sum()) < 3:
        return None
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    if count <= 1:
        return None

    line_h = max(1, int(metrics["ink_h"]))
    line_w = max(1, int(metrics["ink_w"]))
    text = str(el.get("text", "") or "")
    probe_chars = sum(1 for ch in text if _probe_char(ch))
    if probe_chars <= 0:
        return None

    samples: list[dict[str, Any]] = []
    keep_labels: list[int] = []
    total_area = int(mask.sum())
    for label in range(1, count):
        x = int(stats[label, cv2.CC_STAT_LEFT])
        y = int(stats[label, cv2.CC_STAT_TOP])
        w = int(stats[label, cv2.CC_STAT_WIDTH])
        h = int(stats[label, cv2.CC_STAT_HEIGHT])
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area < 2 or h < 3 or w < 1:
            continue
        # Reject nearby rules/borders that slipped into the OCR bbox.
        if w > max(8, line_w * 0.70) and len(text.strip()) > 1:
            continue
        if w > h * 3.8 and h <= max(5, line_h * 0.45):
            continue
        if h > w * 4.5 and w <= max(3, line_h * 0.25):
            continue
        if area > max(20, total_area * 0.45) and (
            w > line_h * 2.2 or h > line_h * 1.8
        ):
            continue
        comp = labels == label
        dims = _refined_mask_dims(comp)
        if dims is None:
            continue
        rw, rh, rarea = dims
        if rh < 3 or rw < 1:
            continue
        samples.append({"width": rw, "height": rh, "area": rarea})
        keep_labels.append(label)

    if not samples:
        return None
    heights = np.array([s["height"] for s in samples], dtype=np.float32)
    widths = np.array([s["width"] for s in samples], dtype=np.float32)
    med_h = float(np.median(heights))
    # Drop tiny punctuation/dust components. Punctuation follows the
    # nearest text because it does not participate in this size signal.
    filtered = [
        s for s in samples
        if s["height"] >= max(3.0, med_h * 0.45)
    ]
    if filtered:
        samples = filtered
        heights = np.array([s["height"] for s in samples], dtype=np.float32)
        widths = np.array([s["width"] for s in samples], dtype=np.float32)
    kept = np.isin(labels, keep_labels)
    ys, xs = np.where(kept)
    if xs.size < 2 or ys.size < 2:
        return None
    ix1, iy1, ix2, iy2 = (int(v) for v in metrics["ink_bbox"])
    return {
        "count": len(samples),
        "basis": "connected_components",
        # A slight upper percentile is more stable for Latin text, where
        # dots and narrow punctuation are separate small components.
        "median_h": float(np.percentile(heights, 60)),
        "median_w": float(np.median(widths)),
        "ink_bbox": [
            ix1 + int(xs.min()),
            iy1 + int(ys.min()),
            ix1 + int(xs.max()) + 1,
            iy1 + int(ys.max()) + 1,
        ],
    }


# Deck-level canvas size, stamped in main(). build_pptx_from_layout
# letterboxes every slide into this one canvas, so glyph metrics must be
# derived from the same scale — a slide's own aspect-derived width_in
# diverges from the deck canvas in mixed-aspect decks.
_DECK_SLIDE_SIZE: dict[str, Any] | None = None


def _slide_pt_per_px(slide: dict[str, Any]) -> float:
    source_w = float(slide.get("source_width") or 1280)
    source_h = float(slide.get("source_height") or 720)
    slide_size = _DECK_SLIDE_SIZE or slide.get("slide_size") or {}
    width_in = float(slide_size.get("width_in") or (source_w / 96.0))
    height_in = float(slide_size.get("height_in") or (source_h / 96.0))
    return max(0.01, min(width_in * 72.0 / max(1.0, source_w),
                         height_in * 72.0 / max(1.0, source_h)))


def _element_font_path(el: dict[str, Any]) -> str | None:
    font_name = str(el.get("font") or "Microsoft YaHei")
    style = "Bold" if el.get("bold") else "Regular"
    return (
        _fontconfig_font_path(f"{font_name}:style={style}")
        or _fontconfig_font_path(font_name)
        or _fontconfig_font_path("Microsoft YaHei")
    )


def _local_char_render_metrics(
    el: dict[str, Any],
    slide: dict[str, Any],
    size_override: int | None = None,
) -> dict[str, Any] | None:
    text = str(el.get("text") or "")
    chars = [ch for ch in text if _probe_char(ch)]
    if not chars:
        return None
    font_path = _element_font_path(el)
    if not font_path:
        return None
    size = float(size_override or el.get("size") or el.get("font_size") or 12.0)
    if size <= 0:
        return None
    pt_per_px = _slide_pt_per_px(slide)
    bold = bool(el.get("bold"))

    heights: list[float] = []
    widths: list[float] = []
    for ch in chars[:80]:
        metrics = _render_text_metrics(
            ch, font_path, int(round(size)), pt_per_px, bold)
        if not metrics:
            continue
        if metrics["ink_h"] >= 2 and metrics["ink_w"] >= 1:
            heights.append(float(metrics["ink_h"]))
            widths.append(float(metrics["ink_w"]))
    full = _render_text_metrics(
        text, font_path, int(round(size)), pt_per_px, bold)
    if not heights or not widths or not full:
        return None
    ascii_count = sum(1 for ch in chars if ord(ch) < 128)
    return {
        "median_h": float(np.median(np.array(heights, dtype=np.float32))),
        "median_w": float(np.median(np.array(widths, dtype=np.float32))),
        "full_w": float(full["ink_w"]),
        "full_h": float(full["ink_h"]),
        "ascii_ratio": ascii_count / max(1, len(chars)),
    }


def _best_integer_size_from_char_metrics(
    char_metrics: dict[str, Any],
    rendered: tuple[float, float, float, float],
    el: dict[str, Any],
    slide: dict[str, Any],
    min_size: float,
    max_size: float,
) -> int | None:
    old_size = int(round(float(el.get("size") or el.get("font_size") or 12.0)))
    lo = max(int(math.ceil(min_size)), old_size - 4)
    hi = min(int(math.floor(max_size)), old_size + 4)
    if hi < lo:
        return None

    target_box = char_metrics.get("ink_bbox") or [0, 0, 0, 0]
    target_full_w = max(1.0, float(target_box[2]) - float(target_box[0]))
    target_full_h = max(1.0, float(target_box[3]) - float(target_box[1]))
    target_h = max(1.0, float(char_metrics["median_h"]))
    target_w = max(1.0, float(char_metrics.get("median_w") or 1.0))

    current_local = _local_char_render_metrics(el, slide, old_size)
    if current_local is None:
        return None
    # Avoid point-size churn when the per-character height is already
    # close. OCR boxes often include extra horizontal spacing or line
    # padding; changing size for a sub-10% height delta tends to create
    # more visible drift than it fixes.
    current_h_ratio = float(current_local["median_h"]) / target_h
    current_full_h_ratio = float(current_local["full_h"]) / target_full_h
    current_full_w_ratio = float(current_local["full_w"]) / target_full_w
    if (
        current_full_w_ratio <= 1.08
        and (
            0.90 <= current_h_ratio <= 1.10
            or 0.90 <= current_full_h_ratio <= 1.08
        )
    ):
        return old_size
    rx1, ry1, rx2, ry2 = rendered
    actual_full_w = max(1.0, float(rx2) - float(rx1))
    actual_full_h = max(1.0, float(ry2) - float(ry1))
    w_factor = _clamp(actual_full_w / max(1.0, current_local["full_w"]),
                      0.70, 1.30)
    h_factor = _clamp(actual_full_h / max(1.0, current_local["full_h"]),
                      0.70, 1.30)

    ascii_heavy = float(current_local.get("ascii_ratio") or 0.0) >= 0.5
    basis = str(char_metrics.get("basis") or "")
    visible_chars = _visible_char_count(str(el.get("text") or ""))
    compact_label = visible_chars <= 6 and target_full_w <= 140
    # Font size should be inferred primarily from glyph height. Full-line
    # width varies with font metrics, OCR grouping, and textbox width; using
    # it as the dominant signal incorrectly shrinks long single-line text.
    if basis == "connected_components":
        width_weight = 0.08 if not ascii_heavy else 0.16
        height_weight = 0.84 if not ascii_heavy else 0.76
    else:
        width_weight = 0.06 if not ascii_heavy else 0.14
        height_weight = 0.86 if not ascii_heavy else 0.78
    if compact_label:
        width_weight = max(width_weight, 0.22 if not ascii_heavy else 0.30)
        height_weight = min(height_weight, 0.70 if not ascii_heavy else 0.62)
    median_width_weight = max(0.0, 1.0 - width_weight - height_weight)

    best: tuple[float, int] | None = None
    for size in range(lo, hi + 1):
        local = _local_char_render_metrics(el, slide, size)
        if local is None:
            continue
        pred_full_w = max(1.0, float(local["full_w"]) * w_factor)
        pred_full_h = max(1.0, float(local["full_h"]) * h_factor)
        pred_h = max(1.0, float(local["median_h"]) * h_factor)
        pred_w = max(1.0, float(local["median_w"]) * w_factor)
        full_w_err = abs(math.log(pred_full_w / target_full_w))
        full_h_err = abs(math.log(pred_full_h / target_full_h))
        h_err = abs(math.log(pred_h / target_h))
        w_err = abs(math.log(pred_w / target_w))
        score = (
            width_weight * full_w_err
            + height_weight * 0.72 * h_err
            + height_weight * 0.28 * full_h_err
            + median_width_weight * w_err
            + 0.03 * abs(size - old_size) / max(1.0, float(old_size))
        )
        over_limit = 1.35 if basis == "connected_components" else 1.40
        if compact_label:
            over_limit = 1.22 if not ascii_heavy else 1.28
        under_limit = 0.70 if basis == "connected_components" else 0.68
        if pred_full_w > target_full_w * over_limit:
            overflow_weight = 0.32 if compact_label else 0.06
            score += overflow_weight * (
                pred_full_w / (target_full_w * over_limit) - 1.0)
        # Compact and header-like single lines are visually bounded by their
        # source ink region. If the chosen size spills a few percent wider,
        # the tail can leave a coloured header/card and appear clipped even
        # though the glyph height is perfect.
        if (visible_chars <= 20
                and pred_full_w > target_full_w * 1.03):
            score += 2.4 * (
                pred_full_w / (target_full_w * 1.03) - 1.0)
        if pred_full_w < target_full_w * under_limit:
            score += 0.04 * (
                (target_full_w * under_limit) / pred_full_w - 1.0)
        candidate = (float(score), int(size))
        if best is None or candidate < best:
            best = candidate
    return best[1] if best is not None else None


def _char_size_ratio(char_metrics: dict[str, Any],
                     rendered: tuple[float, float, float, float],
                     max_scale_step: float,
                     min_ratio_change: float,
                     el: dict[str, Any],
                     slide: dict[str, Any]) -> float:
    rx1, ry1, rx2, ry2 = rendered
    actual_render_h = max(1.0, ry2 - ry1)
    actual_render_w = max(1.0, rx2 - rx1)
    render_h = actual_render_h
    render_w = actual_render_w
    source_h = max(1.0, float(char_metrics["median_h"]))
    source_w = max(1.0, float(char_metrics.get("median_w") or 1.0))
    target_box = char_metrics.get("ink_bbox") or [0, 0, 0, 0]
    target_full_w = max(1.0, float(target_box[2]) - float(target_box[0]))

    local = _local_char_render_metrics(el, slide)
    if local is not None:
        render_h = max(1.0, float(local["median_h"]))
        render_w = max(1.0, float(local["median_w"]))

    height_ratio = _clamp(source_h / render_h, 0.35, 2.50)
    width_ratio = _clamp(source_w / render_w, 0.35, 2.50)
    if 0.90 <= render_h / source_h <= 1.10:
        return 1.0
    basis = str(char_metrics.get("basis") or "")
    ascii_heavy = bool(local and float(local["ascii_ratio"]) >= 0.5)
    width_weight = 0.08
    if basis == "connected_components":
        width_weight = 0.12
    if ascii_heavy:
        width_weight = max(width_weight, 0.20)
    if source_h <= 8 or render_h <= 8:
        width_weight = max(width_weight, 0.22)
    ratio = math.exp(
        (1.0 - width_weight) * math.log(height_ratio)
        + width_weight * math.log(width_ratio)
    )

    target_full_h = max(1.0, float(target_box[3]) - float(target_box[1]))
    if actual_render_h > target_full_h * 1.18:
        ratio = min(ratio, target_full_h / actual_render_h)

    step = max(0.01, min(0.60, float(max_scale_step)))
    ratio = _clamp(ratio, 1.0 - step, 1.0 + step)
    if abs(ratio - 1.0) < float(min_ratio_change):
        return 1.0
    return ratio


def _total_size_bounds(el: dict[str, Any],
                       old_size: float,
                       min_size: float,
                       max_size: float) -> tuple[float, float, float]:
    existing = el.get("size_calibration") or {}
    initial = float(existing.get("initial_size", old_size) or old_size)
    text = str(el.get("text", "") or "")
    visible_chars = sum(1 for c in text if not c.isspace())
    # Source ink bboxes for compact labels are often contaminated by
    # nearby rules/icons. Let calibration nudge those, but do not let a
    # tiny chip label balloon over multiple iterations.
    if old_size <= 14 or visible_chars <= 8:
        max_total = 1.08
    elif visible_chars <= 16:
        max_total = 1.12
    else:
        max_total = 1.18
    min_total = 0.82
    return (
        initial,
        max(min_size, initial * min_total),
        min(max_size, initial * max_total),
    )


def _scale_element_size(el: dict[str, Any],
                        ratio: float,
                        min_size: float,
                        max_size: float,
                        constrained: bool = True) -> tuple[float, float, bool]:
    old_size = float(el.get("size") or el.get("font_size") or 12.0)
    if constrained:
        _initial_size, min_allowed, max_allowed = _total_size_bounds(
            el, old_size, min_size, max_size)
    else:
        min_allowed, max_allowed = min_size, max_size
    new_size = _snap_size(old_size * ratio, min_allowed, max_allowed)
    actual_ratio = float(new_size) / old_size if old_size else 1.0
    changed = int(round(old_size)) != int(new_size)
    el["size"] = new_size

    for run in el.get("runs") or []:
        if run.get("size") is None:
            continue
        old_run = float(run["size"])
        new_run = _snap_size(old_run * actual_ratio, min_size, max_size)
        if int(round(old_run)) != int(new_run):
            changed = True
        run["size"] = new_run
    return old_size, float(new_size), changed


def _visible_char_count(text: str) -> int:
    return sum(1 for ch in text if not ch.isspace())


def _width_char_spacing(
    el: dict[str, Any],
    slide: dict[str, Any],
    old_size: float,
    new_size: float,
    target: list[int],
    rendered: tuple[float, float, float, float],
) -> int | None:
    text = str(el.get("text") or "")
    if "\n" in text:
        return None
    visible = _visible_char_count(text)
    compact_label = 2 <= visible <= 8
    if visible < 24 and not compact_label:
        return None
    tx1, _ty1, tx2, _ty2 = (float(v) for v in target)
    rx1, _ry1, rx2, _ry2 = rendered
    target_w = max(1.0, tx2 - tx1)
    if target_w < 300.0 and not compact_label:
        return None
    if float(new_size) > float(old_size):
        return None
    rendered_w = max(1.0, rx2 - rx1)
    scaled_w = rendered_w * (float(new_size) / max(1.0, float(old_size)))
    overflow = scaled_w - target_w
    min_overflow = max(8.0, target_w * 0.06)
    if compact_label:
        min_overflow = max(6.0, target_w * 0.12)
    if overflow < min_overflow:
        return None
    # DrawingML run property `spc` is an em-relative tracking value.
    # LibreOffice's rendered effect is about 70% of the nominal em
    # spacing for mixed CJK/Latin text, so use a conservative conversion
    # and let later position calibration measure the result.
    font_px = float(new_size) / max(0.01, _slide_pt_per_px(slide))
    gaps = max(1, visible - 1)
    denom = max(1.0, font_px * gaps * 0.70)
    spacing = int(round(-overflow * 1000.0 / denom))
    if compact_label:
        spacing = int(round(spacing * 1.8))
    lower_bound = -320 if compact_label else -160
    spacing = max(lower_bound, min(-10, spacing))
    return spacing


def _record_size_calibration(el: dict[str, Any],
                             iteration: int,
                             old_size: float,
                             new_size: float,
                             ratio: float,
                             target: list[int],
                             rendered: tuple[float, float, float, float],
                             basis: str) -> None:
    existing = el.get("size_calibration") or {}
    previous_source = existing.get("previous_size_source",
                                   el.get("size_source"))
    initial_size = existing.get("initial_size", old_size)
    el["size_source"] = "preview_calibrated"
    el["size_calibration"] = {
        "iteration": iteration,
        "previous_size_source": previous_source,
        "initial_size": _clean_size(float(initial_size)),
        "old_size": _clean_size(old_size),
        "new_size": _clean_size(new_size),
        "scale": round(float(ratio), 4),
        "basis": basis,
        "target_ink": [int(round(v)) for v in target],
        "rendered_ink": [round(float(v), 3) for v in rendered],
    }
    if el.get("char_spacing") is not None:
        el["size_calibration"]["char_spacing"] = int(el["char_spacing"])


def _apply_iteration(layout: dict[str, Any],
                     colours: dict[tuple[int, int], tuple[int, int, int]],
                     preview_dir: Path,
                     source_dir: Path,
                     iteration: int,
                     min_size: float,
                     max_size: float,
                     max_scale_step: float,
                     min_ratio_change: float) -> list[dict[str, Any]]:
    report: list[dict[str, Any]] = []
    slides = layout.get("slides") or [layout]
    for s_idx, slide in enumerate(slides):
        slide_colours = [
            c for (slide_no, _text_no), c in colours.items()
            if slide_no == s_idx
        ]
        preview_path = _preview_for_slide(preview_dir, s_idx)
        preview_bgr = cv2.imread(str(preview_path)) if preview_path else None
        source_path = _source_for_slide(slide, source_dir, s_idx)
        source_bgr = cv2.imread(str(source_path)) if source_path else None
        if preview_bgr is None or source_bgr is None:
            continue
        preview_rgb = cv2.cvtColor(preview_bgr, cv2.COLOR_BGR2RGB)
        scale, ox, oy = _source_to_preview_transform(slide, preview_rgb.shape[:2])
        for t_idx, el in enumerate(_text_elements(slide)):
            colour = colours.get((s_idx, t_idx))
            if colour is None or el.get("size") is None:
                continue
            px1, py1, px2, py2 = _box_to_preview(el["box"], scale, ox, oy)
            box_w = max(1, px2 - px1)
            box_h = max(1, py2 - py1)
            mx = int(round(max(36, min(180, box_w * 0.45))))
            my = int(round(max(36, min(140, box_h * 3.2))))
            detected = _detect_colour_bbox(
                preview_rgb, colour,
                (px1 - mx, py1 - my, px2 + mx, py2 + my),
                (px1, py1, px2, py2),
                competing_colours=slide_colours,
            )
            target = _target_ink_for_element(el, source_bgr)
            if detected is None or target is None:
                report.append({
                    "slide": s_idx + 1,
                    "iteration": iteration,
                    "name": el.get("name"),
                    "text": el.get("text"),
                    "status": "skipped",
                })
                continue
            rendered = (
                (detected[0] - ox) / scale,
                (detected[1] - oy) / scale,
                (detected[2] - ox) / scale,
                (detected[3] - oy) / scale,
            )
            char_metrics = _char_ink_samples(source_bgr, el)
            if char_metrics is not None:
                candidate_size = _best_integer_size_from_char_metrics(
                    char_metrics, rendered, el, slide, min_size, max_size)
                if candidate_size is not None:
                    old_size_for_ratio = float(
                        el.get("size") or el.get("font_size") or 12.0)
                    ratio = candidate_size / max(1.0, old_size_for_ratio)
                    if abs(ratio - 1.0) < min_ratio_change:
                        ratio = 1.0
                    basis = f"{char_metrics.get('basis') or 'char_boxes'}_integer_candidates"
                else:
                    ratio = _char_size_ratio(
                        char_metrics, rendered, max_scale_step,
                        min_ratio_change, el, slide)
                    basis = str(char_metrics.get("basis") or "char_boxes")
                target = [int(v) for v in char_metrics["ink_bbox"]]
                constrained = False
            else:
                ratio = _size_ratio(
                    el, target, rendered, max_scale_step, min_ratio_change)
                basis = "bbox"
                constrained = True
            old_size, new_size, changed = _scale_element_size(
                el, ratio, min_size, max_size, constrained=constrained)
            spacing_delta = _width_char_spacing(
                el, slide, old_size, new_size, target, rendered)
            if spacing_delta is not None:
                old_spacing = int(round(float(el.get("char_spacing") or 0)))
                new_spacing = max(-160, min(80, old_spacing + spacing_delta))
                if new_spacing != old_spacing:
                    el["char_spacing"] = new_spacing
                    changed = True
            _record_size_calibration(
                el, iteration, old_size, new_size, ratio, target, rendered,
                basis)

            tx1, ty1, tx2, ty2 = (float(v) for v in target)
            box = el.get("box") or [0, 0, 0, 0]
            if (el.get("align") or "").lower() != "center":
                render_w = max(1.0, float(rendered[2]) - float(rendered[0]))
                scale_after = float(new_size) / max(1.0, float(old_size))
                needed_w = max(tx2 - tx1, render_w * scale_after)
                box[2] = int(round(max(float(box[2]), needed_w + 12)))
            box[3] = int(round(max(float(box[3]), (ty2 - ty1) + 8)))

            report.append({
                "slide": s_idx + 1,
                "iteration": iteration,
                "name": el.get("name"),
                "text": el.get("text"),
                "status": "calibrated",
                "old_size": _clean_size(old_size),
                "new_size": _clean_size(new_size),
                "scale": round(float(ratio), 4),
                "basis": basis,
                "changed": changed,
            })
    return report


def main() -> int:
    args = parse_args()
    layout_path = Path(args.layout)
    source_dir = Path(args.source_dir)
    work_dir = Path(args.work_dir)
    assets_root = Path(args.assets_root) if args.assets_root else work_dir
    out_layout = Path(args.out_layout) if args.out_layout else layout_path
    debug_dir = work_dir / "debug" / "text_size_calibration"
    debug_dir.mkdir(parents=True, exist_ok=True)

    layout = json.loads(layout_path.read_text(encoding="utf-8-sig"))
    global _DECK_SLIDE_SIZE
    _DECK_SLIDE_SIZE = layout.get("slide_size") or None
    all_reports: list[dict[str, Any]] = []

    for iteration in range(1, max(1, int(args.iterations)) + 1):
        iter_dir = debug_dir / f"iter_{iteration:02d}"
        iter_dir.mkdir(parents=True, exist_ok=True)
        cal_layout, colours = _calibration_layout(copy.deepcopy(layout))
        cal_layout_path = iter_dir / "calibration.layout.json"
        cal_pptx_path = iter_dir / "calibration.pptx"
        cal_layout_path.write_text(
            json.dumps(cal_layout, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        _build_pptx(cal_layout_path, cal_pptx_path, assets_root)
        _render_pptx(cal_pptx_path, iter_dir, args.dpi)
        report = _apply_iteration(
            layout, colours, iter_dir / "previews", source_dir, iteration,
            float(args.min_size), float(args.max_size),
            float(args.max_scale_step), float(args.min_ratio_change),
        )
        all_reports.extend(report)

    out_layout.write_text(json.dumps(layout, ensure_ascii=False, indent=2),
                          encoding="utf-8")
    report_path = debug_dir / "size_calibration_report.json"
    report_path.write_text(json.dumps(all_reports, ensure_ascii=False, indent=2),
                           encoding="utf-8")
    changed = sum(1 for r in all_reports
                  if r.get("status") == "calibrated" and r.get("changed"))
    print(json.dumps({
        "layout": str(out_layout),
        "report": str(report_path),
        "records": len(all_reports),
        "changed": changed,
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
