#!/usr/bin/env python
"""Erase text regions from a slide image, preserving icons/logos.

Reads OCR JSON (from ocr_paddle.py) and writes a "cleaned" image where
ordinary text regions are replaced with their local background color
via solid fill (not blur).

Three OCR results are PRESERVED (not erased) — they're likely part of
icons/logos:
  1. Likely-icon misclassifications (low confidence single chars,
     high-edge-density 1-2 char detections, square colored bg).
  2. Text inside small square colored regions (text labels INSIDE
     icons, e.g. "数据" + binary code in a highlighted card).
  3. Text inside logo strips (>=4 short texts in same y-band, in
     bottom 15 % of the slide).

Background fill uses SIDE-AGREEMENT voting: each side
(top/bot/left/right) gets ONE vote (not pixel-weighted). Avoids the
bug where a wide white area above a colored banner outweighs the
banner color from the other 3 sides.

File layout (search the section banners):
  - CLI: parse_args
  - OCR pre-processing: preserve / logo / trim helpers, preprocess_ocr
  - Icon-vs-text heuristics: is_likely_icon and stroke-component filter
  - Background colour fill: _strip_bg_median, fill_color
  - Erasure (main work): erase_text
  - Icon-review packet I/O: _load_icon_decisions, _dump_icon_review_packet, …
  - Driver: main

Usage:
    python erase_text.py --image slide.png --ocr ocr.json --out clean.png
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import cv2
import numpy as np


# =============================================================================
# CLI
# =============================================================================


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Erase text from slide image.")
    parser.add_argument("--image", required=True, help="Source image path.")
    parser.add_argument("--ocr", required=True, help="OCR JSON path (from ocr_vision.swift).")
    parser.add_argument("--out", required=True, help="Output cleaned image path.")
    parser.add_argument("--debug-dir", help="Optional debug output directory. Writes "
                                            "<stem>_mask.png, <stem>_clean.png, "
                                            "<stem>_compare.png (side-by-side).")
    parser.add_argument(
        "--icon-review-dump",
        metavar="DIR",
        help="Emit an LLM icon-review packet alongside the normal "
             "outputs: DIR/icon_review.json + DIR/crops/*.png + "
             "DIR/contact.png. Each entry holds an icon-vs-text decision "
             "the heuristic made in a gray zone (low_conf_short_text, "
             "isolated_single_char, outlined_icon_glyph, "
             "square_edge_icon, …) so an agent can second-guess each "
             "call. Skipped silently when no ambiguous decisions fire.")
    parser.add_argument(
        "--icon-decisions",
        metavar="JSON",
        help="Path to an icon_review.json filled in by the reviewer. "
             "Entries whose `decision` field is `icon` or `text` "
             "override the heuristic for that bbox; entries with "
             "`decision` left as null fall back to the heuristic.")
    return parser.parse_args()


# =============================================================================
# OCR-item heuristics: imported from a shared module so build_inventory.py and
# run_pipeline.py see the same icon/logo/preprocess verdicts that erase_text
# uses internally. The actual rule code lives in _heuristics.py.
# =============================================================================

from _heuristics import (  # noqa: E402 — sys.path arranged at runtime via subdir layout
    DEFINITIVE_ICON_REASONS,
    detect_logo_strips,
    is_in_logo_zone,
    is_likely_icon,
    preprocess_ocr,
    s_area,
    s_length,
    should_preserve_visual,
    split_icon_prefix,
    trim_ocr_bbox_off_trailing_visual,
)



def _filter_short_text_stroke_components(
        stroke: np.ndarray,
        text: str,
        scale: float = 1.0) -> tuple[np.ndarray, list[dict]]:
    """Remove icon/card-border components accidentally caught by short OCR.

    PaddleOCR sometimes returns `16小时` with a clock icon inside the same
    bbox, or `7分钟` with a few clock pixels caught at the left edge. Those
    marks are small connected components spatially separate from the text
    run. Keep the text strokes, drop edge dividers and square icon blobs.

    Returns (filtered_mask, review_records). `review_records` lists edge
    components in the `square_edge_icon` decision band (kept-as-text and
    removed-as-icon) — these are the calls the LLM icon-review packet
    second-guesses. Each record is a dict with rel_bbox `[x, y, w, h]` in
    the stroke-mask's local coords, a `reason` tag, and the current
    `verdict` (`icon` = removed from mask, will be preserved in image;
    `text` = kept in mask, will be erased).
    """
    if stroke.size == 0 or len(text) > 5:
        return stroke, []
    h, w = stroke.shape[:2]
    # Pixel constants below were tuned at 720-tall sources; route them
    # through the scale helpers so high-DPI inputs keep the same
    # physical behaviour (digit strokes at 1440p exceed the raw 75 px
    # icon cap and the rules silently stop firing).
    big_area = s_area(100, scale)
    small_area = s_area(80, scale)
    dust_area = s_area(8, scale)
    edge_px = max(3, int(round(s_length(5, scale))))
    narrow_w = max(2, int(round(s_length(4, scale))))
    icon_min_side = s_length(16, scale)
    icon_max_side = s_length(75, scale)
    near_px = max(3, int(round(s_length(4, scale))))
    n, labels, stats, _ = cv2.connectedComponentsWithStats(
        stroke.astype(np.uint8) * 255, 8)
    if n <= 1:
        return stroke, []

    keep = np.zeros(n, dtype=bool)
    keep[0] = False
    removed = np.zeros(n, dtype=bool)
    review_records: list[dict] = []
    # Size baseline from inner (non-edge) components, used below to tell a
    # trailing/leading unit suffix (`个`, `B`, `%`, rendered smaller than
    # the digits they follow) apart from an adjacent pictogram of digit-
    # comparable height (`5分钟` + clock).
    inner_max_h = 0
    for i in range(1, n):
        x, y, ww, hh, area = (int(v) for v in stats[i])
        if area >= big_area and not (x <= edge_px or x + ww >= w - edge_px):
            if hh > inner_max_h:
                inner_max_h = hh
    has_digit = any(c.isdigit() for c in text)
    # Pure-digit text like "13", "18", "20" has digit strokes that may
    # land near the bbox edges and look square. Don't treat any of those
    # strokes as an embedded icon — they're part of the number. The
    # square_edge_icon rule was for mixed icon+text (e.g. "5分钟" with a
    # clock pictogram at one edge), so a non-digit anchor still needs to
    # be in the text for it to fire.
    compact_text = "".join(c for c in text if not c.isspace())
    has_non_digit_anchor = any(
        not c.isdigit() and not c in ".,%·～~/+-:：()（）"
        for c in compact_text
    )
    # Direction-aware digit-trailing check. The original square_edge_icon
    # rule was meant for OCR bboxes that pull in an adjacent pictogram
    # (e.g. clock icon next to `5分钟`). Icons sit at one edge, distinct
    # from the text. When the OCR text itself ENDS with digits (e.g.
    # `最高229`), the trailing digit strokes look square-ish and live at
    # the right edge — the rule fires on them and leaves the digits
    # baked into the cleaned image. Similarly, a number followed by a
    # unit (`8.2倍`, `15%`) keeps digits at the left/middle, not edges,
    # so this only matters when digits are AT the bbox edge.
    text_trailing_digit = bool(compact_text and compact_text[-1].isdigit())
    text_leading_digit = bool(compact_text and compact_text[0].isdigit())
    component_boxes: list[tuple[int, int, int, int, int, int]] = []
    for i in range(1, n):
        x, y, ww, hh, area = (int(v) for v in stats[i])
        if area < dust_area:
            removed[i] = True
            continue
        aspect = ww / max(1, hh)
        cx = x + ww / 2.0
        touches_side = x <= edge_px or x + ww >= w - edge_px
        narrow_edge_rule = (touches_side and ww <= narrow_w
                            and hh >= 0.45 * h)
        tiny_edge_dust = touches_side and area < small_area
        # If the text trails with digits AND the suspect sits at the right
        # edge, it's likely the trailing digit run — erase, don't preserve.
        # Mirror for leading-digit text and left edge. Together with the
        # has_non_digit_anchor guard these cover the common stat-callout
        # patterns (`最高229`, `+ 229`, `△23`, `99.4%`).
        suspect_at_right = cx >= 0.62 * w
        suspect_at_left = cx <= 0.38 * w
        suspect_likely_digit = (
            (suspect_at_right and text_trailing_digit)
            or (suspect_at_left and text_leading_digit)
        )
        # Edge component at or below the inner-text height is almost
        # certainly a trailing/leading text glyph (CJK unit suffix like
        # `个`, Latin unit like `B` in `29 PB`), not an adjacent icon.
        # Real pictograms in stat callouts are sized to match the
        # digits, so a clearly-smaller edge blob is text. Without an
        # inner baseline (every component is at an edge, e.g. trimmed
        # bbox so tight that `5` and `24个` flank the bbox) the rule
        # has no way to tell pictogram from text and must bail —
        # otherwise it eats `24个` whenever the OCR run happens to
        # have only two connected components.
        if inner_max_h <= 0:
            suspect_text_sized = True
        else:
            suspect_text_sized = hh <= inner_max_h * 1.05
        square_edge_icon = (
            has_digit
            and has_non_digit_anchor
            and not suspect_likely_digit
            and not suspect_text_sized
            and 0.65 <= aspect <= 1.55
            and icon_min_side <= min(ww, hh)
            and max(ww, hh) <= icon_max_side
            and area >= big_area
            and (suspect_at_left or suspect_at_right)
        )
        # Gray-zone flag for LLM review: this component would have fired
        # `square_edge_icon` except the trailing-text-sized guard held it
        # back. (Left-edge digits in digit-leading text and right-edge
        # digits in digit-trailing text are excluded by suspect_likely_
        # digit — those are unambiguously text and not worth reviewing.)
        square_shape_at_edge = (
            has_digit
            and has_non_digit_anchor
            and not suspect_likely_digit
            and suspect_text_sized
            and 0.65 <= aspect <= 1.55
            and icon_min_side <= min(ww, hh)
            and max(ww, hh) <= icon_max_side
            and area >= big_area
            and (suspect_at_left or suspect_at_right)
        )
        if narrow_edge_rule or tiny_edge_dust or square_edge_icon:
            removed[i] = True
            if square_edge_icon:
                review_records.append({
                    "rel_bbox": [x, y, ww, hh],
                    "reason": "square_edge_icon",
                    "verdict": "icon",
                })
            continue
        if square_shape_at_edge and not square_edge_icon:
            review_records.append({
                "rel_bbox": [x, y, ww, hh],
                "reason": "trailing_text_sized_at_edge",
                "verdict": "text",
            })
        keep[i] = True
        component_boxes.append((x, y, x + ww, y + hh, area, i))

    # Once the main text run is known, drop tiny components floating well
    # outside it. This removes clock/chevron flecks while preserving dots
    # and antialias fragments inside numbers such as `20.26%`.
    anchors = [b for b in component_boxes if b[4] >= small_area]
    if anchors:
        ax1 = min(b[0] for b in anchors)
        ax2 = max(b[2] for b in anchors)
        for x1, _y1, x2, _y2, area, i in component_boxes:
            if area >= small_area:
                continue
            if x2 < ax1 - near_px or x1 > ax2 + near_px:
                removed[i] = True
                keep[i] = False

    out = np.zeros_like(stroke, dtype=np.uint8)
    for i in range(1, n):
        if keep[i] and not removed[i]:
            out[labels == i] = 1
    # If the filter somehow removed everything, keep the original text mask
    # rather than leaving text unerased.
    if int(out.sum()) < 4:
        return stroke, review_records
    return out, review_records


# =============================================================================
# Background-colour fill
#
# Each side of a text bbox votes once with its median (after dropping
# pixels that look like text strokes leaking into the sample). The mode
# of the four votes wins — see fill_color.
# =============================================================================


def _strip_bg_median(strip: np.ndarray) -> np.ndarray | None:
    """Median of a thin ring strip, after excluding pixels that look like
    text strokes leaking into the sample. The leak detection assumes the
    bg colour is the majority colour in the strip: quantise to 16-step
    bins, take the modal bin, then keep only pixels within ±25 per channel
    of the mode. Median of the surviving pixels is the clean bg colour."""
    if strip.size == 0:
        return None
    pix = strip.reshape(-1, 3)
    if len(pix) == 0:
        return None
    quant = (pix // 16) * 16
    counts = Counter(map(tuple, quant))
    mode_q = np.array(counts.most_common(1)[0][0])
    diff = np.abs(pix.astype(int) - mode_q).max(axis=1)
    close = pix[diff <= 25]
    if len(close) >= max(3, int(len(pix) * 0.2)):
        return np.median(close, axis=0)
    return np.median(pix, axis=0)


def fill_color(img: np.ndarray, x1: int, y1: int, x2: int, y2: int) -> np.ndarray:
    """Pick the local background color used to overwrite the text bbox.

    Default: side-agreement voting on the 2-px ring just outside the bbox.
    Each side (top/bot/left/right) reports its own background-only median
    (anti-aliased text strokes that bleed across the bbox boundary are
    excluded — see `_strip_bg_median`). The mode across sides wins.

    Special case — text inside a tight colored container (e.g. `Rhombus`
    on a dark-blue rounded badge whose bbox almost exactly matches the
    OCR text bbox): the outer ring lands OUTSIDE the badge on the white
    slide, so plain side-agreement would erase the badge to white. Detect
    this by sampling INSIDE the bbox: if a clear majority of pixels
    (>55 %) have a saturated colour distinct from the outer ring colour,
    that saturated cluster IS the container fill — use it.
    """
    h, w = img.shape[:2]
    sides: dict[str, np.ndarray] = {}
    if y1 >= 2:
        m = _strip_bg_median(img[y1 - 2:y1, x1:x2])
        if m is not None:
            sides["top"] = m
    if y2 + 2 <= h:
        m = _strip_bg_median(img[y2:y2 + 2, x1:x2])
        if m is not None:
            sides["bot"] = m
    if x1 >= 2:
        m = _strip_bg_median(img[y1:y2, x1 - 2:x1])
        if m is not None:
            sides["left"] = m
    if x2 + 2 <= w:
        m = _strip_bg_median(img[y1:y2, x2:x2 + 2])
        if m is not None:
            sides["right"] = m

    # Aspect-aware preference: a wide bbox (3:1 +) is usually text sitting
    # inside a horizontal band. Its top/bottom rings sample outside the
    # band, but its left/right rings sample inside the band (the real bg
    # behind the text). Weight left/right over top/bottom when the bbox is
    # wide, and vice versa when tall.
    bbox_w, bbox_h = x2 - x1, y2 - y1
    pairs = []
    if bbox_w >= bbox_h * 2.5:
        if "left" in sides:  pairs.append(sides["left"])
        if "right" in sides: pairs.append(sides["right"])
    elif bbox_h >= bbox_w * 2.5:
        if "top" in sides:   pairs.append(sides["top"])
        if "bot" in sides:   pairs.append(sides["bot"])
    if pairs and len(pairs) >= 2:
        a, b = pairs[0], pairs[1]
        if int(np.max(np.abs(a.astype(int) - b.astype(int)))) <= 25:
            # The two ring sides aligned with the bbox's long axis agree —
            # they ARE the local bg. Trust them.
            ring_color = np.mean([a, b], axis=0).astype(np.uint8)
        else:
            ring_color = None
    else:
        ring_color = None

    if ring_color is None:
        side_colors = list(sides.values())
        if not side_colors:
            ring_color = np.array([255, 255, 255], dtype=np.uint8)
        else:
            quantized = [tuple(((c // 16) * 16).astype(int)) for c in side_colors]
            votes = Counter(quantized)
            winner_q, _ = votes.most_common(1)[0]
            close = [c for c in side_colors if max(abs(c - np.array(winner_q))) < 25]
            if close:
                ring_color = np.median(close, axis=0).astype(np.uint8)
            else:
                ring_color = np.array(winner_q, dtype=np.uint8)

    # Special case: tight colored container around the text.
    inner = img[y1:y2, x1:x2]
    if inner.size == 0:
        return ring_color
    hsv = cv2.cvtColor(inner, cv2.COLOR_BGR2HSV)
    saturated_mask = hsv[:, :, 1] > 50
    total = inner.shape[0] * inner.shape[1]
    # Threshold 0.55 distinguishes "saturated text on neutral bg" (~20-30 %
    # saturated, leave alone) from "saturated container fill with light text"
    # (~60-80 % saturated, treat container as the local bg).
    is_container_like = total > 0 and int(saturated_mask.sum()) > 0.55 * total
    if is_container_like:
        # Dense CJK text (e.g. `网络运维智能体` — 7 bold glyphs in a tight
        # PP-OCRv5 bbox) can also exceed 55 % saturated coverage. Distinguish
        # by topology: a real container fill is ONE big saturated blob; dense
        # strokes are many smaller blobs. If the largest saturated component
        # covers a clear majority of saturated pixels, accept as container;
        # otherwise stick with the ring sample. Without this guard, erase
        # picks the dark text-stroke colour as the bg and paints a solid bar
        # over the text region in the cleaned image.
        n_comp, _, stats, _ = cv2.connectedComponentsWithStats(
            saturated_mask.astype(np.uint8) * 255, 8)
        if n_comp > 1:
            largest_area = int(stats[1:, cv2.CC_STAT_AREA].max())
            total_sat = int(saturated_mask.sum())
            if largest_area < 0.55 * total_sat:
                is_container_like = False
    if is_container_like:
        # A single dense CJK/Latin stroke run can still pass the topology
        # guard when adjacent glyphs touch. Real container fill reaches the
        # text bbox edges; glyph strokes leave at least some edge bands
        # mostly background. Require saturated coverage on 3+ bbox sides
        # before overriding the side-ring colour.
        band = max(1, min(inner.shape[:2]) // 6)
        edge_coverages = (
            float(saturated_mask[:band, :].mean()),
            float(saturated_mask[-band:, :].mean()),
            float(saturated_mask[:, :band].mean()),
            float(saturated_mask[:, -band:].mean()),
        )
        if sum(v >= 0.60 for v in edge_coverages) < 3:
            is_container_like = False
    if is_container_like:
        sat_pixels = inner[saturated_mask]
        container = np.median(sat_pixels, axis=0).astype(np.uint8)
        # Only override the ring colour when the container is clearly
        # different from the ring (otherwise the bbox sits on a normal
        # colored background and the ring already returned the right hue).
        if int(np.max(np.abs(container.astype(int) - ring_color.astype(int)))) > 40:
            return container
    return ring_color


def _ocr_guard_boxes(item: dict) -> list[tuple[int, int, int, int]]:
    """Return OCR sub-boxes that tightly describe the text geometry.

    PaddleOCR can emit per-character/per-word boxes. Those are much safer
    erasure bounds than the line bbox when a nearby connector, card edge,
    or dashed border shares the text colour and happens to fall inside
    the line bbox.
    """
    for key in ("char_boxes", "word_boxes"):
        raw_boxes = item.get(key)
        if not isinstance(raw_boxes, list) or not raw_boxes:
            continue
        boxes: list[tuple[int, int, int, int]] = []
        for raw in raw_boxes:
            if not isinstance(raw, (list, tuple)) or len(raw) != 4:
                continue
            try:
                bx1, by1, bx2, by2 = (int(round(float(v))) for v in raw)
            except (TypeError, ValueError):
                continue
            if bx2 <= bx1 or by2 <= by1:
                continue
            boxes.append((bx1, by1, bx2, by2))
        if boxes:
            return boxes
    return []


def _text_geometry_guard_mask(
    item: dict,
    x1: int,
    y1: int,
    x2: int,
    y2: int,
    scale: float,
) -> np.ndarray | None:
    boxes = _ocr_guard_boxes(item)
    if not boxes:
        return None
    h, w = y2 - y1, x2 - x1
    if h <= 0 or w <= 0:
        return None
    guard = np.zeros((h, w), dtype=bool)
    heights = [by2 - by1 for _bx1, by1, _bx2, by2 in boxes]
    median_h = float(np.median(heights)) if heights else float(h)
    # Paddle word boxes can be horizontally tight on bold Latin/CJK runs
    # (notably mixed `NEV让...` titles). Use a wider x guard so antialias
    # and overhanging strokes are erased; keep y padding more conservative
    # to avoid eating same-colour horizontal rules above/below the line.
    pad_x = max(1, int(round(2 * scale)), int(round(0.35 * median_h)))
    pad_y = max(1, int(round(2 * scale)), int(round(0.18 * median_h)))
    for bx1, by1, bx2, by2 in boxes:
        gx1 = max(0, bx1 - x1 - pad_x)
        gy1 = max(0, by1 - y1 - pad_y)
        gx2 = min(w, bx2 - x1 + pad_x)
        gy2 = min(h, by2 - y1 + pad_y)
        if gx2 > gx1 and gy2 > gy1:
            guard[gy1:gy2, gx1:gx2] = True
    if int(guard.sum()) < 4:
        return None
    return guard


def _drop_thin_non_text_row_bands(stroke: np.ndarray, text: str) -> np.ndarray:
    """Remove thin horizontal decorations accidentally classified as text.

    Blue connector dashes and card borders can sit in the same OCR line
    bbox as blue text. They form a very short row band separated from the
    taller glyph band, while real multi-line text forms bands of similar
    height. This keeps the taller text bands and drops only wide, thin
    extras.
    """
    compact = "".join(c for c in (text or "") if c.strip())
    if len(compact) < 3 or stroke.size == 0:
        return stroke
    h, w = stroke.shape[:2]
    if h < 8 or w < 8:
        return stroke

    rows = (stroke.sum(axis=1) > 0).astype(np.uint8)[:, None]
    rows = cv2.morphologyEx(rows, cv2.MORPH_CLOSE,
                            np.ones((3, 1), np.uint8))[:, 0] > 0
    bands: list[tuple[int, int]] = []
    start: int | None = None
    for idx, has_ink in enumerate(rows):
        if has_ink and start is None:
            start = idx
        elif not has_ink and start is not None:
            bands.append((start, idx))
            start = None
    if start is not None:
        bands.append((start, len(rows)))
    if len(bands) < 2:
        return stroke

    heights = [end - start for start, end in bands]
    major_h = max(heights)
    if major_h < max(6, int(round(0.25 * h))):
        return stroke

    out = stroke.copy()
    thin_limit = max(3, int(round(0.32 * major_h)))
    for start, end in bands:
        band_h = end - start
        if band_h > thin_limit:
            continue
        ys, xs = np.where(stroke[start:end] > 0)
        if len(xs) == 0:
            continue
        span = int(xs.max() - xs.min() + 1)
        if span >= max(16, int(round(0.18 * w))):
            out[start:end, :] = 0
    return out


# =============================================================================
# Erasure (the main work)
#
# erase_text walks every OCR item, classifies it, and either fills its
# bbox with the local background colour or preserves the underlying
# pixels. Also records the icon/text decision per item for the LLM
# review packet.
# =============================================================================


def erase_text(
        img: np.ndarray,
        ocr_data: list[dict],
        overrides: dict | None = None,
) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    """Replace text STROKE pixels (not the whole bbox) with the local bg
    colour.

    OCR bboxes are tight on the glyphs but often clip a pixel or two of
    adjacent content — a flat bbox fill paints over icon edges, banner
    borders, or stroke fringes that sit just inside the OCR rectangle.
    Stroke-only fill keeps those bystanders intact:

    1. Sample the bg color from the 4-side ring (existing fill_color
       logic, with side-agreement voting + tight-container fallback).
    2. Inside the bbox, find pixels noticeably different from bg
       (max-channel diff > 30) — these are stroke + antialiased
       neighbourhood + any non-bg content the bbox accidentally covers.
    3. From those candidates take the MEDIAN colour = the dominant
       stroke colour. Pixels close to that median (per-channel diff
       <=40) are the text strokes; everything else (a contrasting icon
       pixel that happened to fall inside the OCR rectangle) is kept.
    4. When OCR emitted per-word/per-character boxes, constrain the mask
       to those geometry boxes so same-colour borders or connector lines
       in the broader OCR line bbox survive.
    5. Dilate the stroke mask by 1 px to cover antialiased fringe that
       sits just below the colour threshold, then paint only those
       pixels with bg.

    `overrides` lets the LLM icon-review stage replace specific
    heuristic decisions. Shape:

        {"ocr_items": {(x1,y1,x2,y2): "icon"|"text", ...},
         "components": {(x1,y1,x2,y2): "icon"|"text", ...}}

    `icon` = preserve as visual (no erase); `text` = erase. Bboxes are
    in absolute image coordinates and must match the values emitted by
    `--icon-review-dump`. Returns (cleaned_img, mask, decisions) where
    `decisions` is the list of icon-vs-text calls made on this pass —
    used to build the LLM review packet.
    """
    overrides = overrides or {}
    ocr_overrides: dict[tuple, str] = overrides.get("ocr_items", {}) or {}
    component_overrides: dict[tuple, str] = overrides.get("components", {}) or {}

    out = img.copy()
    h, w = img.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    decisions: list[dict] = []

    logo_bands = detect_logo_strips(ocr_data, h)
    text_to_remove: list[dict] = []
    for it in ocr_data:
        item_bbox = (it["x1"], it["y1"], it["x2"], it["y2"])
        in_logo = is_in_logo_zone(it, logo_bands)
        forced = ocr_overrides.get(item_bbox)
        if forced == "icon":
            decisions.append({
                "kind": "ocr_item",
                "bbox": list(item_bbox),
                "text": str(it.get("text", "")),
                "reason": "agent_override",
                "verdict": "icon",
            })
            continue
        if forced == "text":
            decisions.append({
                "kind": "ocr_item",
                "bbox": list(item_bbox),
                "text": str(it.get("text", "")),
                "reason": "agent_override",
                "verdict": "text",
            })
            if not in_logo:
                text_to_remove.append(it)
            continue
        is_icon, reason = is_likely_icon(it, ocr_data, img)
        if is_icon:
            if reason not in DEFINITIVE_ICON_REASONS:
                decisions.append({
                    "kind": "ocr_item",
                    "bbox": list(item_bbox),
                    "text": str(it.get("text", "")),
                    "reason": reason or "icon",
                    "verdict": "icon",
                })
            continue
        if not in_logo:
            text_to_remove.append(it)

    dilate_k = np.ones((3, 3), np.uint8)
    scale = h / 720.0
    # OCR bboxes are tight on the glyphs but often miss the last 1-2 px
    # of the descender / underline / bottom row. Expand the processing
    # window by 2 px on every side so those edge pixels go through the
    # same stroke-vs-bg test. The stroke mask itself still excludes any
    # pixel that isn't on the bg→text colour axis, so unrelated content
    # in the padding ring (icon strokes, banner borders) survives.
    pad = 2
    for item in text_to_remove:
        # The bg sample uses the ORIGINAL OCR bbox so the ring sits
        # cleanly outside the glyphs; the stroke search uses the padded
        # bbox to catch the missed-by-OCR fringe.
        ox1 = max(0, item["x1"])
        oy1 = max(0, item["y1"])
        ox2 = min(w, item["x2"])
        oy2 = min(h, item["y2"])
        x1 = max(0, item["x1"] - pad)
        y1 = max(0, item["y1"] - pad)
        x2 = min(w, item["x2"] + pad)
        y2 = min(h, item["y2"] + pad)
        if x2 <= x1 or y2 <= y1:
            continue
        bg = fill_color(img, ox1, oy1, ox2, oy2).astype(float)
        region = img[y1:y2, x1:x2]
        guard_box_count = len(_ocr_guard_boxes(item))
        guard = _text_geometry_guard_mask(item, x1, y1, x2, y2, scale)
        diff_bg = np.abs(region.astype(int) - bg.astype(int)).max(axis=2)
        non_bg = diff_bg > 30
        if int(non_bg.sum()) < 4:
            # Almost no strokes found. Either bg estimate was wrong or
            # text is invisible against bg — fall back to a guarded fill
            # when OCR sub-boxes are available, otherwise the old bbox
            # fill fallback.
            paint = guard if guard is not None else np.ones(
                region.shape[:2], dtype=bool)
            out[y1:y2, x1:x2][paint] = bg.astype(np.uint8)
            mask[y1:y2, x1:x2][paint] = 255
            continue
        # Multi-colour text detection: a single OCR bbox can carry glyphs
        # of more than one colour. Quantise the non-bg pixels into
        # 32-step bins and keep modal colours that are likely to be text.
        # For short bboxes we intentionally keep only major colour
        # clusters: small nearby arrows/icons can fall inside the OCR bbox,
        # and treating their colour as text erases the adjacent visual. For
        # longer text, allow smaller saturated clusters so inline emphasis
        # still gets cleaned.
        non_bg_pixels = region[non_bg]
        quant = (non_bg_pixels // 32) * 32 + 16
        from collections import Counter as _Counter
        mode_counts = _Counter(map(tuple, quant)).most_common(8)
        total_non_bg = max(1, len(non_bg_pixels))
        item_text = str(item.get("text", "") or "")
        is_short_text = len(item_text) <= 5
        modes = []
        for c, count in mode_counts:
            m = np.array(c, dtype=float)
            if float(np.max(np.abs(m - bg))) <= 30:
                continue
            frac = count / total_non_bg
            # Always keep major clusters. On short OCR boxes this excludes
            # small adjacent icon/arrow colours while preserving the main
            # number/unit text.
            if frac >= 0.10:
                modes.append(m)
                continue
            if is_short_text:
                continue
            # Long text can include small emphasized words in a different
            # colour. Keep sufficiently represented saturated/dark clusters.
            bgr = np.uint8([[c]])
            hsv_c = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)[0, 0]
            lum = 0.114 * c[0] + 0.587 * c[1] + 0.299 * c[2]
            if count >= max(20, int(0.025 * total_non_bg)) and (
                int(hsv_c[1]) > 45 or lum < 90
            ):
                modes.append(m)
        if not modes and mode_counts:
            modes = [np.array(mode_counts[0][0], dtype=float)]
        # Drop modes that are basically bg (sometimes the bg ring sample
        # is slightly off the actual bg colour and the modal cluster sits
        # in between).
        text_colors = [m for m in modes
                       if float(np.max(np.abs(m - bg))) > 30]
        if not text_colors:
            text_colors = [np.median(non_bg_pixels, axis=0).astype(float)]
        offset = region.astype(float) - bg
        stroke = np.zeros(region.shape[:2], dtype=bool)
        for text_color in text_colors:
            text_dir = text_color - bg
            text_norm_sq = float((text_dir * text_dir).sum())
            if text_norm_sq < 1:
                continue
            t = (offset * text_dir).sum(axis=2) / text_norm_sq
            projection = t[..., None] * text_dir
            perp = offset - projection
            perp_dist = np.max(np.abs(perp), axis=2)
            # core: clearly on this bg→text axis; fade: antialias fringe.
            core = (t >= 0.18) & (perp_dist < 35)
            fade = (t >= 0.06) & (perp_dist < 50)
            stroke |= core | fade
        has_guard = guard is not None
        if has_guard:
            # Inside Paddle's word/char boxes, be more aggressive than the
            # colour-axis test: antialiased glyph edges often drift off the
            # main bg→text vector and otherwise remain as faint residue.
            stroke = (stroke | ((diff_bg > 12) & guard)) & guard
        stroke = stroke.astype(np.uint8)
        if not has_guard:
            stroke = _drop_thin_non_text_row_bands(stroke, item_text)
        # A light dilation absorbs sub-pixel antialias that lives below
        # the t=0.06 threshold. Short numeric bboxes often sit next to
        # icons/arrows, so keep their dilation tighter to avoid collateral
        # erasure of neighbouring visuals.
        stroke = cv2.dilate(stroke, dilate_k,
                            iterations=1 if is_short_text else 2)
        if has_guard:
            stroke = (stroke.astype(bool) & guard).astype(np.uint8)
        else:
            stroke = _drop_thin_non_text_row_bands(stroke, item_text)
        if is_short_text and not (has_guard and guard_box_count >= 2):
            stroke_before_filter = stroke.copy()
            stroke, review_records = _filter_short_text_stroke_components(
                stroke, item_text, scale)
            # Translate the local (per-region) review records into the
            # global picture, applying any agent overrides that cover
            # this OCR item's region. Each component is logged as a
            # decision so the review packet has full context.
            item_bbox = (item["x1"], item["y1"], item["x2"], item["y2"])
            for rec in review_records:
                rx, ry, rw, rh = rec["rel_bbox"]
                gx1 = x1 + rx
                gy1 = y1 + ry
                gx2 = x1 + rx + rw
                gy2 = y1 + ry + rh
                comp_bbox = (gx1, gy1, gx2, gy2)
                verdict = rec["verdict"]
                reason = rec["reason"]
                forced_c = component_overrides.get(comp_bbox)
                if forced_c in ("icon", "text"):
                    if forced_c != verdict:
                        if forced_c == "icon":
                            # Force-remove from mask: blank out the
                            # component pixels.
                            stroke[ry:ry + rh, rx:rx + rw] = 0
                        else:
                            # Force-add: bring back the stroke pixels
                            # the filter took out.
                            stroke[ry:ry + rh, rx:rx + rw] = (
                                stroke_before_filter[ry:ry + rh, rx:rx + rw])
                    verdict = forced_c
                    reason = "agent_override"
                decisions.append({
                    "kind": "component",
                    "bbox": [gx1, gy1, gx2, gy2],
                    "ocr_bbox": list(item_bbox),
                    "text": item_text,
                    "reason": reason,
                    "verdict": verdict,
                })
        stroke_bool = stroke.astype(bool)
        out[y1:y2, x1:x2][stroke_bool] = bg.astype(np.uint8)
        mask[y1:y2, x1:x2] = stroke * 255
    return out, mask, decisions


# =============================================================================
# Icon-review packet I/O
#
# When the heuristic falls into a gray zone, emit a packet of crops the
# agent can second-guess. apply_ocr_item_overrides feeds the agent's
# verdicts back into the OCR list before erase_text re-runs.
# =============================================================================


def _load_icon_decisions(path: str) -> dict:
    """Read an icon_review.json (post-agent-review) into the override
    shape that `erase_text` expects. Entries with `decision` null are
    ignored — those fall back to the heuristic."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    entries = data.get("entries", [])
    ocr_overrides: dict[tuple, str] = {}
    component_overrides: dict[tuple, str] = {}
    for e in entries:
        decision = e.get("decision")
        if decision not in ("icon", "text"):
            continue
        bbox = tuple(e["bbox"])
        if e.get("kind") == "ocr_item":
            ocr_overrides[bbox] = decision
        elif e.get("kind") == "component":
            component_overrides[bbox] = decision
    return {"ocr_items": ocr_overrides, "components": component_overrides}


def apply_ocr_item_overrides(ocr_data: list[dict],
                             overrides: dict | None) -> list[dict]:
    """Bake OCR-item-level icon-review decisions into the OCR list so
    every downstream call site (erase, build_inventory, layout) makes
    the same icon-vs-text call.

    For each item the reviewer flipped:
      - `text` → set `_force_text: True` (read by `is_likely_icon`)
      - `icon` → set `preserve_visual: True` (read by
        `should_preserve_visual`)

    Component-level overrides stay out of band — they act on stroke
    pixels inside an already-text-flagged OCR item.
    """
    if not overrides:
        return ocr_data
    ocr_overrides = overrides.get("ocr_items") or {}
    if not ocr_overrides:
        return ocr_data
    out = []
    for item in ocr_data:
        bbox = (item["x1"], item["y1"], item["x2"], item["y2"])
        ovr = ocr_overrides.get(bbox)
        if ovr is None:
            out.append(item)
            continue
        patched = dict(item)
        if ovr == "text":
            patched["_force_text"] = True
            patched.pop("preserve_visual", None)
        elif ovr == "icon":
            patched["preserve_visual"] = True
            patched.pop("_force_text", None)
        out.append(patched)
    return out


def _crop_with_pad(img: np.ndarray, bbox: list[int], pad: int) -> np.ndarray:
    h, w = img.shape[:2]
    x1, y1, x2, y2 = bbox
    return img[max(0, y1 - pad):min(h, y2 + pad),
               max(0, x1 - pad):min(w, x2 + pad)]


def _dump_icon_review_packet(img: np.ndarray, decisions: list[dict],
                             dump_dir: Path, image_path: Path) -> None:
    """Write icon_review.json + per-entry crops + a contact sheet.

    Only `decisions` whose verdict came from a non-definitive heuristic
    or sits in the trailing-text-sized gray zone are written. Definitive
    cases (preserve_visual / decorative_binary) are filtered upstream.
    """
    reviewable = [d for d in decisions
                  if d.get("reason") not in DEFINITIVE_ICON_REASONS]
    if not reviewable:
        return
    dump_dir.mkdir(parents=True, exist_ok=True)
    crops_dir = dump_dir / "crops"
    crops_dir.mkdir(exist_ok=True)
    entries = []
    stem = image_path.stem
    for i, d in enumerate(reviewable):
        bbox = d["bbox"]
        # The suspect crop: tight on the component / OCR-item, padded
        # for context so the reviewer can see the surrounding glyphs
        # and any nearby pictogram.
        crop_pad = 8
        suspect = _crop_with_pad(img, bbox, crop_pad)
        crop_name = f"{stem}_icon_{i:03d}.png"
        cv2.imwrite(str(crops_dir / crop_name), suspect)
        # A wider context crop helps for component-level decisions
        # where the agent needs to see the whole OCR text run.
        ctx_bbox = d.get("ocr_bbox", bbox)
        ctx = _crop_with_pad(img, ctx_bbox, 24)
        ctx_name = f"{stem}_icon_{i:03d}_ctx.png"
        cv2.imwrite(str(crops_dir / ctx_name), ctx)
        entries.append({
            "idx": i,
            "kind": d["kind"],
            "bbox": d["bbox"],
            "ocr_bbox": d.get("ocr_bbox"),
            "ocr_text": d.get("text", ""),
            "reason": d["reason"],
            "heuristic_verdict": d["verdict"],
            "decision": None,
            "notes": None,
            "crop_path": f"crops/{crop_name}",
            "context_crop_path": f"crops/{ctx_name}",
        })
    packet = {
        "instructions": (
            "For each entry, look at crop_path (and context_crop_path for "
            "OCR-context) and decide whether the suspect is an ICON "
            "(preserve as a visual element, do not erase) or TEXT (erase "
            "as text). Fill `decision` with `\"icon\"` or `\"text\"`. "
            "`heuristic_verdict` is what the script chose without you — "
            "if you agree, you can either copy it into `decision` or "
            "leave `decision` as null. Set `decision` only when you want "
            "to OVERRIDE the heuristic. After filling, re-run erase_text "
            "with --icon-decisions pointing at this file."),
        "source_image": str(image_path),
        "entries": entries,
    }
    (dump_dir / "icon_review.json").write_text(
        json.dumps(packet, ensure_ascii=False, indent=2), encoding="utf-8")
    _render_icon_contact_sheet(reviewable, crops_dir, dump_dir / "contact.png")


def _render_icon_contact_sheet(decisions: list[dict], crops_dir: Path,
                               out_path: Path) -> None:
    from PIL import Image, ImageDraw, ImageFont
    if not decisions:
        return
    cell_w, cell_h, label_h = 220, 200, 38
    cols = 5
    rows = (len(decisions) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * cell_w, rows * cell_h),
                      (245, 245, 245))
    draw = ImageDraw.Draw(sheet)
    try:
        font = ImageFont.truetype("/System/Library/Fonts/PingFang.ttc", 12)
    except Exception:
        font = ImageFont.load_default()
    for i, d in enumerate(decisions):
        r, c = divmod(i, cols)
        x0, y0 = c * cell_w, r * cell_h
        img_h = cell_h - label_h
        crop_path = crops_dir / f"{Path(out_path).stem.replace('contact', '')}".rstrip("_")
        # Just glob: we know the naming scheme.
        candidates = sorted(crops_dir.glob(f"*_icon_{i:03d}.png"))
        if candidates:
            im = Image.open(candidates[0]).convert("RGB")
            im.thumbnail((cell_w - 4, img_h - 4))
            sheet.paste(im, (x0 + (cell_w - im.width) // 2,
                              y0 + (img_h - im.height) // 2))
        label = f"#{i} {d['reason']}\n{d['verdict']}: {d.get('text','')[:14]}"
        draw.text((x0 + 4, y0 + img_h + 2), label, fill=(20, 20, 20),
                  font=font)
        draw.rectangle([x0, y0, x0 + cell_w - 1, y0 + cell_h - 1],
                       outline=(180, 180, 180))
    sheet.save(out_path)


# =============================================================================
# Driver
# =============================================================================


def run(*, image: str, ocr: str, out: str,
        debug_dir: str | None = None,
        icon_review_dump: str | None = None,
        icon_decisions: str | None = None) -> None:
    """Programmatic entry point — same contract as the CLI flags.

    Used by run_pipeline.process_page to skip the subprocess fork.
    """
    img = cv2.imread(image)
    if img is None:
        raise SystemExit(f"Could not read image: {image}")
    ocr_data = json.loads(Path(ocr).read_text(encoding="utf-8"))
    ocr_data = preprocess_ocr(ocr_data, img=img)
    overrides = (_load_icon_decisions(icon_decisions)
                 if icon_decisions else None)
    ocr_data = apply_ocr_item_overrides(ocr_data, overrides)
    cleaned, mask, decisions = erase_text(img, ocr_data, overrides=overrides)
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(out, cleaned)
    # Sidecar of every pixel the eraser painted. Downstream foreground
    # detection treats these as background by definition — the eraser's
    # fill colour and the detector's canvas estimate use different
    # models, and their disagreement used to re-surface erased regions
    # as phantom image elements.
    erased_mask_path = Path(out).with_name(
        f"{Path(out).stem}.erased_mask.png")
    cv2.imwrite(str(erased_mask_path), mask)

    if icon_review_dump:
        _dump_icon_review_packet(img, decisions, Path(icon_review_dump),
                                 Path(image))

    if debug_dir:
        debug_path = Path(debug_dir)
        debug_path.mkdir(parents=True, exist_ok=True)
        stem = Path(image).stem
        cv2.imwrite(str(debug_path / f"{stem}_mask.png"), mask)
        cv2.imwrite(str(debug_path / f"{stem}_clean.png"), cleaned)
        # OCR-bbox overlay: original image with each detected text drawn as
        # a coloured rectangle (green = will be erased, orange = preserved
        # as part of icon/logo/etc.).
        h = img.shape[0]
        logo_bands = detect_logo_strips(ocr_data, h)
        ocr_vis = img.copy()
        for item in ocr_data:
            x1, y1, x2, y2 = item["x1"], item["y1"], item["x2"], item["y2"]
            is_icon, _ = is_likely_icon(item, ocr_data, img)
            preserved = is_icon or is_in_logo_zone(item, logo_bands)
            color = (0, 165, 255) if preserved else (0, 200, 0)  # BGR
            cv2.rectangle(ocr_vis, (x1, y1), (x2, y2), color, 1)
            label = item["text"][:12]
            cv2.putText(ocr_vis, label, (x1 + 1, max(10, y1 - 2)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.32, color, 1, cv2.LINE_AA)
        cv2.imwrite(str(debug_path / f"{stem}_ocr.png"), ocr_vis)


def main() -> None:
    args = parse_args()
    run(image=args.image, ocr=args.ocr, out=args.out,
        debug_dir=args.debug_dir,
        icon_review_dump=args.icon_review_dump,
        icon_decisions=args.icon_decisions)
    print(args.out)


if __name__ == "__main__":
    main()
