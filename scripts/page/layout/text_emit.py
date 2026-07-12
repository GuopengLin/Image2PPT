"""Build one layout text-element dict from an inventory text entry.

This is the text branch of the per-page main loop, extracted as a
single function so the builder can stay focused on the image branch
and orchestration.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

SCRIPTS_ROOT = Path(__file__).resolve().parents[2]    # scripts/
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

from text_safety import ppt_safe_text  # noqa: E402

from layout.bbox_shape import _trim_short_stat_text_bbox
from layout.render_font import default_ppt_font
from layout.render_ink import _target_text_metrics
from layout.text_mixed import mixed_size_runs_from_char_boxes
from layout.text_sizing import (
    CAPTURE_FACTOR_SHORT,
    _inherit_punct_sizes,
    _should_apply_run_font_sizes,
    _snap_to_ladder,
    _unify_run_sizes_by_color,
    font_size_pt,
)
from layout.font_predict import predict_font_from_bbox
from layout.text_style import detect_text_style

# When the ML font classifier returns family confidence below this floor we
# keep the heuristic font instead of trusting a low-confidence guess. The
# value was picked from the holdout confusion analysis: confident predictions
# above ~0.6 are >99% correct, low-confidence ones often confuse visually
# similar pairs (Helvetica/Arial, SimSun/Songti) where the heuristic default
# is at least as safe.
_FONT_CONF_THRESHOLD = 0.60
from layout.text_units import (
    _derive_source_char_boxes,
    _estimated_runs_width_px,
    _estimated_text_width_px,
)


def _emit_text_element_record(
    el: dict, x1: int, y1: int, x2: int, y2: int,
    source: np.ndarray, pt_per_px: float,
) -> dict | None:
    """Build the layout dict for one inventory `text` entry.

    Returns the record dict, or None when the entry has no visible text
    to emit.
    """
    if not str(el.get("text", "") or "").strip():
        return None
    # Look up fit_text_render through the facade so tests can patch it
    # (`itl.fit_text_render = fake_fit`).
    import inventory_to_layout as _itl
    fit_text_render = _itl.fit_text_render

    x1, y1, x2, y2 = _trim_short_stat_text_bbox(
        str(el.get("text", "") or ""),
        (int(x1), int(y1), int(x2), int(y2)),
        source,
    )
    # PaddleOCR per-char bboxes when present. ocr_paddle attaches these
    # only when len(chars) == len(text); re-verify against the current
    # text (OCR-review may have corrected it).
    raw_text = el["text"]
    safe_text = ppt_safe_text(raw_text)
    cb = el.get("char_boxes")
    if cb is not None and len(cb) != len(raw_text):
        cb = None
    wd = el.get("words")
    wb = el.get("word_boxes")
    if (wd is not None and wb is not None
            and (len(wd) != len(wb) or "".join(wd) != raw_text)):
        wd = wb = None
    source_char_boxes = _derive_source_char_boxes(
        raw_text, (int(x1), int(y1), int(x2), int(y2)), cb, wd, wb)
    style = detect_text_style([x1, y1, x2, y2], source,
                              text=raw_text,
                              char_boxes=source_char_boxes,
                              words=wd, word_boxes=wb)
    font_pred = predict_font_from_bbox(source, [x1, y1, x2, y2], bgr=True)
    apply_run_sizes = _should_apply_run_font_sizes(raw_text)
    if "runs" in style:
        for r in style["runs"]:
            gh = r.pop("glyph_h", None)
            if apply_run_sizes and gh and gh > 0:
                raw = gh * pt_per_px * CAPTURE_FACTOR_SHORT
                r["size"] = _snap_to_ladder(max(8.0, min(36.0, raw)))
        if apply_run_sizes:
            # Same-colour runs share a font size; punctuation inherits
            # its neighbour's em-box.
            _unify_run_sizes_by_color(style["runs"])
            _inherit_punct_sizes(style["runs"])
    mixed_size = None
    if not apply_run_sizes:
        mixed_size = mixed_size_runs_from_char_boxes(
            raw_text, source, (int(x1), int(y1), int(x2), int(y2)),
            source_char_boxes, style.get("runs"), style["color"],
            bold=bool(style["bold"]),
            pt_per_px=pt_per_px,
        )
        if mixed_size is not None:
            style["runs"] = mixed_size["runs"]
    if "runs" in style:
        for r in style["runs"]:
            r["text"] = ppt_safe_text(r.get("text", ""))
    # Decoration-padding probe: when a multi-char line's ink vertical
    # extent is well under bbox_h (≤ 50 %), the bbox likely includes
    # shadow/glow above and below the actual glyphs. Use ink_h.
    ink_h = style.pop("ink_h", None)
    ink_band_h = style.pop("ink_band_h", None)
    ink_band_y0 = style.pop("ink_band_y0", None)
    effective_h = int(y2 - y1)
    visible_chars = sum(1 for c in raw_text if c.strip() and c != "\n")
    if (ink_h and effective_h > 30 and visible_chars >= 3
            and ink_h <= 0.5 * effective_h):
        effective_h = ink_h
    elif (ink_band_h and ink_band_y0 is not None
            and effective_h > 26 and visible_chars >= 3
            and ink_band_h <= 0.72 * effective_h):
        # The bbox swallowed separated non-glyph rows (dashed arrows,
        # icon strips beside the title). The dominant contiguous ink
        # band is the glyph line itself — tighten the whole record to
        # it. Adjusting only the font size is not enough: render_fit
        # scores candidates against the source ink inside the bbox, so
        # a dirty bbox pushes the size right back up, and a middle
        # vertical anchor in the tall box also mis-centres the text.
        y1 = int(y1 + ink_band_y0)
        y2 = int(min(y2, y1 + ink_band_h + 2))
        effective_h = int(y2 - y1)
        # Clamp the per-char boxes to the band too: the size calibration
        # loop measures its target ink from them, and dirty char boxes
        # would pull the size straight back up.
        if source_char_boxes:
            source_char_boxes = [
                [bx1, max(by1, y1), bx2, min(by2, y2)]
                for bx1, by1, bx2, by2 in source_char_boxes
            ]
    size = font_size_pt(safe_text, x2 - x1, effective_h,
                        pt_per_px=pt_per_px)
    if mixed_size is not None:
        size = max(int(size), int(mixed_size["base_size"]))
    ppt_font = default_ppt_font()
    text_bold = bool(style["bold"])
    text_italic = False
    # ML font classifier (ResNet-34 INT8 ONNX). Only override the heuristic
    # font/bold/italic when the prediction is confident enough; otherwise the
    # downstream render_fit + heuristic path stays in charge.
    if font_pred is not None and font_pred["family_confidence"] >= _FONT_CONF_THRESHOLD:
        ppt_font = font_pred["family"]
        text_bold = bool(font_pred["is_bold"])
        text_italic = bool(font_pred["is_italic"])
    run_sized = any(r.get("size") is not None
                    for r in style.get("runs", []))
    compact_text = "".join(c for c in safe_text if not c.isspace())
    bbox_w = max(1, int(x2 - x1))
    bbox_h = max(1, int(y2 - y1))
    badge_like_number = (
        compact_text.isdigit()
        and len(compact_text) <= 3
        and bbox_h >= 24
        and 0.65 <= (bbox_w / float(bbox_h)) <= 1.60
    )
    render_fit = None
    if not run_sized and not badge_like_number:
        render_fit = fit_text_render(
            safe_text, source, (int(x1), int(y1), int(x2), int(y2)),
            initial_size=int(size),
            initial_bold=text_bold,
            initial_font=ppt_font,
            pt_per_px=pt_per_px,
        )
        if render_fit is not None:
            size = int(render_fit["size"])
            text_bold = bool(render_fit["bold"])
    # Tight bbox: do NOT inflate the text box beyond what OCR found.
    # 6 px right + 2 px bottom safety margin avoids font-metric clip.
    align = "left"
    base_w = int(x2 - x1)
    text_w = max(
        base_w + 6,
        _estimated_text_width_px(
            safe_text, int(size),
            pt_per_px=pt_per_px, bold=text_bold) + 8,
    )
    if run_sized:
        text_w = max(
            text_w,
            _estimated_runs_width_px(
                style.get("runs", []), int(size),
                pt_per_px=pt_per_px, bold=text_bold) + 8,
        )
    text_h = int(y2 - y1) + 2
    if run_sized:
        text_h = max(
            text_h,
            int(round((int(size) / max(0.01, pt_per_px)) * 1.20)),
        )
    text_x = int(x1)
    text_y = int(y1)
    valign_mode = "middle"
    if render_fit is not None:
        text_x, text_y, text_w, text_h = render_fit["box"]
        # render_fit picks a tight box around the glyph metrics; with
        # valign=top in PowerPoint/LibreOffice the ascent line is pinned to
        # the box top, which makes the rendered text sit ~half the line
        # leading above where the OCR found it (especially obvious on
        # short banner-style titles like "① 数据收集"). Center vertically
        # inside the render_fit box instead — the box height already
        # tracks the glyph + a small padding, so MIDDLE puts the ink right
        # back on the source y-axis.
        valign_mode = "middle"
    elif mixed_size is not None:
        valign_mode = "top"
    text_w = min(int(text_w),
                 max(1, int(source.shape[1]) - int(text_x)))
    record = {
        "type": "text", "name": el["id"], "text": safe_text,
        "box": [text_x, text_y, text_w, text_h],
        "source_bbox": [int(x1), int(y1), int(x2), int(y2)],
        "font": ppt_font, "size": int(size),
        "bold": text_bold, "italic": text_italic,
        "color": style["color"], "align": align, "valign": valign_mode,
        "line_spacing": 1.0,
    }
    if font_pred is not None:
        record["font_pred"] = {
            "family": font_pred["family"],
            "family_confidence": font_pred["family_confidence"],
            "bold_confidence": font_pred["bold_confidence"],
            "italic_confidence": font_pred["italic_confidence"],
        }
    if (source_char_boxes is not None
            and len(source_char_boxes) == len(raw_text)
            and len(safe_text) == len(raw_text)):
        record["source_chars"] = list(safe_text)
        record["source_char_boxes"] = [
            [int(v) for v in box] for box in source_char_boxes
        ]
    target_metrics = _target_text_metrics(
        source, (int(x1), int(y1), int(x2), int(y2)))
    if target_metrics is not None:
        record["target_ink"] = target_metrics["ink_bbox"]
    if render_fit is not None:
        record["size_source"] = "render_fit"
        record["font_fit_score"] = render_fit["score"]
        record["render_fit_font"] = render_fit["font"]
        record["fit_target_ink"] = render_fit["target_ink"]
        record["fit_render_ink"] = render_fit["render_ink"]
    elif mixed_size is not None:
        record["size_source"] = "mixed_runs"
        record["mixed_size"] = {
            "base_size": mixed_size["base_size"],
            "suffix_size": mixed_size["suffix_size"],
            "prefix_h": mixed_size["prefix_h"],
            "suffix_h": mixed_size["suffix_h"],
            "prefix_bold": mixed_size.get("prefix_bold"),
            "suffix_bold": mixed_size.get("suffix_bold"),
            "prefix_density": mixed_size.get("prefix_density"),
            "prefix_fit_score": mixed_size["prefix_fit_score"],
            "suffix_fit_score": mixed_size["suffix_fit_score"],
        }
    # Optional in-bbox per-character color runs.
    if "runs" in style:
        record["runs"] = style["runs"]
    return record
