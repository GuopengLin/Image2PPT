"""Line-level text style detection: colour, bold flag, optional runs.

Samples the BACKGROUND from a ring outside the OCR bbox first; pixels
inside the bbox far from the ring colour are text strokes. Their median
gives the text colour, the distance-transform median of the ink mask
gives bold/regular. When `text` is supplied, also runs per-character
sampling and emits a `runs` list if more than one colour group survives.
"""
from __future__ import annotations

from collections import Counter

import cv2
import numpy as np

from layout.color import (
    _classify_text_color,
    _normalize_run_colors,
)
from layout.text_runs import _per_char_runs


def detect_text_style(bbox: list[int], orig_img: np.ndarray,
                      text: str | None = None,
                      char_boxes: list[list[int]] | None = None,
                      words: list[str] | None = None,
                      word_boxes: list[list[int]] | None = None) -> dict:
    """Sample text region in original image; return color + bold flag.

    Sample the BACKGROUND from a 4 px ring just outside the bbox first.
    Pixels INSIDE the bbox close to the bg color (within ±35 per channel)
    are considered background; the rest are text strokes. Median of the
    text-stroke pixels gives the text color; their density gives bold.

    When `text` is provided, also runs per-character color sampling and,
    if multiple distinct color runs are detected in the bbox, returns
    them under the `runs` key.
    """
    x1, y1, x2, y2 = bbox
    h_img, w_img = orig_img.shape[:2]
    region = orig_img[y1:y2, x1:x2]
    if region.size == 0:
        return {"color": "#054798", "bold": False}

    # 6-px ring offset 2 px from the bbox. The offset keeps the inner
    # row of the ring outside the antialiased character edge, which
    # would otherwise pollute the bg median with text-colour pixels.
    gap = 2
    ring = 6
    inner_top = max(0, y1 - gap)
    inner_bot = min(h_img, y2 + gap)
    inner_left = max(0, x1 - gap)
    inner_right = min(w_img, x2 + gap)
    bg_samples = []
    if inner_top - ring >= 0:
        bg_samples.append(
            orig_img[inner_top - ring:inner_top, inner_left:inner_right])
    if inner_bot + ring <= h_img:
        bg_samples.append(
            orig_img[inner_bot:inner_bot + ring, inner_left:inner_right])
    if inner_left - ring >= 0:
        bg_samples.append(
            orig_img[inner_top:inner_bot, inner_left - ring:inner_left])
    if inner_right + ring <= w_img:
        bg_samples.append(
            orig_img[inner_top:inner_bot, inner_right:inner_right + ring])
    if bg_samples:
        bg_pixels = np.concatenate(
            [s.reshape(-1, 3) for s in bg_samples if s.size > 0])
        # Vote majority bg via quantised mode, keep only pixels close
        # to mode, then take their median. Excludes text-colour leaks
        # from antialiased character strokes bleeding into the ring.
        quant = (bg_pixels // 16) * 16
        mode_q = np.array(Counter(map(tuple, quant)).most_common(1)[0][0])
        diff = np.abs(bg_pixels.astype(int) - mode_q).max(axis=1)
        close = bg_pixels[diff <= 30]
        if len(close) >= max(8, int(len(bg_pixels) * 0.2)):
            bg = np.median(close, axis=0)
        else:
            bg = np.median(bg_pixels, axis=0)
    else:
        bg = np.array([255.0, 255.0, 255.0])

    # Tight-coloured-container case: when an OCR bbox almost exactly
    # fills a coloured badge (e.g. white `Rhombus` text on a dark-blue
    # rounded rect), the ring sample lands OUTSIDE the badge and reports
    # white. Detect via inside-bbox saturation majority.
    inner_hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
    sat_mask = inner_hsv[:, :, 1] > 50
    total_px = region.shape[0] * region.shape[1]
    # Threshold tuned for: saturated TEXT on neutral bg ~20-30 %;
    # saturated CONTAINER fill with light text ~60-80 %. 0.55 sits
    # between the two.
    is_container_like = int(sat_mask.sum()) > 0.55 * total_px
    if is_container_like:
        # Disambiguate dense CJK text from real containers: dense
        # strokes are many small blobs, real container is one big blob.
        n_comp, _, stats, _ = cv2.connectedComponentsWithStats(
            sat_mask.astype(np.uint8) * 255, 8)
        if n_comp > 1:
            largest_area = int(stats[1:, cv2.CC_STAT_AREA].max())
            total_sat = int(sat_mask.sum())
            if largest_area < 0.55 * total_sat:
                is_container_like = False
    if is_container_like:
        container_bg = np.median(region[sat_mask], axis=0)
        if int(np.max(np.abs(
                container_bg.astype(int) - bg.astype(int)))) > 40:
            bg = container_bg

    # Pixels within bbox far from bg are text strokes.
    diff = np.abs(region.astype(int) - bg).max(axis=2)
    text_mask = diff > 35
    if int(text_mask.sum()) < 5:
        # Fallback: take darkest 25 % vs bg.
        thr = max(15, int(np.percentile(diff, 75)))
        text_mask = diff > thr
        if int(text_mask.sum()) < 5:
            return {"color": "#054798", "bold": False}

    text_pixels = region[text_mask]
    m = np.median(text_pixels, axis=0).astype(int)
    b, g, r = int(m[0]), int(m[1]), int(m[2])
    color = _classify_text_color(r, g, b, bg)
    # Raw RGB hex of the line median: keeps the same colour SPACE as
    # the per-char raw samples so _per_char_runs' anchor step can merge
    # close-but-not-identical samples back to the line dominant.
    raw_line_hex = f"#{r:02X}{g:02X}{b:02X}"

    h, w = region.shape[:2]
    # ink_h: tight vertical extent of strokes inside the bbox. Used as
    # (a) a decoration-padding probe by the caller, and (b) the
    # denominator for stroke-width-based bold detection below.
    # ink_band_h: height of the DOMINANT contiguous row band. When the
    # OCR bbox swallowed extra content below/above the glyphs (dashed
    # arrows, icon rows), the full extent spans glyphs+decoration while
    # the dominant band is just the glyph line — the caller sizes the
    # font from it.
    ink_h: int | None = None
    ink_band_h: int | None = None
    row_ink = text_mask.sum(axis=1) >= 2
    rows = np.where(row_ink)[0]
    ink_band_y0: int | None = None
    if rows.size >= 2:
        ink_h = int(rows.max() - rows.min() + 1)
        bands: list[tuple[int, int]] = []      # (start_row, height)
        start = int(rows[0])
        run = 1
        for i in range(1, len(rows)):
            if rows[i] - rows[i - 1] <= 3:      # bridge AA gaps
                run += int(rows[i] - rows[i - 1])
            else:
                bands.append((start, run))
                start = int(rows[i])
                run = 1
        bands.append((start, run))
        ink_band_y0, ink_band_h = max(bands, key=lambda b: b[1])
        ink_band_h = int(ink_band_h)

    # Bold detection via absolute stroke half-width on the distance
    # transform of the ink mask. Density misfires across font sizes;
    # the 1.2 cutoff sits in the gap between regular (≤ 1 px median)
    # and bold (≥ 1.5 px) strokes.
    bold = False
    if int(text_mask.sum()) >= 8 and ink_h and ink_h >= 4:
        try:
            mask_u8 = text_mask.astype(np.uint8)
            dist = cv2.distanceTransform(mask_u8, cv2.DIST_L2, 3)
            stroke_half_median = float(np.median(dist[text_mask]))
            bold = stroke_half_median > 1.2
        except cv2.error:
            density = float(text_mask.sum()) / max(1, h * w)
            bold = density > 0.27

    result = {"color": color, "bold": bool(bold), "ink_h": ink_h,
              "ink_band_h": ink_band_h, "ink_band_y0": ink_band_y0}

    if text and len(text) >= 2:
        runs = _per_char_runs(bbox, text, orig_img, bg, raw_line_hex,
                              char_boxes=char_boxes,
                              words=words, word_boxes=word_boxes)
        _normalize_run_colors(runs, color, bg)
        if len(runs) > 1:
            result["runs"] = runs
    return result


def _glyph_height_in_bbox(orig_img: np.ndarray, bg: np.ndarray,
                          bbox: tuple[int, int, int, int]) -> int | None:
    """Measure vertical ink extent inside a text bbox.

    PaddleOCR's detection bbox often pads vertically for decorated stat
    numbers — drop shadows, glow effects, gradient borders inflate the
    box past the actual glyph extent. Returns the ink-row count (rows
    containing ≥ 2 stroke px), or None if sampling fails.
    """
    x1, y1, x2, y2 = bbox
    h_img, w_img = orig_img.shape[:2]
    x1 = max(0, min(w_img, int(x1))); x2 = max(0, min(w_img, int(x2)))
    y1 = max(0, min(h_img, int(y1))); y2 = max(0, min(h_img, int(y2)))
    if x2 <= x1 or y2 <= y1:
        return None
    region = orig_img[y1:y2, x1:x2]
    if region.size == 0:
        return None
    diff = np.abs(region.astype(int) - bg).max(axis=2)
    mask = diff > 35
    if int(mask.sum()) < 3:
        return None
    row_ink = mask.sum(axis=1) >= 2
    rows = np.where(row_ink)[0]
    if rows.size < 2:
        return None
    return int(rows.max() - rows.min() + 1)
