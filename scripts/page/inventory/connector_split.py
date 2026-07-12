"""Recover connector strokes merged into card components.

Diagram connectors frequently touch — or cross — the cards they join.
Connected-component detection then yields one blob containing cards and
stroke together, so the stroke never gets the `connector` role and the
whole diagram ships as a single flat PNG.

This module decomposes such a blob:

1. Build a "card region" mask from rectangles already known to the
   inventory (outline/container records) plus solid card cores found by
   eroding the foreground (erosion removes thin strokes, keeps cards).
2. Subtract the (slightly grown) card regions from the component
   foreground; fit each residual piece with the straight/elbow line
   models. Pieces that fit are connectors.
3. For a stroke *crossing* a card the subtraction yields two collinear
   pieces. If the original pixels along the corridor between them still
   show the stroke colour, the line runs *over* the card → merge the
   pieces into one connector (rendered in front). If not, the line runs
   *behind* the card → keep two pieces ending at the card edges, which
   reproduces the occlusion exactly.

Everything here is crop-local and pure; the inventory builder does the
record bookkeeping and in-place inpainting.
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

PAGE_DIR = Path(__file__).resolve().parents[1]     # scripts/page
if str(PAGE_DIR) not in sys.path:
    sys.path.insert(0, str(PAGE_DIR))

from _heuristics import s_area, s_length  # noqa: E402
from layout.connector_geometry import (  # noqa: E402
    MIN_STROKE_CONTRAST,
    fit_line_geometry,
    stroke_background_contrast,
)
from shared.geometry import connector_on_container_border  # noqa: E402


def _solid_card_cores(fg: np.ndarray, scale: float) -> list[tuple[int, int, int, int]]:
    """Filled card bodies inside a component mask.

    Erosion removes structures thinner than the kernel (strokes), so the
    surviving blobs are solid fills. Size gates are applied to the
    ERODED blobs — nearby parallel strokes can fuse into a thin band
    that survives erosion, and judging the dilated-back extent would let
    that band masquerade as a card.
    """
    h, w = fg.shape
    k = max(7, 2 * int(round(s_length(5, scale))) + 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k, k))
    core = cv2.erode(fg.astype(np.uint8), kernel)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(core, 8)
    boxes = []
    min_area = s_area(1200, scale)
    min_side = s_length(20, scale)
    pad = k // 2
    for i in range(1, n):
        x, y, cw, ch, area = (int(v) for v in stats[i])
        if area < min_area or min(cw, ch) < min_side:
            continue
        comp = labels[y:y + ch, x:x + cw] == i
        # Fused parallel strokes can survive erosion as a thin band
        # hanging off the card body. A card's per-column/row occupancy
        # is uniform, the band's is a fraction of it — trim extents
        # whose occupancy falls below half the body's.
        col_h = comp.sum(axis=0)
        row_w = comp.sum(axis=1)
        col_ref = float(np.percentile(col_h[col_h > 0], 90))
        row_ref = float(np.percentile(row_w[row_w > 0], 90))
        good_cols = np.flatnonzero(col_h >= 0.5 * col_ref)
        good_rows = np.flatnonzero(row_w >= 0.5 * row_ref)
        if not len(good_cols) or not len(good_rows):
            continue
        tx1 = x + int(good_cols.min())
        tx2 = x + int(good_cols.max()) + 1
        ty1 = y + int(good_rows.min())
        ty2 = y + int(good_rows.max()) + 1
        tw, th = tx2 - tx1, ty2 - ty1
        if min(tw, th) < min_side:
            continue
        trimmed = labels[ty1:ty2, tx1:tx2] == i
        trimmed_px = float(trimmed.sum())
        # Trimming may only shave a thin tail off a rectangle. When it
        # removes a big fraction of the blob (a triangle's occupancy
        # rises toward its dense centre, so trimming keeps only the
        # middle slab), this is a solid non-rectangular shape, not a
        # card — cutting it up would butcher the artwork.
        if trimmed_px < 0.80 * float(area):
            continue
        # Cards are rectangles — a straggly eroded blob is texture.
        if trimmed_px < 0.60 * tw * th:
            continue
        x1 = max(0, tx1 - pad)
        y1 = max(0, ty1 - pad)
        x2 = min(w, tx2 + pad)
        y2 = min(h, ty2 + pad)
        # A "core" the size of the whole component is a photo/texture,
        # not a card sitting next to connectors.
        if (x2 - x1) >= 0.96 * w and (y2 - y1) >= 0.96 * h:
            continue
        boxes.append((x1, y1, x2, y2))
    return boxes


def _stroke_color(probe_crop: np.ndarray, mask: np.ndarray) -> np.ndarray:
    pixels = probe_crop[mask]
    if not len(pixels):
        return np.array([0, 0, 0], dtype=np.int16)
    return np.median(pixels.reshape(-1, 3), axis=0).astype(np.int16)


def _corridor_between(a_box, b_box, axis: str):
    """Rectangle spanning the gap between two facing piece bboxes."""
    ax1, ay1, ax2, ay2 = a_box
    bx1, by1, bx2, by2 = b_box
    if axis == "x":
        left, right = (a_box, b_box) if ax1 <= bx1 else (b_box, a_box)
        gx1, gx2 = left[2], right[0]
        gy1 = min(left[1], right[1])
        gy2 = max(left[3], right[3])
        if gx2 <= gx1:
            return None
        return (gx1, gy1, gx2, gy2)
    top, bottom = (a_box, b_box) if ay1 <= by1 else (b_box, a_box)
    gy1, gy2 = top[3], bottom[1]
    gx1 = min(top[0], bottom[0])
    gx2 = max(top[2], bottom[2])
    if gy2 <= gy1:
        return None
    return (gx1, gy1, gx2, gy2)


def _pieces_collinear(a, b, scale: float) -> str | None:
    """'x'/'y' when two straight pieces line up along that axis."""
    (ax1, ay1, ax2, ay2), (bx1, by1, bx2, by2) = a["bbox"], b["bbox"]
    tol = max(4.0, s_length(6, scale))
    a_horiz = (ax2 - ax1) >= 2 * (ay2 - ay1)
    b_horiz = (bx2 - bx1) >= 2 * (by2 - by1)
    a_vert = (ay2 - ay1) >= 2 * (ax2 - ax1)
    b_vert = (by2 - by1) >= 2 * (bx2 - bx1)
    if a_horiz and b_horiz:
        acy = (ay1 + ay2) / 2.0
        bcy = (by1 + by2) / 2.0
        if abs(acy - bcy) <= tol:
            return "x"
    if a_vert and b_vert:
        acx = (ax1 + ax2) / 2.0
        bcx = (bx1 + bx2) / 2.0
        if abs(acx - bcx) <= tol:
            return "y"
    return None


def _corridor_shows_stroke(probe_crop: np.ndarray, corridor, color,
                           tol: int = 45) -> bool:
    """Sample the corridor centreline for the stroke colour."""
    gx1, gy1, gx2, gy2 = corridor
    h, w = probe_crop.shape[:2]
    gx1 = max(0, min(w - 1, gx1)); gx2 = max(0, min(w, gx2))
    gy1 = max(0, min(h - 1, gy1)); gy2 = max(0, min(h, gy2))
    if gx2 <= gx1 or gy2 <= gy1:
        return False
    cy = (gy1 + gy2) // 2
    cx = (gx1 + gx2) // 2
    if (gx2 - gx1) >= (gy2 - gy1):
        band = probe_crop[max(gy1, cy - 2):min(gy2, cy + 3), gx1:gx2]
        diff = np.abs(band.astype(np.int16) - color).max(axis=2)
        hit = (diff <= tol).any(axis=0)          # per column
    else:
        band = probe_crop[gy1:gy2, max(gx1, cx - 2):min(gx2, cx + 3)]
        diff = np.abs(band.astype(np.int16) - color).max(axis=2)
        hit = (diff <= tol).any(axis=1)          # per row
    if not len(hit):
        return False
    return float(hit.mean()) >= 0.60


def _merge_pieces(a, b, corridor, axis: str) -> dict:
    mask = a["mask"] | b["mask"]
    gx1, gy1, gx2, gy2 = corridor
    # Fill a stroke-wide band along the corridor so the merged mask is a
    # continuous line.
    if axis == "x":
        cy = (gy1 + gy2) // 2
        half = max(1, (min(a["bbox"][3] - a["bbox"][1],
                           b["bbox"][3] - b["bbox"][1])) // 2)
        mask[max(0, cy - half):cy + half + 1, gx1:gx2] = True
    else:
        cx = (gx1 + gx2) // 2
        half = max(1, (min(a["bbox"][2] - a["bbox"][0],
                           b["bbox"][2] - b["bbox"][0])) // 2)
        mask[gy1:gy2, max(0, cx - half):cx + half + 1] = True
    ys, xs = np.where(mask)
    bbox = (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)
    return {"bbox": bbox, "mask": mask, "z_front": True,
            "geom_kind": "straight"}


def split_connectors_from_component(
    fg_crop: np.ndarray,
    probe_crop: np.ndarray,
    known_card_boxes: list[tuple[int, int, int, int]],
    scale: float,
) -> dict | None:
    """Decompose a merged card+connector component.

    Args are crop-local: `fg_crop` bool foreground of the component
    region, `probe_crop` the pristine (pre-mutation) BGR pixels,
    `known_card_boxes` rectangles already recorded by the inventory.

    Returns None when nothing line-like was recovered, else::

        {"connectors": [{bbox, mask, z_front}, ...],
         "card_mask": bool array,
         "core_boxes": [...],
         "explained_frac": float}
    """
    h, w = fg_crop.shape
    if h * w == 0 or not bool(fg_crop.any()):
        return None

    core_boxes = _solid_card_cores(fg_crop, scale)
    card_mask = np.zeros((h, w), dtype=bool)
    all_cards = list(known_card_boxes) + core_boxes
    for x1, y1, x2, y2 in all_cards:
        x1 = max(0, int(x1)); y1 = max(0, int(y1))
        x2 = min(w, int(x2)); y2 = min(h, int(y2))
        if x2 > x1 and y2 > y1:
            card_mask[y1:y2, x1:x2] = True
    if not bool(card_mask.any()):
        return None

    # Grow the card regions a little so the touching joint is cut clean.
    grow = max(2, int(round(s_length(3, scale))))
    grown = cv2.dilate(card_mask.astype(np.uint8),
                       np.ones((2 * grow + 1, 2 * grow + 1), np.uint8)) > 0
    residual = fg_crop & ~grown
    if not bool(residual.any()):
        return None

    min_len = s_length(30, scale)
    max_stroke = max(9.0, s_length(9, scale))
    n, labels, stats, _ = cv2.connectedComponentsWithStats(
        residual.astype(np.uint8), 8)
    pieces: list[dict] = []
    for i in range(1, n):
        x, y, cw, ch, area = (int(v) for v in stats[i])
        if max(cw, ch) < min_len:
            continue
        piece_mask = labels == i
        # A dash lying on a card border is the card's own frame, not an
        # independent connector.
        bbox = (x, y, x + cw, y + ch)
        if any(connector_on_container_border(cb, bbox) for cb in all_cards):
            continue
        geom = fit_line_geometry(piece_mask)
        if geom is None:
            continue
        # Residual slivers of solid artwork (a triangle's corner wedge)
        # can satisfy the line models — real connector strokes are thin.
        if geom["width_px"] > max_stroke:
            continue
        # …and visible: near-zero contrast pieces are anti-aliased card
        # fringes or erased regions, not lines.
        if stroke_background_contrast(
                probe_crop, piece_mask) < MIN_STROKE_CONTRAST:
            continue
        pieces.append({"bbox": bbox, "mask": piece_mask, "z_front": False,
                       "geom_kind": geom["kind"]})
    if not pieces:
        return None

    # Bridge collinear pieces separated by a card (crossing stroke).
    merged: list[dict] = []
    used = [False] * len(pieces)
    for i in range(len(pieces)):
        if used[i]:
            continue
        current = pieces[i]
        for j in range(i + 1, len(pieces)):
            if used[j]:
                continue
            if current["geom_kind"] != "straight" \
                    or pieces[j]["geom_kind"] != "straight":
                continue
            axis = _pieces_collinear(current, pieces[j], scale)
            if axis is None:
                continue
            corridor = _corridor_between(current["bbox"],
                                         pieces[j]["bbox"], axis)
            if corridor is None:
                continue
            gx1, gy1, gx2, gy2 = corridor
            corr = card_mask[max(0, gy1):gy2, max(0, gx1):gx2]
            if corr.size == 0 or float(corr.mean()) < 0.55:
                continue
            color = _stroke_color(probe_crop,
                                  current["mask"] | pieces[j]["mask"])
            if not _corridor_shows_stroke(probe_crop, corridor, color):
                continue
            current = _merge_pieces(current, pieces[j], corridor, axis)
            used[j] = True
        merged.append(current)

    explained = grown.copy()
    for piece in merged:
        explained |= piece["mask"]
    fg_total = float(fg_crop.sum())
    explained_frac = float((fg_crop & explained).sum()) / max(1.0, fg_total)
    return {
        "connectors": merged,
        "card_mask": card_mask,
        "core_boxes": core_boxes,
        "explained_frac": explained_frac,
    }
