"""Extract native line geometry from connector crops.

Inventory tags elongated stroke components as ``role: "connector"``, but
until now they were rasterised into transparent PNGs. This module
recovers editable geometry — endpoints, an optional elbow corner,
stroke colour/width, dash pattern and arrowheads — so the layout can
emit a native PPT ``type: "line"`` element instead. Extraction is
conservative: any crop the models below can't explain confidently
returns ``None`` and the caller keeps the raster fallback.

All coordinates are crop-local pixels; the caller offsets them into
page space.
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

SCRIPTS_ROOT = Path(__file__).resolve().parents[2]    # scripts/
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

from shared.geometry import bgr_to_hex  # noqa: E402


# Minimum projected length (px) for a crop to count as a line at all.
_MIN_LINE_LEN = 18.0
# Fraction of foreground pixels the fitted model must explain.
_MIN_COVERAGE = 0.90


def _foreground_points(
        alpha: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
    ys, xs = np.where(alpha > 40)
    if len(xs) < 12:
        return None
    weights = alpha[ys, xs].astype(np.float32) / 255.0
    return np.column_stack([xs, ys]).astype(np.float32), weights


def _stroke_color_hex(crop_bgr: np.ndarray, alpha: np.ndarray) -> str:
    core = alpha >= max(120, int(np.percentile(alpha[alpha > 40], 60)))
    pixels = crop_bgr[core] if bool(core.any()) else crop_bgr[alpha > 40]
    return bgr_to_hex(np.median(pixels.reshape(-1, 3), axis=0))


def _coverage_runs(t: np.ndarray) -> tuple[list[int], list[int], int]:
    """(occupied run lengths, interior gap lengths, span) along the axis."""
    ti = np.unique(np.round(t).astype(int))
    if len(ti) < 2:
        return [], [], 0
    span = int(ti.max() - ti.min()) + 1
    covered = np.zeros(span, dtype=bool)
    covered[ti - ti.min()] = True
    runs: list[int] = []
    gaps: list[int] = []
    run_len = gap_len = 0
    for c in covered:
        if c:
            run_len += 1
            if gap_len:
                gaps.append(gap_len)
                gap_len = 0
        else:
            gap_len += 1
            if run_len:
                runs.append(run_len)
                run_len = 0
    if run_len:
        runs.append(run_len)
    return runs, gaps, span


def _dash_pattern(t: np.ndarray) -> bool:
    """Detect interior gaps in the integer projection coverage."""
    runs, gaps, span = _coverage_runs(t)
    big_gaps = [g for g in gaps if g >= 3]
    return bool(len(big_gaps) >= 2 and sum(big_gaps) >= 0.08 * span)


def _irregular_glyph_row(t: np.ndarray) -> bool:
    """True when the coverage looks like a text row, not a line.

    An unerased text line (missed by OCR) is elongated and sparse
    enough to pass the connector role test, and the straight fit then
    flattens it into a solid bar. Real dashes have near-uniform segment
    and gap lengths; glyph runs vary wildly.
    """
    runs, gaps, _span = _coverage_runs(t)
    if len(runs) < 4 or len(gaps) < 3:
        return False
    runs_a = np.array(runs, dtype=float)
    gaps_a = np.array(gaps, dtype=float)
    run_cv = float(runs_a.std() / max(1e-3, runs_a.mean()))
    gap_cv = float(gaps_a.std() / max(1e-3, gaps_a.mean()))
    return bool(run_cv > 0.65 or gap_cv > 0.90)


def _fit_straight(pts: np.ndarray, weights: np.ndarray) -> dict | None:
    """PCA line fit with per-bin width profile.

    Returns endpoints, stroke width, dash flag and arrowhead flags, or
    None when the point cloud is not a single straight stroke.
    """
    center = pts.mean(axis=0)
    centered = pts - center
    cov = (centered.T @ centered) / max(1, len(pts) - 1)
    vals, vecs = np.linalg.eigh(cov)
    u = vecs[:, 1]                      # principal axis
    v = vecs[:, 0]                      # perpendicular
    t = centered @ u
    d = centered @ v
    length = float(t.max() - t.min())
    if length < _MIN_LINE_LEN:
        return None

    nb = int(np.clip(length / 5.0, 8, 48))
    edges = np.linspace(float(t.min()), float(t.max()) + 1e-3, nb + 1)
    bin_idx = np.clip(np.digitize(t, edges) - 1, 0, nb - 1)
    extents = np.full(nb, np.nan)
    counts = np.zeros(nb, dtype=int)
    for b in range(nb):
        sel = bin_idx == b
        counts[b] = int(sel.sum())
        if counts[b] >= 2:
            db = d[sel]
            extents[b] = float(np.percentile(db, 97)
                               - np.percentile(db, 3)) + 1.0
    if not bool(np.isfinite(extents).any()):
        return None
    # Profile extent (AA halo included) drives arrow/straightness
    # decisions; the emitted stroke width is the ink-mass integral,
    # which matches the drawn thickness instead of the halo.
    stroke_w = float(np.nanmedian(extents))
    if stroke_w <= 0 or stroke_w > max(14.0, length * 0.30):
        return None
    ink_cols = max(1, len(np.unique(np.round(t).astype(int))))
    # -0.8 compensates the matte's close/blur spread.
    ink_width = float(weights.sum()) / float(ink_cols) - 0.8
    # A solid stroke's ink mass fills its profile extent. Two nearby
    # parallel strokes, hollow rails, or a dense unerased TEXT ROW all
    # leave the band mostly empty — none of them may collapse into one
    # fat bar.
    if stroke_w > 4.0 and (ink_width + 0.8) < 0.45 * stroke_w:
        return None
    ink_width = float(np.clip(ink_width, 0.75, stroke_w))

    # Arrowheads widen the profile near an end.
    zone = max(2, int(round(nb * 0.18)))
    finite = np.isfinite(extents)

    def _zone_widening(sl: slice) -> bool:
        zone_ext = extents[sl][finite[sl]]
        if not len(zone_ext):
            return False
        return bool(zone_ext.max() >= max(2.1 * stroke_w, stroke_w + 4.0))

    arrow_start = _zone_widening(slice(0, zone))
    arrow_end = _zone_widening(slice(nb - zone, nb))

    # Straightness on the shaft only (arrow zones excluded).
    shaft = np.ones(len(pts), dtype=bool)
    if arrow_start:
        shaft &= bin_idx >= zone
    if arrow_end:
        shaft &= bin_idx < nb - zone
    if int(shaft.sum()) < 8:
        return None
    d_shaft = d[shaft]
    d_med = float(np.median(d_shaft))
    residual = float(np.percentile(np.abs(d_shaft - d_med), 96))
    if residual > max(2.2, 0.90 * stroke_w):
        return None

    if _irregular_glyph_row(t[shaft]):
        return None
    dash = _dash_pattern(t[shaft])

    p1 = center + float(t.min()) * u + d_med * v
    p2 = center + float(t.max()) * u + d_med * v
    points = [(float(p1[0]), float(p1[1])), (float(p2[0]), float(p2[1]))]
    # Normalise direction (left→right, then top→bottom) so downstream
    # output is stable regardless of the PCA eigenvector sign.
    if (points[0][0], points[0][1]) > (points[1][0], points[1][1]):
        points.reverse()
        arrow_start, arrow_end = arrow_end, arrow_start
    return {
        "kind": "straight",
        "points": points,
        "width_px": ink_width,
        "extent_px": stroke_w,
        "dash": "dash" if dash else None,
        "arrow_start": bool(arrow_start),
        "arrow_end": bool(arrow_end),
    }


def _fit_elbow(alpha: np.ndarray) -> dict | None:
    """Axis-aligned L-shape: one horizontal arm + one vertical arm."""
    fg = alpha > 40
    ys, xs = np.where(fg)
    if len(xs) < 24:
        return None
    x1, x2 = int(xs.min()), int(xs.max())
    y1, y2 = int(ys.min()), int(ys.max())
    w_fg, h_fg = x2 - x1 + 1, y2 - y1 + 1
    if w_fg < _MIN_LINE_LEN or h_fg < _MIN_LINE_LEN:
        return None

    row_occ = fg.sum(axis=1).astype(float)
    col_occ = fg.sum(axis=0).astype(float)
    nz_rows = row_occ[row_occ > 0]
    nz_cols = col_occ[col_occ > 0]
    if not len(nz_rows) or not len(nz_cols):
        return None
    stroke_r = float(np.median(nz_rows))    # ≈ vertical-arm thickness
    stroke_c = float(np.median(nz_cols))    # ≈ horizontal-arm thickness

    def _one_band(occ: np.ndarray, base: float, min_len: float):
        hot = occ >= max(base * 2.5, 12.0, min_len * 0.45)
        idx = np.flatnonzero(hot)
        if not len(idx):
            return None
        if int(idx.max() - idx.min()) + 1 != len(idx):    # must be contiguous
            return None
        if len(idx) > max(10.0, 3.0 * base):              # too thick = not an arm
            return None
        return float(idx.mean()), int(idx.min()), int(idx.max())

    h_band = _one_band(row_occ, stroke_r, w_fg)
    v_band = _one_band(col_occ, stroke_c, h_fg)
    if h_band is None or v_band is None:
        return None
    hy, hr1, hr2 = h_band
    vx, vc1, vc2 = v_band

    in_h = np.zeros_like(fg)
    in_h[hr1:hr2 + 1, :] = True
    in_v = np.zeros_like(fg)
    in_v[:, vc1:vc2 + 1] = True
    coverage = float((fg & (in_h | in_v)).sum()) / float(fg.sum())
    if coverage < _MIN_COVERAGE:
        return None

    h_xs = xs[(ys >= hr1) & (ys <= hr2)]
    v_ys = ys[(xs >= vc1) & (xs <= vc2)]
    if not len(h_xs) or not len(v_ys):
        return None
    hx1, hx2 = float(h_xs.min()), float(h_xs.max())
    vy1, vy2 = float(v_ys.min()), float(v_ys.max())
    # The horizontal free end is whichever extreme lies away from the
    # vertical arm; same for the vertical arm vs the horizontal band.
    h_far = hx1 if abs(hx1 - vx) >= abs(hx2 - vx) else hx2
    v_far = vy1 if abs(vy1 - hy) >= abs(vy2 - hy) else vy2
    if abs(h_far - vx) < _MIN_LINE_LEN * 0.7:
        return None
    if abs(v_far - hy) < _MIN_LINE_LEN * 0.7:
        return None

    path_len = abs(h_far - vx) + abs(v_far - hy)
    mass = float(alpha[fg].astype(np.float32).sum()) / 255.0
    band_w = float(min(hr2 - hr1 + 1, vc2 - vc1 + 1))
    ink_width = float(np.clip(mass / max(1.0, path_len) - 0.8,
                              0.75, band_w))
    return {
        "kind": "elbow",
        "points": [(h_far, hy), (float(vx), hy), (float(vx), v_far)],
        "width_px": ink_width,
        "extent_px": band_w,
        "dash": None,
        "arrow_start": False,
        "arrow_end": False,
    }


def fit_line_geometry(fg: np.ndarray) -> dict | None:
    """Fit straight/elbow line geometry to a binary foreground mask.

    Same models as `extract_connector_geometry` but without needing a
    colour crop — used by inventory-side decomposition to decide whether
    a residual stroke piece is really a line.
    """
    alpha = fg.astype(np.uint8) * 255
    fg_pts = _foreground_points(alpha)
    if fg_pts is None:
        return None
    pts, weights = fg_pts
    geom = _fit_straight(pts, weights)
    if geom is None:
        geom = _fit_elbow(alpha)
    return geom


def is_elbow_stroke_mask(fg: np.ndarray) -> bool:
    """True when a binary mask is an axis-aligned L-shaped thin stroke.

    Used by inventory role classification: PCA elongation misses elbow
    connectors (they spread along two axes), so they'd otherwise stay
    role-less and ship as opaque icon crops.
    """
    alpha = fg.astype(np.uint8) * 255
    return _fit_elbow(alpha) is not None


def merge_collinear_dash_runs(entries: list[dict]) -> list[dict]:
    """Fuse runs of small collinear fragments into single connector records.

    A dashed connector arrives from component detection as one tiny
    box per dash. Downstream that means a dozen 20×8 px PNG assets
    instead of one editable dashed line. Detect evenly-spaced collinear
    runs of small role-less fragments and replace them with one
    `role: "connector"` record spanning the union bbox; the dash
    pattern is then recovered by the straight-line fit.

    Returns the modified list (entries not in a run are untouched).
    """
    def _small_fragment(e: dict) -> bool:
        if e.get("type") != "image" or e.get("role") not in (None, ""):
            return False
        bbox = e.get("bbox") or []
        if len(bbox) != 4:
            return False
        x1, y1, x2, y2 = (int(v) for v in bbox)
        w, h = x2 - x1, y2 - y1
        return 2 <= min(w, h) <= 14 and max(w, h) <= 60

    frags = [e for e in entries if _small_fragment(e)]
    if len(frags) < 3:
        return entries

    def _center(e: dict) -> tuple[float, float]:
        x1, y1, x2, y2 = (float(v) for v in e["bbox"])
        return (x1 + x2) / 2.0, (y1 + y2) / 2.0

    used: set[int] = set()
    merged_records: list[dict] = []
    removed_ids: set[int] = set()
    for axis in ("x", "y"):
        order = sorted(range(len(frags)),
                       key=lambda i: _center(frags[i])[0 if axis == "x" else 1])
        chain: list[int] = []

        def _flush(chain: list[int]) -> None:
            if len(chain) < 3:
                return
            boxes = [tuple(float(v) for v in frags[i]["bbox"]) for i in chain]
            perp = ([(b[1] + b[3]) / 2.0 for b in boxes] if axis == "x"
                    else [(b[0] + b[2]) / 2.0 for b in boxes])
            if max(perp) - min(perp) > 4.0:
                return
            lo = [b[0] if axis == "x" else b[1] for b in boxes]
            hi = [b[2] if axis == "x" else b[3] for b in boxes]
            lengths = [h - l for l, h in zip(lo, hi)]
            gaps = [lo[k + 1] - hi[k] for k in range(len(boxes) - 1)]
            if min(gaps) < 1.0:
                return
            med_len = float(np.median(lengths))
            # Dash gaps are commensurate with the dash length; sparse
            # decorative dot columns are not.
            if max(gaps) > max(2.0 * med_len, 18.0):
                return
            med_gap = float(np.median(gaps))
            if max(gaps) > 2.5 * max(med_gap, 1.0):
                return
            if (max(hi) - min(lo)) < 50.0:
                return
            for i in chain:
                used.add(i)
                removed_ids.add(id(frags[i]))
            first = frags[chain[0]]
            union = [min(b[0] for b in boxes), min(b[1] for b in boxes),
                     max(b[2] for b in boxes), max(b[3] for b in boxes)]
            record = dict(first)
            record["id"] = f"{first.get('id', 'dash')}_dashrun"
            record["role"] = "connector"
            record["bbox"] = [int(round(v)) for v in union]
            record.pop("mask_path", None)
            merged_records.append(record)

        for i in order:
            if i in used:
                continue
            if not chain:
                chain = [i]
                continue
            prev = frags[chain[-1]]
            cur = frags[i]
            pc, cc = _center(prev), _center(cur)
            main_gap = (cc[0] - pc[0]) if axis == "x" else (cc[1] - pc[1])
            perp_off = abs(cc[1] - pc[1]) if axis == "x" else abs(cc[0] - pc[0])
            px1, py1, px2, py2 = (float(v) for v in prev["bbox"])
            plen = (px2 - px1) if axis == "x" else (py2 - py1)
            if perp_off <= 3.0 and 0.0 < main_gap <= max(3.0 * plen, 26.0):
                chain.append(i)
            else:
                _flush(chain)
                chain = [i]
        _flush(chain)

    if not merged_records:
        return entries
    out = [e for e in entries if id(e) not in removed_ids]
    out.extend(merged_records)
    return out


def stroke_background_contrast(crop_bgr: np.ndarray,
                               fg: np.ndarray) -> float:
    """Max-channel difference between stroke and surrounding colours.

    A real connector is visible against its background; a near-zero
    contrast "line" is an artefact of residual masks (anti-aliased card
    fringes, erased regions) and must not become a drawn shape.
    """
    if not bool(fg.any()):
        return 0.0
    stroke = np.median(crop_bgr[fg].reshape(-1, 3), axis=0)
    ring = cv2.dilate(fg.astype(np.uint8), np.ones((3, 3), np.uint8),
                      iterations=3) > 0
    ring &= ~fg
    if not bool(ring.any()):
        ring = ~fg
    if not bool(ring.any()):
        return 0.0
    bg = np.median(crop_bgr[ring].reshape(-1, 3), axis=0)
    return float(np.abs(stroke - bg).max())


# Below this stroke-vs-surround contrast a fitted line would be
# invisible in the source — it is a residual-mask artefact.
MIN_STROKE_CONTRAST = 20.0


def _looks_like_glyph_row(crop_bgr: np.ndarray, alpha: np.ndarray) -> bool:
    """Component-density test separating text rows from dashed lines.

    Dash/dot segments are SOLID (ink fills each segment's own bbox);
    glyphs are stroke art (~40% fill). A dense bold CJK row that slipped
    past the connector role test defeats every projection statistic —
    per-component density is the discriminator that survives.

    Works on a strict raw ink mask (no morphology, no hole filling):
    the matte's hole-filling turns glyphs into solid blobs and would
    hide exactly the structure this test needs.
    """
    bg_px = crop_bgr[alpha < 16]
    if len(bg_px) < 12:
        return False
    bg = np.median(bg_px.reshape(-1, 3), axis=0)
    ink = np.abs(crop_bgr.astype(np.int16) - bg.astype(np.int16)
                 ).max(axis=2) > 45
    ink &= alpha > 40
    n, _labels, stats, _ = cv2.connectedComponentsWithStats(
        ink.astype(np.uint8), 8)
    densities = []
    widths = []
    for i in range(1, n):
        _x, _y, cw, ch, area = (int(v) for v in stats[i])
        if area < 8:
            continue
        densities.append(area / float(max(1, cw * ch)))
        widths.append(float(cw))
    if len(densities) < 3:
        return False
    dens = np.array(densities)
    ws = np.array(widths)
    w_cv = float(ws.std() / max(1e-3, ws.mean()))
    # Dash segments and dots are UNIFORM stamps (measured width CV
    # ≤0.05); glyphs are sparse AND vary in width (CJK rows ≥0.28,
    # mixed text ≥0.5). Require both signals so anti-aliased dots
    # (density dips below the solid band) stay accepted.
    if float(np.median(dens)) < 0.72 and w_cv > 0.15:
        return True
    # Many solid-but-ragged stamps (disconnected Latin/stroke glyphs)
    # are not a dash pattern either — dashes never vary this much.
    return bool(len(densities) >= 6 and w_cv > 0.40)


def extract_connector_geometry(crop_bgr: np.ndarray,
                               alpha: np.ndarray) -> dict | None:
    """Fit a straight or elbow line model to a connector matte.

    Returns crop-local geometry plus stroke colour, or None when the
    crop is not confidently a line (caller keeps the PNG fallback).
    """
    if crop_bgr.size == 0 or alpha.shape[:2] != crop_bgr.shape[:2]:
        return None
    fg_pts = _foreground_points(alpha)
    if fg_pts is None:
        return None
    pts, weights = fg_pts
    geom = _fit_straight(pts, weights)
    if geom is None:
        geom = _fit_elbow(alpha)
    if geom is None:
        return None
    if stroke_background_contrast(crop_bgr, alpha > 40) < MIN_STROKE_CONTRAST:
        return None
    if _looks_like_glyph_row(crop_bgr, alpha):
        return None
    geom["color"] = _stroke_color_hex(crop_bgr, alpha)
    return geom
