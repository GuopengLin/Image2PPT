"""Icon-asset alpha + text scrub helpers.

Three responsibilities live here:

* `_ICON_PAD_ROLES` / `_SPARSE_VISUAL_ROLES` — the role tags that mark
  an inventory record as "icon-like" (used by bbox_shape padding +
  builder routing).
* `_scrub_text_boxes_from_icon_crop` — a second local fill pass that
  removes faint text residue from saturated icon backgrounds.
* `_line_art_alpha` — alpha mask for sparse stroke icons that keeps
  filled foreground islands + their enclosed holes but lets surrounding
  panel/background pixels go transparent.
* `_simple_background_strip` — used by `_pad_icon_bbox` to decide
  whether a candidate icon padding strip is plain enough to absorb.
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

SCRIPTS_ROOT = Path(__file__).resolve().parents[2]    # scripts/
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

from icon import inpaint_region_inplace  # noqa: E402


_ICON_PAD_ROLES = {
    "small_icon",
    "preserve_visual_icon",
    "subicon",
    "badge_subicon",
    "connector",
    "line_subicon",
}
_SPARSE_VISUAL_ROLES = _ICON_PAD_ROLES | {"thin_rule", "outline"}


def _scrub_text_boxes_from_icon_crop(
    crop_bgr: np.ndarray,
    crop_box: tuple[int, int, int, int],
    text_boxes: list[tuple[int, int, int, int]],
    *,
    alpha: np.ndarray | None = None,
) -> np.ndarray:
    """Remove OCR-text residue from a clean-cropped icon asset.

    The source crop is already text-erased, but high-contrast text on
    saturated icon fills can leave a faint local texture after the
    global eraser. A second, crop-local fill keeps the icon asset
    neutral under the editable PPT text that will render above it.
    """
    if crop_bgr.size == 0 or not text_boxes:
        return crop_bgr
    x1, y1, x2, y2 = crop_box
    h, w = crop_bgr.shape[:2]
    if h == 0 or w == 0:
        return crop_bgr
    support = None
    if alpha is not None and alpha.shape[:2] == (h, w):
        support = alpha > 16
    out = crop_bgr.copy()
    for tx1, ty1, tx2, ty2 in text_boxes:
        lx1 = max(0, tx1 - x1)
        ly1 = max(0, ty1 - y1)
        lx2 = min(w, tx2 - x1)
        ly2 = min(h, ty2 - y1)
        if lx2 <= lx1 or ly2 <= ly1:
            continue
        mask = np.zeros((h, w), dtype=bool)
        mask[ly1:ly2, lx1:lx2] = True
        if support is not None:
            mask &= support
        if int(mask.sum()) < 4:
            continue

        pad = max(4, min(10, int(round(max(lx2 - lx1, ly2 - ly1) * 0.18))))
        rx1, ry1 = max(0, lx1 - pad), max(0, ly1 - pad)
        rx2, ry2 = min(w, lx2 + pad), min(h, ly2 + pad)
        ring = np.zeros((h, w), dtype=bool)
        ring[ry1:ry2, rx1:rx2] = True
        ring &= ~mask
        if support is not None:
            ring &= support
        pixels = out[ring]
        fill = None
        if len(pixels) >= 12:
            flat = pixels.reshape(-1, 3).astype(np.uint8)
            quant = (flat.astype(np.uint16) // 16).astype(np.uint8)
            keys, counts = np.unique(quant, axis=0, return_counts=True)
            key = keys[int(np.argmax(counts))]
            cluster = flat[(quant == key).all(axis=1)]
            if len(cluster) < max(8, int(0.20 * len(flat))):
                cluster = flat
            fill = np.median(cluster.reshape(-1, 3), axis=0).astype(np.uint8)
        inpaint_region_inplace(out, mask, radius=2, fill_color=fill)
    return out


def _simple_background_strip(source: np.ndarray,
                             box: tuple[int, int, int, int]) -> bool:
    """Return True when a candidate padding strip looks like plain bg.

    The strip can be white slide bg or a uniform card/bg fill. We avoid
    requiring near-white because many icons sit on coloured panels.
    """
    x1, y1, x2, y2 = box
    h_img, w_img = source.shape[:2]
    x1 = max(0, min(w_img, int(x1)))
    x2 = max(0, min(w_img, int(x2)))
    y1 = max(0, min(h_img, int(y1)))
    y2 = max(0, min(h_img, int(y2)))
    if x2 <= x1 or y2 <= y1:
        return False
    strip = source[y1:y2, x1:x2]
    if strip.size == 0:
        return False

    flat = strip.reshape(-1, 3).astype(np.int16)
    med = np.median(flat, axis=0)
    diff = np.abs(flat - med).max(axis=1)
    stable = (
        float(np.percentile(diff, 90)) <= 22
        or float((diff <= 28).sum()) / max(1, len(diff)) >= 0.90
    )

    gray = cv2.cvtColor(strip, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(strip, cv2.COLOR_BGR2HSV)
    light_neutral = (
        float(((gray > 238) & (hsv[:, :, 1] < 35)).sum())
        / max(1, gray.size)
        >= 0.85
    )
    return bool(stable or light_neutral)


def _border_median_bgr(crop_bgr: np.ndarray) -> np.ndarray:
    h, w = crop_bgr.shape[:2]
    ring = max(1, min(4, min(h, w) // 8))
    border = np.concatenate([
        crop_bgr[:ring, :].reshape(-1, 3),
        crop_bgr[-ring:, :].reshape(-1, 3),
        crop_bgr[:, :ring].reshape(-1, 3),
        crop_bgr[:, -ring:].reshape(-1, 3),
    ])
    if not len(border):
        return np.array([255.0, 255.0, 255.0])
    return np.median(border, axis=0)


def _border_is_uniform(crop_bgr: np.ndarray) -> bool:
    """True when the border ring is one colour (flat-bg hypothesis holds)."""
    h, w = crop_bgr.shape[:2]
    ring = max(1, min(4, min(h, w) // 8))
    border = np.concatenate([
        crop_bgr[:ring, :].reshape(-1, 3),
        crop_bgr[-ring:, :].reshape(-1, 3),
        crop_bgr[:, :ring].reshape(-1, 3),
        crop_bgr[:, -ring:].reshape(-1, 3),
    ]).astype(np.int16)
    if not len(border):
        return True
    med = np.median(border, axis=0)
    spread = np.abs(border - med).max(axis=1)
    return float(np.percentile(spread, 90)) <= 14.0


def _estimate_background_image(crop_bgr: np.ndarray,
                               fg_mask: np.ndarray) -> np.ndarray:
    """Per-pixel background estimate underneath the foreground.

    A single border-median colour breaks on gradient fills and on icons
    that straddle a card edge (the ring straddles two colours and the
    median matches neither). Inpainting the dilated foreground from its
    surroundings gives every pixel a local background colour instead.
    """
    h, w = crop_bgr.shape[:2]
    grow = cv2.dilate(fg_mask.astype(np.uint8),
                      np.ones((3, 3), np.uint8), iterations=2)
    covered = float(grow.sum()) / float(max(1, h * w))
    if covered >= 0.90:
        # Almost no background left to sample from — fall back to the
        # flat border estimate.
        flat = _border_median_bgr(crop_bgr)
        return np.tile(flat.astype(np.float32), (h, w, 1))
    bg = cv2.inpaint(crop_bgr, grow * 255, 3, cv2.INPAINT_TELEA)
    # Smooth inpaint streaks; the mask is gone so this can't smear
    # foreground colours back in.
    k = 5 if min(h, w) >= 5 else max(1, min(h, w) // 2 * 2 + 1)
    return cv2.blur(bg, (k, k)).astype(np.float32)


def _local_foreground_seed(crop_bgr: np.ndarray) -> np.ndarray:
    """Bg-hypothesis-free foreground seed for non-uniform backgrounds.

    A wide median filter removes thin strokes but follows gradients, so
    the residual highlights line art without assuming any global bg
    colour. Canny edges add stroke/blob boundaries the residual may
    miss.
    """
    h, w = crop_bgr.shape[:2]
    k = min(15, max(9, (min(h, w) // 6) | 1))
    if min(h, w) < 9:
        k = max(3, (min(h, w) // 2) | 1)
    bg_med = cv2.medianBlur(crop_bgr, k)
    residual = np.abs(
        crop_bgr.astype(np.int16) - bg_med.astype(np.int16)).max(axis=2)
    seed = residual > 10
    edges = cv2.Canny(crop_bgr, 40, 120) > 0
    seed |= edges
    return cv2.dilate(seed.astype(np.uint8),
                      np.ones((3, 3), np.uint8), 1).astype(bool)


def _decontaminate_edges(crop_bgr: np.ndarray, alpha: np.ndarray,
                         bg_est: np.ndarray) -> np.ndarray:
    """Un-blend anti-aliased edge pixels from the baked page background.

    An edge pixel is physically `a*fg + (1-a)*bg_page`. Keeping the
    blended colour under a partial alpha leaves a page-coloured fringe
    once PowerPoint composites the PNG over a different fill. Solving
    for fg removes the halo.
    """
    a = alpha.astype(np.float32) / 255.0
    partial = (alpha > 24) & (alpha < 250)
    if not bool(partial.any()):
        return crop_bgr
    out = crop_bgr.astype(np.float32)
    a3 = a[..., None]
    fg = (out - (1.0 - a3) * bg_est) / np.maximum(a3, 0.10)
    fg = np.clip(fg, 0, 255)
    out[partial] = fg[partial]
    return out.astype(np.uint8)


def _linear_edge_alpha(crop_bgr: np.ndarray, bg_est: np.ndarray,
                       alpha: np.ndarray, core: np.ndarray,
                       edge_px: np.ndarray) -> np.ndarray:
    """Physically-motivated alpha for anti-aliased boundary pixels.

    The soft ramp saturates fast, so a pixel that is only ~20% fg can
    get ~90% opacity; decontamination then divides by a too-large alpha
    and the page colour survives. Projecting the pixel onto the
    bg→nearest-core-fg colour axis recovers the true blend fraction.
    """
    if not bool(core.any()) or not bool(edge_px.any()):
        return alpha
    inv = np.where(core, 0, 1).astype(np.uint8)
    _dist, labels = cv2.distanceTransformWithLabels(
        inv, cv2.DIST_L2, 3, labelType=cv2.DIST_LABEL_PIXEL)
    lut = np.zeros((int(labels.max()) + 1, 3), np.float32)
    lut[labels[core]] = crop_bgr[core].astype(np.float32)
    f = lut[labels]
    c = crop_bgr.astype(np.float32)
    num = ((c - bg_est) * (f - bg_est)).sum(axis=2)
    den = ((f - bg_est) ** 2).sum(axis=2)
    a_lin = np.clip(num / np.maximum(den, 1e-3), 0.0, 1.0)
    out = alpha.copy()
    out[edge_px] = np.minimum(
        alpha[edge_px],
        np.clip(a_lin[edge_px] * 255.0 + 12.0, 0, 255).astype(np.uint8))
    return out


def _line_art_keep_mask(crop_bgr: np.ndarray,
                        diff: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2HSV)
    sat = hsv[:, :, 1]
    keep = (
        (diff > 16)
        & ((sat > 16) | (gray < 238) | (gray > 245))
    )
    return cv2.morphologyEx(
        keep.astype(np.uint8) * 255,
        cv2.MORPH_CLOSE,
        np.ones((3, 3), np.uint8),
        iterations=1,
    ) > 0


def _line_art_matte(crop_bgr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(decontaminated BGR, alpha) matte for sparse line icons.

    The foreground can be a pure stroke icon, a white badge on a coloured
    title bar, or a filled warning triangle with white holes. We preserve
    filled foreground islands and their enclosed holes, but leave
    ordinary surrounding panel/background pixels transparent. Background
    is estimated per pixel (gradient/card-edge tolerant) and anti-aliased
    edges are un-blended from it so no page-colour halo survives.
    """
    h, w = crop_bgr.shape[:2]
    if h == 0 or w == 0:
        return crop_bgr, np.zeros((h, w), dtype=np.uint8)

    if _border_is_uniform(crop_bgr):
        # Flat background: the proven global border-median key. The bg
        # estimate is still materialised per-pixel for decontamination.
        bg_flat = _border_median_bgr(crop_bgr)
        bg_est = np.tile(bg_flat.astype(np.float32), (h, w, 1))
        diff = np.abs(crop_bgr.astype(int) - bg_flat.astype(int)).max(axis=2)
        keep = _line_art_keep_mask(crop_bgr, diff)
    else:
        # Gradient / card-edge background: seed the foreground without a
        # bg-colour hypothesis, then re-key against a local per-pixel
        # background estimate.
        seed = _local_foreground_seed(crop_bgr)
        bg_est = _estimate_background_image(crop_bgr, seed)
        diff = np.abs(crop_bgr.astype(np.float32) - bg_est).max(axis=2)
        keep = _line_art_keep_mask(crop_bgr, diff)

    alpha_mask = np.zeros((h, w), dtype=bool)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(
        keep.astype(np.uint8), 8)
    total = max(1, h * w)
    for i in range(1, n):
        x, y, cw, ch, area = (int(v) for v in stats[i])
        if area < max(3, int(round(0.0006 * total))):
            continue
        component = labels == i
        density = area / float(max(1, cw * ch))
        filled_component = component
        if density >= 0.24:
            comp_u8 = component.astype(np.uint8) * 255
            contours, _ = cv2.findContours(
                comp_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                filled = np.zeros_like(comp_u8)
                cv2.drawContours(filled, contours, -1, 255, cv2.FILLED)
                filled_component = filled > 0
        alpha_mask |= filled_component

    alpha = np.zeros((h, w), dtype=np.uint8)
    soft = np.clip((diff.astype(int) - 8) * 14, 0, 255).astype(np.uint8)
    alpha[keep] = soft[keep]
    alpha[alpha_mask] = np.maximum(alpha[alpha_mask], 245)
    # Confident-core = eroded filled islands (their outermost ring is
    # anti-aliased and must stay refinable) plus pixels near the top of
    # the crop's own contrast range.
    interior = cv2.erode(alpha_mask.astype(np.uint8),
                         np.ones((3, 3), np.uint8), iterations=1) > 0
    d95 = float(np.percentile(diff[keep], 95)) if bool(keep.any()) else 0.0
    core = interior | (diff >= max(24.0, 0.6 * d95))
    alpha = _linear_edge_alpha(crop_bgr, bg_est, alpha, core,
                               (keep | alpha_mask) & ~core)
    alpha = cv2.GaussianBlur(alpha, (3, 3), 0)
    # The blur bleeds opacity one pixel outward onto pure-background
    # pixels. Their colour IS the page background, so any alpha there
    # renders as a page-coloured halo on other fills — drop it.
    stray = (~alpha_mask) & (diff < 8)
    alpha[stray] = 0
    fg = _decontaminate_edges(crop_bgr, alpha, bg_est)
    return fg, alpha


def _line_art_alpha(crop_bgr: np.ndarray) -> np.ndarray:
    """Alpha-only view of `_line_art_matte` for mask consumers."""
    _fg, alpha = _line_art_matte(crop_bgr)
    return alpha


def refine_masked_asset_edges(crop_bgr: np.ndarray,
                              alpha: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Soften + decontaminate the boundary of a hard sidecar mask.

    Detector masks are near-binary; their boundary pixels keep the page
    background blended into the RGB, which shows as a fringe on any
    other fill. Feather only the outermost band and un-blend it from a
    local background estimate — interior pixels are left untouched.
    """
    h, w = crop_bgr.shape[:2]
    if h == 0 or w == 0 or alpha.shape[:2] != (h, w) or h * w > 400_000:
        return crop_bgr, alpha
    hard = alpha > 128
    if not bool(hard.any()) or bool(hard.all()):
        return crop_bgr, alpha
    kernel = np.ones((3, 3), np.uint8)
    interior = cv2.erode(hard.astype(np.uint8), kernel, iterations=1) > 0
    band = hard & ~interior
    if not bool(band.any()):
        return crop_bgr, alpha
    bg_est = _estimate_background_image(crop_bgr, hard)
    diff = np.abs(crop_bgr.astype(np.float32) - bg_est).max(axis=2)
    soft = np.clip((diff - 6.0) * 16.0, 0, 255).astype(np.uint8)
    out_alpha = alpha.copy()
    out_alpha[band] = np.minimum(out_alpha[band], soft[band])
    fg = _decontaminate_edges(crop_bgr, out_alpha, bg_est)
    return fg, out_alpha
