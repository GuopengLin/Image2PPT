"""Per-component shape classifiers used by the inventory builder.

These decide whether a connected component looks like a container
(card / panel) or a connector (line / arrow). The builder uses the
returned role tag to pick crop sources and routing further down.
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

PAGE_DIR = Path(__file__).resolve().parents[1]
if str(PAGE_DIR) not in sys.path:
    sys.path.insert(0, str(PAGE_DIR))

from _heuristics import s_area, s_length  # noqa: E402
from layout.connector_geometry import is_elbow_stroke_mask  # noqa: E402


def _connector_fg_mask(crop: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    diff_white = np.abs(crop.astype(np.int16) - 255).max(axis=2)
    return (
        ((gray < 235) & (diff_white > 8))
        | ((hsv[:, :, 1] > 30) & (diff_white > 6))
    )


def is_connector_like(crop: np.ndarray, scale: float) -> bool:
    if crop.size == 0:
        return False
    h, w = crop.shape[:2]
    if max(w, h) < s_length(26, scale):
        return False
    if w * h > s_area(24000, scale):
        # PCA elongation can't see elbow connectors (they spread along
        # two axes and their bbox area is large); give L-shaped thin
        # strokes their own bounded check.
        if w * h > s_area(160000, scale) or max(w, h) > s_length(640, scale):
            return False
        fg = _connector_fg_mask(crop)
        fg_count = int(fg.sum())
        if fg_count < s_area(40, scale):
            return False
        if fg_count / float(max(1, w * h)) > 0.12:
            return False
        return is_elbow_stroke_mask(fg)
    fg = _connector_fg_mask(crop)
    fg_count = int(fg.sum())
    if fg_count < s_area(12, scale):
        return False
    density = fg_count / float(max(1, w * h))
    aspect = max(w, h) / float(max(1, min(w, h)))
    if aspect >= 5.0 and density <= 0.60:
        return True
    if density > 0.28:
        return False
    ys, xs = np.where(fg)
    if len(xs) < 8:
        return False
    pts = np.column_stack([xs.astype(np.float32), ys.astype(np.float32)])
    pts -= pts.mean(axis=0, keepdims=True)
    cov = (pts.T @ pts) / max(1, len(pts) - 1)
    vals = np.linalg.eigvalsh(cov)
    if vals[0] <= 1e-3:
        return True
    return bool(vals[1] / vals[0] >= 9.0)


def is_container_like(crop: np.ndarray, scale: float) -> bool:
    if crop.size == 0:
        return False
    h, w = crop.shape[:2]
    if w < s_length(42, scale) or h < s_length(22, scale):
        return False
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    diff_white = np.abs(crop.astype(np.int16) - 255).max(axis=2)
    fg = (
        ((gray < 248) & (diff_white > 8))
        | ((hsv[:, :, 1] > 10) & (diff_white > 6))
    )
    band = max(2, min(max(2, min(w, h) // 10), s_length(10, scale)))
    top = float(np.any(fg[:band, :], axis=0).sum()) / max(1, w)
    bottom = float(np.any(fg[-band:, :], axis=0).sum()) / max(1, w)
    left = float(np.any(fg[:, :band], axis=1).sum()) / max(1, h)
    right = float(np.any(fg[:, -band:], axis=1).sum()) / max(1, h)
    edge_like = (
        sum(v >= 0.24 for v in (top, bottom, left, right)) >= 3
        and max(top, bottom) >= 0.45
        and max(left, right) >= 0.45
    )
    fill_density = float(fg.sum()) / float(max(1, w * h))
    filled_shape = (
        fill_density >= 0.45
        and max(w, h) >= s_length(70, scale)
        and max(w, h) / float(max(1, min(w, h))) <= 8.0
    )
    return bool(edge_like or filled_shape)


def foreground_role_for_box(img: np.ndarray, x1: int, y1: int,
                            x2: int, y2: int, scale: float) -> str | None:
    crop = img[y1:y2, x1:x2]
    if is_container_like(crop, scale):
        return "container"
    if is_connector_like(crop, scale):
        return "connector"
    return None
