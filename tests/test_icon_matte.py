"""Matting quality tests for the per-pixel background + decontamination path."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
PAGE_DIR = REPO_ROOT / "scripts" / "page"
SCRIPTS_DIR = REPO_ROOT / "scripts"
for path in (SCRIPTS_DIR, PAGE_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from layout.icon_alpha import (  # noqa: E402
    _line_art_alpha,
    _line_art_matte,
    refine_masked_asset_edges,
)


class GradientBackgroundMatteTests(unittest.TestCase):
    def test_icon_on_vertical_gradient_bg_is_extracted(self) -> None:
        # Background sweeps 250→170 gray top to bottom; a single global
        # border-median would sit at ~210 and mis-key both halves.
        h, w = 90, 90
        crop = np.zeros((h, w, 3), dtype=np.uint8)
        for y in range(h):
            v = int(250 - 80 * y / (h - 1))
            crop[y, :] = (v, v, v)
        blue = (150, 70, 10)
        cv2.circle(crop, (45, 45), 22, blue, 4, cv2.LINE_AA)
        cv2.line(crop, (33, 45), (43, 55), blue, 4, cv2.LINE_AA)

        alpha = _line_art_alpha(crop)

        # Stroke pixels opaque at both the light top and dark bottom arc.
        self.assertGreater(int(alpha[24, 45]), 180)   # top of circle
        self.assertGreater(int(alpha[66, 45]), 180)   # bottom of circle
        # Background corners stay transparent in both gradient halves.
        self.assertLess(int(alpha[3, 3]), 24)
        self.assertLess(int(alpha[h - 4, w - 4]), 24)

    def test_matte_edges_are_decontaminated_from_page_bg(self) -> None:
        # Red disc anti-aliased over a saturated blue panel. Composite the
        # matte over white: without un-blending, edge pixels keep a blue
        # tint (halo).
        crop = np.zeros((80, 80, 3), dtype=np.uint8)
        crop[:] = np.array([180, 90, 20], dtype=np.uint8)  # blue-ish BGR
        cv2.circle(crop, (40, 40), 24, (40, 40, 210), -1, cv2.LINE_AA)

        fg, alpha = _line_art_matte(crop)

        a = alpha.astype(np.float32)[..., None] / 255.0
        comp = (fg.astype(np.float32) * a
                + 255.0 * (1.0 - a)).astype(np.uint8)
        edge_band = (alpha > 40) & (alpha < 220)
        self.assertGreater(int(edge_band.sum()), 10)
        b = comp[..., 0].astype(int)
        r = comp[..., 2].astype(int)
        # Edge pixels composited on white must stay red-dominant — a blue
        # halo would push B above R.
        blue_halo = edge_band & (b > r + 30)
        self.assertLess(
            int(blue_halo.sum()), max(3, int(0.05 * edge_band.sum())))

    def test_refine_masked_asset_edges_softens_hard_mask(self) -> None:
        crop = np.zeros((60, 60, 3), dtype=np.uint8)
        crop[:] = np.array([200, 200, 200], dtype=np.uint8)
        cv2.circle(crop, (30, 30), 18, (30, 30, 30), -1, cv2.LINE_AA)
        hard = np.zeros((60, 60), dtype=np.uint8)
        cv2.circle(hard, (30, 30), 20, 255, -1)  # 2px looser than the ink

        fg, alpha = refine_masked_asset_edges(crop, hard)

        # Interior stays opaque, the over-wide mask rim goes transparent
        # because those pixels match the local background.
        self.assertEqual(int(alpha[30, 30]), 255)
        ys, xs = np.where(hard > 0)
        rim = (np.hypot(ys - 30.0, xs - 30.0) > 18.8)
        rim_alpha = alpha[ys[rim], xs[rim]]
        self.assertLess(float(np.median(rim_alpha)), 80.0)


if __name__ == "__main__":
    unittest.main()
