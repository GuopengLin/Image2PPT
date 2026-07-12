"""Eraser-painted pixels must not resurface as phantom foreground."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
PAGE_DIR = REPO_ROOT / "scripts" / "page"
if str(PAGE_DIR) not in sys.path:
    sys.path.insert(0, str(PAGE_DIR))

from build_inventory import _foreground_mask, detect_components  # noqa: E402


class ErasedMaskSuppressionTests(unittest.TestCase):
    def _page_with_offcolor_patch(self):
        # White canvas; the eraser filled a text region with a colour that
        # disagrees with the corner-estimated canvas background.
        img = np.full((720, 1280, 3), 255, np.uint8)
        img[300:340, 400:700] = (235, 220, 205)
        suppress = np.zeros((720, 1280), dtype=bool)
        suppress[300:340, 400:700] = True
        return img, suppress

    def test_phantom_without_suppression(self) -> None:
        img, _ = self._page_with_offcolor_patch()
        comps = detect_components(img, 12, 4)
        self.assertTrue(comps)  # documents the failure mode

    def test_suppression_removes_phantom(self) -> None:
        img, suppress = self._page_with_offcolor_patch()
        comps = detect_components(img, 12, 4, suppress=suppress)
        self.assertEqual(comps, [])

    def test_real_content_survives_suppression(self) -> None:
        img, suppress = self._page_with_offcolor_patch()
        img[500:560, 100:220] = (30, 60, 200)  # a real icon elsewhere
        comps = detect_components(img, 12, 4, suppress=suppress)
        self.assertEqual(len(comps), 1)
        x1, y1, x2, y2, _area = comps[0]
        self.assertLess(abs(x1 - 100), 6)
        self.assertLess(abs(y1 - 500), 6)

    def test_foreground_mask_kills_boundary_edge_ring(self) -> None:
        # The Canny layer marks the colour boundary just outside the
        # painted region; the grown suppression must remove that ring too.
        img, suppress = self._page_with_offcolor_patch()
        mask = _foreground_mask(img, 4, suppress=suppress)
        self.assertEqual(int(mask.sum()), 0)


if __name__ == "__main__":
    unittest.main()
