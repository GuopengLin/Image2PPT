"""Guards that keep non-line content out of native connector emission."""
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

from layout.connector_geometry import extract_connector_geometry  # noqa: E402
from layout.icon_alpha import _line_art_matte  # noqa: E402
from inventory.connector_split import _solid_card_cores  # noqa: E402


def _geometry_for(crop: np.ndarray):
    _fg, alpha = _line_art_matte(crop)
    return extract_connector_geometry(crop, alpha)


class ConnectorGuardTests(unittest.TestCase):
    def test_near_invisible_stroke_rejected(self) -> None:
        # A "line" whose colour barely differs from the page background
        # is a residual-mask artefact, not a drawable connector.
        crop = np.full((20, 200, 3), 255, np.uint8)
        cv2.line(crop, (5, 10), (195, 10), (246, 248, 239), 3, cv2.LINE_AA)
        self.assertIsNone(_geometry_for(crop))

    def test_unerased_text_row_rejected(self) -> None:
        # Uniform CJK-like glyph stamps: elongated, sparse, but the
        # per-component width/density signature is text, not dashes.
        rng = np.random.default_rng(11)
        crop = np.full((26, 300, 3), 255, np.uint8)
        x = 6
        while x < 280:
            w = int(rng.integers(11, 19))
            for _ in range(5):
                sx = x + int(rng.integers(0, max(1, w - 3)))
                sy = int(rng.integers(4, 18))
                if rng.random() < 0.5:
                    cv2.line(crop, (sx, sy),
                             (min(x + w, sx + int(rng.integers(4, w))), sy),
                             (40, 40, 90), 2)
                else:
                    cv2.line(crop, (sx, sy),
                             (sx, min(21, sy + int(rng.integers(4, 14)))),
                             (40, 40, 90), 2)
            x += w + 3
        self.assertIsNone(_geometry_for(crop))

    def test_dotted_line_accepted(self) -> None:
        crop = np.full((20, 220, 3), 255, np.uint8)
        for x in range(10, 210, 14):
            cv2.circle(crop, (x, 10), 3, (90, 60, 30), -1, cv2.LINE_AA)
        geom = _geometry_for(crop)
        self.assertIsNotNone(geom)
        self.assertEqual(geom["dash"], "dash")

    def test_solid_triangle_not_carved_into_cores(self) -> None:
        # Occupancy trimming keeps a triangle's dense centre slab; the
        # trimmed-area guard must reject it so the artwork stays whole.
        fg = np.zeros((300, 340), dtype=bool)
        pts = np.array([[170, 20], [30, 280], [310, 280]], np.int32)
        m = np.zeros((300, 340), np.uint8)
        cv2.fillPoly(m, [pts], 255)
        fg |= m > 0
        self.assertEqual(_solid_card_cores(fg, 1.0), [])


if __name__ == "__main__":
    unittest.main()
