"""Decomposition of components where connectors touch/cross cards."""
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

from inventory.connector_split import (  # noqa: E402
    _solid_card_cores,
    split_connectors_from_component,
)


def _diagram(crossing_visible: bool) -> tuple[np.ndarray, np.ndarray]:
    """Two cards joined by a touching arrow + a line at card height.

    When `crossing_visible` the line is drawn over the card; otherwise it
    stops at the card edges (line runs behind the card).
    """
    img = np.full((400, 900, 3), 255, np.uint8)
    for (x, y) in [(60, 60), (560, 60)]:
        cv2.rectangle(img, (x, y), (x + 240, y + 200), (240, 228, 214), -1)
        cv2.rectangle(img, (x, y), (x + 240, y + 200), (160, 120, 60), 2)
    # arrow touching both cards
    cv2.arrowedLine(img, (300, 120), (560, 120), (90, 90, 90), 3,
                    cv2.LINE_AA, tipLength=0.06)
    # horizontal line at card-2 height
    if crossing_visible:
        cv2.line(img, (330, 160), (880, 160), (40, 40, 200), 3, cv2.LINE_AA)
    else:
        cv2.line(img, (330, 160), (560, 160), (40, 40, 200), 3, cv2.LINE_AA)
        cv2.line(img, (800, 160), (880, 160), (40, 40, 200), 3, cv2.LINE_AA)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    fg = (gray < 250) | (hsv[:, :, 1] > 8)
    return img, fg


class SolidCardCoreTests(unittest.TestCase):
    def test_cores_found_and_strokes_excluded(self) -> None:
        img, fg = _diagram(crossing_visible=False)
        cores = _solid_card_cores(fg, 1.0)
        self.assertEqual(len(cores), 2)
        for x1, y1, x2, y2 in cores:
            self.assertGreater(x2 - x1, 200)
            self.assertGreater(y2 - y1, 160)
            # No core may swallow the connector corridor between cards.
            self.assertFalse(x1 < 320 < x2 and x1 < 540 < x2)


class ConnectorSplitTests(unittest.TestCase):
    def test_touching_arrow_recovered(self) -> None:
        img, fg = _diagram(crossing_visible=False)
        result = split_connectors_from_component(fg, img, [], 1.0)
        self.assertIsNotNone(result)
        boxes = [p["bbox"] for p in result["connectors"]]
        # An arrow-height piece between the two cards must exist.
        arrow = [b for b in boxes if b[1] < 140 and b[3] > 100]
        self.assertTrue(arrow)
        self.assertGreaterEqual(result["explained_frac"], 0.85)

    def test_crossing_visible_line_is_bridged_and_front(self) -> None:
        img, fg = _diagram(crossing_visible=True)
        result = split_connectors_from_component(fg, img, [], 1.0)
        self.assertIsNotNone(result)
        lines = [p for p in result["connectors"]
                 if p["bbox"][1] > 140 and p["bbox"][3] < 185]
        self.assertEqual(len(lines), 1)
        piece = lines[0]
        self.assertTrue(piece["z_front"])
        self.assertLess(piece["bbox"][0], 340)
        self.assertGreater(piece["bbox"][2], 860)

    def test_occluded_line_stays_two_pieces(self) -> None:
        img, fg = _diagram(crossing_visible=False)
        result = split_connectors_from_component(fg, img, [], 1.0)
        self.assertIsNotNone(result)
        lines = [p for p in result["connectors"]
                 if p["bbox"][1] > 140 and p["bbox"][3] < 185]
        self.assertEqual(len(lines), 2)
        self.assertFalse(any(p["z_front"] for p in lines))

    def test_dense_photo_like_component_yields_nothing(self) -> None:
        rng = np.random.default_rng(7)
        img = rng.integers(0, 255, (300, 500, 3), dtype=np.uint8)
        fg = np.ones((300, 500), dtype=bool)
        result = split_connectors_from_component(fg, img, [], 1.0)
        self.assertTrue(result is None or not result["connectors"])


if __name__ == "__main__":
    unittest.main()
