"""Preview-calibrated text must not be moved by later class snapping.

calibrate_text_positions measures rendered ink against the source image
and stamps `position_source: "preview_calibrated"`. classify_text_slots
runs once more after that calibration; its axis-snapping passes must
leave the measured coordinates alone (uncalibrated peers still snap).
"""
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
DECK_DIR = ROOT / "scripts" / "deck"
if str(DECK_DIR) not in sys.path:
    sys.path.insert(0, str(DECK_DIR))

from classify_text_slots import (  # noqa: E402
    _apply_class_position_priors,
    _box_xyxy,
    _position_locked,
)


def _text_item(name: str, box: list[float], *, calibrated: bool = False,
               align: str = "left") -> dict:
    el = {
        "type": "text",
        "name": name,
        "text": name,
        "box": list(box),
        "align": align,
    }
    if calibrated:
        el["position_source"] = "preview_calibrated"
    return {
        "el": el,
        "align": align,
        "box": _box_xyxy(box),
        "direct_parent": None,
    }


class CalibratedPositionLockTests(unittest.TestCase):
    def test_position_locked_flag(self) -> None:
        self.assertTrue(_position_locked(
            {"position_source": "preview_calibrated"}))
        self.assertFalse(_position_locked({}))
        self.assertFalse(_position_locked({"position_source": "ocr"}))

    def test_class_priors_skip_calibrated_elements(self) -> None:
        # Three same-class labels almost on one left axis; the middle one
        # was preview-calibrated 4px off the median — it must stay put.
        texts = [
            _text_item("a", [100.0, 50.0, 80.0, 20.0]),
            _text_item("b", [104.0, 100.0, 80.0, 20.0], calibrated=True),
            _text_item("c", [101.0, 150.0, 80.0, 20.0]),
        ]
        _apply_class_position_priors(texts, [0, 1, 2], "class_0")

        self.assertEqual(float(texts[1]["el"]["box"][0]), 104.0)
        # Uncalibrated members snapped to the shared axis.
        self.assertEqual(float(texts[0]["el"]["box"][0]),
                         float(texts[2]["el"]["box"][0]))


if __name__ == "__main__":
    unittest.main()
