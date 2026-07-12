"""InventoryBuilder: orchestrates cleaned image + OCR → inventory.json.

Lifts the ~20 previously-nested closures inside build_inventory._run
into instance methods on a single class, so each step can be navigated
and tested independently. Runtime behaviour is unchanged — this is a
pure refactor of where the code lives.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np

PAGE_DIR = Path(__file__).resolve().parents[1]    # scripts/page
SCRIPTS_ROOT = PAGE_DIR.parent                     # scripts
for _p in (PAGE_DIR, SCRIPTS_ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from shared.geometry import connector_on_container_border  # noqa: E402
from erase_text import (  # noqa: E402
    detect_logo_strips,
    is_in_logo_zone,
    is_likely_icon,
    preprocess_ocr,
    should_preserve_visual,
)
from _heuristics import pixel_scale, s_area, s_kernel, s_length  # noqa: E402
from icon import (  # noqa: E402
    detect_internal_shapes,
    detect_line_art_subicons,
    detect_white_subicons,
    inpaint_region_inplace,
)

from inventory.badge_trim import trim_filled_badge  # noqa: E402
from inventory.connector_split import (  # noqa: E402
    split_connectors_from_component,
)
from inventory.child_fill import (  # noqa: E402
    clean_child_fill,
    ring_bg_quality,
    simulated_fill_quality,
)
from inventory.role_classify import (  # noqa: E402
    foreground_role_for_box,
    is_connector_like,
    is_container_like,
)


def _box4(record: tuple) -> tuple[int, int, int, int]:
    return (int(record[0]), int(record[1]),
            int(record[2]), int(record[3]))


def _inventory_sort_key(e: dict) -> tuple:
    """Stable order: image first, by role priority, then top-to-bottom."""
    role_order = {
        "background": 0, "container": 1, "outline": 2, "internal": 3,
        "badge_subicon": 4, "connector": 4, "line_subicon": 4, "subicon": 4,
    }
    if e.get("type") == "image":
        role = e.get("role")
        return (0, role_order.get(role, 1), e["bbox"][1], e["bbox"][0])
    return (1, 0, e["bbox"][1], e["bbox"][0])


class InventoryBuilder:
    """Drive cleaned + OCR → inventory.json + per-component masks.

    Owns the per-page state the previous closure-heavy `_run` function
    carried implicitly: source/cleaned/icon_probe images, OCR splits,
    scaling factors, and the six record buckets (background, outline,
    foreground, subicon, line_subicon, internal_shape).
    """

    def __init__(self, args):
        from build_inventory import (
            detect_components,
            detect_outline_mask,
            detect_outline_rects,
            conservative_split,
            _foreground_mask,
        )
        self._detect_components = detect_components
        self._detect_outline_mask = detect_outline_mask
        self._detect_outline_rects = detect_outline_rects
        self._conservative_split = conservative_split
        self._foreground_mask = _foreground_mask

        self.args = args
        self.cleaned = cv2.imread(args.clean)
        self.source = cv2.imread(args.source)
        if self.cleaned is None or self.source is None:
            raise SystemExit("Could not load images.")
        self.ocr_data = preprocess_ocr(
            json.loads(Path(args.ocr).read_text(encoding="utf-8")),
            img=self.source,
        )
        # Icon detectors want only real text glyphs, not OCR-misclassified
        # visuals like numbered badges read as "②".
        self.ocr_text_items = [
            it for it in self.ocr_data
            if not should_preserve_visual(it)
        ]

        # Every pixel constant in this pipeline was tuned at 720-tall source.
        # Silently scaling user-facing knobs keeps default-flow runs correct.
        self.scale = pixel_scale(self.cleaned)
        args.min_area = s_area(args.min_area, self.scale)
        args.dilate = s_kernel(args.dilate, self.scale)
        args.split_gap = s_length(args.split_gap, self.scale)

        # Save a sidecar TEXT-ONLY-erased cleaned image BEFORE we modify
        # `cleaned` in place. Downstream asset cropping uses this sidecar.
        clean_path = Path(args.clean)
        self.text_only_path = clean_path.with_name(
            f"{clean_path.stem}.text_only.png")
        cv2.imwrite(str(self.text_only_path), self.cleaned)
        # Icon detection must run on the text-erased view, not the source.
        self.icon_probe = self.cleaned.copy()

        # Pre-filter OCR for likely-icons / logo-band texts.
        logo_bands = detect_logo_strips(self.ocr_data, self.cleaned.shape[0])
        self.candidate_texts = [
            item for item in self.ocr_data
            if not is_likely_icon(item, self.ocr_data, self.source)[0]
            and not is_in_logo_zone(item, logo_bands)
        ]

        self.components = self._detect_components(
            self.cleaned, args.min_area, args.dilate)

        self.inventory: list[dict] = []
        self.visual_idx = 0
        self.img_h, self.img_w = self.cleaned.shape[:2]
        self.background_records: list[tuple] = []
        self.outline_records: list[tuple] = []
        self.foreground_records: list[tuple] = []
        self.subicon_records: list[tuple] = []
        self.line_subicon_records: list[tuple] = []
        # (x1, y1, x2, y2, full_mask, z_front) strokes recovered from
        # components where a connector touches/crosses the cards it joins.
        self.connector_records: list[tuple] = []
        self.internal_shape_records: list[tuple] = []
        self.inpainted_children: set[tuple[int, int, int, int]] = set()
        self.unclean_nested_children: set[tuple[int, int, int, int]] = set()

        # Pre-compute scaled gates used by the scan helpers.
        self.parent_min_side = s_length(120, self.scale)
        self.line_min_area = s_area(1200, self.scale)
        self.line_min_side = s_length(24, self.scale)

    # ------------------------------------------------------------------
    # Record-append helpers
    # ------------------------------------------------------------------

    def _append_foreground_record(self, x1: int, y1: int,
                                  x2: int, y2: int) -> None:
        """Append a foreground bbox unless an equivalent one exists."""
        area = max(1, (x2 - x1) * (y2 - y1))
        for existing in self.foreground_records:
            ex1, ey1, ex2, ey2 = _box4(existing)
            earea = max(1, (ex2 - ex1) * (ey2 - ey1))
            size_ratio = min(area, earea) / max(area, earea)
            if size_ratio < 0.82:
                continue
            ix1, iy1 = max(x1, ex1), max(y1, ey1)
            ix2, iy2 = min(x2, ex2), min(y2, ey2)
            if ix2 <= ix1 or iy2 <= iy1:
                continue
            if (ix2 - ix1) * (iy2 - iy1) >= 0.92 * min(area, earea):
                return
        self.foreground_records.append(
            (int(x1), int(y1), int(x2), int(y2)))

    def _append_outline_record(self, x1: int, y1: int, x2: int, y2: int,
                               outline_mask: np.ndarray) -> None:
        """Dedup against existing outlines before appending."""
        area = max(1, (x2 - x1) * (y2 - y1))
        for ox1, oy1, ox2, oy2, _om in self.outline_records:
            ix1, iy1 = max(x1, ox1), max(y1, oy1)
            ix2, iy2 = min(x2, ox2), min(y2, oy2)
            if ix2 <= ix1 or iy2 <= iy1:
                continue
            oarea = max(1, (ox2 - ox1) * (oy2 - oy1))
            similar_size = min(area, oarea) / max(area, oarea) >= 0.55
            if similar_size and (ix2 - ix1) * (iy2 - iy1) >= 0.85 * min(area, oarea):
                return
        self.outline_records.append((x1, y1, x2, y2, outline_mask))

    # ------------------------------------------------------------------
    # Sub-shape scans
    # ------------------------------------------------------------------

    def _scan_internal_shapes_inplace(self, sx1: int, sy1: int,
                                      sx2: int, sy2: int) -> None:
        """Detect internal sub-shapes inside a parent component."""
        if (sx2 - sx1 < self.parent_min_side
                or sy2 - sy1 < self.parent_min_side):
            return
        shapes, fill_jobs = detect_internal_shapes(
            self.icon_probe, sx1, sy1, sx2, sy2,
            ocr_text_items=self.ocr_text_items,
            min_dim=s_length(20, self.scale),
            max_dim=s_length(220, self.scale),
            min_area=s_area(400, self.scale),
            scale=self.scale,
        )
        if not shapes:
            return
        source_local = self.icon_probe[sy1:sy2, sx1:sx2]
        for (ix1, iy1, ix2, iy2), (mask_in_crop, color) in zip(shapes, fill_jobs):
            ok, fill = clean_child_fill(source_local, mask_in_crop, self.scale, color)
            if not ok:
                continue
            if self._shape_box_already_covered(ix1, iy1, ix2, iy2):
                continue
            self.internal_shape_records.append((ix1, iy1, ix2, iy2))
            local = self.cleaned[sy1:sy2, sx1:sx2]
            inpaint_region_inplace(local, mask_in_crop, scale=self.scale,
                                   fill_color=fill)

    def _scan_subicons_inplace(self, sx1: int, sy1: int,
                               sx2: int, sy2: int) -> None:
        """Detect white pictograms inside dark uniform sub-shapes."""
        subs, fill_jobs = detect_white_subicons(
            self.icon_probe, sx1, sy1, sx2, sy2,
            ocr_text_items=self.ocr_text_items,
            min_dim=s_length(15, self.scale),
            max_dim=s_length(220, self.scale),
            min_area=s_area(200, self.scale),
            scale=self.scale,
        )
        if not subs:
            return
        source_local = self.icon_probe[sy1:sy2, sx1:sx2]
        for (ix1, iy1, ix2, iy2), (mask_in_crop, color) in zip(subs, fill_jobs):
            ok, fill = clean_child_fill(source_local, mask_in_crop, self.scale, color)
            if not ok:
                continue
            if any(self._box_overlap_ratio(
                    (ix1, iy1, ix2, iy2), _box4(r)) >= 0.5
                   for r in self.subicon_records):
                continue
            color_lum = float(
                0.114 * int(fill[0])
                + 0.587 * int(fill[1])
                + 0.299 * int(fill[2])
            )
            role = "badge_subicon" if color_lum > 160 else "subicon"
            self.subicon_records.append((ix1, iy1, ix2, iy2, role))
            local = self.cleaned[sy1:sy2, sx1:sx2]
            inpaint_region_inplace(local, mask_in_crop, scale=self.scale,
                                   fill_color=fill)

    def _scan_line_subicons_inplace(self, sx1: int, sy1: int,
                                    sx2: int, sy2: int) -> None:
        """Detect sparse line-art children inside this parent bbox."""
        sw = sx2 - sx1
        sh = sy2 - sy1
        if sw * sh < self.line_min_area or min(sw, sh) < self.line_min_side:
            return
        subs, fill_jobs = detect_line_art_subicons(
            self.icon_probe, sx1, sy1, sx2, sy2,
            ocr_text_items=self.ocr_text_items,
            min_dim=s_length(12, self.scale),
            max_dim=s_length(120, self.scale),
            min_area=s_area(35, self.scale),
            scale=self.scale,
        )
        if not subs:
            return
        for (ix1, iy1, ix2, iy2), (mask_in_crop, _color) in zip(subs, fill_jobs):
            if self._line_subicon_carried_by_container(ix1, iy1, ix2, iy2):
                continue
            source_local = self.icon_probe[sy1:sy2, sx1:sx2]
            ix1, iy1, ix2, iy2, child_mask = trim_filled_badge(self.icon_probe, self.ocr_text_items, self.scale, 
                ix1, iy1, ix2, iy2, mask_in_crop, (sx1, sy1))
            ok, fill = clean_child_fill(source_local, child_mask, self.scale, _color)
            if not ok:
                continue
            if self._shape_box_already_covered(ix1, iy1, ix2, iy2):
                continue
            full_mask = np.zeros((self.img_h, self.img_w), dtype=np.uint8)
            full_mask[sy1:sy2, sx1:sx2] = child_mask.astype(np.uint8) * 255
            self.line_subicon_records.append((ix1, iy1, ix2, iy2, full_mask))
            local = self.cleaned[sy1:sy2, sx1:sx2]
            inpaint_region_inplace(local, child_mask, scale=self.scale,
                                   fill_color=fill)

    def _shape_box_already_covered(self, ix1: int, iy1: int,
                                   ix2: int, iy2: int) -> bool:
        existing = (
            [_box4(r) for r in self.subicon_records]
            + [_box4(r) for r in self.line_subicon_records]
            + [_box4(r) for r in self.internal_shape_records]
        )
        for ex1, ey1, ex2, ey2 in existing:
            if self._box_overlap_ratio(
                    (ix1, iy1, ix2, iy2), (ex1, ey1, ex2, ey2)) >= 0.5:
                return True
        return False

    @staticmethod
    def _box_overlap_ratio(a, b) -> float:
        ax1, ay1, ax2, ay2 = a
        bx1, by1, bx2, by2 = b
        ox1, oy1 = max(ax1, bx1), max(ay1, by1)
        ox2, oy2 = min(ax2, bx2), min(ay2, by2)
        if ox2 <= ox1 or oy2 <= oy1:
            return 0.0
        inter = (ox2 - ox1) * (oy2 - oy1)
        a1 = max(1, (ax2 - ax1) * (ay2 - ay1))
        a2 = max(1, (bx2 - bx1) * (by2 - by1))
        return inter / min(a1, a2)

    def _line_subicon_carried_by_container(self, ix1, iy1, ix2, iy2) -> bool:
        for parent_record in self.foreground_records:
            parent_box = _box4(parent_record)
            if not connector_on_container_border(
                    parent_box, (ix1, iy1, ix2, iy2)):
                continue
            px1, py1, px2, py2 = parent_box
            if foreground_role_for_box(self.icon_probe, px1, py1, px2, py2, self.scale) == "container":
                return True
        return False

    # ------------------------------------------------------------------
    # Pipeline orchestration
    # ------------------------------------------------------------------

    def _emit_text_entries(self) -> None:
        for i, item in enumerate(self.candidate_texts):
            entry = {
                "id": f"t{i:03d}",
                "type": "text",
                "text": item["text"],
                "bbox": [item["x1"], item["y1"], item["x2"], item["y2"]],
                "confidence": item.get("confidence", 1.0),
            }
            # PaddleOCR per-segment data: chars/char_boxes for pure-CJK,
            # words/word_boxes for every line.
            if "chars" in item and "char_boxes" in item:
                entry["chars"] = list(item["chars"])
                entry["char_boxes"] = [list(b) for b in item["char_boxes"]]
            if "words" in item and "word_boxes" in item:
                entry["words"] = list(item["words"])
                entry["word_boxes"] = [list(b) for b in item["word_boxes"]]
            self.inventory.append(entry)

    def _detect_visual_components(self) -> None:
        # Global contour pass first — surfaces card outlines that a single
        # merged CC would hide.
        for x1, y1, x2, y2, outline_mask in self._detect_outline_rects(
                self.cleaned):
            self._append_outline_record(x1, y1, x2, y2, outline_mask)
        for x1, y1, x2, y2, _area in self.components:
            crop = self.cleaned[y1:y2, x1:x2]
            outline_mask = self._detect_outline_mask(
                self.cleaned, x1, y1, x2, y2)
            if outline_mask is not None:
                self._append_outline_record(x1, y1, x2, y2, outline_mask)
                if not (x2 - x1 >= self.img_w * 0.95
                        and y2 - y1 >= self.img_h * 0.95):
                    self._append_foreground_record(x1, y1, x2, y2)
            sub = self._conservative_split(crop, min_gap=self.args.split_gap)
            if len(sub) == 1:
                cw, ch = x2 - x1, y2 - y1
                if cw >= self.img_w * 0.95 and ch >= self.img_h * 0.95:
                    # Whole-image component → BACKGROUND.
                    self.background_records.append((x1, y1, x2, y2))
                    continue
                self._append_foreground_record(x1, y1, x2, y2)
                self._scan_subicons_inplace(x1, y1, x2, y2)
                self._scan_line_subicons_inplace(x1, y1, x2, y2)
                self._scan_internal_shapes_inplace(x1, y1, x2, y2)
            else:
                for sx1, sy1, sx2, sy2 in sub:
                    ax1, ay1, ax2, ay2 = (
                        x1 + sx1, y1 + sy1, x1 + sx2, y1 + sy2)
                    self._append_foreground_record(ax1, ay1, ax2, ay2)
                    self._scan_subicons_inplace(ax1, ay1, ax2, ay2)
                    self._scan_line_subicons_inplace(ax1, ay1, ax2, ay2)
                    self._scan_internal_shapes_inplace(ax1, ay1, ax2, ay2)

    def _known_card_boxes_within(
            self, box: tuple[int, int, int, int]) -> list[tuple]:
        """Outline/container rects already recorded inside `box`."""
        bx1, by1, bx2, by2 = box
        b_area = max(1, (bx2 - bx1) * (by2 - by1))
        cards = []
        candidates = (
            [(r[0], r[1], r[2], r[3]) for r in self.outline_records]
            + [_box4(r) for r in self.foreground_records
               if _box4(r) != box]
        )
        for cx1, cy1, cx2, cy2 in candidates:
            c_area = max(1, (cx2 - cx1) * (cy2 - cy1))
            if c_area >= 0.90 * b_area:
                continue
            ox1, oy1 = max(bx1, cx1), max(by1, cy1)
            ox2, oy2 = min(bx2, cx2), min(by2, cy2)
            if ox2 <= ox1 or oy2 <= oy1:
                continue
            if (ox2 - ox1) * (oy2 - oy1) < 0.80 * c_area:
                continue
            cards.append((cx1 - bx1, cy1 - by1, cx2 - bx1, cy2 - by1))
        return cards

    def _split_connectors_touching_cards(self) -> None:
        """Recover connector strokes merged into card components.

        A connector that touches or crosses a card fuses with it into one
        connected component, so `foreground_role_for_box` never sees a
        connector. Decompose such components (see connector_split.py),
        record the strokes as independent connector records, inpaint
        their pixels out of `cleaned`, and — when cards + strokes explain
        the component — drop the merged blob in favour of its parts.
        """
        min_area = s_area(20000, self.scale)
        min_side = s_length(120, self.scale)
        page_fg = self._foreground_mask(self.cleaned, self.args.dilate) > 0
        kept_records: list[tuple] = []
        new_card_boxes: list[tuple] = []
        dropped_boxes: list[tuple] = []
        for record in list(self.foreground_records):
            x1, y1, x2, y2 = _box4(record)
            w, h = x2 - x1, y2 - y1
            if w * h < min_area or max(w, h) < min_side:
                kept_records.append(record)
                continue
            crop_probe = self.icon_probe[y1:y2, x1:x2]
            if is_connector_like(crop_probe, self.scale):
                # Already a clean connector component; the normal role
                # path handles it.
                kept_records.append(record)
                continue
            result = split_connectors_from_component(
                page_fg[y1:y2, x1:x2],
                crop_probe,
                self._known_card_boxes_within((x1, y1, x2, y2)),
                self.scale,
            )
            if result is None or not result["connectors"]:
                kept_records.append(record)
                continue
            for piece in result["connectors"]:
                px1, py1, px2, py2 = piece["bbox"]
                full_mask = np.zeros((self.img_h, self.img_w),
                                     dtype=np.uint8)
                full_mask[y1:y2, x1:x2] = (
                    piece["mask"].astype(np.uint8) * 255)
                self.connector_records.append((
                    x1 + px1, y1 + py1, x1 + px2, y1 + py2,
                    full_mask, bool(piece.get("z_front"))))
                local = self.cleaned[y1:y2, x1:x2]
                inpaint_region_inplace(local, piece["mask"],
                                       scale=self.scale)
            drop_component = (
                result["explained_frac"] >= 0.85
                and bool(result["core_boxes"]
                         or self._known_card_boxes_within(
                             (x1, y1, x2, y2)))
            )
            if drop_component:
                dropped_boxes.append((x1, y1, x2, y2))
                for cx1, cy1, cx2, cy2 in result["core_boxes"]:
                    new_card_boxes.append(
                        (x1 + cx1, y1 + cy1, x1 + cx2, y1 + cy2))
            else:
                kept_records.append(record)
        self.foreground_records = kept_records
        for cx1, cy1, cx2, cy2 in new_card_boxes:
            self._append_foreground_record(cx1, cy1, cx2, cy2)
        if dropped_boxes:
            # The merged blob's own contour often registered as a fake
            # whole-diagram outline ring. It used to be culled as a
            # duplicate of the (now removed) full crop — cull it here.
            def _is_phantom(o: tuple) -> bool:
                ox1, oy1, ox2, oy2 = int(o[0]), int(o[1]), int(o[2]), int(o[3])
                oarea = max(1, (ox2 - ox1) * (oy2 - oy1))
                for dx1, dy1, dx2, dy2 in dropped_boxes:
                    darea = max(1, (dx2 - dx1) * (dy2 - dy1))
                    if min(oarea, darea) / max(oarea, darea) < 0.80:
                        continue
                    ix1, iy1 = max(ox1, dx1), max(oy1, dy1)
                    ix2, iy2 = min(ox2, dx2), min(oy2, dy2)
                    if ix2 <= ix1 or iy2 <= iy1:
                        continue
                    if (ix2 - ix1) * (iy2 - iy1) >= 0.92 * min(oarea, darea):
                        return True
                return False

            self.outline_records = [
                o for o in self.outline_records if not _is_phantom(o)
            ]

    def _drop_foregrounds_covered_by_shapes(self) -> None:
        """Drop foregrounds whose bbox is largely covered by sub-icons."""
        shape_boxes = (
            [_box4(r) for r in self.subicon_records]
            + [_box4(r) for r in self.line_subicon_records]
            + [_box4(r) for r in self.internal_shape_records]
        )

        def covered(fx1, fy1, fx2, fy2) -> bool:
            farea = max(1, (fx2 - fx1) * (fy2 - fy1))
            for sx1, sy1, sx2, sy2 in shape_boxes:
                ox1, oy1 = max(fx1, sx1), max(fy1, sy1)
                ox2, oy2 = min(fx2, sx2), min(fy2, sy2)
                if ox2 <= ox1 or oy2 <= oy1:
                    continue
                if (ox2 - ox1) * (oy2 - oy1) >= 0.8 * farea:
                    return True
            return False

        self.foreground_records = [
            r for r in self.foreground_records
            if not covered(*_box4(r))
        ]

    def _drop_outlines_duplicated_by_full_crop(self) -> None:
        """Drop outline masks when an equivalent full crop already exists."""
        def duplicated(ox1, oy1, ox2, oy2) -> bool:
            oarea = max(1, (ox2 - ox1) * (oy2 - oy1))
            for rec in self.background_records + self.foreground_records:
                fx1, fy1, fx2, fy2 = _box4(rec)
                farea = max(1, (fx2 - fx1) * (fy2 - fy1))
                size_ratio = min(oarea, farea) / max(oarea, farea)
                if size_ratio < 0.80:
                    continue
                ix1, iy1 = max(ox1, fx1), max(oy1, fy1)
                ix2, iy2 = min(ox2, fx2), min(oy2, fy2)
                if ix2 <= ix1 or iy2 <= iy1:
                    continue
                inter = (ix2 - ix1) * (iy2 - iy1)
                if inter >= 0.92 * min(oarea, farea):
                    return True
            return False

        self.outline_records = [
            r for r in self.outline_records
            if not duplicated(r[0], r[1], r[2], r[3])
        ]

    def _inpaint_nested_foreground_in_parents(self) -> None:
        """Erase nested children from their parent crops.

        When a small foreground sits inside a larger one, the parent's
        asset would still carry the child as baked pixels — moving or
        deleting the child in PPT would reveal an identical ghost
        underneath. Erase the child from the parent and route the child's
        own crop to the text-only sidecar (strokes intact).
        """
        if not self.foreground_records:
            return
        scale = self.scale
        child_min_side = s_length(4, scale)
        child_max_area = s_area(18000, scale)
        child_max_side = s_length(180, scale)
        child_min_fg_pixels = s_area(20, scale)
        child_masks: list[tuple] = []
        for child_record in self.foreground_records:
            cx1, cy1, cx2, cy2 = _box4(child_record)
            cw, ch = cx2 - cx1, cy2 - cy1
            if cw <= child_min_side or ch <= child_min_side:
                continue
            c_area = cw * ch
            if c_area > child_max_area or max(cw, ch) > child_max_side:
                continue
            crop = self.cleaned[cy1:cy2, cx1:cx2]
            if crop.size == 0:
                continue
            gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
            hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
            local_fg = (gray < 245) | (hsv[:, :, 1] > 12)
            if int(local_fg.sum()) < child_min_fg_pixels:
                continue
            is_connector = is_connector_like(crop, self.scale)
            child_masks.append(((cx1, cy1, cx2, cy2), c_area,
                                local_fg, is_connector))

        for child, c_area, local_fg, is_connector in child_masks:
            cx1, cy1, cx2, cy2 = child
            for parent_record in self.foreground_records:
                parent = _box4(parent_record)
                if parent == child:
                    continue
                px1, py1, px2, py2 = parent
                p_area = (px2 - px1) * (py2 - py1)
                # Parent must be meaningfully bigger; 2× catches icons
                # sitting in their tight bounding card.
                if p_area <= c_area * 2:
                    continue
                ox1, oy1 = max(cx1, px1), max(cy1, py1)
                ox2, oy2 = min(cx2, px2), min(cy2, py2)
                if ox2 <= ox1 or oy2 <= oy1:
                    continue
                if (ox2 - ox1) * (oy2 - oy1) < 0.85 * c_area:
                    continue
                if (
                    is_connector
                    and foreground_role_for_box(self.cleaned, px1, py1, px2, py2, self.scale) == "container"
                    and connector_on_container_border(parent, child)
                ):
                    continue
                parent_local = self.cleaned[py1:py2, px1:px2]
                ph, pw = parent_local.shape[:2]
                pm = np.zeros((ph, pw), dtype=bool)
                ty1 = cy1 - py1
                tx1 = cx1 - px1
                dy1, dx1 = max(0, ty1), max(0, tx1)
                dy2 = min(ph, ty1 + local_fg.shape[0])
                dx2 = min(pw, tx1 + local_fg.shape[1])
                if dy2 <= dy1 or dx2 <= dx1:
                    continue
                sy1, sx1 = dy1 - ty1, dx1 - tx1
                sy2 = sy1 + (dy2 - dy1)
                sx2 = sx1 + (dx2 - dx1)
                pm[dy1:dy2, dx1:dx2] = local_fg[sy1:sy2, sx1:sx2]
                clean, bg = clean_child_fill(parent_local, pm, self.scale)
                if clean:
                    inpaint_region_inplace(parent_local, pm, scale=scale,
                                           fill_color=bg)
                    self.inpainted_children.add(tuple(child))
                elif not is_connector:
                    self.unclean_nested_children.add(tuple(child))

        if self.unclean_nested_children:
            self.foreground_records = [
                r for r in self.foreground_records
                if (_box4(r) not in self.unclean_nested_children
                    or _box4(r) in self.inpainted_children)
            ]

    def _coverage_residual_pass(self) -> None:
        """Catch isolated decorations that all detectors missed."""
        text_only = cv2.imread(str(self.text_only_path))
        if text_only is None:
            return
        residual_mask = self._foreground_mask(text_only, self.args.dilate)
        all_bboxes = (
            [_box4(r) for r in self.foreground_records]
            + [_box4(r) for r in self.subicon_records]
            + [_box4(r) for r in self.line_subicon_records]
            + [_box4(r) for r in self.connector_records]
            + [_box4(r) for r in self.internal_shape_records]
            + [(x1, y1, x2, y2) for (x1, y1, x2, y2, _) in self.outline_records]
        )
        covered = np.zeros_like(residual_mask, dtype=np.uint8)
        for bx1, by1, bx2, by2 in all_bboxes:
            covered[by1:by2, bx1:bx2] = 255
        leftover = cv2.bitwise_and(residual_mask, cv2.bitwise_not(covered))
        ln, _, lstats, _ = cv2.connectedComponentsWithStats(leftover, 8)
        residual_min_area = s_area(400, self.scale)
        residual_min_side = s_length(18, self.scale)
        for i in range(1, ln):
            lx, ly, lw, lh, larea = lstats[i]
            if larea < residual_min_area:
                continue
            if lw < residual_min_side and lh < residual_min_side:
                continue
            self._append_foreground_record(
                int(lx), int(ly), int(lx + lw), int(ly + lh))

    def _emit_image_entries(self) -> None:
        """Append visual records to inventory in role-priority order."""
        # Backgrounds first (so they sort to the back via y-position tiebreak).
        for x1, y1, x2, y2 in self.background_records:
            self.inventory.append({
                "id": f"v{self.visual_idx:03d}",
                "type": "image",
                "bbox": [x1, y1, x2, y2],
                "source": "cleaned",
                "role": "background",
            })
            self.visual_idx += 1
        masks_out_dir = (Path(self.args.masks_dir)
                         if self.args.masks_dir else None)
        needs_masks = (self.outline_records or self.line_subicon_records
                       or self.connector_records)
        if needs_masks and masks_out_dir is None:
            out_path = Path(self.args.out)
            masks_out_dir = out_path.with_name(f"{out_path.stem}_masks")
        if masks_out_dir is not None:
            masks_out_dir.mkdir(parents=True, exist_ok=True)
        # Outlines next so the rounded border renders behind interiors.
        for x1, y1, x2, y2, outline_mask in self.outline_records:
            comp_id = f"v{self.visual_idx:03d}"
            entry = {
                "id": comp_id, "type": "image",
                "bbox": [int(x1), int(y1), int(x2), int(y2)],
                "source": "cleaned", "role": "outline",
            }
            if masks_out_dir is not None:
                full_mask = np.zeros(
                    (self.img_h, self.img_w), dtype=np.uint8)
                full_mask[y1:y2, x1:x2] = outline_mask
                mask_path = masks_out_dir / f"{comp_id}.mask.png"
                cv2.imwrite(str(mask_path), full_mask)
                entry["mask_path"] = str(mask_path)
            self.inventory.append(entry)
            self.visual_idx += 1
        for record in self.foreground_records:
            x1, y1, x2, y2 = _box4(record)
            comp_id = f"v{self.visual_idx:03d}"
            # Inpainted nested children must crop from the text-only sidecar
            # (strokes intact); other foregrounds crop from cleaned.
            is_nested_child = (x1, y1, x2, y2) in self.inpainted_children
            role_probe = self.source if is_nested_child else self.cleaned
            role = foreground_role_for_box(role_probe, x1, y1, x2, y2, self.scale)
            entry = {
                "id": comp_id, "type": "image",
                "bbox": [int(x1), int(y1), int(x2), int(y2)],
                "source": "source" if is_nested_child else "cleaned",
            }
            if role:
                entry["role"] = role
            self.inventory.append(entry)
            self.visual_idx += 1
        # Subicons last: they render IN FRONT of their parent card.
        for x1, y1, x2, y2 in self.internal_shape_records:
            self.inventory.append({
                "id": f"v{self.visual_idx:03d}", "type": "image",
                "bbox": [int(x1), int(y1), int(x2), int(y2)],
                "source": "source", "role": "internal",
            })
            self.visual_idx += 1
        for x1, y1, x2, y2, line_mask in self.line_subicon_records:
            comp_id = f"v{self.visual_idx:03d}"
            entry = {
                "id": comp_id, "type": "image",
                "bbox": [int(x1), int(y1), int(x2), int(y2)],
                "source": "source", "role": "line_subicon",
            }
            if masks_out_dir is not None:
                mask_path = masks_out_dir / f"{comp_id}.mask.png"
                cv2.imwrite(str(mask_path), line_mask)
                entry["mask_path"] = str(mask_path)
            self.inventory.append(entry)
            self.visual_idx += 1
        # Strokes recovered from card-touching/crossing components. Their
        # pixels were inpainted out of `cleaned`, so the crop must come
        # from the text-only sidecar (source="source" routes there).
        for x1, y1, x2, y2, conn_mask, z_front in self.connector_records:
            comp_id = f"v{self.visual_idx:03d}"
            entry = {
                "id": comp_id, "type": "image",
                "bbox": [int(x1), int(y1), int(x2), int(y2)],
                "source": "source", "role": "connector",
            }
            if z_front:
                entry["z"] = "front"
            if masks_out_dir is not None:
                mask_path = masks_out_dir / f"{comp_id}.mask.png"
                cv2.imwrite(str(mask_path), conn_mask)
                entry["mask_path"] = str(mask_path)
            self.inventory.append(entry)
            self.visual_idx += 1
        for record in self.subicon_records:
            x1, y1, x2, y2 = _box4(record)
            role = (
                str(record[4])
                if len(record) >= 5 and str(record[4])
                else "subicon"
            )
            self.inventory.append({
                "id": f"v{self.visual_idx:03d}", "type": "image",
                "bbox": [x1, y1, x2, y2],
                "source": "source", "role": role,
            })
            self.visual_idx += 1

    def _write_outputs(self) -> None:
        self.inventory.sort(key=_inventory_sort_key)
        out = Path(self.args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(
            json.dumps(self.inventory, ensure_ascii=False, indent=2),
            encoding="utf-8")
        # Persist the icon-filled cleaned image so the parent's asset crop
        # comes out without embedded pictograms.
        cv2.imwrite(self.args.clean, self.cleaned)
        text_count = sum(1 for e in self.inventory if e["type"] == "text")
        img_count = sum(1 for e in self.inventory if e["type"] == "image")
        sub_count = sum(1 for e in self.inventory
                        if e.get("role") in {"subicon", "badge_subicon"})
        line_count = sum(1 for e in self.inventory
                         if e.get("role") == "line_subicon")
        print(json.dumps({"text": text_count, "image": img_count,
                          "subicon": sub_count, "line_subicon": line_count,
                          "out": str(out)}, ensure_ascii=False))

    def _write_debug(self) -> None:
        if not self.args.debug_dir:
            return
        debug_dir = Path(self.args.debug_dir)
        debug_dir.mkdir(parents=True, exist_ok=True)
        stem = Path(self.args.source).stem
        vis = self.source.copy()
        for x1, y1, x2, y2 in self.background_records:
            cv2.rectangle(vis, (x1, y1), (x2, y2), (160, 160, 160), 1)
        for x1, y1, x2, y2, _outline_mask in self.outline_records:
            cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 220, 220), 1)
        for record in self.foreground_records:
            x1, y1, x2, y2 = _box4(record)
            cv2.rectangle(vis, (x1, y1), (x2, y2), (255, 100, 0), 1)
        for it in self.candidate_texts:
            cv2.rectangle(vis, (it["x1"], it["y1"]),
                          (it["x2"], it["y2"]), (0, 200, 0), 1)
        for x1, y1, x2, y2 in [_box4(r) for r in self.subicon_records]:
            cv2.rectangle(vis, (x1, y1), (x2, y2), (255, 0, 255), 2)
            cv2.putText(vis, "sub", (x1 + 1, max(10, y1 - 2)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 0, 255), 1,
                        cv2.LINE_AA)
        for x1, y1, x2, y2 in [_box4(r) for r in self.line_subicon_records]:
            cv2.rectangle(vis, (x1, y1), (x2, y2), (180, 0, 180), 2)
            cv2.putText(vis, "line", (x1 + 1, max(10, y1 - 2)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (180, 0, 180), 1,
                        cv2.LINE_AA)
        for x1, y1, x2, y2 in self.internal_shape_records:
            cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 100, 255), 2)
            cv2.putText(vis, "int", (x1 + 1, max(10, y1 - 2)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 100, 255), 1,
                        cv2.LINE_AA)
        legend = [
            ("foreground (img element)", (255, 100, 0)),
            ("background (whole-slide)", (160, 160, 160)),
            ("outline (card border)", (0, 220, 220)),
            ("editable text", (0, 200, 0)),
            ("subicon (movable)", (255, 0, 255)),
            ("line subicon (movable)", (180, 0, 180)),
            ("internal shape (movable)", (0, 100, 255)),
        ]
        for li, (label, col) in enumerate(legend):
            cv2.rectangle(vis, (8, 8 + li * 18), (24, 22 + li * 18), col, -1)
            cv2.putText(vis, label, (28, 20 + li * 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1,
                        cv2.LINE_AA)
        cv2.imwrite(str(debug_dir / f"{stem}_inv.png"), vis)

        # _cv2.png — shows the raw foreground mask + every component bbox
        # before filtering. Useful for "missing card border" diagnosis.
        cv2_mask = self._foreground_mask(self.cleaned, self.args.dilate)
        vis2 = self.source.copy()
        mask_red = np.zeros_like(vis2)
        mask_red[..., 2] = cv2_mask
        vis2 = cv2.addWeighted(vis2, 0.4, mask_red, 0.6, 0)
        n_cv, _, stats_cv, _ = cv2.connectedComponentsWithStats(cv2_mask, 8)
        kept = 0
        for i in range(1, n_cv):
            x, y, ww, hh, area = stats_cv[i]
            if area < self.args.min_area:
                continue
            cv2.rectangle(vis2, (int(x), int(y)),
                          (int(x + ww), int(y + hh)), (200, 200, 0), 1)
            kept += 1
        cv2.putText(
            vis2, f"{kept} components (min_area={self.args.min_area})",
            (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
            (255, 255, 255), 1, cv2.LINE_AA)
        cv2.imwrite(str(debug_dir / f"{stem}_cv2.png"), vis2)

    def build_and_write(self) -> None:
        self._emit_text_entries()
        self._detect_visual_components()
        self._split_connectors_touching_cards()
        self._drop_foregrounds_covered_by_shapes()
        self._drop_outlines_duplicated_by_full_crop()
        self._inpaint_nested_foreground_in_parents()
        self._coverage_residual_pass()
        self._emit_image_entries()
        self._write_outputs()
        self._write_debug()
