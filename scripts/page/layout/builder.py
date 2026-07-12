"""LayoutBuilder: orchestrates inventory → manifest + layout output.

Lifts the previously-nested closures inside inventory_to_layout._run into
instance methods on a single class, so each step can be navigated and
tested independently. The runtime behaviour is unchanged — this is a
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

from shared.bg_sample import estimate_canvas_hex  # noqa: E402
from shared.geometry import connector_on_container_border  # noqa: E402
from icon import inpaint_region_inplace  # noqa: E402
from text_safety import ppt_safe_text  # noqa: E402

# Direct sibling imports — no longer routed through the inventory_to_layout
# facade.
from layout.bbox_shape import _pad_icon_bbox  # noqa: E402
from layout.bullet_images import restore_ignored_bullet_marker_images  # noqa: E402
from layout.grouping import unify_group_sizes  # noqa: E402
from layout.icon_alpha import (  # noqa: E402
    _ICON_PAD_ROLES,
    _SPARSE_VISUAL_ROLES,
    _border_is_uniform,
    _decontaminate_edges,
    _estimate_background_image,
    _line_art_alpha,
    _line_art_matte,
    _scrub_text_boxes_from_icon_crop,
    refine_masked_asset_edges,
)
from layout.connector_geometry import (  # noqa: E402
    extract_connector_geometry,
    merge_collinear_dash_runs,
)
from layout.lists import strip_leading_list_markers  # noqa: E402
from layout.outline import (  # noqa: E402
    _outline_should_be_native_shape,
    _outline_should_keep_full_crop,
    _sample_card_fill_color,
    _sample_outline_color,
    _split_filled_outline_rows,
)
from layout.text_emit import _emit_text_element_record  # noqa: E402
from layout.zorder import topo_sort_by_containment  # noqa: E402


_CHILD_ROLES = {
    "container",
    "internal",
    "subicon",
    "badge_subicon",
    "connector",
    "line_subicon",
    "preserve_visual_icon",
}


class LayoutBuilder:
    """Drive inventory → asset PNGs + manifest + layout JSON.

    Owns the per-page state that the previous closure-heavy `_run`
    function carried implicitly: the source/cleaned/text_only images,
    parsed inventory split by role, scaling factors, and the output
    accumulator lists.
    """

    def __init__(self, args):
        # The feature toggles live on the facade because they are the
        # switches that may be flipped externally for QA.
        from inventory_to_layout import (
            ENABLE_NATIVE_CONNECTORS,
            ENABLE_NATIVE_OUTLINE_SHAPES,
        )
        self._enable_native_outline = ENABLE_NATIVE_OUTLINE_SHAPES
        self._enable_native_connectors = ENABLE_NATIVE_CONNECTORS

        self.args = args
        self.inventory = json.loads(
            Path(args.inventory).read_text(encoding="utf-8"))
        if self._enable_native_connectors:
            self.inventory = merge_collinear_dash_runs(self.inventory)
        self.source = cv2.imread(args.source)
        self.cleaned = cv2.imread(args.cleaned)
        if self.source is None or self.cleaned is None:
            raise SystemExit("Could not load images.")
        # build_inventory publishes its icon-filled view as a sidecar
        # (it no longer overwrites the clean image in place); asset
        # crops must come from that view when it exists.
        icon_filled_path = Path(args.cleaned).with_name(
            f"{Path(args.cleaned).stem}.icon_filled.png")
        if icon_filled_path.exists():
            filled = cv2.imread(str(icon_filled_path))
            if filled is not None and filled.shape == self.cleaned.shape:
                self.cleaned = filled

        source_h_px, source_w_px = self.source.shape[:2]
        self.inpaint_scale = source_h_px / 720.0
        if args.slide_width_in is None:
            args.slide_width_in = (
                args.slide_height_in * (source_w_px / source_h_px))
        # Every source pixel maps to this many PPT points.
        self.pt_per_px = (args.slide_height_in * 72.0) / source_h_px

        cleaned_path = Path(args.cleaned)
        text_only_path = cleaned_path.with_name(
            f"{cleaned_path.stem}.text_only.png")
        self.text_only = (
            cv2.imread(str(text_only_path))
            if text_only_path.exists() else None
        )

        self.asset_dir = Path(args.out_assets_dir)
        self.asset_dir.mkdir(parents=True, exist_ok=True)
        for f in self.asset_dir.glob("*.png"):
            f.unlink()

        self.manifest_assets: list[dict] = []
        self.image_elements: list[dict] = []
        self.shape_elements: list[dict] = []
        self.text_elements: list[dict] = []
        self.text_boxes = [
            tuple(int(v) for v in el["bbox"])
            for el in self.inventory
            if el.get("type") == "text"
            and str(el.get("text", "") or "").strip()
        ]
        self.image_inventory = [
            el for el in self.inventory
            if el.get("type") == "image" and len(el.get("bbox", [])) == 4
        ]
        self.outline_inventory = [
            el for el in self.image_inventory if el.get("role") == "outline"
        ]

    # ------------------------------------------------------------------
    # Child / parent crop helpers (formerly nested closures in _run)
    # ------------------------------------------------------------------

    def _child_source_image(self, child: dict) -> np.ndarray:
        role = child.get("role")
        if role in _ICON_PAD_ROLES or role == "preserve_visual_icon":
            return self.text_only if self.text_only is not None else self.cleaned
        if child.get("source") == "source":
            return self.text_only if self.text_only is not None else self.source
        if child.get("source") == "original":
            return self.source
        return self.text_only if self.text_only is not None else self.cleaned

    def _child_inpaint_mask(self, child: dict):
        cx1, cy1, cx2, cy2 = (int(v) for v in child["bbox"])
        if cx2 <= cx1 or cy2 <= cy1:
            return None
        src = self._child_source_image(child)
        h_src, w_src = src.shape[:2]
        cx1 = max(0, min(w_src, cx1)); cx2 = max(0, min(w_src, cx2))
        cy1 = max(0, min(h_src, cy1)); cy2 = max(0, min(h_src, cy2))
        if cx2 <= cx1 or cy2 <= cy1:
            return None
        crop_child = src[cy1:cy2, cx1:cx2]
        role = child.get("role")
        mask_path = child.get("mask_path")
        if mask_path and Path(mask_path).exists():
            m = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if m is not None and m.shape[:2] == src.shape[:2]:
                mask = m[cy1:cy2, cx1:cx2] > 16
                if int(mask.sum()) >= 4:
                    return (cx1, cy1, cx2, cy2), mask
        if role == "container":
            # A nested card/panel is itself an independent object. Remove
            # the full child bbox from any larger parent asset so deleting
            # the card in PowerPoint does not reveal an identical baked copy.
            mask = np.ones(crop_child.shape[:2], dtype=bool)
        elif role == "subicon":
            gray = cv2.cvtColor(crop_child, cv2.COLOR_BGR2GRAY)
            hsv_ = cv2.cvtColor(crop_child, cv2.COLOR_BGR2HSV)
            mask = (gray > 185) & (hsv_[:, :, 1] < 70)
        elif role in {"badge_subicon", "connector", "line_subicon"}:
            mask = _line_art_alpha(crop_child) > 12
        elif role in {"internal", "preserve_visual_icon"}:
            mask = np.ones(crop_child.shape[:2], dtype=bool)
        else:
            gray = cv2.cvtColor(crop_child, cv2.COLOR_BGR2GRAY)
            hsv_ = cv2.cvtColor(crop_child, cv2.COLOR_BGR2HSV)
            diff_white = np.abs(crop_child.astype(int) - 255).max(axis=2)
            mask = ((gray < 245) & (diff_white > 5)) | (hsv_[:, :, 1] > 12)
            if int(mask.sum()) < 20:
                mask = np.ones(crop_child.shape[:2], dtype=bool)
        mask = cv2.dilate(mask.astype(np.uint8), np.ones((3, 3), np.uint8),
                          iterations=1).astype(bool)
        return (cx1, cy1, cx2, cy2), mask

    def _contained_child_entries(
        self, parent: dict,
        parent_box: tuple[int, int, int, int],
    ) -> list[dict]:
        px1, py1, px2, py2 = parent_box
        p_area = max(1, (px2 - px1) * (py2 - py1))
        pw = max(1, px2 - px1)
        ph = max(1, py2 - py1)
        out: list[dict] = []
        for child in self.image_inventory:
            if child is parent:
                continue
            role = child.get("role")
            if role in {"background", "outline"}:
                continue
            cx1, cy1, cx2, cy2 = (int(v) for v in child["bbox"])
            c_area = max(1, (cx2 - cx1) * (cy2 - cy1))
            if c_area >= p_area * 0.55:
                continue
            cw, ch = cx2 - cx1, cy2 - cy1
            if (
                parent.get("role") == "container"
                and role == "connector"
                and connector_on_container_border(
                    parent_box, (cx1, cy1, cx2, cy2))
            ):
                continue
            explicit_child = role in _CHILD_ROLES
            implicit_small_child = c_area <= 18000 and max(cw, ch) <= 180
            implicit_flat_child = (
                c_area <= 30000
                and min(cw, ch) <= 90
                and cw <= 0.95 * pw
                and ch <= 0.80 * ph
            )
            if (not explicit_child and not implicit_small_child
                    and not implicit_flat_child):
                continue
            ox1, oy1 = max(px1, cx1), max(py1, cy1)
            ox2, oy2 = min(px2, cx2), min(py2, cy2)
            if ox2 <= ox1 or oy2 <= oy1:
                continue
            if (ox2 - ox1) * (oy2 - oy1) >= 0.85 * c_area:
                out.append(child)
        return out

    def _connector_carried_by_container(self, el: dict) -> bool:
        if el.get("role") != "connector":
            return False
        child_box = tuple(int(v) for v in el["bbox"])
        for parent in self.image_inventory:
            if parent is el or parent.get("role") != "container":
                continue
            parent_box = tuple(int(v) for v in parent["bbox"])
            if connector_on_container_border(parent_box, child_box):
                return True
        return False

    def _inpaint_children_for_parent_asset(
        self,
        crop_bgr: np.ndarray,
        parent: dict,
        parent_box: tuple[int, int, int, int],
    ) -> np.ndarray:
        role = parent.get("role")
        if role in {
            "background", "internal", "subicon",
            "badge_subicon", "connector", "line_subicon",
        }:
            return crop_bgr
        children = self._contained_child_entries(parent, parent_box)
        if not children:
            return crop_bgr
        px1, py1, px2, py2 = parent_box
        mask = np.zeros(crop_bgr.shape[:2], dtype=bool)
        for child in children:
            child_mask_info = self._child_inpaint_mask(child)
            if child_mask_info is None:
                continue
            (cx1, cy1, cx2, cy2), child_mask = child_mask_info
            ox1, oy1 = max(px1, cx1), max(py1, cy1)
            ox2, oy2 = min(px2, cx2), min(py2, cy2)
            if ox2 <= ox1 or oy2 <= oy1:
                continue
            dst_x1, dst_y1 = ox1 - px1, oy1 - py1
            dst_x2, dst_y2 = ox2 - px1, oy2 - py1
            src_x1, src_y1 = ox1 - cx1, oy1 - cy1
            src_x2 = src_x1 + (dst_x2 - dst_x1)
            src_y2 = src_y1 + (dst_y2 - dst_y1)
            mask[dst_y1:dst_y2, dst_x1:dst_x2] |= (
                child_mask[src_y1:src_y2, src_x1:src_x2])
        if int(mask.sum()) < 4:
            return crop_bgr
        patched = crop_bgr.copy()
        inpaint_region_inplace(patched, mask, radius=3,
                               scale=self.inpaint_scale)
        return patched

    # ------------------------------------------------------------------
    # Outline routing helpers (formerly nested closures in _run)
    # ------------------------------------------------------------------

    def _is_redundant_multi_card_outline(self, el: dict) -> bool:
        x1, y1, x2, y2 = (int(v) for v in el["bbox"])
        area = max(1, (x2 - x1) * (y2 - y1))
        contained_boxes: list[tuple[int, int, int, int]] = []
        for other in self.outline_inventory:
            if other is el:
                continue
            ox1, oy1, ox2, oy2 = (int(v) for v in other["bbox"])
            oarea = max(1, (ox2 - ox1) * (oy2 - oy1))
            if oarea >= area * 0.85:
                continue
            ix1, iy1 = max(x1, ox1), max(y1, oy1)
            ix2, iy2 = min(x2, ox2), min(y2, oy2)
            if ix2 <= ix1 or iy2 <= iy1:
                continue
            inter = (ix2 - ix1) * (iy2 - iy1)
            if inter >= 0.72 * oarea:
                contained_boxes.append((ox1, oy1, ox2, oy2))
        if len(contained_boxes) < 2:
            return False
        # Redundant contour artefacts usually wrap two real cards placed
        # side-by-side (high vertical overlap, little horizontal overlap).
        # A true outer card can legitimately contain two stacked inner
        # panels; keep those.
        for i, a in enumerate(contained_boxes):
            ax1, ay1, ax2, ay2 = a
            aw, ah = ax2 - ax1, ay2 - ay1
            for b in contained_boxes[i + 1:]:
                bx1, by1, bx2, by2 = b
                bw, bh = bx2 - bx1, by2 - by1
                x_overlap = max(0, min(ax2, bx2) - max(ax1, bx1))
                y_overlap = max(0, min(ay2, by2) - max(ay1, by1))
                side_by_side = (
                    y_overlap >= 0.70 * min(ah, bh)
                    and x_overlap <= 0.20 * min(aw, bw)
                )
                if side_by_side:
                    return True
        return False

    def _outline_carried_by_large_parent(self, el: dict) -> bool:
        """Skip outline records already baked into a larger parent asset."""
        ox1, oy1, ox2, oy2 = (int(v) for v in el["bbox"])
        outline_area = max(1, (ox2 - ox1) * (oy2 - oy1))
        slide_area = max(1, self.source.shape[0] * self.source.shape[1])
        for parent in self.image_inventory:
            if parent is el:
                continue
            role = parent.get("role")
            if role in {
                "outline", "background", "internal", "subicon",
                "badge_subicon", "connector", "line_subicon",
            }:
                continue
            px1, py1, px2, py2 = (int(v) for v in parent["bbox"])
            parent_area = max(1, (px2 - px1) * (py2 - py1))
            if parent_area > 0.85 * slide_area:
                continue
            if parent_area < 2.8 * outline_area:
                continue
            ix1, iy1 = max(ox1, px1), max(oy1, py1)
            ix2, iy2 = min(ox2, px2), min(oy2, py2)
            if ix2 <= ix1 or iy2 <= iy1:
                continue
            inter = (ix2 - ix1) * (iy2 - iy1)
            if inter >= 0.92 * outline_area:
                return True
        return False

    def _outline_has_top_badge(self, el: dict) -> bool:
        """Return True when a small banner overlaps the outline's top edge."""
        ox1, oy1, ox2, oy2 = (int(v) for v in el["bbox"])
        ow = max(1, ox2 - ox1)
        oh = max(1, oy2 - oy1)

        h_img, w_img = self.source.shape[:2]
        sx1 = max(0, ox1 - 10)
        sx2 = min(w_img, ox2 + 10)
        sy1 = max(0, oy1 - 38)
        sy2 = min(h_img, oy1 + min(70, max(34, oh // 3)))
        if sx2 > sx1 and sy2 > sy1:
            band = self.source[sy1:sy2, sx1:sx2]
            hsv = cv2.cvtColor(band, cv2.COLOR_BGR2HSV)
            # Deck title tabs are saturated dark-blue rounded rectangles.
            # Detect from original pixels because some slides crop the tab
            # together with the illustration (no separate inventory entry).
            blue = (
                (hsv[:, :, 0] >= 88) & (hsv[:, :, 0] <= 124)
                & (hsv[:, :, 1] >= 70)
                & (hsv[:, :, 2] >= 40) & (hsv[:, :, 2] <= 230)
            ).astype(np.uint8) * 255
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 5))
            blue = cv2.morphologyEx(blue, cv2.MORPH_CLOSE, kernel,
                                    iterations=1)
            cnts, _ = cv2.findContours(blue, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
            for c in cnts:
                bx, by, bw, bh = cv2.boundingRect(c)
                if bw < max(55, 0.18 * ow) or bw > 0.78 * ow:
                    continue
                if bh < 18 or bh > min(72, max(28, 0.35 * oh)):
                    continue
                ax1, ay1 = sx1 + bx, sy1 + by
                ax2, ay2 = ax1 + bw, ay1 + bh
                cx = (ax1 + ax2) / 2.0
                overlaps_top = ay1 <= oy1 + 36 and ay2 >= oy1 - 16
                inside_x = ox1 - 8 <= cx <= ox2 + 8
                if overlaps_top and inside_x:
                    return True

        for other in self.image_inventory:
            if other is el or other.get("role") in {"outline", "background"}:
                continue
            bx1, by1, bx2, by2 = (int(v) for v in other["bbox"])
            bw = max(1, bx2 - bx1)
            bh = max(1, by2 - by1)
            if bw > 0.45 * ow or bh > 0.18 * oh:
                continue
            cx = (bx1 + bx2) / 2.0
            overlaps_top = by1 <= oy1 + 24 and by2 >= oy1 - 18
            inside_x = ox1 - 6 <= cx <= ox2 + 6
            if overlaps_top and inside_x:
                return True
        return False

    def _contained_child_outline_boxes(self, el: dict) -> list[tuple[int, int, int, int]]:
        """Return nested outline boxes carried by an outer outline mask."""
        ox1, oy1, ox2, oy2 = (int(v) for v in el["bbox"])
        outline_area = max(1, (ox2 - ox1) * (oy2 - oy1))
        boxes: list[tuple[int, int, int, int]] = []
        for other in self.image_inventory:
            if other is el or other.get("role") != "outline":
                continue
            cx1, cy1, cx2, cy2 = (int(v) for v in other["bbox"])
            child_area = max(1, (cx2 - cx1) * (cy2 - cy1))
            if child_area >= 0.85 * outline_area:
                continue
            ix1, iy1 = max(ox1, cx1), max(oy1, cy1)
            ix2, iy2 = min(ox2, cx2), min(oy2, cy2)
            if ix2 <= ix1 or iy2 <= iy1:
                continue
            inter = (ix2 - ix1) * (iy2 - iy1)
            if inter >= 0.88 * child_area:
                boxes.append((cx1, cy1, cx2, cy2))
        return boxes

    # ------------------------------------------------------------------
    # Asset empty-check (formerly nested closure)
    # ------------------------------------------------------------------

    def _is_empty_asset(self, crop_bgr: np.ndarray,
                        alpha: np.ndarray | None,
                        *, sparse_ok: bool = False) -> bool:
        """Return True when the asset has no visible content to render.

        Two failure modes we want to skip:
          - The crop sits on an erased region — almost every pixel is the
            slide bg colour (near-white), so the asset PNG is a blank
            rectangle on top of whatever's actually behind it.
          - We have an alpha mask, but every opaque pixel is also near-
            white. This happens when an upstream detector leaves a thin
            frame around an emptied card; the rendered PNG then looks like
            a washed-out rectangle on top of the real children below.

        "Near-white" = max channel diff vs white ≤ 16 — catches the
        anti-aliased card-bg tints typical of light UI cards.
        """
        h, w = crop_bgr.shape[:2]
        if h == 0 or w == 0:
            return True
        diff = np.abs(crop_bgr.astype(int) - 255).max(axis=2)
        near_white = diff <= 16
        if alpha is not None:
            opaque = alpha > 16
            opaque_count = int(opaque.sum())
            opaque_frac = opaque_count / float(h * w)
            if opaque_count == 0:
                return True
            if sparse_ok:
                # Thin card borders / line-art masks are intentionally sparse.
                return opaque_count < max(8, int(0.00015 * h * w))
            if opaque_frac < 0.02:
                return True
            opaque_white = int((near_white & opaque).sum())
            opaque_white_frac = opaque_white / float(opaque_count)
            # Big region segments (>30k px) get a pass even at high
            # white-frac: their visual identity is the rounded corners +
            # faint tint, not the interior fill.
            if opaque_white_frac >= 0.92 and (h * w) < 30000:
                return True
            return False
        near_white_frac = float(near_white.sum()) / float(h * w)
        visible = ~near_white
        if near_white_frac >= 0.95:
            elongated_rule = (
                (w >= 35 and h <= 4) or (h >= 35 and w <= 4)
            )
            if elongated_rule and int(diff.max()) > 5:
                return False
            median_bgr = np.median(crop_bgr.reshape(-1, 3), axis=0)
            median_diff_white = int(
                np.max(np.abs(median_bgr.astype(int) - 255)))
            if h * w >= 3000 and median_diff_white > 6:
                return False
            gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
            hsv_ = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2HSV)
            meaningful = visible & ((gray < 235) | (hsv_[:, :, 1] > 18))
            meaningful_count = int(meaningful.sum())
            if meaningful_count >= max(80, int(0.001 * h * w)):
                return False
            return True
        if sparse_ok:
            visible_count = int(visible.sum())
            return visible_count < max(4, int(0.001 * h * w))
        return False

    # ------------------------------------------------------------------
    # Element emitters (split out of the old _run main loop)
    # ------------------------------------------------------------------

    def _emit_image_element(self, el: dict, x1: int, y1: int,
                            x2: int, y2: int) -> None:
        asset_name = f"{el['id']}.png"
        role = el.get("role")
        if self._connector_carried_by_container(el):
            return
        if role == "container":
            src_img = self.text_only if self.text_only is not None else self.cleaned
        elif role in _ICON_PAD_ROLES or role == "preserve_visual_icon":
            src_img = self.text_only if self.text_only is not None else self.cleaned
        elif el.get("source") == "original":
            src_img = self.source
        elif el.get("source") == "source":
            # Prefer text-erased sidecar; fall back to source if absent.
            src_img = self.text_only if self.text_only is not None else self.source
        else:
            src_img = self.cleaned
        mask_path = el.get("mask_path")
        has_mask = bool(mask_path and Path(mask_path).exists())
        if role == "outline" and self._outline_carried_by_large_parent(el):
            return
        native_outline = (
            self._enable_native_outline
            and role == "outline"
            and _outline_should_be_native_shape(
                (int(x1), int(y1), int(x2), int(y2)))
            and not self._outline_has_top_badge(el)
        )
        if native_outline:
            if self._is_redundant_multi_card_outline(el):
                return
            line_color = _sample_outline_color(
                self.source, (int(x1), int(y1), int(x2), int(y2)), mask_path)
            fill_color = _sample_card_fill_color(
                self.source, (int(x1), int(y1), int(x2), int(y2)))
            self.shape_elements.append({
                "type": "shape", "name": el["id"], "shape": "round_rect",
                "box": [int(x1), int(y1), int(x2 - x1), int(y2 - y1)],
                "fill": fill_color, "line": line_color,
                "line_width": 0.9, "radius": 0.08,
            })
            return
        text_erased_crop = (
            src_img is self.cleaned
            or (self.text_only is not None and src_img is self.text_only)
        )
        original_w, original_h = int(x2 - x1), int(y2 - y1)
        implicit_cv2_icon = (
            role in {None, ""}
            and original_w <= 340 and original_h <= 340
            and original_w * original_h <= 80000
        )
        sparse_visual = (role in _SPARSE_VISUAL_ROLES
                         or implicit_cv2_icon)
        scrub_icon_text = (
            role in _ICON_PAD_ROLES
            or role == "preserve_visual_icon"
            or implicit_cv2_icon
        )
        x1, y1, x2, y2 = _pad_icon_bbox(
            (x1, y1, x2, y2), src_img, self.text_boxes, role,
            allow_text_overlap=text_erased_crop)
        crop = src_img[y1:y2, x1:x2].copy()
        crop = self._inpaint_children_for_parent_asset(
            crop, el, (int(x1), int(y1), int(x2), int(y2)))
        contained_outline_boxes = (
            self._contained_child_outline_boxes(el)
            if role == "outline" else []
        )
        keep_outline_full_crop = (
            role == "outline"
            and not contained_outline_boxes
            and _outline_should_keep_full_crop(
                self.source, (int(x1), int(y1), int(x2), int(y2)))
        )
        if keep_outline_full_crop:
            split_boxes = _split_filled_outline_rows(
                self.source, (int(x1), int(y1), int(x2), int(y2)))
            if len(split_boxes) > 1:
                self._emit_split_outline_rows(el, split_boxes,
                                              src_img, sparse_visual)
                return
        if (role == "connector" and self._enable_native_connectors
                and self._try_emit_native_connector(el, crop,
                                                    int(x1), int(y1))):
            return
        if has_mask and not keep_outline_full_crop:
            if self._emit_masked_image(el, asset_name, mask_path, src_img,
                                       crop, contained_outline_boxes,
                                       x1, y1, x2, y2, sparse_visual,
                                       scrub_icon_text):
                return
        if self._is_empty_asset(crop, None, sparse_ok=sparse_visual):
            return
        if not self._emit_role_specific_crop(el, asset_name, crop,
                                             x1, y1, x2, y2,
                                             scrub_icon_text,
                                             try_matte=implicit_cv2_icon):
            # No asset PNG was written (e.g. line-art alpha came out
            # empty) — registering the element would make the PPTX
            # build crash on the missing file.
            return
        self.manifest_assets.append({
            "name": asset_name,
            "box": [int(x1), int(y1), int(x2), int(y2)],
            "mode": "keep",
        })
        self.image_elements.append({
            "type": "image", "name": el["id"],
            "path": f"{self.args.asset_prefix}/{asset_name}",
            "box": [int(x1), int(y1), int(x2 - x1), int(y2 - y1)],
        })

    def _emit_split_outline_rows(self, el: dict, split_boxes: list,
                                 src_img: np.ndarray,
                                 sparse_visual: bool) -> None:
        for idx, split_box in enumerate(split_boxes):
            sx1, sy1, sx2, sy2 = split_box
            split_asset_name = f"{el['id']}_s{idx:02d}.png"
            split_crop = src_img[sy1:sy2, sx1:sx2].copy()
            split_crop = self._inpaint_children_for_parent_asset(
                split_crop, el, split_box)
            if self._is_empty_asset(split_crop, None, sparse_ok=sparse_visual):
                continue
            cv2.imwrite(str(self.asset_dir / split_asset_name), split_crop)
            self.manifest_assets.append({
                "name": split_asset_name,
                "box": [int(sx1), int(sy1), int(sx2), int(sy2)],
                "mode": "keep",
            })
            self.image_elements.append({
                "type": "image",
                "name": f"{el['id']}_s{idx:02d}",
                "path": f"{self.args.asset_prefix}/{split_asset_name}",
                "box": [int(sx1), int(sy1),
                        int(sx2 - sx1), int(sy2 - sy1)],
            })

    def _emit_masked_image(self, el: dict, asset_name: str,
                           mask_path: str, src_img: np.ndarray,
                           crop: np.ndarray,
                           contained_outline_boxes: list,
                           x1: int, y1: int, x2: int, y2: int,
                           sparse_visual: bool,
                           scrub_icon_text: bool) -> bool:
        """Use the mask as alpha for a transparent-bg asset PNG."""
        m = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if m is None or m.shape[:2] != src_img.shape[:2]:
            return False
        alpha = m[y1:y2, x1:x2].copy()
        for cx1, cy1, cx2, cy2 in contained_outline_boxes:
            rx1 = max(0, cx1 - int(x1) - 3)
            ry1 = max(0, cy1 - int(y1) - 3)
            rx2 = min(alpha.shape[1], cx2 - int(x1) + 3)
            ry2 = min(alpha.shape[0], cy2 - int(y1) + 3)
            if rx2 > rx1 and ry2 > ry1:
                alpha[ry1:ry2, rx1:rx2] = 0
        if self._is_empty_asset(crop, alpha, sparse_ok=sparse_visual):
            return True
        if scrub_icon_text:
            crop = _scrub_text_boxes_from_icon_crop(
                crop, (int(x1), int(y1), int(x2), int(y2)),
                self.text_boxes, alpha=alpha)
        crop, alpha = refine_masked_asset_edges(crop, alpha)
        rgba = np.dstack([crop, alpha])
        cv2.imwrite(str(self.asset_dir / asset_name), rgba)
        self.manifest_assets.append({
            "name": asset_name,
            "box": [int(x1), int(y1), int(x2), int(y2)],
            "mode": "keep",
        })
        self.image_elements.append({
            "type": "image", "name": el["id"],
            "path": f"{self.args.asset_prefix}/{asset_name}",
            "box": [int(x1), int(y1), int(x2 - x1), int(y2 - y1)],
        })
        return True

    def _connector_mask_alpha(self, el: dict, crop: np.ndarray,
                              x1: int, y1: int) -> np.ndarray | None:
        """Sidecar-mask alpha for connectors recovered from merged blobs.

        Their crops can contain card corners near the touching joint, so
        keying the whole crop would poison the geometry fit — the
        detector's own mask is the authoritative stroke footprint.
        """
        mask_path = el.get("mask_path")
        if not mask_path or not Path(mask_path).exists():
            return None
        m = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if m is None or m.shape[:2] != self.source.shape[:2]:
            return None
        h, w = crop.shape[:2]
        alpha = m[y1:y1 + h, x1:x1 + w]
        if alpha.shape[:2] != (h, w) or int((alpha > 40).sum()) < 12:
            return None
        return alpha

    def _try_emit_native_connector(self, el: dict, crop: np.ndarray,
                                   x1: int, y1: int) -> bool:
        """Emit a connector as a native PPT line when the fit is confident.

        Returns True when a `type: "line"` element was appended (the
        raster fallback must then be skipped). The element also carries
        the bbox as `box` so containment z-ordering places it above the
        background/cards it lies on.
        """
        alpha = self._connector_mask_alpha(el, crop, x1, y1)
        if alpha is None:
            _fg, alpha = _line_art_matte(crop)
        geom = extract_connector_geometry(crop, alpha)
        if geom is None:
            return False
        points = [[round(x1 + px, 1), round(y1 + py, 1)]
                  for px, py in geom["points"]]
        pxs = [p[0] for p in points]
        pys = [p[1] for p in points]
        width_pt = max(0.5, round(geom["width_px"] * self.pt_per_px, 2))
        element = {
            "type": "line",
            "name": el["id"],
            "points": points,
            "box": [min(pxs), min(pys),
                    max(1.0, max(pxs) - min(pxs)),
                    max(1.0, max(pys) - min(pys))],
            "line": geom["color"],
            "line_width": width_pt,
        }
        if geom.get("dash"):
            element["dash"] = geom["dash"]
        if geom.get("arrow_start"):
            element["arrow_start"] = True
        if geom.get("arrow_end"):
            element["arrow_end"] = True
        # Goes into the image z-order machinery (not shape_elements) so
        # topo_sort_by_containment keeps it above any containing asset.
        self.image_elements.append(element)
        return True

    def _icon_matte_alpha(self, crop: np.ndarray) -> np.ndarray | None:
        """Conservative auto-matte for small role-less icon crops.

        These crops used to ship fully opaque, so the page background
        baked into the rectangle shows as a colour block whenever the
        slide/card fill underneath differs even slightly. Matte only
        when the border ring is one flat colour and the recovered
        foreground looks icon-like; otherwise keep the opaque emit.
        """
        h, w = crop.shape[:2]
        if h * w == 0 or not _border_is_uniform(crop):
            return None
        fg, alpha = _line_art_matte(crop)
        opaque = alpha > 16
        frac = float(opaque.sum()) / float(h * w)
        if not 0.005 <= frac <= 0.65:
            return None
        # The ring must really be background — a photo-like crop keeps
        # content at the border and must stay opaque.
        ring = np.zeros((h, w), dtype=bool)
        r = max(1, min(2, min(h, w) // 10))
        ring[:r, :] = ring[-r:, :] = ring[:, :r] = ring[:, -r:] = True
        if float(alpha[ring].mean()) > 20.0:
            return None
        crop[:] = fg
        return alpha

    def _emit_role_specific_crop(self, el: dict, asset_name: str,
                                 crop: np.ndarray,
                                 x1: int, y1: int, x2: int, y2: int,
                                 scrub_icon_text: bool,
                                 try_matte: bool = False) -> bool:
        """Write the asset PNG; returns False when nothing was written."""
        role = el.get("role")
        if role == "subicon":
            # White icon on dark bg → keep only near-white pixels opaque,
            # everything else transparent. Matches detect_white_subicons.
            gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
            hsv_ = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
            alpha = np.clip((gray.astype(int) - 180) * 3,
                            0, 255).astype(np.uint8)
            alpha[hsv_[:, :, 1] > 60] = 0
            if int((alpha > 24).sum()) >= 4:
                bg_est = _estimate_background_image(crop, alpha > 24)
                crop = _decontaminate_edges(crop, alpha, bg_est)
            crop = _scrub_text_boxes_from_icon_crop(
                crop, (int(x1), int(y1), int(x2), int(y2)),
                self.text_boxes, alpha=alpha)
            rgba = np.dstack([crop, alpha])
            cv2.imwrite(str(self.asset_dir / asset_name), rgba)
        elif role in {"badge_subicon", "connector", "line_subicon"}:
            crop, alpha = _line_art_matte(crop)
            if int((alpha > 16).sum()) < 4:
                return False
            crop = _scrub_text_boxes_from_icon_crop(
                crop, (int(x1), int(y1), int(x2), int(y2)),
                self.text_boxes, alpha=alpha)
            rgba = np.dstack([crop, alpha])
            cv2.imwrite(str(self.asset_dir / asset_name), rgba)
        else:
            alpha = self._icon_matte_alpha(crop) if try_matte else None
            if scrub_icon_text:
                crop = _scrub_text_boxes_from_icon_crop(
                    crop, (int(x1), int(y1), int(x2), int(y2)),
                    self.text_boxes, alpha=alpha)
            if alpha is not None:
                cv2.imwrite(str(self.asset_dir / asset_name),
                            np.dstack([crop, alpha]))
            else:
                cv2.imwrite(str(self.asset_dir / asset_name), crop)
        return True

    # ------------------------------------------------------------------
    # Build orchestration
    # ------------------------------------------------------------------

    def build(self) -> None:
        for el in self.inventory:
            x1, y1, x2, y2 = el["bbox"]
            if el["type"] == "image":
                self._emit_image_element(el, int(x1), int(y1),
                                         int(x2), int(y2))
            else:
                record = _emit_text_element_record(
                    el, int(x1), int(y1), int(x2), int(y2),
                    self.source, self.pt_per_px)
                if record is not None:
                    self.text_elements.append(record)

        self.text_elements = strip_leading_list_markers(self.text_elements)
        self.image_elements.extend(
            restore_ignored_bullet_marker_images(
                self.text_elements, self.source,
                self.asset_dir, self.args.asset_prefix)
        )
        # Size unification normalises font-size drift between
        # visually-identical labels (`1000亿元` vs `24.22%`).
        self.text_elements = unify_group_sizes(self.text_elements)
        self.image_elements = topo_sort_by_containment(self.image_elements)
        # Native connector lines render above every image: strokes that
        # should be occluded were already clipped to the card edges at
        # extraction time, so whatever remains is visually on top.
        lines = [e for e in self.image_elements if e.get("type") == "line"]
        if lines:
            self.image_elements = [
                e for e in self.image_elements if e.get("type") != "line"
            ] + lines

    def write(self) -> None:
        manifest = {
            "source": str(Path(self.args.cleaned).absolute()),
            "assets": self.manifest_assets,
        }
        Path(self.args.out_manifest).parent.mkdir(parents=True, exist_ok=True)
        Path(self.args.out_manifest).write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2),
            encoding="utf-8")

        h, w = self.source.shape[:2]
        layout = {
            "slide_size": {
                "width_in": self.args.slide_width_in,
                "height_in": self.args.slide_height_in,
            },
            "source_width": w,
            "source_height": h,
            "background": estimate_canvas_hex(self.source),
            "elements": (self.shape_elements
                         + self.image_elements
                         + self.text_elements),
        }
        Path(self.args.out_layout).parent.mkdir(parents=True, exist_ok=True)
        Path(self.args.out_layout).write_text(
            json.dumps(layout, ensure_ascii=False, indent=2),
            encoding="utf-8")
        print(json.dumps({
            "text_elements": len(self.text_elements),
            "image_elements": len(self.image_elements),
            "shape_elements": len(self.shape_elements),
            "manifest": str(self.args.out_manifest),
            "layout": str(self.args.out_layout),
        }, ensure_ascii=False))
