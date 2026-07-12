#!/usr/bin/env python
"""Classify text boxes into repeated structural slots.

The classifier is intentionally conservative: it builds text classes from
shared/nested containers, axis alignment, and compatible visual style.
When --apply is used, high-confidence class priors can also normalize
font size, alignment, and list bullet indentation.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


MAX_STYLE_SIZE_DELTA = 2.0


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--layout", required=True, help="Combined layout JSON.")
    p.add_argument("--out", help="Output JSON report.")
    p.add_argument("--out-layout",
                   help="Output layout path when --apply is used. "
                        "Default: update --layout in place.")
    p.add_argument("--apply", action="store_true",
                   help="Annotate/apply high-confidence slot classes.")
    p.add_argument("--min-group-size", type=int, default=2)
    p.add_argument("--min-apply-size", type=int, default=3,
                   help="Minimum class size required for size/align priors.")
    return p.parse_args()


def _box_xyxy(box: list[float]) -> tuple[float, float, float, float]:
    x, y, w, h = (float(v) for v in box)
    return x, y, x + w, y + h


def _area_xyxy(box: tuple[float, float, float, float]) -> float:
    x1, y1, x2, y2 = box
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def _center(box: tuple[float, float, float, float]) -> tuple[float, float]:
    x1, y1, x2, y2 = box
    return (x1 + x2) / 2.0, (y1 + y2) / 2.0


def _intersection_area(
    a: tuple[float, float, float, float],
    b: tuple[float, float, float, float],
) -> float:
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    return _area_xyxy((x1, y1, x2, y2))


def _contains_text(
    container: tuple[float, float, float, float],
    text: tuple[float, float, float, float],
    margin: float = 6.0,
) -> bool:
    cx, cy = _center(text)
    x1, y1, x2, y2 = container
    center_inside = (
        x1 - margin <= cx <= x2 + margin
        and y1 - margin <= cy <= y2 + margin
    )
    if not center_inside:
        return False
    text_area = max(1.0, _area_xyxy(text))
    return _intersection_area(container, text) / text_area >= 0.55


def _parse_rgb(value: Any) -> tuple[int, int, int] | None:
    if not isinstance(value, str):
        return None
    text = value.strip().lstrip("#")
    if len(text) != 6:
        return None
    try:
        return int(text[0:2], 16), int(text[2:4], 16), int(text[4:6], 16)
    except ValueError:
        return None


def _colour_family(rgb: tuple[int, int, int] | None) -> str:
    if rgb is None:
        return "unknown"
    r, g, b = rgb
    if min(rgb) >= 225:
        return "white"
    if max(rgb) <= 70:
        return "black"
    if r > g + 35 and r > b + 35:
        return "red"
    if b > r + 32 and b > g + 18:
        return "blue"
    if max(rgb) - min(rgb) <= 42:
        return "gray"
    return "other"


def _colour_distance(a: tuple[int, int, int] | None,
                     b: tuple[int, int, int] | None) -> float:
    if a is None or b is None:
        return 999.0
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


def _colour_compatible(a: dict[str, Any], b: dict[str, Any]) -> bool:
    ar = a["rgb"]
    br = b["rgb"]
    af = a["colour_family"]
    bf = b["colour_family"]
    if af != bf:
        return False
    if af in {"white", "black"}:
        return _colour_distance(ar, br) <= 38
    if af in {"blue", "red", "gray"}:
        return _colour_distance(ar, br) <= 78
    return _colour_distance(ar, br) <= 45


def _style_compatible(a: dict[str, Any], b: dict[str, Any]) -> bool:
    if a["align"] != b["align"]:
        return False
    font_a = str(a["el"].get("font") or a["el"].get("font_name") or "")
    font_b = str(b["el"].get("font") or b["el"].get("font_name") or "")
    if font_a and font_b and font_a != font_b:
        return False
    size_a = float(a["el"].get("size") or a["el"].get("font_size") or 0)
    size_b = float(b["el"].get("size") or b["el"].get("font_size") or 0)
    if size_a <= 0 or size_b <= 0:
        return False
    if abs(size_a - size_b) > MAX_STYLE_SIZE_DELTA:
        return False
    if max(size_a, size_b) / max(1.0, min(size_a, size_b)) > 1.75:
        return False
    return _colour_compatible(a, b)


def _compact_latin_label(text: str) -> bool:
    stripped = str(text or "").strip()
    if not stripped:
        return False
    visible = _visible_len(stripped)
    if visible > 16:
        return False
    if any(ch.isspace() for ch in stripped):
        return False
    if any(ch in stripped for ch in "，。！？；;,:：、/\\()（）[]{}<>《》"):
        return False
    latin_letters = sum(
        1 for ch in stripped
        if ("A" <= ch <= "Z") or ("a" <= ch <= "z")
    )
    return latin_letters >= 2 and latin_letters / max(1, visible) >= 0.55


def _compact_latin_title_compatible(a: dict[str, Any],
                                    b: dict[str, Any]) -> bool:
    """Same-row product/framework labels may be one slot with kept sizes."""
    if a["align"] != b["align"]:
        return False
    if a["colour_family"] != "blue" or b["colour_family"] != "blue":
        return False
    if not _colour_compatible(a, b):
        return False
    if not bool(a["el"].get("bold")) or not bool(b["el"].get("bold")):
        return False
    if _has_list_marker(a["el"]) or _has_list_marker(b["el"]):
        return False
    if not _compact_latin_label(str(a["el"].get("text") or "")):
        return False
    if not _compact_latin_label(str(b["el"].get("text") or "")):
        return False
    font_a = str(a["el"].get("font") or a["el"].get("font_name") or "")
    font_b = str(b["el"].get("font") or b["el"].get("font_name") or "")
    if font_a and font_b and font_a != font_b:
        return False
    size_a = float(a["el"].get("size") or a["el"].get("font_size") or 0)
    size_b = float(b["el"].get("size") or b["el"].get("font_size") or 0)
    if size_a < 20 or size_b < 20:
        return False
    if abs(size_a - size_b) <= MAX_STYLE_SIZE_DELTA:
        return False
    return (
        max(size_a, size_b) / max(1.0, min(size_a, size_b)) <= 1.45
        and abs(size_a - size_b) <= 14.0
    )


def _alignment_style_compatible(a: dict[str, Any],
                                b: dict[str, Any]) -> bool:
    if a["align"] != b["align"]:
        return False
    if bool(a["el"].get("bold")) != bool(b["el"].get("bold")):
        return False
    font_a = str(a["el"].get("font") or a["el"].get("font_name") or "")
    font_b = str(b["el"].get("font") or b["el"].get("font_name") or "")
    if font_a and font_b and font_a != font_b:
        return False
    size_a = float(a["el"].get("size") or a["el"].get("font_size") or 0)
    size_b = float(b["el"].get("size") or b["el"].get("font_size") or 0)
    if (
        size_a <= 0
        or size_b <= 0
        or abs(size_a - size_b) > MAX_STYLE_SIZE_DELTA
    ):
        return False
    families = {a["colour_family"], b["colour_family"]}
    if len(families) == 1:
        return True
    # Paragraph bodies often alternate dark and gray emphasis, but blue/red
    # labels are usually structural headers and should not pull body text.
    return families <= {"black", "gray"}


def _align_value(el: dict[str, Any]) -> str:
    return str(el.get("align") or "left").lower()


def _row_aligned(a: dict[str, Any], b: dict[str, Any]) -> bool:
    ax1, ay1, ax2, ay2 = a["box"]
    bx1, by1, bx2, by2 = b["box"]
    ah = ay2 - ay1
    bh = by2 - by1
    acy = (ay1 + ay2) / 2.0
    bcy = (by1 + by2) / 2.0
    return abs(acy - bcy) <= max(4.0, min(ah, bh) * 0.42)


def _column_aligned(a: dict[str, Any], b: dict[str, Any]) -> bool:
    ax1, ay1, ax2, ay2 = a["box"]
    bx1, by1, bx2, by2 = b["box"]
    aw = ax2 - ax1
    bw = bx2 - bx1
    y_gap = max(0.0, max(ay1, by1) - min(ay2, by2))
    left_close = abs(ax1 - bx1) <= max(7.0, min(18.0, min(aw, bw) * 0.14))
    center_close = abs(((ax1 + ax2) / 2.0) - ((bx1 + bx2) / 2.0)) <= 10.0
    return y_gap <= 42.0 and (left_close or center_close)


def _container_siblings(a: dict[str, Any], b: dict[str, Any]) -> bool:
    if a["id"] == b["id"]:
        return False
    aw = a["box"][2] - a["box"][0]
    ah = a["box"][3] - a["box"][1]
    bw = b["box"][2] - b["box"][0]
    bh = b["box"][3] - b["box"][1]
    if min(aw, ah, bw, bh) <= 0:
        return False
    if max(aw, bw) / min(aw, bw) > 1.18:
        return False
    if max(ah, bh) / min(ah, bh) > 1.22:
        return False
    acx, acy = _center(a["box"])
    bcx, bcy = _center(b["box"])
    same_row = abs(acy - bcy) <= max(12.0, min(ah, bh) * 0.18)
    same_col = abs(acx - bcx) <= max(12.0, min(aw, bw) * 0.18)
    return same_row or same_col


def _sibling_relative_metrics(
    a: dict[str, Any],
    b: dict[str, Any],
    containers: dict[int, dict[str, Any]],
) -> list[tuple[tuple[float, float, float, float],
                tuple[float, float, float, float]]]:
    ax1, ay1, ax2, ay2 = a["box"]
    bx1, by1, bx2, by2 = b["box"]
    out: list[tuple[tuple[float, float, float, float],
                    tuple[float, float, float, float]]] = []
    for pa in a.get("ancestors") or []:
        ca = containers.get(pa)
        if ca is None:
            continue
        for pb in b.get("ancestors") or []:
            if pa == pb:
                continue
            cb = containers.get(pb)
            if cb is None or not _container_siblings(ca, cb):
                continue
            cax1, cay1, cax2, cay2 = ca["box"]
            cbx1, cby1, cbx2, cby2 = cb["box"]
            caw = max(1.0, cax2 - cax1)
            cah = max(1.0, cay2 - cay1)
            cbw = max(1.0, cbx2 - cbx1)
            cbh = max(1.0, cby2 - cby1)
            a_rel = ((ax1 - cax1) / caw, (ay1 - cay1) / cah,
                     (ax2 - ax1) / caw, (ay2 - ay1) / cah)
            b_rel = ((bx1 - cbx1) / cbw, (by1 - cby1) / cbh,
                     (bx2 - bx1) / cbw, (by2 - by1) / cbh)
            out.append((a_rel, b_rel))
    return out


def _same_relative_slot(a: dict[str, Any], b: dict[str, Any],
                        containers: dict[int, dict[str, Any]]) -> bool:
    for a_rel, b_rel in _sibling_relative_metrics(a, b, containers):
        if (
            abs(a_rel[0] - b_rel[0]) <= 0.08
            and abs(a_rel[1] - b_rel[1]) <= 0.10
            and abs(a_rel[2] - b_rel[2]) <= 0.22
            and abs(a_rel[3] - b_rel[3]) <= 0.16
        ):
            return True
    return False


def _same_relative_alert_band(a: dict[str, Any], b: dict[str, Any],
                              containers: dict[int, dict[str, Any]]) -> bool:
    """Match red warning/callout text across sibling cards despite wrapping."""
    if a["colour_family"] != "red" or b["colour_family"] != "red":
        return False
    if not bool(a["el"].get("bold")) or not bool(b["el"].get("bold")):
        return False
    if _has_list_marker(a["el"]) or _has_list_marker(b["el"]):
        return False
    if min(_visible_len(a["el"].get("text")), _visible_len(b["el"].get("text"))) < 6:
        return False
    for a_rel, b_rel in _sibling_relative_metrics(a, b, containers):
        # Warning strips sit in the lower part of sibling cards. Their line
        # widths legitimately differ when one side wraps to two lines and the
        # other to three, so compare x/y band and height but not width.
        if min(a_rel[1], b_rel[1]) < 0.45:
            continue
        if (
            abs(a_rel[0] - b_rel[0]) <= 0.12
            and abs(a_rel[1] - b_rel[1]) <= 0.12
            and abs(a_rel[3] - b_rel[3]) <= 0.12
        ):
            return True
    return False


def _same_vertical_list_slot(
    a: dict[str, Any],
    b: dict[str, Any],
    containers: dict[int, dict[str, Any]],
    scope: int | None,
    same_parent: bool,
) -> bool:
    """Match stacked TOC/section-list items whose text widths vary widely."""
    if a["align"] != "left" or b["align"] != "left":
        return False
    if a["colour_family"] != "blue" or b["colour_family"] != "blue":
        return False
    if not bool(a["el"].get("bold")) or not bool(b["el"].get("bold")):
        return False
    if _has_list_marker(a["el"]) or _has_list_marker(b["el"]):
        return False
    visible_a = _visible_len(str(a["el"].get("text") or ""))
    visible_b = _visible_len(str(b["el"].get("text") or ""))
    if min(visible_a, visible_b) < 2 or max(visible_a, visible_b) > 48:
        return False
    if not (
        same_parent
        or scope is not None
        or _sibling_relative_metrics(a, b, containers)
    ):
        return False
    ax1, ay1, ax2, ay2 = a["box"]
    bx1, by1, bx2, by2 = b["box"]
    ah = max(1.0, ay2 - ay1)
    bh = max(1.0, by2 - by1)
    if ay1 <= by1:
        y_gap = by1 - ay2
    else:
        y_gap = ay1 - by2
    if y_gap < -min(ah, bh) * 0.12:
        return False
    if y_gap > max(86.0, _median_float([ah, bh]) * 1.85):
        return False
    return abs(ax1 - bx1) <= 26.0


def _common_scope(a: dict[str, Any], b: dict[str, Any],
                  containers: dict[int, dict[str, Any]]) -> int | None:
    common = set(a["scope_ancestors"]) & set(b["scope_ancestors"])
    if not common:
        return None
    return min(common, key=lambda cid: containers[cid]["area"])


def _median_size(values: list[float]) -> int:
    if not values:
        return 0
    ordered = sorted(float(v) for v in values)
    mid = len(ordered) // 2
    if len(ordered) % 2:
        median = ordered[mid]
    else:
        median = (ordered[mid - 1] + ordered[mid]) / 2.0
    return int(math.floor(median + 0.5))


def _median_float(values: list[float]) -> float:
    if not values:
        return 0.0
    ordered = sorted(float(v) for v in values)
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[mid]
    return (ordered[mid - 1] + ordered[mid]) / 2.0


def _visible_len(text: str) -> int:
    return sum(1 for ch in str(text or "") if not ch.isspace())


def _has_list_marker(el: dict[str, Any]) -> bool:
    name = str(el.get("name") or "")
    role = str(el.get("role") or "")
    return (
        bool(el.get("ignored_marker_image"))
        or "bullet_marker" in name
        or role in {"bullet_marker", "inferred_bullet_marker"}
    )


def _short_centered_stack_text(text: str) -> bool:
    stripped = str(text or "").strip()
    if not stripped:
        return False
    if _visible_len(stripped) > 14:
        return False
    # Sentence/list bodies with colon/comma punctuation should keep their
    # measured left indent. Compact labels may still contain 、 / + - ().
    if any(ch in stripped for ch in "，。！？；;,:："):
        return False
    return True


def _compact_center_stack_candidate(
    a: dict[str, Any],
    b: dict[str, Any],
) -> bool:
    if _has_list_marker(a["el"]) or _has_list_marker(b["el"]):
        return False
    if not _short_centered_stack_text(str(a["el"].get("text") or "")):
        return False
    if not _short_centered_stack_text(str(b["el"].get("text") or "")):
        return False
    ax1, ay1, ax2, ay2 = a["box"]
    bx1, by1, bx2, by2 = b["box"]
    aw = max(1.0, ax2 - ax1)
    bw = max(1.0, bx2 - bx1)
    ah = max(1.0, ay2 - ay1)
    bh = max(1.0, by2 - by1)
    acy = (ay1 + ay2) / 2.0
    bcy = (by1 + by2) / 2.0
    center_y_gap = abs(acy - bcy)
    if center_y_gap < max(8.0, min(ah, bh) * 0.42):
        return False
    if center_y_gap > max(38.0, _median_float([ah, bh]) * 1.85):
        return False
    overlap = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    if overlap / min(aw, bw) < 0.55:
        return False
    center_gap = abs((ax1 + ax2) / 2.0 - (bx1 + bx2) / 2.0)
    if center_gap <= 1.0:
        return False
    return center_gap <= max(24.0, min(aw, bw) * 0.42)


def _reference_center_for_stack(
    items: list[dict[str, Any]],
) -> float:
    best = max(
        items,
        key=lambda item: (
            _element_size_reliability(item["el"]),
            _visible_len(str(item["el"].get("text") or "")),
        ),
    )
    x1, _y1, x2, _y2 = best["box"]
    return (x1 + x2) / 2.0


def _element_size_reliability(el: dict[str, Any]) -> float:
    score = float(_visible_len(str(el.get("text") or "")))
    bbox = el.get("source_bbox")
    if bbox and len(bbox) == 4:
        x1, y1, x2, y2 = (float(v) for v in bbox)
        score += min(8.0, max(0.0, (x2 - x1) / 40.0))
        score += min(4.0, max(0.0, (y2 - y1) / 8.0))
    target = el.get("target_ink")
    if target and len(target) == 4:
        tx1, ty1, tx2, ty2 = (float(v) for v in target)
        score += min(6.0, max(0.0, (tx2 - tx1) / 45.0))
        score += min(3.0, max(0.0, (ty2 - ty1) / 8.0))
    return score


def _suggested_class_size(values: list[float],
                          force_apply: bool = False,
                          elements: list[dict[str, Any]] | None = None) -> int:
    if not values:
        return 0
    if force_apply and len(values) == 2 and elements and len(elements) == 2:
        scored = sorted(
            zip(values, elements),
            key=lambda pair: (
                -_element_size_reliability(pair[1]),
                abs(float(pair[0]) - _median_float(values)),
                float(pair[0]),
            ),
        )
        return int(math.floor(float(scored[0][0]) + 0.5))
    return _median_size(values)


def _sync_text_box(text: dict[str, Any]) -> None:
    box = text["el"].get("box")
    if box and len(box) == 4:
        text["box"] = _box_xyxy(box)


def _apply_parent_column_alignment(
    texts: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Snap close vertical text stacks inside the same container to one x.

    This is deliberately local: only left-aligned text with compatible body
    style and near-identical x positions is adjusted. It catches OCR/preview
    drift in paragraph stacks and list bodies without merging headers into
    body rows.
    """
    by_parent: dict[int, list[int]] = {}
    for idx, text in enumerate(texts):
        parent = text.get("direct_parent")
        if parent is None or text["align"] != "left":
            continue
        by_parent.setdefault(int(parent), []).append(idx)

    adjustments: list[dict[str, Any]] = []
    for parent, ids in by_parent.items():
        if len(ids) < 2:
            continue
        uf = {idx: idx for idx in ids}

        def find(idx: int) -> int:
            while uf[idx] != idx:
                uf[idx] = uf[uf[idx]]
                idx = uf[idx]
            return idx

        def union(a: int, b: int) -> None:
            ra = find(a)
            rb = find(b)
            if ra != rb:
                uf[rb] = ra

        for pos, i in enumerate(ids):
            for j in ids[pos + 1:]:
                if not _alignment_style_compatible(texts[i], texts[j]):
                    continue
                ax = texts[i]["box"][0]
                bx = texts[j]["box"][0]
                if abs(ax - bx) <= 24.0:
                    union(i, j)

        groups: dict[int, list[int]] = {}
        for idx in ids:
            groups.setdefault(find(idx), []).append(idx)
        for group in groups.values():
            if len(group) < 3:
                continue
            centers_y = [
                (texts[i]["box"][1] + texts[i]["box"][3]) / 2.0
                for i in group
            ]
            heights = [
                max(1.0, texts[i]["box"][3] - texts[i]["box"][1])
                for i in group
            ]
            if max(centers_y) - min(centers_y) <= _median_float(heights) * 0.75:
                continue
            y_sorted = sorted(centers_y)
            max_y_gap = max(
                y_sorted[pos + 1] - y_sorted[pos]
                for pos in range(len(y_sorted) - 1)
            )
            if max_y_gap > max(36.0, _median_float(heights) * 2.2):
                continue
            xs = [texts[i]["box"][0] for i in group]
            if max(xs) - min(xs) > 24.0:
                continue
            target_x = round(_median_float(xs), 3)
            moved: list[dict[str, Any]] = []
            for i in group:
                el = texts[i]["el"]
                box = el.get("box")
                if not box or len(box) != 4:
                    continue
                old_x = float(box[0])
                if abs(old_x - target_x) < 0.15:
                    continue
                box[0] = target_x
                _sync_text_box(texts[i])
                moved.append({
                    "name": el.get("name"),
                    "text": el.get("text"),
                    "old_x": round(old_x, 3),
                    "new_x": target_x,
                })
            if moved:
                adjustments.append({
                    "direct_parent_id": parent,
                    "target_x": target_x,
                    "members": moved,
                })
    return adjustments


def _table_column_text_candidate(text: dict[str, Any]) -> bool:
    """Whether a text box is small enough to behave like a table cell label."""
    el = text["el"]
    if text["align"] != "left" or _has_list_marker(el):
        return False
    if text.get("colour_family") == "white":
        return False
    box = el.get("box")
    if not box or len(box) != 4:
        return False
    value = str(el.get("text") or "").strip()
    if not value:
        return False
    visible = _visible_len(value)
    if visible == 0 or visible > 14:
        return False
    if any(ch in value for ch in "，。！？；;,:："):
        return False
    width = float(box[2])
    height = float(box[3])
    if width <= 0 or height <= 0 or width > 180 or height > 44:
        return False
    return True


def _scope_is_table_like(
    scope_items: list[dict[str, Any]],
    min_rows: int = 3,
) -> bool:
    if len(scope_items) < 6:
        return False
    rows: list[float] = []
    xs: list[float] = []
    for item in sorted(
        scope_items,
        key=lambda text: (
            float(text["el"]["box"][1]) + float(text["el"]["box"][3]) / 2.0
        ),
    ):
        box = item["el"].get("box")
        if not box or len(box) != 4:
            continue
        cy = float(box[1]) + float(box[3]) / 2.0
        if not rows or abs(cy - rows[-1]) > 10.0:
            rows.append(cy)
        else:
            rows[-1] = (rows[-1] + cy) / 2.0
        xs.append(float(box[0]))
    if len(rows) < min_rows or len(xs) < 6:
        return False
    return max(xs) - min(xs) >= 90.0


def _cluster_numeric(values: list[float], threshold: float) -> list[list[float]]:
    if not values:
        return []
    ordered = sorted(float(v) for v in values)
    clusters: list[list[float]] = [[ordered[0]]]
    for value in ordered[1:]:
        if value - clusters[-1][-1] <= threshold:
            clusters[-1].append(value)
        else:
            clusters.append([value])
    return clusters


def _table_column_target_x(items: list[dict[str, Any]]) -> float | None:
    xs = [float(item["el"]["box"][0]) for item in items]
    if not xs:
        return None
    tight_clusters = _cluster_numeric(xs, 7.0)
    max_count = max(len(cluster) for cluster in tight_clusters)
    # A left-shifted icon/OCR remnant often forms a small secondary cluster.
    # Use the dominant modal x; if two modes are almost equally strong,
    # prefer the rightmost one because the failure mode we are correcting is
    # usually a label box starting on the preceding icon.
    contenders = [
        cluster for cluster in tight_clusters
        if len(cluster) >= max(2, max_count - 1)
    ]
    if not contenders:
        return None
    best = max(
        contenders,
        key=lambda cluster: (len(cluster), _median_float(cluster)),
    )
    return round(_median_float(best), 3)


def _apply_table_column_geometry_alignment(
    texts: list[dict[str, Any]],
    containers: dict[int, dict[str, Any]],
) -> list[dict[str, Any]]:
    """Align short vertical table-cell label columns across visual styles.

    This pass is intentionally geometry-first. Unlike style classes, it may
    group red/blue/gray/black variants when they sit in the same table-like
    scope and form a repeated left-x stack. It only changes x, never y,
    size, boldness, or paragraph alignment.
    """
    by_scope: dict[int, list[dict[str, Any]]] = {}
    all_scope_items: dict[int, list[dict[str, Any]]] = {}
    for text in texts:
        scopes = text.get("scope_ancestors") or []
        if not scopes:
            continue
        for raw_scope in scopes:
            scope = int(raw_scope)
            all_scope_items.setdefault(scope, []).append(text)
            if _table_column_text_candidate(text):
                by_scope.setdefault(scope, []).append(text)

    adjustments: list[dict[str, Any]] = []
    aligned_text_ids: set[int] = set()
    for scope, candidates in sorted(
        by_scope.items(),
        key=lambda pair: containers.get(pair[0], {}).get("area", 0.0),
    ):
        candidates = [
            item for item in candidates
            if int(item["id"]) not in aligned_text_ids
        ]
        if len(candidates) < 3:
            continue
        if not _scope_is_table_like(all_scope_items.get(scope, [])):
            continue
        ordered = sorted(candidates, key=lambda item: float(item["el"]["box"][0]))
        columns: list[list[dict[str, Any]]] = []
        for item in ordered:
            x = float(item["el"]["box"][0])
            if not columns:
                columns.append([item])
                continue
            prev_xs = [float(prev["el"]["box"][0]) for prev in columns[-1]]
            if abs(x - _median_float(prev_xs)) <= 34.0:
                columns[-1].append(item)
            else:
                columns.append([item])

        for column in columns:
            if len(column) < 3:
                continue
            centers_y = [
                float(item["el"]["box"][1]) + float(item["el"]["box"][3]) / 2.0
                for item in column
            ]
            heights = [
                max(1.0, float(item["el"]["box"][3]))
                for item in column
            ]
            if max(centers_y) - min(centers_y) <= _median_float(heights) * 1.6:
                continue
            row_clusters = _cluster_numeric(
                centers_y,
                max(8.0, _median_float(heights) * 0.45),
            )
            if len(row_clusters) < 3:
                continue
            xs = [float(item["el"]["box"][0]) for item in column]
            if max(xs) - min(xs) > 34.0:
                continue
            widths = [
                max(1.0, float(item["el"]["box"][2]))
                for item in column
            ]
            if max(widths) / min(widths) > 1.75:
                continue
            target_x = _table_column_target_x(column)
            if target_x is None:
                continue
            moved: list[dict[str, Any]] = []
            for item in column:
                el = item["el"]
                box = el.get("box")
                if not box or len(box) != 4:
                    continue
                old_x = float(box[0])
                if abs(old_x - target_x) > 34.0:
                    continue
                if abs(old_x - target_x) < 0.15:
                    continue
                box[0] = target_x
                el["style_table_column_x"] = {
                    "axis": "left",
                    "target": target_x,
                    "scope": containers.get(scope, {}).get("name"),
                }
                _sync_text_box(item)
                moved.append({
                    "name": el.get("name"),
                    "text": el.get("text"),
                    "old_x": round(old_x, 3),
                    "new_x": target_x,
                })
            if moved:
                aligned_text_ids.update(int(item["id"]) for item in column)
                scope_container = containers.get(scope)
                adjustments.append({
                    "scope_id": scope,
                    "scope_name": (
                        scope_container.get("name") if scope_container else None
                    ),
                    "target_x": target_x,
                    "members": moved,
                })
    return adjustments


def _run_role(run: dict[str, Any], idx: int) -> str:
    text = str(run.get("text") or "").strip()
    if text.startswith(("（", "(")) and text.endswith(("）", ")")):
        return "bracket"
    if text.startswith(("（", "(")):
        return "bracket"
    if idx == 0:
        return "main"
    return f"run_{idx}"


def _explicit_mixed_run_sizes(el: dict[str, Any]) -> bool:
    runs = el.get("runs") or []
    sizes = {
        int(round(float(run["size"])))
        for run in runs
        if run.get("size") is not None
    }
    return len(sizes) > 1


def _run_slot_classes(texts: list[dict[str, Any]],
                      members: list[int],
                      class_id: str,
                      apply: bool,
                      min_apply_size: int,
                      force_apply: bool = False) -> list[dict[str, Any]]:
    buckets: dict[str, list[tuple[dict[str, Any], dict[str, Any], int]]] = {}
    for i in members:
        el = texts[i]["el"]
        for idx, run in enumerate(el.get("runs") or []):
            role = _run_role(run, idx)
            if run.get("size") is None and el.get("size") is None:
                continue
            buckets.setdefault(role, []).append((el, run, idx))

    out: list[dict[str, Any]] = []
    for role, entries in buckets.items():
        if len(entries) < 2:
            continue
        sizes = [
            float(run.get("size") or el.get("size") or 0)
            for el, run, _idx in entries
        ]
        size_elements = [el for el, _run, _idx in entries]
        bolds = [
            bool(run.get("bold", el.get("bold", False)))
            for el, run, _idx in entries
        ]
        suggested = _suggested_class_size(
            sizes, force_apply, size_elements)
        max_delta = max(abs(float(size) - suggested) for size in sizes)
        applied = False
        applied_bold = False
        dominant_bold = sorted(
            set(bolds),
            key=lambda value: (-bolds.count(value), value),
        )[0]
        bold_stable_or_outlier = (
            bolds.count(dominant_bold) >= max(3, len(bolds) - 1)
        )
        # Structural classes are high-priority once they are formed. Keep the
        # guardrails local to a role: we only normalize runs that already have
        # close sizes inside the same structural slot.
        role_limit = MAX_STYLE_SIZE_DELTA
        can_apply_role = role == "bracket" or (
            role == "main"
            and (force_apply or len(entries) >= min_apply_size)
            and bold_stable_or_outlier
            and max_delta <= role_limit
        )
        if (
            apply
            and can_apply_role
            and (force_apply or len(entries) >= min_apply_size)
            and suggested > 0
            and max_delta <= role_limit
        ):
            for el, run, _idx in entries:
                run["size"] = suggested
                run["style_class"] = f"{class_id}.{role}"
                run["style_class_suggested_size"] = suggested
            applied = True
        if (
            apply
            and role in {"bracket", "main"}
            and (force_apply or len(entries) >= min_apply_size)
            and bold_stable_or_outlier
        ):
            for _el, run, _idx in entries:
                run["bold"] = dominant_bold
            applied_bold = True
        out.append({
            "role": role,
            "count": len(entries),
            "suggested_size": suggested,
            "sizes": [int(round(v)) for v in sizes],
            "bold_values": sorted(set(bolds)),
            "applied": applied,
            "applied_bold": applied_bold,
            "members": [
                {
                    "text_name": el.get("name"),
                    "run_index": idx,
                    "text": run.get("text"),
                    "size": run.get("size"),
                    "bold": bool(run.get("bold", el.get("bold", False))),
                }
                for el, run, idx in entries
            ],
        })
    return out


def _group_containers(containers: dict[int, dict[str, Any]]) -> list[list[int]]:
    ids = [cid for cid, c in containers.items() if c["child_text_count"] >= 1]
    parent = {cid: cid for cid in ids}

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra = find(a)
        rb = find(b)
        if ra != rb:
            parent[rb] = ra

    for i, cid in enumerate(ids):
        for other in ids[i + 1:]:
            if _container_siblings(containers[cid], containers[other]):
                union(cid, other)
    groups: dict[int, list[int]] = {}
    for cid in ids:
        groups.setdefault(find(cid), []).append(cid)
    return [g for g in groups.values() if len(g) >= 2]


def _marker_owner_name(marker_name: str) -> str | None:
    suffixes = (
        "_bullet_marker_inferred",
        "_bullet_marker",
    )
    for suffix in suffixes:
        if marker_name.endswith(suffix):
            return marker_name[:-len(suffix)]
    if "_bullet_marker_" in marker_name:
        return marker_name.split("_bullet_marker_", 1)[0]
    return None


def _bullet_markers_by_text(
    elements: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for el in elements:
        if (el.get("type") or "").lower() != "image":
            continue
        name = str(el.get("name") or "")
        role = str(el.get("role") or "")
        if "bullet_marker" not in name and role not in {
            "bullet_marker",
            "inferred_bullet_marker",
        }:
            continue
        owner = _marker_owner_name(name)
        if owner:
            out[owner] = el
    return out


def _good_marker_score(marker: dict[str, Any]) -> tuple[float, float]:
    box = marker.get("box") or [0, 0, 0, 0]
    w = max(1.0, float(box[2]))
    h = max(1.0, float(box[3]))
    aspect = w / h
    aspect_penalty = abs(math.log(max(0.1, min(10.0, aspect))))
    height_penalty = max(0.0, h - 16.0) * 0.25 + max(0.0, 5.0 - h)
    return aspect_penalty + height_penalty, h


def _cluster_texts_by_x(text_items: list[dict[str, Any]],
                        threshold: float = 34.0) -> list[list[dict[str, Any]]]:
    ordered = sorted(text_items, key=lambda t: float(t["el"]["box"][0]))
    clusters: list[list[dict[str, Any]]] = []
    for item in ordered:
        x = float(item["el"]["box"][0])
        if not clusters:
            clusters.append([item])
            continue
        prev_x = float(clusters[-1][-1]["el"]["box"][0])
        if x - prev_x <= threshold:
            clusters[-1].append(item)
        else:
            clusters.append([item])
    return clusters


def _apply_bullet_marker_priors(
    slide: dict[str, Any],
    texts: list[dict[str, Any]],
    members: list[int],
    class_id: str,
) -> dict[str, Any] | None:
    elements = slide.get("elements") or []
    text_items = [texts[i] for i in members if texts[i]["el"].get("box")]
    if len(text_items) < 3:
        return None
    markers = _bullet_markers_by_text(elements)
    marked = [
        item for item in text_items
        if str(item["el"].get("name") or "") in markers
        or item["el"].get("ignored_marker_image")
    ]
    if len(marked) < max(2, math.ceil(len(text_items) * 0.35)):
        return None

    adjustments: list[dict[str, Any]] = []
    created = 0
    normalized = 0
    for cluster in _cluster_texts_by_x(text_items):
        cluster_markers: list[tuple[dict[str, Any], dict[str, Any]]] = []
        for item in cluster:
            name = str(item["el"].get("name") or "")
            marker = markers.get(name)
            if marker is not None and marker.get("box") and marker.get("path"):
                cluster_markers.append((item, marker))
        if not cluster_markers:
            continue
        template_item, template_marker = min(
            cluster_markers,
            key=lambda pair: _good_marker_score(pair[1]),
        )
        template_path = template_marker.get("path")
        template_box = template_marker.get("box") or [0, 0, 10, 10]
        marker_w = float(template_box[2])
        marker_h = float(template_box[3])
        if marker_w <= 0 or marker_h <= 0 or marker_w > 24 or marker_h > 24:
            good_sizes = [
                (float(marker.get("box")[2]), float(marker.get("box")[3]))
                for _item, marker in cluster_markers
                if marker.get("box")
                and 4 <= float(marker.get("box")[2]) <= 24
                and 4 <= float(marker.get("box")[3]) <= 24
            ]
            if good_sizes:
                marker_w = _median_float([w for w, _h in good_sizes])
                marker_h = _median_float([h for _w, h in good_sizes])
            else:
                marker_w = marker_h = 10.0
        body_x = _median_float([
            float(item["el"]["box"][0])
            for item, _marker in cluster_markers
        ])
        marker_x = _median_float([
            float(marker["box"][0])
            for _item, marker in cluster_markers
            if marker.get("box")
        ])
        cluster_report: list[dict[str, Any]] = []
        for item in cluster:
            el = item["el"]
            name = str(el.get("name") or "")
            box = el.get("box")
            if not name or not box or len(box) != 4:
                continue
            old_x = float(box[0])
            if abs(old_x - body_x) <= 34.0:
                box[0] = round(body_x, 3)
                _sync_text_box(item)
            y = float(box[1])
            h = float(box[3])
            marker_y = round(y + h / 2.0 - marker_h / 2.0, 3)
            new_box = [
                round(marker_x, 3),
                marker_y,
                round(marker_w, 3),
                round(marker_h, 3),
            ]
            marker = markers.get(name)
            if marker is None:
                marker = {
                    "type": "image",
                    "name": f"{name}_bullet_marker_inferred",
                    "path": template_path,
                    "box": new_box,
                    "role": "inferred_bullet_marker",
                    "source_text": name,
                    "style_class": class_id,
                }
                elements.append(marker)
                markers[name] = marker
                created += 1
            else:
                marker["path"] = template_path
                marker["box"] = new_box
                marker["style_class"] = class_id
                normalized += 1
            el["ignored_marker_image"] = {
                "box": new_box,
                "path": template_path,
            }
            if abs(old_x - float(box[0])) >= 0.15 or marker is not None:
                cluster_report.append({
                    "name": name,
                    "text": el.get("text"),
                    "old_x": round(old_x, 3),
                    "new_x": round(float(box[0]), 3),
                    "marker_box": new_box,
                    "created_marker": marker.get("role") == "inferred_bullet_marker",
                })
        if cluster_report:
            adjustments.append({
                "body_x": round(body_x, 3),
                "marker_x": round(marker_x, 3),
                "template": template_marker.get("name"),
                "members": cluster_report,
            })
    if not adjustments:
        return None
    return {
        "created": created,
        "normalized": normalized,
        "clusters": adjustments,
    }


def _apply_class_position_priors(
    texts: list[dict[str, Any]],
    members: list[int],
    class_id: str,
    force_apply: bool = False,
) -> dict[str, Any] | None:
    """Give a confident structural class precedence over tiny xy drift."""
    items = [
        texts[i] for i in members
        if texts[i]["align"] == "left"
        and texts[i]["el"].get("box")
        and len(texts[i]["el"]["box"]) == 4
    ]
    if len(items) < (2 if force_apply else 3):
        return None
    x_moved: list[dict[str, Any]] = []
    edges = []
    for item in items:
        x, y, w, h = (float(v) for v in item["el"]["box"])
        edges.append({
            "item": item,
            "left": x,
            "center": x + w / 2.0,
            "right": x + w,
            "top": y,
            "middle": y + h / 2.0,
            "bottom": y + h,
        })
    def best_axis(keys: list[str]) -> tuple[str, float, float]:
        best_key = keys[0]
        best_values = [edge[best_key] for edge in edges]
        best_spread = max(best_values) - min(best_values)
        for key in keys[1:]:
            values = [edge[key] for edge in edges]
            spread = max(values) - min(values)
            if spread < best_spread:
                best_key = key
                best_values = values
                best_spread = spread
        return best_key, _median_float(best_values), best_spread

    x_axis, x_target_raw, x_spread = best_axis(["left", "center", "right"])
    x_spread_limit = 14.0 if force_apply else 8.0
    if x_spread <= x_spread_limit:
        target_axis = round(x_target_raw, 3)
        for edge in edges:
            item = edge["item"]
            el = item["el"]
            box = el["box"]
            old_x = float(box[0])
            if x_axis == "left":
                new_x = target_axis
            elif x_axis == "center":
                new_x = target_axis - float(box[2]) / 2.0
            else:
                new_x = target_axis - float(box[2])
            new_x = round(new_x, 3)
            if abs(old_x - new_x) < 0.15:
                continue
            box[0] = new_x
            el["style_class_position_x"] = {
                "axis": x_axis,
                "target": target_axis,
            }
            _sync_text_box(item)
            x_moved.append({
                "name": el.get("name"),
                "text": el.get("text"),
                "old_x": round(old_x, 3),
                "new_x": new_x,
            })

    row_adjustments: list[dict[str, Any]] = []
    row_items = sorted(
        items,
        key=lambda item: (
            float(item["el"]["box"][1]) + float(item["el"]["box"][3]) / 2.0,
            float(item["el"]["box"][0]),
        ),
    )
    rows: list[list[dict[str, Any]]] = []
    for item in row_items:
        box = item["el"]["box"]
        cy = float(box[1]) + float(box[3]) / 2.0
        if not rows:
            rows.append([item])
            continue
        prev_centers = [
            float(prev["el"]["box"][1]) + float(prev["el"]["box"][3]) / 2.0
            for prev in rows[-1]
        ]
        if abs(cy - _median_float(prev_centers)) <= 7.0:
            rows[-1].append(item)
        else:
            rows.append([item])
    for row in rows:
        if len(row) < 2:
            continue
        row_xs = [float(item["el"]["box"][0]) for item in row]
        if max(row_xs) - min(row_xs) < 48.0:
            continue
        row_edges = []
        for item in row:
            x, y, w, h = (float(v) for v in item["el"]["box"])
            row_edges.append({
                "item": item,
                "top": y,
                "middle": y + h / 2.0,
                "bottom": y + h,
            })
        y_axis = "top"
        y_values = [edge[y_axis] for edge in row_edges]
        y_spread = max(y_values) - min(y_values)
        for candidate in ("middle", "bottom"):
            values = [edge[candidate] for edge in row_edges]
            spread = max(values) - min(values)
            if spread < y_spread:
                y_axis = candidate
                y_values = values
                y_spread = spread
        if y_spread > 7.0:
            continue
        target_axis_y = round(_median_float(y_values), 3)
        moved: list[dict[str, Any]] = []
        for edge in row_edges:
            item = edge["item"]
            el = item["el"]
            box = el["box"]
            old_y = float(box[1])
            if y_axis == "top":
                new_y = target_axis_y
            elif y_axis == "middle":
                new_y = target_axis_y - float(box[3]) / 2.0
            else:
                new_y = target_axis_y - float(box[3])
            new_y = round(new_y, 3)
            if abs(old_y - new_y) < 0.15:
                continue
            box[1] = new_y
            el["style_class_position_y"] = {
                "axis": y_axis,
                "target": target_axis_y,
            }
            _sync_text_box(item)
            moved.append({
                "name": el.get("name"),
                "text": el.get("text"),
                "old_y": round(old_y, 3),
                "new_y": new_y,
            })
        if moved:
            row_adjustments.append({
                "axis": y_axis,
                "target": target_axis_y,
                "spread": round(y_spread, 3),
                "members": moved,
            })

    column_stack_adjustments: list[dict[str, Any]] = []
    by_parent: dict[int, list[dict[str, Any]]] = {}
    for item in items:
        parent = item.get("direct_parent")
        if parent is None:
            continue
        by_parent.setdefault(int(parent), []).append(item)
    for parent, group in by_parent.items():
        if len(group) != 2:
            continue
        pair = sorted(
            group,
            key=lambda item: (
                float(item["el"]["box"][1]) + float(item["el"]["box"][3]) / 2.0,
                float(item["el"]["box"][0]),
            ),
        )
        if not _compact_center_stack_candidate(pair[0], pair[1]):
            continue
        target_center = round(_reference_center_for_stack(pair), 3)
        moved: list[dict[str, Any]] = []
        for item in pair:
            el = item["el"]
            box = el.get("box")
            if not box or len(box) != 4:
                continue
            old_x = float(box[0])
            new_x = round(target_center - float(box[2]) / 2.0, 3)
            if abs(old_x - new_x) < 0.15:
                continue
            box[0] = new_x
            el["style_class_position_x"] = {
                "axis": "center",
                "target": target_center,
                "scope": "compact_stack",
            }
            _sync_text_box(item)
            moved.append({
                "name": el.get("name"),
                "text": el.get("text"),
                "old_x": round(old_x, 3),
                "new_x": new_x,
            })
        if moved:
            column_stack_adjustments.append({
                "direct_parent_id": parent,
                "axis": "center",
                "target": target_center,
                "members": moved,
            })

    if not x_moved and not row_adjustments and not column_stack_adjustments:
        return None
    return {
        "class_id": class_id,
        "x_alignment": (
            {
                "axis": x_axis,
                "target": round(x_target_raw, 3),
                "spread": round(x_spread, 3),
                "members": x_moved,
            }
            if x_moved else None
        ),
        "row_alignment": row_adjustments,
        "column_stack_alignment": column_stack_adjustments,
    }


def _high_confidence_pair(
    texts: list[dict[str, Any]],
    members: list[int],
    edge_reasons: dict[tuple[int, int], list[str]],
) -> bool:
    if len(members) != 2:
        return False
    i, j = members
    a = texts[i]
    b = texts[j]
    if a["align"] != b["align"]:
        return False
    if a["colour_family"] != b["colour_family"]:
        return False
    if bool(a["el"].get("bold")) != bool(b["el"].get("bold")):
        return False
    size_a = float(a["el"].get("size") or a["el"].get("font_size") or 0)
    size_b = float(b["el"].get("size") or b["el"].get("font_size") or 0)
    if (
        size_a <= 0
        or size_b <= 0
        or abs(size_a - size_b) > MAX_STYLE_SIZE_DELTA
    ):
        return False
    if a.get("direct_parent") is None or a.get("direct_parent") != b.get("direct_parent"):
        return False
    reasons = edge_reasons.get((min(i, j), max(i, j)), [])
    return "column" in reasons and "same_parent" in reasons


def _size_exception_slot(
    members: list[int],
    edge_reasons: dict[tuple[int, int], list[str]],
) -> bool:
    """Allow structural classes whose role matches but sizes should stay."""
    if len(members) < 2:
        return False
    covered: set[int] = set()
    for pos, i in enumerate(members):
        for j in members[pos + 1:]:
            reasons = edge_reasons.get((min(i, j), max(i, j)), [])
            if "compact_latin_title" in reasons:
                covered.update((i, j))
    return len(covered) == len(members)


def _clear_style_annotations(slide: dict[str, Any]) -> None:
    slide["elements"] = [
        el for el in slide.get("elements") or []
        if str(el.get("role") or "") != "inferred_bullet_marker"
    ]
    for el in slide.get("elements") or []:
        if (el.get("type") or "").lower() != "text":
            continue
        for key in (
            "style_class",
            "style_class_suggested_size",
            "style_class_align",
            "style_class_colour_family",
            "style_parent_column_align",
            "style_class_position_x",
            "style_class_position_y",
            "style_table_column_x",
        ):
            el.pop(key, None)
        for run in el.get("runs") or []:
            for key in ("style_class", "style_class_suggested_size"):
                run.pop(key, None)


def classify_slide(slide: dict[str, Any],
                   min_group_size: int,
                   apply: bool = False,
                   min_apply_size: int = 3) -> dict[str, Any]:
    if apply:
        _clear_style_annotations(slide)
    elements = slide.get("elements") or []
    page_area = float(slide.get("source_width") or 1280) * float(
        slide.get("source_height") or 720)

    texts: list[dict[str, Any]] = []
    for idx, el in enumerate(elements):
        if (el.get("type") or "").lower() != "text":
            continue
        if not str(el.get("text") or "").strip() or not el.get("box"):
            continue
        rgb = _parse_rgb(el.get("color"))
        texts.append({
            "id": idx,
            "el": el,
            "box": _box_xyxy(el["box"]),
            "rgb": rgb,
            "colour_family": _colour_family(rgb),
            "align": _align_value(el),
            "ancestors": [],
            "scope_ancestors": [],
            "direct_parent": None,
        })

    containers: dict[int, dict[str, Any]] = {}
    for idx, el in enumerate(elements):
        kind = (el.get("type") or "").lower()
        if kind == "text" or not el.get("box") or len(el["box"]) != 4:
            continue
        name = str(el.get("name") or "")
        if "bullet_marker" in name:
            continue
        box = _box_xyxy(el["box"])
        area = _area_xyxy(box)
        if area < 900 or area > page_area * 0.72:
            continue
        containers[idx] = {
            "id": idx,
            "kind": kind,
            "name": name or f"{kind}_{idx}",
            "box": box,
            "area": area,
            "el": el,
            "child_text_count": 0,
        }

    for text in texts:
        containing: list[int] = []
        for cid, container in containers.items():
            if _contains_text(container["box"], text["box"]):
                containing.append(cid)
        containing.sort(key=lambda cid: containers[cid]["area"])
        text["ancestors"] = containing
        if containing:
            text["direct_parent"] = containing[0]
        for cid in containing:
            containers[cid]["child_text_count"] += 1

    for text in texts:
        text["scope_ancestors"] = [
            cid for cid in text["ancestors"]
            if containers[cid]["child_text_count"] >= 3
        ]

    parent_column_alignment = (
        _apply_parent_column_alignment(texts) if apply else []
    )

    parent = list(range(len(texts)))
    edge_reasons: dict[tuple[int, int], list[str]] = {}

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i: int, j: int) -> None:
        ri = find(i)
        rj = find(j)
        if ri != rj:
            parent[rj] = ri

    for i, a in enumerate(texts):
        for j in range(i + 1, len(texts)):
            b = texts[j]
            style_ok = _style_compatible(a, b)
            compact_latin_title = False
            if not style_ok:
                compact_latin_title = _compact_latin_title_compatible(a, b)
            if not style_ok and not compact_latin_title:
                continue
            scope = _common_scope(a, b, containers)
            rel_slot = _same_relative_slot(a, b, containers)
            rel_alert = _same_relative_alert_band(a, b, containers)
            same_parent = (
                a.get("direct_parent") is not None
                and a.get("direct_parent") == b.get("direct_parent")
            )
            vertical_list = _same_vertical_list_slot(
                a, b, containers, scope, same_parent)
            if (
                scope is None
                and not rel_slot
                and not rel_alert
                and not vertical_list
                and not same_parent
            ):
                continue
            reasons: list[str] = []
            if _row_aligned(a, b):
                reasons.append("row")
            if _column_aligned(a, b):
                reasons.append("column")
            if rel_slot:
                reasons.append("relative_slot")
            if rel_alert:
                reasons.append("relative_alert_band")
            if vertical_list:
                reasons.append("vertical_list_slot")
            if compact_latin_title:
                reasons.append("compact_latin_title")
            if not reasons:
                continue
            if scope is not None:
                reasons.append(f"scope:{containers[scope]['name']}")
            if same_parent:
                reasons.append("same_parent")
            edge_reasons[(i, j)] = reasons
            union(i, j)

    groups: dict[int, list[int]] = {}
    for i in range(len(texts)):
        groups.setdefault(find(i), []).append(i)

    classes: list[dict[str, Any]] = []
    for members in groups.values():
        if len(members) < min_group_size:
            continue
        sizes = [
            float(texts[i]["el"].get("size")
                  or texts[i]["el"].get("font_size") or 0)
            for i in members
        ]
        class_size_delta = max(sizes) - min(sizes)
        size_unification_eligible = class_size_delta <= MAX_STYLE_SIZE_DELTA
        size_exception = _size_exception_slot(members, edge_reasons)
        if not size_unification_eligible and not size_exception:
            continue
        class_id = f"slot_{len(classes) + 1:02d}"
        align_values = [texts[i]["align"] for i in members]
        align = sorted(set(align_values), key=lambda v: (-align_values.count(v), v))[0]
        has_explicit_mixed_runs = any(
            _explicit_mixed_run_sizes(texts[i]["el"])
            for i in members
        )
        class_bolds = [bool(texts[i]["el"].get("bold")) for i in members]
        dominant_bold = sorted(
            set(class_bolds),
            key=lambda value: (-class_bolds.count(value), value),
        )[0]
        bold_stable_or_outlier = (
            class_bolds.count(dominant_bold) >= max(3, len(members) - 1)
        )
        force_apply = _high_confidence_pair(texts, members, edge_reasons)
        suggested_size = (
            _suggested_class_size(
                sizes, force_apply, [texts[i]["el"] for i in members])
            if size_unification_eligible else None
        )
        applied_size = False
        applied_bold = False
        if apply:
            for i in members:
                el = texts[i]["el"]
                el["style_class"] = class_id
                if suggested_size is not None:
                    el["style_class_suggested_size"] = suggested_size
                el["style_class_align"] = align
                el["style_class_colour_family"] = texts[i]["colour_family"]
                el["align"] = align
            # Text-level size priors are only safe for classes without
            # explicit mixed run sizes. Mixed title+annotation text is
            # handled by run_slot_classes below.
            deltas = (
                [abs(float(size) - suggested_size) for size in sizes]
                if suggested_size is not None else []
            )
            max_delta = max(deltas) if deltas else class_size_delta
            stable_or_outlier = (
                size_unification_eligible
                and max_delta <= MAX_STYLE_SIZE_DELTA
            )
            if (
                (force_apply or len(members) >= min_apply_size)
                and suggested_size is not None
                and suggested_size > 0
                and stable_or_outlier
                and not has_explicit_mixed_runs
            ):
                for i in members:
                    texts[i]["el"]["size"] = suggested_size
                applied_size = True
            if (
                (force_apply or len(members) >= min_apply_size)
                and bold_stable_or_outlier
            ):
                for i in members:
                    texts[i]["el"]["bold"] = dominant_bold
                applied_bold = True
        run_classes = _run_slot_classes(
            texts, members, class_id, apply, min_apply_size, force_apply)
        class_position_alignment = (
            _apply_class_position_priors(texts, members, class_id, force_apply)
            if apply else None
        )
        bullet_alignment = (
            _apply_bullet_marker_priors(slide, texts, members, class_id)
            if apply else None
        )
        member_edges: list[list[Any]] = []
        for x, i in enumerate(members):
            for j in members[x + 1:]:
                key = (min(i, j), max(i, j))
                if key in edge_reasons:
                    member_edges.append([
                        texts[i]["el"].get("name"),
                        texts[j]["el"].get("name"),
                        edge_reasons[key],
                    ])
        classes.append({
            "class_id": class_id,
            "count": len(members),
            "suggested_size": suggested_size,
            "sizes": [int(round(v)) for v in sizes],
            "bold": bool(texts[members[0]]["el"].get("bold")),
            "align": align,
            "colour_family": texts[members[0]]["colour_family"],
            "size_delta": round(class_size_delta, 3),
            "size_unification_eligible": size_unification_eligible,
            "size_exception": size_exception,
            "applied_text_size": applied_size,
            "applied_text_bold": applied_bold,
            "force_applied": force_apply,
            "bold_values": sorted(set(class_bolds)),
            "edge_evidence": member_edges[:24],
            "run_classes": run_classes,
            "class_position_alignment": class_position_alignment,
            "bullet_alignment": bullet_alignment,
            "members": [
                {
                    "name": texts[i]["el"].get("name"),
                    "text": texts[i]["el"].get("text"),
                    "size": texts[i]["el"].get("size"),
                    "bold": bool(texts[i]["el"].get("bold")),
                    "color": texts[i]["el"].get("color"),
                    "box": texts[i]["el"].get("box"),
                    "direct_parent": (
                        containers[texts[i]["direct_parent"]]["name"]
                        if texts[i]["direct_parent"] is not None else None
                    ),
                    "ancestor_chain": [
                        {
                            "name": containers[cid]["name"],
                            "kind": containers[cid]["kind"],
                            "box": [
                                round(containers[cid]["box"][0], 3),
                                round(containers[cid]["box"][1], 3),
                                round(containers[cid]["box"][2] - containers[cid]["box"][0], 3),
                                round(containers[cid]["box"][3] - containers[cid]["box"][1], 3),
                            ],
                            "child_text_count": containers[cid]["child_text_count"],
                        }
                        for cid in texts[i]["ancestors"][:6]
                    ],
                }
                for i in members
            ],
        })

    table_column_alignment = (
        _apply_table_column_geometry_alignment(texts, containers)
        if apply else []
    )

    classes.sort(key=lambda c: (
        min(float(m["box"][1]) for m in c["members"] if m.get("box")),
        min(float(m["box"][0]) for m in c["members"] if m.get("box")),
    ))

    container_groups = []
    for group in _group_containers(containers):
        container_groups.append([
            {
                "name": containers[cid]["name"],
                "kind": containers[cid]["kind"],
                "box": [
                    round(containers[cid]["box"][0], 3),
                    round(containers[cid]["box"][1], 3),
                    round(containers[cid]["box"][2] - containers[cid]["box"][0], 3),
                    round(containers[cid]["box"][3] - containers[cid]["box"][1], 3),
                ],
                "child_text_count": containers[cid]["child_text_count"],
            }
            for cid in group
        ])

    return {
        "text_count": len(texts),
        "container_count": len(containers),
        "container_groups": container_groups,
        "parent_column_alignment": parent_column_alignment,
        "table_column_alignment": table_column_alignment,
        "classes": classes,
        "unclassified_texts": [
            {
                "name": t["el"].get("name"),
                "text": t["el"].get("text"),
                "size": t["el"].get("size"),
                "color": t["el"].get("color"),
                "box": t["el"].get("box"),
                "direct_parent": (
                    containers[t["direct_parent"]]["name"]
                    if t["direct_parent"] is not None else None
                ),
            }
            for idx, t in enumerate(texts)
            if all(idx not in members for members in groups.values()
                   if len(members) >= min_group_size)
        ],
    }


def main() -> int:
    args = parse_args()
    layout_path = Path(args.layout)
    layout = json.loads(layout_path.read_text(encoding="utf-8-sig"))
    slides = layout.get("slides") or [layout]
    report = {
        "layout": str(layout_path),
        "slides": [
            classify_slide(
                slide,
                int(args.min_group_size),
                apply=bool(args.apply),
                min_apply_size=int(args.min_apply_size),
            )
            for slide in slides
        ],
    }
    if args.apply:
        out_layout = Path(args.out_layout) if args.out_layout else layout_path
        out_layout.write_text(
            json.dumps(layout, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, ensure_ascii=False, indent=2),
                       encoding="utf-8")
    for s_idx, slide_report in enumerate(report["slides"], start=1):
        print(f"slide {s_idx}: {slide_report['text_count']} texts, "
              f"{slide_report['container_count']} containers, "
              f"{len(slide_report['classes'])} classes")
        for cls in slide_report["classes"]:
            names = ", ".join(m["name"] for m in cls["members"])
            texts = " | ".join(str(m["text"]) for m in cls["members"])
            print(f"  {cls['class_id']} n={cls['count']} "
                  f"size={cls['suggested_size']} "
                  f"{cls['colour_family']} bold={cls['bold']} "
                  f"align={cls['align']}: {names}")
            print(f"    {texts}")
            for run_cls in cls.get("run_classes") or []:
                run_texts = " | ".join(
                    str(m["text"]) for m in run_cls["members"])
                print(f"    run.{run_cls['role']} "
                      f"n={run_cls['count']} "
                      f"size={run_cls['suggested_size']} "
                      f"applied={run_cls['applied']}: {run_texts}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
