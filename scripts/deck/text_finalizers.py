"""Final text-style passes applied right before the PPTX is rendered.

These three passes used to live inline at the top of
build_pptx_from_layout. They run after classify_text_slots and the two
calibration steps — they take the final layout as it stands and apply
last-mile corrections:

* `apply_size_unification` — snap same-row / same-style texts to a
  shared font size (different rule set from classify_text_slots').
* `apply_class_alignment` — restore class-derived alignment that may
  have been overwritten by calibration.
* `apply_title_centering` — geometric centring of short text inside
  the smallest containing image.

Separated from `build_pptx_from_layout` so the builder can stay a pure
renderer; the rendering path no longer mutates the layout dict.
"""
from __future__ import annotations

from typing import Any


def apply_size_unification(elements: list[dict[str, Any]],
                           max_ratio: float = 1.6) -> None:
    """Snap visually-related text elements to a common font size.

    Per-bbox font_size_pt takes min(height, width) — robust against the
    α miscalibration that overshoots stat bboxes, but still wobbles when
    OCR returns slightly different per-glyph widths for visually
    identical stats (e.g. `524个` = 48 px wide → 12 pt vs `63个` = 54 px
    wide → 18 pt). This pass cleans that up at PPTX-build time using
    final layout geometry.

    Two texts cluster when ALL of:
      - same colour (per-channel ±25, anti-alias tolerant)
      - same bold flag
      - same LENGTH CATEGORY: short ≤4 / medium 5-12 / long >12.
      - SAME ROW: y-overlap ≥ 50 % of the shorter box.
      - SIZE-ratio: max(a.size, b.size) / min(...) ≤ max_ratio

    Edges compose transitively, but a merge is rejected when it would
    push the cluster's overall ratio past max_ratio. Each cluster is
    snapped to its MIN size.
    """
    texts = [el for el in elements if (el.get("type") or "").lower() == "text"
             and el.get("size") is not None
             and el.get("box") and len(el["box"]) == 4]
    texts = [
        el for el in texts
        if el.get("size_source")
        not in {"render_fit", "mixed_runs", "preview_calibrated",
                "style_class"}
    ]
    if len(texts) < 2:
        return

    def colour_close(a: str, b: str, tol: int = 25) -> bool:
        a, b = (a or "").lstrip("#"), (b or "").lstrip("#")
        if len(a) != 6 or len(b) != 6:
            return a == b
        try:
            return all(abs(int(a[i:i+2], 16) - int(b[i:i+2], 16)) <= tol
                       for i in (0, 2, 4))
        except ValueError:
            return False

    def same_row(a: dict, b: dict) -> bool:
        ax, ay, aw, ah = a["box"]
        bx, by, bw, bh = b["box"]
        y_overlap = max(0, min(ay + ah, by + bh) - max(ay, by))
        return y_overlap >= 0.5 * min(ah, bh)

    def length_bucket(text: str) -> str:
        n = len(text or "")
        if n <= 4:
            return "short"
        if n <= 12:
            return "medium"
        return "long"

    # Union-Find with cluster size-ratio cap (max/min ≤ max_ratio).
    parent = list(range(len(texts)))
    sizes = [int(t["size"]) for t in texts]
    root_min = list(sizes)
    root_max = list(sizes)

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union_if_within_ratio(i: int, j: int) -> None:
        ri, rj = find(i), find(j)
        if ri == rj:
            return
        new_min = min(root_min[ri], root_min[rj])
        new_max = max(root_max[ri], root_max[rj])
        if new_min <= 0 or new_max / new_min > max_ratio:
            return
        parent[ri] = rj
        root_min[rj] = new_min
        root_max[rj] = new_max

    buckets = [length_bucket(t.get("text", "")) for t in texts]
    for i in range(len(texts)):
        a = texts[i]
        for j in range(i + 1, len(texts)):
            b = texts[j]
            if bool(a.get("bold")) != bool(b.get("bold")):
                continue
            if buckets[i] != buckets[j]:
                continue
            if not colour_close(a.get("color"), b.get("color")):
                continue
            ratio = max(sizes[i], sizes[j]) / max(1, min(sizes[i], sizes[j]))
            if ratio > max_ratio:
                continue
            if not same_row(a, b):
                continue
            union_if_within_ratio(i, j)

    clusters: dict[int, list[int]] = {}
    for i in range(len(texts)):
        clusters.setdefault(find(i), []).append(i)

    for members in clusters.values():
        if len(members) < 2:
            continue
        unified = min(sizes[i] for i in members)
        for i in members:
            texts[i]["size"] = unified


def apply_class_alignment(elements: list[dict[str, Any]]) -> None:
    """Make structural classification the final paragraph alignment source.

    classify_text_slots writes `style_class_align` alongside `align`. The
    calibration passes that run between classify and PPTX build can
    overwrite `align`. This restores the class-derived value as the last
    word — matching the layered priority documented in classify_text_slots.
    """
    for el in elements:
        if (el.get("type") or "").lower() != "text":
            continue
        class_align = el.get("style_class_align") or el.get(
            "style_parent_column_align")
        if class_align:
            el["align"] = str(class_align).lower()


def apply_title_centering(elements: list[dict[str, Any]],
                          tol_frac: float = 0.05,
                          tol_max_px: int = 25,
                          max_title_chars: int = 8) -> None:
    """Re-anchor text boxes whose centre lines up with a containing image.

    Runs once per slide on the final layout right before shapes are
    emitted. For each text element, applies three filters in order:

      1. LENGTH — text must be ≤ max_title_chars (default 8). Long
         strings (paragraphs, quotes) are not titles.
      2. SMALLEST containing image — the text's "wrapper" is whichever
         image element strictly contains the box and has the smallest
         area. Centre-test runs only against that single parent.
      3. CENTRE TOLERANCE — `min(tol_frac × parent_width, tol_max_px)`.
         The 25-px absolute cap stops a 1100-px-wide bottom-strip
         background from grabbing every midline-ish text as its title.

    When all four pass, x is re-anchored to the parent x-centre and the
    paragraph alignment is set to centre. y is left unchanged; vertical
    placement belongs to OCR/layout calibration.
    """
    images: list[tuple[float, float, float, float, float]] = []
    for el in elements:
        if (el.get("type") or "").lower() != "image":
            continue
        box = el.get("box")
        if not box or len(box) != 4:
            continue
        x, y, w, h = (float(v) for v in box)
        if w <= 0 or h <= 0:
            continue
        images.append((x, y, x + w, y + h, w * h))
    if not images:
        return

    for el in elements:
        if (el.get("type") or "").lower() != "text":
            continue
        # Position calibration measured rendered ink against the source,
        # and class snapping is the pipeline's final authority on
        # positions. Don't re-center over either of them.
        if el.get("position_source") in {"preview_calibrated",
                                         "style_class"}:
            continue
        if len(str(el.get("text", "") or "")) > max_title_chars:
            continue
        box = el.get("box")
        if not box or len(box) != 4:
            continue
        bx, by, bw, bh = (float(v) for v in box)
        tx1, ty1, tx2, ty2 = bx, by, bx + bw, by + bh
        tcx = (tx1 + tx2) / 2.0
        # Pick the absolutely smallest image that strictly contains the
        # text — that's the visual "wrapper". Centre-test only against
        # that one parent.
        best: tuple[float, float, float, float] | None = None
        best_area: float | None = None
        for px1, py1, px2, py2, parea in images:
            if not (px1 < tx1 and py1 < ty1 and px2 > tx2 and py2 > ty2):
                continue
            if best_area is None or parea < best_area:
                best = (px1, py1, px2, py2)
                best_area = parea
        if best is None:
            continue
        px1, py1, px2, py2 = best
        tol = min(tol_frac * (px2 - px1), float(tol_max_px))
        if abs(tcx - (px1 + px2) / 2.0) > tol:
            continue
        new_x = (px1 + px2) / 2.0 - bw / 2.0
        el["box"] = [int(round(new_x)), int(round(by)),
                     int(round(bw)), int(round(bh))]
        el["align"] = "center"


def apply_all(elements: list[dict[str, Any]]) -> None:
    """Run the three text-finalizer passes in the canonical order."""
    apply_title_centering(elements)
    apply_size_unification(elements)
    apply_class_alignment(elements)
