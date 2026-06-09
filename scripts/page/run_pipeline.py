#!/usr/bin/env python
"""Batch pipeline runner: slide image → editable PPT layout (cv2 path).

This is the orchestrator that walks every `page_NN.ocr.json` in a work
directory and produces, per page, a cleaned image, an inventory JSON, a
crop manifest, a layout JSON, and the extracted asset PNGs.

The three core stages are imported and called in-process to avoid
per-page Python startup. Each script also exposes a CLI for hand-runs:

    erase_text.py       remove OCR text from the slide
    build_inventory.py  cv2 connected-component detection → inventory
    inventory_to_layout cv2-cropped asset PNGs + editable layout JSON

Optional pre-pass: `--detect-tables` runs the OCR-grid table detector
and verifies candidates with SLANet_plus. Accepted regions become
native PPT table elements; their cells are dropped from the editable
text stream so they don't double-render.

Optional icon-review packet: `--icon-review` emits per-page crops of
every icon-vs-text gray-zone call so an agent can second-guess them;
re-run with `--icon-decisions` to feed verdicts back into erase_text.

Usage:
    python scripts/page/run_pipeline.py \\
        --source-dir <slides_image_dir> \\
        --work-dir output/<name>_<YYYYMMDD>/

The work dir is expected to already contain
`<work>/ocr/page_NN.ocr.json` files (produced by prepare_ocr.py +
ocr_review_apply.py). The runner writes:

    inventory/page_NN.clean.png
    inventory/page_NN.clean.text_only.png
    inventory/page_NN.inventory.json
    inventory/masks/page_NN/v###.mask.png
    manifests/page_NN.assets.json
    layouts/page_NN.layout.json
    assets/page_NN/*.png
    debug/page_NN_*.png

Note: a FastSAM instance-segmentation path used to live alongside the
cv2 path here. It was removed — the cv2 + subicon + internal-shape
stack in build_inventory.py is more predictable on slide-style layouts
and is now the only path.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

# Resolve sibling subdir paths so `import erase_text` (in pipeline/) and
# `import detect_tables` (in ../tables/) both work whatever cwd we're
# launched from.
SCRIPTS = Path(__file__).resolve().parent           # .../scripts/pipeline
SCRIPTS_ROOT = SCRIPTS.parent                       # .../scripts
sys.path.insert(0, str(SCRIPTS))
sys.path.insert(0, str(SCRIPTS_ROOT))
sys.path.insert(0, str(SCRIPTS_ROOT / "tables"))

import erase_text as et
import build_inventory as bi
import inventory_to_layout as i2l
import _heuristics as heur
import detect_tables as dt
from shared.geometry import intersection_area as _box_intersection
from image_sources import find_page_image, supported_image_formats


# =============================================================================
# CLI
# =============================================================================


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Batch pipeline runner")
    p.add_argument("--source-dir", required=True,
                   help=f"Directory of page_NN images "
                        f"({supported_image_formats()}).")
    p.add_argument("--work-dir", required=True,
                   help="Work dir with ocr/page_NN.ocr.json.")
    p.add_argument("--pages",
                   help="Comma-separated page numbers (e.g. 1,4,8). "
                        "Default: all page_NN.ocr.json files found.")
    p.add_argument("--detect-tables", dest="detect_tables",
                   action="store_true",
                   help="Run the heuristic OCR-grid table detector "
                        "(detect_tables.py). When a candidate region "
                        "passes SLANet_plus structure verification "
                        "(score >= 0.85) it's emitted as a native PPT "
                        "table element and its cells are removed from "
                        "the editable-text stream so they don't render "
                        "twice. Off by default — most decks don't "
                        "contain literal grid tables.")
    p.add_argument("--no-detect-tables", dest="detect_tables",
                   action="store_false",
                   help="Disable automatic table detection.")
    # Default OFF: SLANet hits high structure scores on styled banner
    # regions (dark card + decorative columns) and consumes their OCR
    # text into a fake 2xN grid that renders mostly empty. Pass
    # `--detect-tables` only when the deck has literal grid tables.
    p.set_defaults(detect_tables=False)
    p.add_argument("--table-score-threshold", type=float, default=0.85,
                   help="Minimum SLANet_plus structure confidence to "
                        "accept a detected region as a table.")
    p.add_argument("--icon-review", action="store_true",
                   help="Emit an LLM icon-review packet per page: "
                        "<work>/ocr/icon_review/page_NN/"
                        "icon_review.json + crops + contact.png. The "
                        "packet lists every icon-vs-text decision the "
                        "heuristic made in a gray zone so an agent can "
                        "second-guess each call. Re-run with "
                        "--icon-decisions to apply the agent's verdicts.")
    p.add_argument("--icon-decisions", action="store_true",
                   help="After the agent has filled `decision` fields in "
                        "<work>/ocr/icon_review/page_NN/"
                        "icon_review.json, pass this flag on a re-run to "
                        "make erase_text consume those overrides "
                        "instead of pure heuristics. Implies the same "
                        "packet location as --icon-review.")
    return p.parse_args()


def page_numbers(args) -> list[str]:
    if args.pages:
        return [n.strip().zfill(2) for n in args.pages.split(",")]
    inv_dir = Path(args.work_dir) / "ocr"
    nums = []
    for p in sorted(inv_dir.glob("page_*.ocr.json")):
        nums.append(p.stem.split("_")[1].split(".")[0])
    return nums


# =============================================================================
# Geometry helpers
# =============================================================================


def _coeff_var(values: list[float]) -> float:
    if not values:
        return float("inf")
    mean = sum(values) / len(values)
    if mean <= 0:
        return float("inf")
    return (sum((v - mean) ** 2 for v in values) / len(values)) ** 0.5 / mean


# =============================================================================
# Ruled-line detection (used by image-based table candidates + prefilter)
# =============================================================================


def _group_line_segments(mask: np.ndarray,
                         orientation: str,
                         *,
                         min_len: int,
                         merge_gap: int = 12) -> list[tuple[int, int, int, int]]:
    """Extract long horizontal/vertical line segments from a binary mask."""
    n, _, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    raw: list[tuple[int, int, int, int]] = []
    for i in range(1, n):
        x, y, w_, h_, area = (int(v) for v in stats[i])
        if area < 20:
            continue
        if orientation == "h":
            if w_ >= min_len and h_ <= 10:
                raw.append((x, y, x + w_, y + h_))
        else:
            if h_ >= min_len and w_ <= 10:
                raw.append((x, y, x + w_, y + h_))

    groups: list[dict] = []
    def coord(seg: tuple[int, int, int, int]) -> float:
        return ((seg[1] + seg[3]) / 2.0
                if orientation == "h" else (seg[0] + seg[2]) / 2.0)

    for seg in sorted(raw, key=lambda b: (coord(b), b[0], b[1])):
        c = coord(seg)
        for g in groups:
            if abs(c - g["c"]) <= 3:
                g["segs"].append(seg)
                g["c"] = (g["c"] * (len(g["segs"]) - 1) + c) / len(g["segs"])
                break
        else:
            groups.append({"c": c, "segs": [seg]})

    out: list[tuple[int, int, int, int]] = []
    for g in groups:
        if orientation == "h":
            spans = sorted((s[0], s[2], s[1], s[3]) for s in g["segs"])
        else:
            spans = sorted((s[1], s[3], s[0], s[2]) for s in g["segs"])
        cur: list[int] | None = None
        for a, b, c1, c2 in spans:
            if cur is None:
                cur = [a, b, c1, c2]
            elif a <= cur[1] + merge_gap:
                cur[1] = max(cur[1], b)
                cur[2] = min(cur[2], c1)
                cur[3] = max(cur[3], c2)
            else:
                if cur[1] - cur[0] >= min_len:
                    out.append(
                        (cur[0], cur[2], cur[1], cur[3])
                        if orientation == "h"
                        else (cur[2], cur[0], cur[3], cur[1])
                    )
                cur = [a, b, c1, c2]
        if cur is not None and cur[1] - cur[0] >= min_len:
            out.append(
                (cur[0], cur[2], cur[1], cur[3])
                if orientation == "h"
                else (cur[2], cur[0], cur[3], cur[1])
            )
    return out


def _ruled_lines(img: np.ndarray) -> tuple[list[tuple[int, int, int, int]],
                                           list[tuple[int, int, int, int]]]:
    """Detect long ruling lines using both Canny and light-line thresholding."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    diff_white = np.abs(img.astype(int) - 255).max(axis=2)
    masks = [
        cv2.Canny(gray, 50, 150),
        (diff_white > 6).astype(np.uint8) * 255,
    ]
    h_lines: list[tuple[int, int, int, int]] = []
    v_lines: list[tuple[int, int, int, int]] = []
    for base in masks:
        h_mask = cv2.morphologyEx(
            base, cv2.MORPH_OPEN,
            cv2.getStructuringElement(cv2.MORPH_RECT, (45, 1)))
        v_mask = cv2.morphologyEx(
            base, cv2.MORPH_OPEN,
            cv2.getStructuringElement(cv2.MORPH_RECT, (1, 30)))
        h_lines.extend(_group_line_segments(h_mask, "h", min_len=120))
        v_lines.extend(_group_line_segments(v_mask, "v", min_len=70))

    # Re-group the union so Canny and threshold detections don't double-count.
    H, W = img.shape[:2]
    h_canvas = np.zeros((H, W), dtype=np.uint8)
    v_canvas = np.zeros((H, W), dtype=np.uint8)
    for x1, y1, x2, y2 in h_lines:
        cv2.rectangle(h_canvas, (x1, y1), (x2 - 1, y2 - 1), 255, -1)
    for x1, y1, x2, y2 in v_lines:
        cv2.rectangle(v_canvas, (x1, y1), (x2 - 1, y2 - 1), 255, -1)
    return (
        _group_line_segments(h_canvas, "h", min_len=120, merge_gap=14),
        _group_line_segments(v_canvas, "v", min_len=70, merge_gap=14),
    )


# =============================================================================
# Table detection (OCR-grid + image-line candidates, SLANet verification)
# =============================================================================


def _dedupe_candidates(candidates: list[dict]) -> list[dict]:
    out: list[dict] = []
    for cand in sorted(candidates,
                       key=lambda c: ((c["bbox"][2] - c["bbox"][0])
                                      * (c["bbox"][3] - c["bbox"][1])),
                       reverse=True):
        box = tuple(int(v) for v in cand["bbox"])
        area = max(1, (box[2] - box[0]) * (box[3] - box[1]))
        duplicate = False
        for kept in out:
            kbox = tuple(int(v) for v in kept["bbox"])
            karea = max(1, (kbox[2] - kbox[0]) * (kbox[3] - kbox[1]))
            if _box_intersection(box, kbox) >= 0.75 * min(area, karea):
                duplicate = True
                break
        if not duplicate:
            out.append(cand)
    return out


def _table_grid_metrics(raw_cells: list[dict],
                        bbox: tuple[int, int, int, int],
                        rows: int,
                        cols: int) -> tuple[list[int], list[int]]:
    """Estimate native-table column widths/row heights from SLANet cells."""
    bx1, by1, bx2, by2 = bbox
    xs: list[list[int]] = [[] for _ in range(cols + 1)]
    ys: list[list[int]] = [[] for _ in range(rows + 1)]
    xs[0].append(bx1); xs[-1].append(bx2)
    ys[0].append(by1); ys[-1].append(by2)
    for c in raw_cells:
        if not isinstance(c, dict) or "bbox" not in c:
            continue
        try:
            r = int(c.get("row", 0)); col = int(c.get("col", 0))
            rs = max(1, int(c.get("rowspan", 1)))
            cs = max(1, int(c.get("colspan", 1)))
            x1, y1, x2, y2 = (int(v) for v in c["bbox"])
        except (TypeError, ValueError):
            continue
        if cs == 1 and 0 <= col < cols:
            xs[col].append(x1)
            xs[col + 1].append(x2)
        if rs == 1 and 0 <= r < rows:
            ys[r].append(y1)
            ys[r + 1].append(y2)

    def boundaries(samples: list[list[int]], lo: int, hi: int) -> list[int]:
        out: list[int | None] = []
        for vals in samples:
            if vals:
                vals_sorted = sorted(vals)
                out.append(vals_sorted[len(vals_sorted) // 2])
            else:
                out.append(None)
        n = len(samples) - 1
        for i, v in enumerate(out):
            if v is None:
                out[i] = int(round(lo + (hi - lo) * i / max(1, n)))
        fixed = [int(v) for v in out]
        fixed[0] = lo
        fixed[-1] = hi
        for i in range(1, len(fixed)):
            if fixed[i] <= fixed[i - 1] + 4:
                fixed[i] = fixed[i - 1] + 5
        if fixed[-1] != hi:
            fixed[-1] = hi
        return fixed

    xb = boundaries(xs, bx1, bx2)
    yb = boundaries(ys, by1, by2)
    return (
        [max(5, xb[i + 1] - xb[i]) for i in range(cols)],
        [max(5, yb[i + 1] - yb[i]) for i in range(rows)],
    )


def _style_table_cells(cells: list[dict], rows: int, cols: int) -> list[dict]:
    styled: list[dict] = []
    for c in cells:
        out = dict(c)
        r = int(out.get("row", 0))
        col = int(out.get("col", 0))
        text = str(out.get("text", "") or "")
        is_header = r == 0
        is_total = r == rows - 1
        is_numeric = col > 0 and any(ch.isdigit() for ch in text)
        out.setdefault("font", "Microsoft YaHei")
        out["valign"] = "middle"
        out["margin"] = 1.0
        if is_header or (is_total and col == 0):
            out["fill"] = "#08498D"
            out["color"] = "#FFFFFF"
            out["bold"] = True
            out["align"] = "center"
            out["size"] = 8
        elif is_total:
            out["fill"] = "#EEF4FB"
            out["bold"] = True
            out["align"] = "center"
            out["size"] = 18 if is_numeric else 9
            out["color"] = "#C01818" if col in {1, 2, 4} else "#054798"
        else:
            out["fill"] = "#F7FAFE" if r % 2 else "#EEF3FA"
            out["align"] = "center" if col > 0 else "left"
            if is_numeric:
                out["bold"] = True
                out["size"] = 16
                if text in {"5", "3", "1"} and col in {1, 4}:
                    out["color"] = "#D43E3E"
                elif text == "0":
                    out["color"] = "#777777"
                else:
                    out["color"] = "#054798"
            else:
                out["size"] = 8
                out["color"] = "#666666" if col == 0 else "#222222"
        styled.append(out)
    return styled


def _image_table_candidates(src_path: Path) -> list[dict]:
    """Find ruled tables from pixels when OCR-grid clustering misses them."""
    img = cv2.imread(str(src_path))
    if img is None:
        return []
    H, W = img.shape[:2]
    h_lines, v_lines = _ruled_lines(img)
    h_lines = sorted(h_lines, key=lambda l: ((l[1] + l[3]) / 2.0, l[0]))
    candidates: list[dict] = []
    for i, first in enumerate(h_lines):
        run = [first]
        for cur in h_lines[i + 1:]:
            prev = run[-1]
            prev_y = (prev[1] + prev[3]) / 2.0
            cur_y = (cur[1] + cur[3]) / 2.0
            if cur_y - prev_y > 60:
                break
            overlap = max(0, min(prev[2], cur[2]) - max(prev[0], cur[0]))
            required = 0.50 * min(prev[2] - prev[0], cur[2] - cur[0])
            if overlap < required:
                # Another card/table segment on the same row. Ignore it
                # and keep scanning for the next row of this table.
                continue
            run.append(cur)
        if len(run) < 4:
            continue
        ys = [(l[1] + l[3]) / 2.0 for l in run]
        gaps = [ys[j + 1] - ys[j] for j in range(len(ys) - 1)]
        if _coeff_var(gaps) > 0.55:
            continue
        x1 = min(l[0] for l in run)
        x2 = max(l[2] for l in run)
        y1 = min(l[1] for l in run)
        y2 = max(l[3] for l in run)
        if x2 - x1 < 250 or y2 - y1 < 90:
            continue
        v_in = []
        for vl in v_lines:
            vx = (vl[0] + vl[2]) / 2.0
            overlap_y = max(0, min(y2, vl[3]) - max(y1, vl[1]))
            if x1 - 8 <= vx <= x2 + 8 and overlap_y >= 0.30 * (y2 - y1):
                v_in.append(vl)
        # Pixel-only candidates are conservative: require a real grid, not
        # just a two-column callout box with one divider.
        if len(v_in) < 4:
            continue
        bx1 = max(0, x1 - 2)
        by1 = max(0, y1 - 2)
        bx2 = min(W, x2 + 2)
        by2 = min(H, y2 + 3)
        candidates.append({
            "bbox": [int(bx1), int(by1), int(bx2), int(by2)],
            "rows": max(2, len(run) - 1),
            "cols": max(2, len(v_in) - 1),
            "cells_used": max(1, (len(run) - 1) * (len(v_in) - 1)),
            "source": "image_lines",
        })
    return _dedupe_candidates(candidates)


def _looks_like_ruled_table(src_path: Path,
                            bbox: tuple[int, int, int, int],
                            rows: int,
                            cols: int) -> bool:
    """Conservative image prefilter for table candidates.

    OCR alignment alone misreads stat cards and comparison callouts as
    tables. Before invoking SLANet, require visible ruling lines that span
    most of the candidate region. This keeps wired tables while rejecting
    slide layouts that are merely grid-like.
    """
    img = cv2.imread(str(src_path))
    if img is None:
        return False
    H, W = img.shape[:2]
    x1, y1, x2, y2 = (int(v) for v in bbox)
    x1 = max(0, min(W, x1)); x2 = max(0, min(W, x2))
    y1 = max(0, min(H, y1)); y2 = max(0, min(H, y2))
    if x2 - x1 < 80 or y2 - y1 < 50:
        return False
    crop = img[y1:y2, x1:x2]
    ch, cw = crop.shape[:2]
    h_lines, v_lines = _ruled_lines(crop)
    h_count = sum(1 for l in h_lines if l[2] - l[0] >= 0.45 * cw)
    v_count = sum(1 for l in v_lines if l[3] - l[1] >= 0.35 * ch)
    return h_count >= min(2, max(1, rows - 1)) and \
        v_count >= min(2, max(1, cols - 1))


def _detect_tables_for_page(num: str, src_path: Path, ocr: list[dict],
                            ocr_path: Path, work: Path,
                            score_threshold: float) -> tuple[list[dict],
                                                              set[int]]:
    """Run the OCR grid detector + SLANet verification for this page.

    Returns (table_elements, consumed_ocr_indices) where:
      - table_elements are ready-to-merge `type: table` entries for the
        layout JSON,
      - consumed_ocr_indices is the set of OCR-item indices whose center
        falls inside an accepted table cell. Caller drops these from the
        editable-text stream so they don't double-render under the
        native table.

    The detection step is intentionally conservative — see detect_tables
    and the SLANet score gate. If anything fails (paddlex unavailable,
    SLANet errors), we log to stderr and return empty results so the
    rest of the pipeline keeps producing the previous output.
    """
    candidates = dt.detect(
        ocr, min_rows=3, min_cols=2,
        col_tolerance=20, padding=6,
    )
    candidates = _dedupe_candidates(candidates + _image_table_candidates(src_path))
    if not candidates:
        return [], set()

    try:
        import subprocess
    except ImportError:
        return [], set()

    accepted_cell_bboxes: list[tuple[int, int, int, int]] = []
    table_elements: list[dict] = []
    tables_path = work / "ocr" / f"page_{num}.table.json"
    for idx, cand in enumerate(candidates):
        x1, y1, x2, y2 = cand["bbox"]
        if not _looks_like_ruled_table(
            src_path, (x1, y1, x2, y2),
            int(cand.get("rows") or 0),
            int(cand.get("cols") or 0),
        ):
            continue
        out_path = work / "ocr" / f"page_{num}.table_{idx:02d}.json"
        try:
            r = subprocess.run([
                sys.executable,
                str(SCRIPTS_ROOT / "tables" / "table_recognize.py"),
                str(src_path),
                "--bbox", f"{x1},{y1},{x2},{y2}",
                "--ocr", str(ocr_path),
                "--out", str(out_path),
            ], capture_output=True, timeout=60)
        except (subprocess.SubprocessError, OSError) as exc:
            print(f"page {num}: table_recognize failed ({exc})",
                  file=sys.stderr)
            continue
        if r.returncode != 0 or not out_path.exists():
            continue
        try:
            tbl = json.loads(out_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        if not isinstance(tbl, dict):
            continue
        try:
            score = float(tbl.get("score") or 0.0)
        except (TypeError, ValueError):
            continue
        # Reject NaN/inf scores explicitly — float() will happily return
        # them and they pass the threshold comparison silently.
        if score != score or score == float("inf"):
            continue
        if score < score_threshold:
            # Below confidence — leave this region to the regular cv2/OCR
            # path. Delete the per-table debug JSON so we don't leave
            # stale files lying around.
            try:
                out_path.unlink()
            except OSError:
                pass
            continue
        try:
            rows = int(tbl.get("rows") or 0)
            cols = int(tbl.get("cols") or 0)
        except (TypeError, ValueError):
            continue
        if rows < 2 or cols < 2:
            continue
        # Cell-fill sanity check. SLANet can land a high structure score on
        # a styled decorative banner (white-on-blue text inside a wide card
        # with subtle column gutters), then return only 3-4 cells with
        # text in a declared rows*cols grid — most of the grid is empty
        # because there were no real columns. A real table has text in
        # most of its cells. Require non-empty cell count to cover at
        # least half of the declared rows * cols footprint, accounting
        # for colspan/rowspan. Without this gate, decorative banners can
        # collapse into sparse tables that consume OCR text and render
        # mostly empty.
        raw_cells_for_fill = tbl.get("cells") or []
        if isinstance(raw_cells_for_fill, list):
            covered = 0
            for c in raw_cells_for_fill:
                if not isinstance(c, dict):
                    continue
                if not str(c.get("text") or "").strip():
                    continue
                try:
                    rs = max(1, int(c.get("rowspan", 1)))
                    cs = max(1, int(c.get("colspan", 1)))
                except (TypeError, ValueError):
                    rs = cs = 1
                covered += rs * cs
            fill_ratio = covered / float(rows * cols)
            if fill_ratio < 0.5:
                try:
                    out_path.unlink()
                except OSError:
                    pass
                continue
        bbox = tbl.get("bbox") or cand["bbox"]
        if not (isinstance(bbox, (list, tuple)) and len(bbox) == 4):
            continue
        try:
            bx1, by1, bx2, by2 = (int(v) for v in bbox)
        except (TypeError, ValueError):
            continue
        if bx2 <= bx1 or by2 <= by1:
            continue
        cells = []
        raw_cells = tbl.get("cells") or []
        if not isinstance(raw_cells, list):
            continue
        for c in raw_cells:
            if not isinstance(c, dict):
                continue
            try:
                cell = {
                    "row": int(c.get("row", 0)),
                    "col": int(c.get("col", 0)),
                    "rowspan": max(1, int(c.get("rowspan", 1))),
                    "colspan": max(1, int(c.get("colspan", 1))),
                    "text": str(c.get("text", "") or ""),
                }
                cb = c.get("bbox")
                if isinstance(cb, (list, tuple)) and len(cb) == 4:
                    cell["bbox"] = [int(v) for v in cb]
                cells.append(cell)
            except (TypeError, ValueError):
                continue
        if not cells:
            continue
        col_widths, row_heights = _table_grid_metrics(
            raw_cells, (bx1, by1, bx2, by2), rows, cols)
        cells = _style_table_cells(cells, rows, cols)
        table_elements.append({
            "type": "table",
            "name": f"table_{num}_{idx:02d}",
            "box": [bx1, by1, bx2 - bx1, by2 - by1],
            "rows": rows,
            "cols": cols,
            "col_widths": col_widths,
            "row_heights": row_heights,
            "cells": cells,
            "font": "Microsoft YaHei",
            "size": 9,
            "align": "left",
            "valign": "middle",
        })
        # Collect per-cell bboxes for OCR consumption. The original code
        # consumed every OCR item inside the table bbox, which over-
        # consumed text that fell into row gaps / outside-cell whitespace
        # but was still inside the table outline. Now we only consume
        # items whose center sits inside an actual cell.
        for c in raw_cells:
            if not isinstance(c, dict):
                continue
            cb = c.get("bbox")
            if isinstance(cb, (list, tuple)) and len(cb) == 4:
                try:
                    cx1_, cy1_, cx2_, cy2_ = (int(v) for v in cb)
                except (TypeError, ValueError):
                    continue
                if cx2_ > cx1_ and cy2_ > cy1_:
                    accepted_cell_bboxes.append((cx1_, cy1_, cx2_, cy2_))

    # Aggregate accepted tables into the same summary file the agent
    # would have produced via manual invocation — convenient for QA.
    if table_elements:
        tables_path.write_text(
            json.dumps({"tables": [
                {"bbox": [e["box"][0], e["box"][1],
                          e["box"][0] + e["box"][2],
                          e["box"][1] + e["box"][3]],
                 "rows": e["rows"], "cols": e["cols"],
                 "cells": e["cells"]}
                for e in table_elements
            ]}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    # OCR items whose center falls inside an accepted CELL bbox are
    # already represented as table-cell text; drop them from the editable
    # stream to avoid the duplicate-render artefact. Items inside the
    # outer table bbox but in a row-gap / cell-less region are KEPT —
    # otherwise SLANet's slightly-loose table outline silently eats
    # decorative captions that live just below the last row.
    consumed: set[int] = set()
    for i, it in enumerate(ocr):
        cx = (int(it["x1"]) + int(it["x2"])) / 2.0
        cy = (int(it["y1"]) + int(it["y2"])) / 2.0
        for bx1, by1, bx2, by2 in accepted_cell_bboxes:
            if bx1 <= cx <= bx2 and by1 <= cy <= by2:
                consumed.add(i)
                break
    return table_elements, consumed


def _region_background(img: np.ndarray,
                       x1: int, y1: int, x2: int, y2: int) -> np.ndarray:
    """Median background from a ring around a large region."""
    H, W = img.shape[:2]
    pad = 6
    strips = []
    if y1 - pad >= 0:
        strips.append(img[y1 - pad:y1, x1:x2])
    if y2 + pad <= H:
        strips.append(img[y2:y2 + pad, x1:x2])
    if x1 - pad >= 0:
        strips.append(img[y1:y2, x1 - pad:x1])
    if x2 + pad <= W:
        strips.append(img[y1:y2, x2:x2 + pad])
    pixels = [s.reshape(-1, 3) for s in strips if s.size]
    if not pixels:
        return np.array([255, 255, 255], dtype=np.uint8)
    pix = np.concatenate(pixels, axis=0)
    quant = (pix // 16) * 16
    mode_q = np.array(Counter(map(tuple, quant)).most_common(1)[0][0])
    diff = np.abs(pix.astype(int) - mode_q.astype(int)).max(axis=1)
    close = pix[diff <= 28]
    if len(close) >= max(20, int(len(pix) * 0.2)):
        return np.median(close, axis=0).astype(np.uint8)
    return np.median(pix, axis=0).astype(np.uint8)


def _erase_table_regions(clean_path: Path,
                         table_elements: list[dict]) -> None:
    """Blank accepted table regions before generic cv2 visual detection."""
    if not table_elements:
        return
    img = cv2.imread(str(clean_path))
    if img is None:
        return
    H, W = img.shape[:2]
    for tbl in table_elements:
        x, y, w_, h_ = (int(v) for v in tbl["box"])
        x1, y1 = max(0, x), max(0, y)
        x2, y2 = min(W, x + w_), min(H, y + h_)
        if x2 <= x1 or y2 <= y1:
            continue
        bg = _region_background(img, x1, y1, x2, y2)
        img[y1:y2, x1:x2] = bg
    cv2.imwrite(str(clean_path), img)


# =============================================================================
# Per-page pipeline
# =============================================================================


def process_page(num: str, src_dir: Path, work: Path,
                 *, detect_tables_flag: bool = False,
                 table_score_threshold: float = 0.85,
                 icon_review_dump: bool = False,
                 icon_decisions: bool = False) -> dict:
    """Erase → inventory → layout for one page.

    Shells out to erase_text.py, build_inventory.py, and
    inventory_to_layout.py so each script stays usable standalone for
    hand-runs. The icon-review and table-detection pre-passes are wired
    in here so the subprocess calls see consistent OCR / cleaned
    inputs.
    """
    src_path = find_page_image(src_dir, num)
    ocr_path = work / "ocr" / f"page_{num}.ocr.json"
    clean_path = work / "inventory" / f"page_{num}.clean.png"
    inv_path = work / "inventory" / f"page_{num}.inventory.json"
    masks_dir = work / "inventory" / "masks" / f"page_{num}"
    manifest_path = work / "manifests" / f"page_{num}.assets.json"
    layout_path = work / "layouts" / f"page_{num}.layout.json"
    assets_dir = work / "assets" / f"page_{num}"
    debug_dir = work / "debug"

    debug_dir.mkdir(parents=True, exist_ok=True)
    masks_dir.mkdir(parents=True, exist_ok=True)
    assets_dir.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    layout_path.parent.mkdir(parents=True, exist_ok=True)
    (work / "ocr").mkdir(parents=True, exist_ok=True)

    # ---- preprocess OCR (in-memory) for table detection + icon overrides ----
    ocr = json.loads(ocr_path.read_text(encoding="utf-8"))
    src_img = cv2.imread(str(src_path))
    if src_img is None:
        raise RuntimeError(f"cannot read source image: {src_path}")
    ocr = heur.preprocess_ocr(ocr, img=src_img)

    # ---- optional: table detection (pre-pass on original OCR + image) ----
    table_layout_elements: list[dict] = []
    table_consumed: set[int] = set()
    if detect_tables_flag:
        table_layout_elements, table_consumed = _detect_tables_for_page(
            num, src_path, ocr, ocr_path, work, table_score_threshold,
        )

    # ---- optional: icon-review overrides ----
    # Apply icon-review decisions to the OCR list once, so every
    # downstream tool (erase_text, build_inventory) sees the same
    # icon-vs-text verdict. Without this the override would only affect
    # erase_text's stroke removal and the item-level inventory call
    # would still use the heuristic.
    icon_review_dir = work / "ocr" / "icon_review" / f"page_{num}"
    decisions_json = icon_review_dir / "icon_review.json"
    overrides = None
    if icon_decisions and decisions_json.exists():
        overrides = et._load_icon_decisions(str(decisions_json))
    if overrides:
        ocr = et.apply_ocr_item_overrides(ocr, overrides)
        patched_ocr_path = (work / "ocr"
                            / f"page_{num}.ocr.with_icon_overrides.json")
        patched_ocr_path.write_text(
            json.dumps(ocr, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        ocr_for_subprocess = patched_ocr_path
    else:
        ocr_for_subprocess = ocr_path

    # If tables consumed some OCR items, hand build_inventory a filtered
    # copy so those cells don't render twice (once as native table, once
    # as floating text on top).
    filtered_ocr_path = ocr_for_subprocess
    if table_consumed:
        filtered_ocr_path = work / "ocr" / f"page_{num}.ocr.no_tables.json"
        filtered = [it for i, it in enumerate(ocr) if i not in table_consumed]
        filtered_ocr_path.write_text(
            json.dumps(filtered, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    # ---- stage 1: erase_text (in-process) ----
    et.run(
        image=str(src_path),
        ocr=str(ocr_for_subprocess),
        out=str(clean_path),
        debug_dir=str(debug_dir),
        icon_review_dump=str(icon_review_dir) if icon_review_dump else None,
        icon_decisions=(str(decisions_json)
                        if icon_decisions and decisions_json.exists()
                        else None),
    )
    _erase_table_regions(clean_path, table_layout_elements)
    if table_layout_elements:
        stem = src_path.stem
        clean_dbg = cv2.imread(str(clean_path))
        if clean_dbg is not None:
            cv2.imwrite(str(debug_dir / f"{stem}_clean.png"), clean_dbg)

    # ---- stage 2: build_inventory (in-process) ----
    bi.run(
        clean=str(clean_path),
        source=str(src_path),
        ocr=str(filtered_ocr_path),
        out=str(inv_path),
        debug_dir=str(debug_dir),
        masks_dir=str(masks_dir),
    )

    # ---- stage 3: inventory_to_layout (in-process) ----
    i2l.run(
        inventory=str(inv_path),
        source=str(src_path),
        cleaned=str(clean_path),
        asset_prefix=f"assets/page_{num}",
        out_assets_dir=str(assets_dir),
        out_manifest=str(manifest_path),
        out_layout=str(layout_path),
    )

    # Merge any accepted table elements into the layout. They render
    # after images but interleave with text via z-order; we append at
    # the end which puts them on top of text. Cells that came from the
    # OCR stream were already removed so we don't double-render.
    if table_layout_elements:
        layout = json.loads(layout_path.read_text(encoding="utf-8-sig"))
        layout.setdefault("elements", []).extend(table_layout_elements)
        layout_path.write_text(
            json.dumps(layout, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    try:
        inv = json.loads(inv_path.read_text(encoding="utf-8"))
        text_count = sum(1 for e in inv if e.get("type") == "text")
        img_count = sum(1 for e in inv if e.get("type") == "image")
    except (OSError, json.JSONDecodeError):
        text_count = img_count = 0
    return {"page": num, "text": text_count, "image": img_count,
            "tables": len(table_layout_elements)}


def process_page_simple(num: str, src_dir: Path, work: Path) -> dict:
    """Text-only mode: full mode minus the icon/shape asset extraction.

    Used by build_deck's `--mode text-only`. Runs the same
    erase → build_inventory → inventory_to_layout chain as full mode, so
    editable text keeps the identical color/bold/align styling. The only
    divergence: inventory_to_layout emits the text-erased cleaned page as
    one full-slide background image (icons/shapes baked in) instead of
    extracting each visual as a separate, movable PNG/shape object.

    Table detection and icon-review pre-passes are skipped — those exist
    only to protect the icon/shape extraction this mode drops.
    """
    src_path = find_page_image(src_dir, num)
    ocr_path = work / "ocr" / f"page_{num}.ocr.json"
    clean_path = work / "inventory" / f"page_{num}.clean.png"
    inv_path = work / "inventory" / f"page_{num}.inventory.json"
    masks_dir = work / "inventory" / "masks" / f"page_{num}"
    manifest_path = work / "manifests" / f"page_{num}.assets.json"
    layout_path = work / "layouts" / f"page_{num}.layout.json"
    assets_dir = work / "assets" / f"page_{num}"
    debug_dir = work / "debug"

    for d in (clean_path.parent, masks_dir, assets_dir,
              manifest_path.parent, layout_path.parent, debug_dir):
        d.mkdir(parents=True, exist_ok=True)

    # ---- stage 1: erase_text → clean.png (text removed, icons intact) ----
    et.run(
        image=str(src_path),
        ocr=str(ocr_path),
        out=str(clean_path),
        debug_dir=str(debug_dir),
    )

    # ---- stage 2: build_inventory (we consume only its text items; the
    #              detected image items are dropped by text-only layout) ----
    bi.run(
        clean=str(clean_path),
        source=str(src_path),
        ocr=str(ocr_path),
        out=str(inv_path),
        debug_dir=str(debug_dir),
        masks_dir=str(masks_dir),
    )

    # ---- stage 3: inventory_to_layout in text-only mode ----
    # clean_rel is resolved by build_deck's assets_root (= the work-dir)
    # back to <work>/inventory/page_NN.clean.png.
    clean_rel = f"inventory/page_{num}.clean.png"
    i2l.run(
        inventory=str(inv_path),
        source=str(src_path),
        cleaned=str(clean_path),
        asset_prefix=f"assets/page_{num}",
        out_assets_dir=str(assets_dir),
        out_manifest=str(manifest_path),
        out_layout=str(layout_path),
        text_only=True,
        text_only_bg_rel=clean_rel,
    )

    try:
        inv = json.loads(inv_path.read_text(encoding="utf-8"))
        text_count = sum(1 for e in inv if e.get("type") == "text")
    except (OSError, json.JSONDecodeError):
        text_count = 0
    return {"page": num, "text": text_count, "image": 1, "tables": 0}


# =============================================================================
# Driver
# =============================================================================


def main() -> None:
    args = parse_args()
    src_dir = Path(args.source_dir)
    work = Path(args.work_dir)
    nums = page_numbers(args)
    t0 = time.time()
    for n in nums:
        ts = time.time()
        try:
            r = process_page(
                n, src_dir, work,
                detect_tables_flag=args.detect_tables,
                table_score_threshold=args.table_score_threshold,
                icon_review_dump=args.icon_review,
                icon_decisions=args.icon_decisions,
            )
        except Exception as e:
            print(f"page {n}: ERROR {e}", file=sys.stderr)
            raise
        tables_note = f" tables={r['tables']}" if r.get('tables') else ""
        print(f"page {n}: text={r['text']:>3} image={r['image']:>3}"
              f"{tables_note} ({time.time()-ts:.1f}s)", file=sys.stderr)
    print(f"total: {time.time()-t0:.1f}s for {len(nums)} pages",
          file=sys.stderr)


if __name__ == "__main__":
    main()
