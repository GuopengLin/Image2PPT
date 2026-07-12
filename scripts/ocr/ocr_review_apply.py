#!/usr/bin/env python
"""Merge OCR corrections back into the original OCR JSON.

Reads the review JSON produced by prepare_ocr.py (after the 3-engine
cross-verify pre-fills corrected_text and the agent optionally
overrides specific entries) and writes a corrected OCR JSON. Entries
where corrected_text is null are left at their original OCR text;
entries where corrected_text equals original_text are treated as
"confirmed correct" — the text is unchanged but confidence is bumped to
1.0 so downstream heuristics that rely on confidence won't penalize
them.

Two modes:

  Per-page (legacy):
      python scripts/ocr/ocr_review_apply.py \\
          --ocr <inv>/page_NN.ocr.json \\
          --review <inv>/page_NN.ocr_review.json [--require-filled]

  Batch (one command for an entire run):
      python scripts/ocr/ocr_review_apply.py \\
          --work-dir output/<run>/ [--require-filled]

In batch mode every `inventory/page_NN.ocr.json` paired with the
matching `inventory/page_NN.ocr_review.json` is processed in turn.
With `--require-filled`, the command exits non-zero if any page still
has unfilled corrected_text fields.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Apply agent OCR corrections to OCR JSON."
    )
    parser.add_argument("--ocr", help="Original OCR JSON (per-page mode).")
    parser.add_argument("--review",
                        help="Review JSON filled in by the agent (per-page).")
    parser.add_argument("--work-dir",
                        help="Run dir for batch mode. Loops over every "
                             "inventory/page_NN.ocr.json + .ocr_review.json.")
    parser.add_argument("--out", default=None,
                        help="Output corrected OCR JSON (per-page only). "
                             "Default: overwrite --ocr in place after "
                             "backing up to <ocr>.raw.json.")
    parser.add_argument("--require-filled", action="store_true",
                        help="Exit non-zero if any review entry has "
                             "corrected_text == null.")
    return parser.parse_args()


def apply_one(ocr_path: Path, review_path: Path, out_path: Path | None,
              require_filled: bool) -> tuple[int, int, int, int]:
    """Apply corrections from review_path back into ocr_path.

    Returns (changed, confirmed, skipped, unfilled).
    """
    ocr_data = json.loads(ocr_path.read_text(encoding="utf-8"))
    review = json.loads(review_path.read_text(encoding="utf-8"))
    entries = review.get("entries", [])
    unfilled = [e for e in entries if e.get("corrected_text") is None]
    if unfilled and require_filled:
        idxs = ", ".join(str(e["idx"]) for e in unfilled[:10])
        raise SystemExit(
            f"[ocr_review_apply] ERROR: {len(unfilled)} review entries "
            f"in {review_path.name} still have corrected_text=null "
            f"(idx: {idxs}...)."
        )

    changed = confirmed = skipped = 0
    out_data = [dict(it) for it in ocr_data]
    for e in entries:
        if e.get("corrected_text") is None:
            skipped += 1
            continue
        if e.get("dropped"):
            # Rescued below-min-conf detection: it never existed in
            # ocr.json, so a filled-in correction INSERTS a new item.
            corrected = e["corrected_text"]
            if not corrected:
                skipped += 1
                continue
            bx1, by1, bx2, by2 = (int(v) for v in e["bbox"])
            already = any(
                int(it.get("x1", -1)) == bx1 and int(it.get("y1", -1)) == by1
                and int(it.get("x2", -1)) == bx2
                and int(it.get("y2", -1)) == by2
                for it in out_data
            )
            if already:
                skipped += 1
                continue
            out_data.append({
                "text": corrected,
                "x1": bx1, "y1": by1, "x2": bx2, "y2": by2,
                "confidence": 1.0,
            })
            changed += 1
            continue
        idx = e["idx"]
        if not (0 <= idx < len(out_data)):
            print(f"[ocr_review_apply] WARN: review idx {idx} out of range "
                  f"in {review_path.name}, skipping.", file=sys.stderr)
            continue
        target = out_data[idx]
        # Defensive: bbox must match — guards against the OCR JSON being
        # regenerated between prep and apply.
        mismatch = False
        for k in ("x1", "y1", "x2", "y2"):
            ki = ("x1", "y1", "x2", "y2").index(k)
            if int(target[k]) != int(e["bbox"][ki]):
                print(f"[ocr_review_apply] WARN: bbox mismatch at idx {idx} "
                      f"in {review_path.name} ({k}: OCR has {target[k]}, "
                      f"review has {e['bbox']}). Skipping.",
                      file=sys.stderr)
                mismatch = True
                break
        if mismatch:
            continue
        corrected = e["corrected_text"]
        if corrected == e["original_text"]:
            # Agent confirmed OCR was correct; bump confidence.
            target["confidence"] = 1.0
            target.pop("preserve_visual", None)
            confirmed += 1
        elif corrected == "":
            # Empty correction means "not editable text", not "erase
            # this visual mark". Keep the bbox as a preservation hint so
            # erase_text does not wipe icon-internal glyphs.
            target["text"] = ""
            target["confidence"] = 1.0
            target["preserve_visual"] = True
            changed += 1
        else:
            target["text"] = corrected
            target["confidence"] = 1.0
            target.pop("preserve_visual", None)
            changed += 1

    if out_path is None:
        # In-place mode: back up the raw OCR once, then overwrite ocr.json
        # so the rest of the pipeline picks up corrections transparently.
        raw_path = ocr_path.with_suffix(".raw.json")
        if not raw_path.exists():
            raw_path.write_text(ocr_path.read_text(encoding="utf-8"),
                                encoding="utf-8")
        out_path = ocr_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(out_data, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return changed, confirmed, skipped, len(unfilled)


def main() -> int:
    args = parse_args()
    if args.work_dir:
        if args.ocr or args.review or args.out:
            print("[ocr_review_apply] ERROR: --work-dir is mutually "
                  "exclusive with --ocr/--review/--out.", file=sys.stderr)
            return 2
        ocr_dir = Path(args.work_dir) / "ocr"
        if not ocr_dir.is_dir():
            print(f"[ocr_review_apply] ERROR: no ocr/ dir at {ocr_dir}",
                  file=sys.stderr)
            return 1
        review_paths = sorted(ocr_dir.glob("page_*.ocr_review.json"))
        if not review_paths:
            print(f"[ocr_review_apply] No page_*.ocr_review.json found in "
                  f"{ocr_dir}; nothing to apply.")
            return 0
        totals = [0, 0, 0, 0]
        for rp in review_paths:
            num = rp.stem.replace(".ocr_review", "").split("_")[1]
            ocr_path = ocr_dir / f"page_{num}.ocr.json"
            if not ocr_path.exists():
                print(f"  SKIP page {num}: {ocr_path.name} missing",
                      file=sys.stderr)
                continue
            try:
                changed, confirmed, skipped, unf = apply_one(
                    ocr_path, rp, None, args.require_filled,
                )
            except SystemExit as exc:
                # `--require-filled` failure: surface the message and stop.
                print(str(exc), file=sys.stderr)
                return 1
            totals[0] += changed
            totals[1] += confirmed
            totals[2] += skipped
            totals[3] += unf
            print(f"  page {num}: {changed} corrected, {confirmed} confirmed, "
                  f"{skipped} unfilled-kept",
                  flush=True)
        print(f"\n[ocr_review_apply] TOTAL: {totals[0]} corrected, "
              f"{totals[1]} confirmed, {totals[2]} unfilled "
              f"({totals[3]} would have failed --require-filled).")
        return 0

    # Per-page mode (legacy).
    if not args.ocr or not args.review:
        print("[ocr_review_apply] ERROR: pass --ocr and --review, or "
              "use --work-dir for batch mode.", file=sys.stderr)
        return 2
    ocr_path = Path(args.ocr)
    review_path = Path(args.review)
    out_path = Path(args.out) if args.out else None
    try:
        changed, confirmed, skipped, _ = apply_one(
            ocr_path, review_path, out_path, args.require_filled,
        )
    except SystemExit as exc:
        print(str(exc), file=sys.stderr)
        return 1
    final_path = out_path if out_path is not None else ocr_path
    print(f"[ocr_review_apply] {changed} corrected, {confirmed} confirmed, "
          f"{skipped} left unfilled (kept original).")
    print(f"  Wrote: {final_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
