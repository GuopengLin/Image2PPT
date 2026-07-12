#!/usr/bin/env python
"""End-to-end deck builder: run_pipeline -> combine -> build -> QA.

Single-command orchestrator that runs the five-step build/QA sequence
in one go:

    1. run_pipeline.py    erase + inventory + layout (every page)
    2. combine_layouts.py per-page layouts -> deck-wide combined.layout.json
    3. build_pptx_from_layout.py   layout JSON -> slides.pptx
    4. inspect_pptx.py    package + media QA report -> qa.json
    5. render_preview.py  PPTX -> PDF -> per-page PNG previews

Run AFTER the OCR pre-pass (prepare_ocr.py + ocr_review_apply.py).
The work directory's ocr/ subfolder must already contain
`page_NN.ocr.json` files with the agent's review applied.

Usage:
    python scripts/build_deck.py \\
        --source-dir <slides_image_dir> \\
        --work-dir   output/<name>_<YYYYMMDD>/

The orchestrator imports `run_pipeline` in-process so the per-page
loop reuses one Python session. The other four stages are simple
shell-outs because each one is already a thin standalone tool and
the subprocess overhead is small compared to LibreOffice startup
(render_preview alone is ~10 s).

Flags forwarded to run_pipeline:
    --pages 1,4,8           run only specified pages
    --detect-tables         enable SLANet table verification
    --table-score-threshold 0.85
    --icon-review           emit icon-review packet
    --icon-decisions        apply previously-recorded icon decisions

Stage skip flags (when iterating):
    --skip-pipeline         skip stage 1 (already-built layouts)
    --skip-render           skip stage 5 (no LibreOffice / preview need)
    --skip-calibration      skip preview-based text size/position calibration
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

# Per-page scripts live in scripts/page/; tables/ is consulted by
# run_pipeline. We're at scripts/ root.
SCRIPTS_ROOT = Path(__file__).resolve().parent      # .../scripts
sys.path.insert(0, str(SCRIPTS_ROOT / "page"))
sys.path.insert(0, str(SCRIPTS_ROOT / "tables"))

import run_pipeline as rp  # noqa: E402
from image_sources import supported_image_formats  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--source-dir", required=True,
                   help=f"Directory of page_NN slide images "
                        f"({supported_image_formats()}).")
    p.add_argument("--work-dir", required=True,
                   help="Run dir; expects ocr/page_NN.ocr.json present.")
    p.add_argument("--pages",
                   help="Comma-separated page numbers; default = all "
                        "ocr/page_NN.ocr.json present.")
    p.add_argument("--detect-tables", dest="detect_tables",
                   action="store_true",
                   help="Forward --detect-tables to run_pipeline.")
    p.add_argument("--no-detect-tables", dest="detect_tables",
                   action="store_false",
                   help="Forward --no-detect-tables to run_pipeline.")
    p.set_defaults(detect_tables=False)
    p.add_argument("--table-score-threshold", type=float, default=0.85)
    p.add_argument("--icon-review", action="store_true")
    p.add_argument("--icon-decisions", action="store_true")
    p.add_argument("--skip-pipeline", action="store_true",
                   help="Skip stage 1 (use existing per-page layouts).")
    p.add_argument("--skip-render", action="store_true",
                   help="Skip stage 5 (no preview PNGs produced).")
    p.add_argument("--calibrate-positions", dest="calibrate_text",
                   action="store_true", default=None,
                   help="Force preview-based closed-loop text "
                        "calibration, even with --skip-render.")
    p.add_argument("--skip-calibration", dest="calibrate_text",
                   action="store_false",
                   help="Skip preview-based text size/position calibration.")
    p.add_argument("--font-calibration-iterations", type=int, default=1,
                   help="Text font-size calibration iterations (default: 1).")
    p.add_argument("--calibration-iterations", type=int, default=2,
                   help="Text position calibration iterations (default: 2).")
    p.add_argument("--calibration-max-shift", type=float, default=30.0,
                   help="Max source-pixel shift per text box per calibration "
                        "iteration (default: 30).")
    p.add_argument("--workers", type=int, default=0,
                   help="Parallel workers for stage 1 per-page processing. "
                        "0 = auto (min(physical_cpu, n_pages, 8)). "
                        "1 = serial (in-process, no fork).")
    p.add_argument("--mode", choices=["full", "text-only"],
                   default="full",
                   help="full (default): erase + inventory + icon "
                        "extraction → editable PPT with vector icons. "
                        "text-only: erase only; the cleaned page "
                        "image becomes one full-slide background and "
                        "OCR text is laid on top as editable text boxes. "
                        "Skips inventory, icon extraction, slot "
                        "classification and text calibration. Faster, "
                        "no icon artifacts, but icons aren't editable.")
    p.add_argument("--stop-after",
                   choices=["erase", "layout", "combine", "classify",
                            "calibrate", "build", "qa"],
                   default=None,
                   help="Stop the pipeline after the named stage. "
                        "erase: per-page clean.png written. "
                        "layout: per-page layout.json written. "
                        "combine: combined.layout.json written. "
                        "classify: text-slot classes applied. "
                        "calibrate: text size/position calibration done. "
                        "build: slides.pptx written. "
                        "qa: qa.json written.")
    p.add_argument("--interactive", action="store_true",
                   help="After each stage prompt the user to continue / "
                        "stop. Intermediate output paths are printed so "
                        "the user can inspect before proceeding.")
    return p.parse_args()


def _process_page_worker(payload: dict) -> dict:
    """Top-level worker for ProcessPoolExecutor (needs to be picklable).

    Each spawn-child re-imports this module, which sets up sys.path for
    page/ and tables/ before importing run_pipeline.
    """
    num = payload["num"]
    t = time.time()
    if payload.get("mode") == "text-only":
        result = rp.process_page_simple(
            num, Path(payload["src"]), Path(payload["work"]),
        )
    else:
        result = rp.process_page(
            num, Path(payload["src"]), Path(payload["work"]),
            detect_tables_flag=payload["detect_tables"],
            table_score_threshold=payload["table_score_threshold"],
            icon_review_dump=payload["icon_review"],
            icon_decisions=payload["icon_decisions"],
        )
    result["_elapsed"] = time.time() - t
    return result


# Stage names in pipeline order. Used to decide whether --stop-after
# halts before / after a given stage, and to drive the --interactive
# prompt. Stages not present in a particular mode (e.g. classify and
# calibrate in text-only) are simply skipped without affecting
# the ordering.
_STAGE_ORDER = ["erase", "layout", "combine", "classify",
                "calibrate", "build", "qa", "render"]


def _stage_index(name: str) -> int:
    return _STAGE_ORDER.index(name)


def _should_stop_after(args, stage: str) -> bool:
    if not args.stop_after:
        return False
    return _stage_index(stage) >= _stage_index(args.stop_after)


def _interactive_gate(args, stage: str, *, artifacts: list[Path]) -> bool:
    """Prompt the user after a stage. Returns True to continue, False to stop.

    No-op (returns True) when --interactive isn't set. Printing the
    artifact paths before prompting lets the user open them in another
    terminal / viewer before deciding.
    """
    if not args.interactive:
        return True
    print(f"\n  [interactive] stage `{stage}` complete.")
    for a in artifacts:
        if a is None:
            continue
        marker = "" if Path(a).exists() else "  (missing)"
        print(f"    artifact: {a}{marker}")
    while True:
        resp = input("  continue? [Y/n/abort] ").strip().lower()
        if resp in ("", "y", "yes"):
            return True
        if resp in ("n", "no", "stop"):
            print("  stopping after this stage (user choice).")
            return False
        if resp in ("a", "abort"):
            sys.stderr.write("  aborted by user.\n")
            sys.exit(130)
        print("  please answer y / n / abort")


def _auto_workers(n_pages: int) -> int:
    """Pick a worker count that uses cores without thrashing memory."""
    try:
        cores = os.cpu_count() or 4
    except Exception:
        cores = 4
    # Each worker re-imports cv2/numpy/PaddleX shims — ~200-400 MB resident.
    # Cap at 8 to leave headroom on dev machines; cap by page count too.
    return max(1, min(cores, n_pages, 8))


def run_stage(cmd: list[str]) -> subprocess.CompletedProcess:
    """`subprocess.run(check=True, capture_output=True)` that keeps logs.

    With captured output, a failing stage's traceback lives only in
    `exc.stderr` — without this wrapper the user just sees "returned
    non-zero exit status N" and the real error is lost.
    """
    try:
        return subprocess.run(cmd, check=True, capture_output=True,
                              text=True)
    except subprocess.CalledProcessError as exc:
        if exc.stdout:
            print(exc.stdout, end="", flush=True)
        if exc.stderr:
            print(exc.stderr, end="", file=sys.stderr, flush=True)
        raise


def banner(title: str) -> None:
    print(f"\n=== {title} ===", flush=True)


def main() -> int:
    args = parse_args()
    work = Path(args.work_dir)
    src = Path(args.source_dir)
    layouts_dir = work / "layouts"
    combined_path = layouts_dir / "combined.layout.json"
    pptx_path = work / "slides.pptx"
    qa_path = work / "qa.json"
    previews_dir = work / "previews"
    work.mkdir(parents=True, exist_ok=True)
    layouts_dir.mkdir(parents=True, exist_ok=True)

    t_total = time.time()
    simple_mode = args.mode == "text-only"
    if simple_mode:
        print("  mode: text-only (full-page background + editable text)")

    # ---- Stage 1: run_pipeline ----
    nums: list[str] = []
    if not args.skip_pipeline:
        stage1_label = ("run_pipeline (erase only, simple layout)"
                        if simple_mode
                        else "run_pipeline (erase + inventory + layout)")
        banner(f"1/5  {stage1_label}")
        ts = time.time()
        nums = (
            [n.strip().zfill(2) for n in args.pages.split(",")]
            if args.pages else rp.page_numbers(args)
        )
        if not nums:
            sys.stderr.write(
                f"ERROR: no page_NN.ocr.json under {work / 'ocr'}\n"
            )
            return 1
        workers = args.workers if args.workers > 0 else _auto_workers(len(nums))
        if workers > 1 and len(nums) > 1:
            print(f"  parallel: {workers} workers × {len(nums)} pages",
                  flush=True)
            ctx = mp.get_context("spawn")
            payloads = [{
                "num": n, "src": str(src), "work": str(work),
                "mode": args.mode,
                "detect_tables": args.detect_tables,
                "table_score_threshold": args.table_score_threshold,
                "icon_review": args.icon_review,
                "icon_decisions": args.icon_decisions,
            } for n in nums]
            results: dict[str, dict] = {}
            with ProcessPoolExecutor(max_workers=workers,
                                     mp_context=ctx) as ex:
                futures = {ex.submit(_process_page_worker, p): p["num"]
                           for p in payloads}
                for fut in as_completed(futures):
                    n = futures[fut]
                    try:
                        r = fut.result()
                    except Exception as exc:
                        sys.stderr.write(f"page {n}: ERROR {exc}\n")
                        raise
                    results[n] = r
                    tables_note = (f" tables={r['tables']}"
                                   if r.get("tables") else "")
                    print(f"page {n}: text={r['text']:>3} "
                          f"image={r['image']:>3}{tables_note} "
                          f"({r['_elapsed']:.1f}s)", flush=True)
        else:
            for n in nums:
                t = time.time()
                try:
                    if simple_mode:
                        r = rp.process_page_simple(n, src, work)
                    else:
                        r = rp.process_page(
                            n, src, work,
                            detect_tables_flag=args.detect_tables,
                            table_score_threshold=args.table_score_threshold,
                            icon_review_dump=args.icon_review,
                            icon_decisions=args.icon_decisions,
                        )
                except Exception as exc:
                    sys.stderr.write(f"page {n}: ERROR {exc}\n")
                    raise
                tables_note = (f" tables={r['tables']}"
                               if r.get("tables") else "")
                print(f"page {n}: text={r['text']:>3} image={r['image']:>3}"
                      f"{tables_note} ({time.time() - t:.1f}s)", flush=True)
        print(f"  stage 1 done in {time.time() - ts:.1f}s", flush=True)
    else:
        banner("1/5  run_pipeline SKIPPED")

    # process_page / process_page_simple are monolithic — by the time
    # stage 1 returns, both the cleaned PNG and the per-page layout JSON
    # are on disk. So `--stop-after erase` and `--stop-after layout`
    # both halt here. The interactive gate fires twice so the user can
    # inspect cleaned images and then layout JSON separately if they
    # want to.
    clean_dir = work / "inventory"
    if _should_stop_after(args, "erase"):
        print(f"\n--stop-after=erase: cleaned pages at {clean_dir}/, "
              f"layouts at {layouts_dir}/")
        return 0
    if not _interactive_gate(args, "erase", artifacts=[clean_dir]):
        return 0
    if _should_stop_after(args, "layout"):
        print(f"\n--stop-after=layout: per-page layouts at {layouts_dir}/")
        return 0
    if not _interactive_gate(args, "layout", artifacts=[layouts_dir]):
        return 0

    # ---- Stage 2: combine_layouts ----
    banner("2/5  combine_layouts")
    ts = time.time()
    r = run_stage(
        [sys.executable, str(SCRIPTS_ROOT / "deck" / "combine_layouts.py"),
         "--layouts", str(layouts_dir),
         "--out", str(combined_path)])
    print(r.stdout.strip())
    print(f"  stage 2 done in {time.time() - ts:.1f}s", flush=True)

    if _should_stop_after(args, "combine"):
        print(f"\n--stop-after=combine: {combined_path}")
        return 0
    if not _interactive_gate(args, "combine", artifacts=[combined_path]):
        return 0

    # ---- Stage 2b: structural text-slot classes ----
    # In text-only mode the editable text is laid out from raw OCR
    # bboxes with no structural grouping, so style-class clustering has
    # nothing meaningful to merge — skip it.
    if simple_mode:
        banner("2b/5  classify_text_slots SKIPPED (text-only mode)")
    else:
        banner("2b/5  classify_text_slots")
        ts = time.time()
        slot_report = work / "debug" / "text_slot_classes.json"
        r = run_stage(
            [sys.executable,
             str(SCRIPTS_ROOT / "deck" / "classify_text_slots.py"),
             "--layout", str(combined_path),
             "--out", str(slot_report),
             "--apply",
             "--min-group-size", "2",
             "--min-apply-size", "3"])
        if r.stdout.strip():
            print(r.stdout.strip())
        print(f"  -> {slot_report}")
        print(f"  stage 2b done in {time.time() - ts:.1f}s", flush=True)
        if _should_stop_after(args, "classify"):
            print(f"\n--stop-after=classify: {slot_report}")
            return 0
        if not _interactive_gate(args, "classify", artifacts=[slot_report]):
            return 0

    # In text-only mode the cleaned PNG already carries the
    # original glyphs visually, so even if our overlay sizing is off
    # by a few points it doesn't move ink underneath. Calibration is
    # expensive (re-renders PPTX → PDF → PNG twice) and only refines
    # things the user can't see — skip by default.
    should_calibrate = False if simple_mode else (
        (not args.skip_render)
        if args.calibrate_text is None else args.calibrate_text
    )

    # ---- Stage 3: build_pptx_from_layout ----
    # When calibrating, calibration scripts only consume the layout JSON
    # (they build their own measurement-only cal.pptx internally). The
    # final pptx is produced by stage 3c. So the intermediate draft pptx
    # would just be overwritten — skip it.
    if should_calibrate:
        banner("3/5  build_pptx_from_layout SKIPPED "
               "(draft is unused; final pptx built in stage 3c)")
    else:
        banner("3/5  build_pptx_from_layout")
        ts = time.time()
        r = run_stage(
            [sys.executable,
             str(SCRIPTS_ROOT / "deck" / "build_pptx_from_layout.py"),
             "--layout", str(combined_path),
             "--assets-root", str(work),
             "--out", str(pptx_path)])
        if r.stdout.strip():
            print(r.stdout.strip())
        print(f"  -> {pptx_path}")
        print(f"  stage 3 done in {time.time() - ts:.1f}s", flush=True)

    # ---- Stage 3b/3c: closed-loop text calibration ----
    if should_calibrate:
        banner("3b/5  calibrate_text_sizes")
        ts = time.time()
        r = run_stage(
            [sys.executable,
             str(SCRIPTS_ROOT / "deck" / "calibrate_text_sizes.py"),
             "--layout", str(combined_path),
             "--source-dir", str(src),
             "--work-dir", str(work),
             "--assets-root", str(work),
             "--iterations", str(args.font_calibration_iterations)])
        if r.stdout.strip():
            print(r.stdout.strip())
        slot_report = work / "debug" / "text_slot_classes.after_size.json"
        r = run_stage(
            [sys.executable,
             str(SCRIPTS_ROOT / "deck" / "classify_text_slots.py"),
             "--layout", str(combined_path),
             "--out", str(slot_report),
             "--apply",
             "--min-group-size", "2",
             "--min-apply-size", "3"])
        if r.stdout.strip():
            print(r.stdout.strip())
        # Intermediate draft pptx between size and position calibration
        # was built here for debug visibility but never consumed —
        # calibrate_text_positions reads the layout JSON, not the pptx.
        # The final pptx is built in stage 3c.
        print(f"  stage 3b done in {time.time() - ts:.1f}s", flush=True)

        banner("3c/5  calibrate_text_positions")
        ts = time.time()
        r = run_stage(
            [sys.executable,
             str(SCRIPTS_ROOT / "deck" / "calibrate_text_positions.py"),
             "--layout", str(combined_path),
             "--source-dir", str(src),
             "--work-dir", str(work),
             "--assets-root", str(work),
             "--iterations", str(args.calibration_iterations),
             "--max-shift", str(args.calibration_max_shift)])
        if r.stdout.strip():
            print(r.stdout.strip())
        slot_report = work / "debug" / "text_slot_classes.after_position.json"
        r = run_stage(
            [sys.executable,
             str(SCRIPTS_ROOT / "deck" / "classify_text_slots.py"),
             "--layout", str(combined_path),
             "--out", str(slot_report),
             "--apply",
             "--min-group-size", "2",
             "--min-apply-size", "3"])
        if r.stdout.strip():
            print(r.stdout.strip())
        print(f"  -> {slot_report}")
        r = run_stage(
            [sys.executable,
             str(SCRIPTS_ROOT / "deck" / "build_pptx_from_layout.py"),
             "--layout", str(combined_path),
             "--assets-root", str(work),
             "--out", str(pptx_path)])
        if r.stdout.strip():
            print(r.stdout.strip())
        print(f"  -> {pptx_path}")
        print(f"  stage 3c done in {time.time() - ts:.1f}s", flush=True)
    else:
        banner("3b/5  calibrate_text_sizes SKIPPED")
        banner("3c/5  calibrate_text_positions SKIPPED")

    if _should_stop_after(args, "calibrate"):
        print(f"\n--stop-after=calibrate: layout={combined_path}, "
              f"pptx={pptx_path}")
        return 0
    if not _interactive_gate(args, "calibrate",
                             artifacts=[combined_path, pptx_path]):
        return 0
    if _should_stop_after(args, "build"):
        print(f"\n--stop-after=build: {pptx_path}")
        return 0
    if not _interactive_gate(args, "build", artifacts=[pptx_path]):
        return 0

    # ---- Stage 4: inspect_pptx ----
    banner("4/5  inspect_pptx")
    ts = time.time()
    r = run_stage(
        [sys.executable, str(SCRIPTS_ROOT / "verify" / "inspect_pptx.py"),
         "--pptx", str(pptx_path),
         "--report", str(qa_path)])
    print(r.stdout.strip())
    print(f"  stage 4 done in {time.time() - ts:.1f}s", flush=True)
    try:
        report = json.loads(qa_path.read_text(encoding="utf-8"))
        failures = report.get("failures", [])
        if failures:
            sys.stderr.write(
                f"WARNING: {len(failures)} QA failures — see {qa_path}\n"
            )
    except (OSError, json.JSONDecodeError):
        pass

    if _should_stop_after(args, "qa"):
        print(f"\n--stop-after=qa: {qa_path}")
        return 0
    if not _interactive_gate(args, "qa", artifacts=[qa_path]):
        return 0

    # ---- Stage 5: render_preview ----
    if not args.skip_render:
        banner("5/5  render_preview (PPTX -> PDF -> PNG)")
        ts = time.time()
        previews_dir.mkdir(parents=True, exist_ok=True)
        for stale in previews_dir.glob("page-*.png"):
            stale.unlink()
        try:
            # render_preview writes <out-dir>/previews/page-NN.png,
            # so we point --out-dir at <work> to land at <work>/previews/.
            r = run_stage(
                [sys.executable, str(SCRIPTS_ROOT / "verify" / "render_preview.py"),
                 "--pptx", str(pptx_path),
                 "--out-dir", str(work)])
            if r.stdout.strip():
                print(r.stdout.strip())
        except subprocess.CalledProcessError as exc:
            sys.stderr.write(
                f"WARNING: render_preview failed: {exc.stderr or exc}\n"
                "         (LibreOffice / pdftoppm may not be installed.)\n"
            )
        print(f"  stage 5 done in {time.time() - ts:.1f}s", flush=True)
    else:
        banner("5/5  render_preview SKIPPED")

    print(f"\n=== build_deck total: {time.time() - t_total:.1f}s ===")
    print(f"  PPTX:     {pptx_path}")
    print(f"  QA:       {qa_path}")
    if not args.skip_render:
        print(f"  Previews: {previews_dir}/")

    # ---- Post-build: OCR review opportunity summary ----
    # Lists pages where 3-engine consensus left yellow/red entries the
    # agent might want to verify. The deck already uses the consensus
    # picks, so this is OPTIONAL — surface it so the calling agent can
    # ask the user whether to do a manual review pass.
    _report_review_pending(work)

    return 0


def _report_review_pending(work: Path) -> None:
    """Scan ocr/page_*.ocr_review.json and print per-page yellow/red counts.

    Skips silently if no review JSONs exist (e.g. legacy workdir or the
    user ran build_deck.py without prepare_ocr.py first).
    """
    ocr_dir = work / "ocr"
    review_paths = sorted(ocr_dir.glob("page_*.ocr_review.json"))
    if not review_paths:
        return

    rows = []
    total_y = total_r = 0
    for rp in review_paths:
        try:
            d = json.loads(rp.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        # Count YELLOW + RED entries that aren't auto-cleared. These
        # have a pre-filled corrected_text but are visually flagged on
        # the annotated page for the agent to consider.
        y = r = 0
        for e in d.get("entries", []):
            if (e.get("notes") or "").startswith("auto-cleared"):
                continue
            if e.get("tier") == "yellow":
                y += 1
            elif e.get("tier") == "red":
                r += 1
        if y or r:
            num = rp.stem.replace(".ocr_review", "").split("_")[1]
            rows.append((num, y, r))
            total_y += y
            total_r += r

    if not rows:
        return

    print("\n=== Optional: OCR review ===")
    print(f"  {total_y + total_r} entries across {len(rows)} pages have "
          f"3-engine consensus pre-fills the PPTX is using.")
    print(f"  ({total_y} 🟡 weak consensus — glance to verify; "
          f"{total_r} 🔴 no consensus — review carefully)")
    print()
    print("  Pages with pending review entries:")
    for num, y, r in rows:
        ann = work / "ocr" / f"page_{num}.ocr_review.annotated.png"
        print(f"    page {num}: {y} 🟡 + {r} 🔴   {ann}")
    print()
    print("  ASK THE USER whether to manually review these. If yes:")
    print("    1. open each ocr_review.annotated.png above")
    print("    2. for any wrong pre-fill, edit `corrected_text` in the")
    print("       matching ocr_review.json")
    print("    3. rerun (FULL rebuild — OCR text propagates through")
    print("       erase + layout → PPTX):")
    print(f"         scripts/ocr/ocr_review_apply.py --work-dir {work}")
    print(f"         scripts/build_deck.py --source-dir <src> "
          f"--work-dir {work}")


if __name__ == "__main__":
    raise SystemExit(main())
