#!/usr/bin/env python3
"""One-command converter: source image(s) -> editable PPTX."""
from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

from image_sources import (
    SUPPORTED_IMAGE_EXTENSIONS,
    discover_page_images,
    supported_image_formats,
)


SCRIPTS_ROOT = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Convert one image or a folder of images into an editable PPTX."
    )
    p.add_argument(
        "--source", "-s", required=True,
        help=f"Source image or source directory ({supported_image_formats()}).",
    )
    p.add_argument(
        "--work-dir", "-o",
        help="Output run directory. Default: output/<source-stem-or-dir-name>_<YYYYMMDD>.",
    )
    p.add_argument(
        "--name",
        help="Run name used when --work-dir is omitted. Default: source name.",
    )
    p.add_argument(
        "--pages",
        help="Comma-separated page numbers to process after source normalization.",
    )
    p.add_argument(
        "--skip-render", action="store_true",
        help="Skip LibreOffice preview rendering.",
    )
    p.add_argument(
        "--calibrate-positions", action="store_true",
        help="Force preview-based text size/position calibration, even with "
             "--skip-render.",
    )
    p.add_argument(
        "--skip-calibration", action="store_true",
        help="Skip preview-based text size/position calibration.",
    )
    p.add_argument(
        "--font-calibration-iterations", type=int, default=None,
        help="Text font-size calibration iterations.",
    )
    p.add_argument(
        "--calibration-iterations", type=int, default=None,
        help="Text position calibration iterations.",
    )
    p.add_argument(
        "--calibration-max-shift", type=float, default=30.0,
        help="Max source-pixel shift per text box per calibration iteration.",
    )
    p.add_argument(
        "--detect-tables", action="store_true",
        help="Enable optional native table reconstruction.",
    )
    p.add_argument(
        "--table-score-threshold", type=float, default=0.85,
        help="Minimum table-structure confidence when --detect-tables is set.",
    )
    p.add_argument(
        "--icon-review", action="store_true",
        help="Emit icon-vs-text review packets.",
    )
    p.add_argument(
        "--icon-decisions", action="store_true",
        help="Apply filled icon review decisions on a rerun.",
    )
    p.add_argument(
        "--ocr-threshold", type=float, default=0.95,
        help="Paddle confidence threshold for OCR review queueing.",
    )
    p.add_argument(
        "--max-review-entries", type=int, default=50,
        help="Maximum OCR review entries per page.",
    )
    p.add_argument(
        "--skip-cross-verify", action="store_true",
        help="Skip EasyOCR/Tesseract cross-verification for faster local checks.",
    )
    p.add_argument(
        "--workers", type=int, default=0,
        help="Parallel workers for per-page processing (0 = auto, 1 = serial).",
    )
    p.add_argument(
        "--pdf-dpi", type=int, default=300,
        help="Render DPI when --source is a .pdf (default 300).",
    )
    p.add_argument(
        "--mode", choices=["full", "text-only"], default="full",
        help="full (default): full pipeline with vector icon extraction. "
             "text-only: cleaned page becomes a background image with "
             "editable OCR text on top (faster, no icon extraction).",
    )
    return p.parse_args()


def slugify(value: str) -> str:
    text = re.sub(r"[^\w.-]+", "_", value.strip(), flags=re.UNICODE)
    text = text.strip("._-")
    return text or "deck"


def default_work_dir(source: Path, name: str | None) -> Path:
    """Default run dir under `output/`.

    Naming convention (matches user request):
      * single-image input  → `output/<image_stem>_<YYYYMMDD>/`
      * directory input     → `output/<dir_name>_<YYYYMMDD>/`
    """
    default_name = source.stem if source.is_file() else source.name
    run_name = slugify(name or default_name)
    date = time.strftime("%Y%m%d")
    return Path("output") / f"{run_name}_{date}"


def supported_file(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS


def copy_as_pages(files: list[Path], dest_dir: Path) -> Path:
    dest_dir.mkdir(parents=True, exist_ok=True)
    for i, src in enumerate(files, 1):
        dst = dest_dir / f"page_{i:02d}{src.suffix.lower()}"
        shutil.copy2(src, dst)
    return dest_dir


def prepare_source(source: Path, work: Path) -> Path:
    """Return a directory containing page_NN source images.

    If the user already provides a page_NN directory, use it in place.
    Otherwise copy the image(s) into <work>/source/ as page_01, page_02,
    etc. so the lower-level pipeline can stay strict and reproducible.
    """
    if source.is_file():
        if source.suffix.lower() not in SUPPORTED_IMAGE_EXTENSIONS:
            raise SystemExit(
                f"Unsupported source image: {source}\n"
                f"Supported formats: {supported_image_formats()}"
            )
        return copy_as_pages([source], work / "source")

    if not source.is_dir():
        raise SystemExit(f"Source does not exist: {source}")

    files = [p for p in sorted(source.iterdir()) if supported_file(p)]
    if not files:
        raise SystemExit(
            f"No supported images found in {source}\n"
            f"Supported formats: {supported_image_formats()}"
        )

    try:
        page_images = discover_page_images(source)
    except ValueError:
        page_images = {}
    if len(page_images) == len(files):
        return source

    return copy_as_pages(files, work / "source")


def run(cmd: list[str]) -> None:
    print("\n$ " + " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def main() -> int:
    args = parse_args()
    source = Path(args.source).expanduser()
    work = Path(args.work_dir).expanduser() if args.work_dir else default_work_dir(source, args.name)
    work.mkdir(parents=True, exist_ok=True)

    pdf_mode = source.is_file() and source.suffix.lower() == ".pdf"
    if pdf_mode:
        ingest_cmd = [
            sys.executable, str(SCRIPTS_ROOT / "ocr" / "pdf_ingest.py"),
            "--pdf", str(source),
            "--work-dir", str(work),
            "--dpi", str(args.pdf_dpi),
        ]
        if args.pages:
            ingest_cmd += ["--pages", args.pages]
        run(ingest_cmd)
        source_dir = work / "source"
    else:
        source_dir = prepare_source(source, work)

    prepare_cmd = [
        sys.executable, str(SCRIPTS_ROOT / "ocr" / "prepare_ocr.py"),
        "--source-dir", str(source_dir),
        "--work-dir", str(work),
        "--threshold", str(args.ocr_threshold),
        "--max-entries", str(args.max_review_entries),
    ]
    if args.pages:
        prepare_cmd += ["--pages", args.pages]
    if args.skip_cross_verify:
        prepare_cmd.append("--skip-cross-verify")

    apply_cmd = [
        sys.executable, str(SCRIPTS_ROOT / "ocr" / "ocr_review_apply.py"),
        "--work-dir", str(work),
    ]

    build_cmd = [
        sys.executable, str(SCRIPTS_ROOT / "build_deck.py"),
        "--source-dir", str(source_dir),
        "--work-dir", str(work),
        "--table-score-threshold", str(args.table_score_threshold),
    ]
    if args.pages and not pdf_mode:
        # In PDF mode pdf_ingest already selected the requested pages AND
        # renumbered them sequentially (page_01, page_02, ...), so
        # re-filtering build_deck on the original numbers would look for
        # pages that no longer exist.
        build_cmd += ["--pages", args.pages]
    if args.detect_tables:
        build_cmd.append("--detect-tables")
    if args.icon_review:
        build_cmd.append("--icon-review")
    if args.icon_decisions:
        build_cmd.append("--icon-decisions")
    if args.skip_render:
        build_cmd.append("--skip-render")
    if args.calibrate_positions:
        build_cmd.append("--calibrate-positions")
    if args.skip_calibration:
        build_cmd.append("--skip-calibration")
    if args.font_calibration_iterations is not None:
        build_cmd += ["--font-calibration-iterations",
                      str(args.font_calibration_iterations)]
    if args.calibration_iterations is not None:
        build_cmd += ["--calibration-iterations",
                      str(args.calibration_iterations)]
    if args.calibration_max_shift != 30.0:
        build_cmd += ["--calibration-max-shift",
                      str(args.calibration_max_shift)]
    if args.workers != 0:
        build_cmd += ["--workers", str(args.workers)]
    if args.mode != "full":
        build_cmd += ["--mode", args.mode]

    if pdf_mode:
        # PDF text layer is already exact; skip OCR + review-apply.
        run(build_cmd)
    else:
        run(prepare_cmd)
        run(apply_cmd)
        run(build_cmd)

    print("\nDone.")
    print(f"  PPTX:     {work / 'slides.pptx'}")
    print(f"  QA:       {work / 'qa.json'}")
    if not args.skip_render:
        print(f"  Previews: {work / 'previews'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
