"""Subprocess wrapper around scripts/convert.py.

Streams stdout, parses progress markers, calls back into the queue.
The subprocess is launched with CWD = repo root so its relative paths
behave identically to a manual CLI invocation, and run through the
sandbox module so it can only write to its own job dirs.
"""
from __future__ import annotations

import asyncio
import re
from collections import deque
from pathlib import Path
from typing import Awaitable, Callable

from .config import REPO_ROOT, get_settings
from . import sandbox

# Progress protocol — emitted on stdout by the CLI scripts (see
# scripts/convert.py, prepare_ocr.py, build_deck.py):
#   "[progress] band <lo> <hi> <label>"  — current phase owns the bar
#                                           range [lo, hi] (0-99).
#   "[progress] tick <done> <total>"     — fractional progress *within*
#                                           the current band.
# These lines are consumed here (not surfaced as log lines). `page N:`
# is still parsed to drive the "page N / M" counter in the UI.
BAND_RE = re.compile(r"^\[progress\] band (\d+) (\d+)")
TICK_RE = re.compile(r"^\[progress\] tick (\d+) (\d+)")
PAGE_RE = re.compile(r"^page\s+(\d+):")
LOG_TAIL_LINES = 200


async def run_convert(
    *,
    source: Path,
    work_dir: Path,
    upload_dir: Path,
    mode: str,
    on_band: Callable[[int, int], Awaitable[None]],
    on_tick: Callable[[int, int], Awaitable[None]],
    on_page: Callable[[int], Awaitable[None]],
    on_line: Callable[[str], Awaitable[None]],
) -> tuple[int, str]:
    """Run convert.py. Returns (exit_code, full_tail_log)."""
    s = get_settings()
    cmd = [
        s.python_bin,
        str(s.convert_script),
        "--source", str(source),
        "--work-dir", str(work_dir),
        # EasyOCR + Tesseract are optional cross-verifiers; skipping
        # them keeps the prod install (and the sandbox) lean. Set
        # DECKWEAVER_CROSS_VERIFY=true if you've installed them and
        # want belt-and-suspenders OCR confidence.
    ]
    if not s.cross_verify:
        cmd.append("--skip-cross-verify")
    if mode != "full":
        cmd += ["--mode", mode]

    wrapped, cleanup = sandbox.wrap_command(
        cmd, upload_dir=upload_dir, output_dir=work_dir,
    )
    env = sandbox.safe_env()
    preexec = sandbox.make_preexec(
        memory_mb=s.subprocess_memory_mb,
        cpu_seconds=s.subprocess_cpu_seconds,
        output_mb=s.subprocess_output_mb,
    )

    try:
        proc = await asyncio.create_subprocess_exec(
            *wrapped,
            cwd=str(REPO_ROOT),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            env=env,
            preexec_fn=preexec,
            start_new_session=False,  # preexec already does setsid
        )

        tail: deque[str] = deque(maxlen=LOG_TAIL_LINES)
        assert proc.stdout is not None
        while True:
            raw = await proc.stdout.readline()
            if not raw:
                break
            line = raw.decode("utf-8", errors="replace").rstrip()
            # Progress markers are control lines — consume without
            # logging so they don't clutter the streamed log / tail.
            m = BAND_RE.match(line)
            if m:
                await on_band(int(m.group(1)), int(m.group(2)))
                continue
            m = TICK_RE.match(line)
            if m:
                await on_tick(int(m.group(1)), int(m.group(2)))
                continue
            tail.append(line)
            await on_line(line)
            m = PAGE_RE.match(line)
            if m:
                await on_page(int(m.group(1)))

        code = await proc.wait()
        return code, "\n".join(tail)
    finally:
        for p in cleanup:
            try:
                p.unlink(missing_ok=True)
            except Exception:
                pass
