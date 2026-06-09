"""Single-worker job queue.

One asyncio.Task drains the queue. Re-enqueues queued jobs on startup;
running jobs from a previous run are marked failed (we have no way to
attach back to the dead subprocess).

The queue is paused while github_sync is pulling an update so the next
restart starts from a clean state.
"""
from __future__ import annotations

import asyncio
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path

from .config import get_settings
from .db import SessionLocal
from .eta import per_page_seconds
from .models import Job
from .runner import run_convert
from .ws import broker


_paused = False
_queue: asyncio.Queue[str] = asyncio.Queue()


def pause() -> None:
    global _paused
    _paused = True


def resume() -> None:
    global _paused
    _paused = False


def is_paused() -> bool:
    return _paused


async def enqueue(job_id: str) -> None:
    await _queue.put(job_id)


async def _notify(job: Job, extra: dict | None = None) -> None:
    payload = {
        "type": "job",
        "id": job.id,
        "status": job.status,
        "progress_pct": job.progress_pct,
        "current_page": job.current_page,
        "page_count": job.page_count,
        "owner_id": job.owner_id,
    }
    if extra:
        payload.update(extra)
    await broker.broadcast(payload, owner_id=job.owner_id)


async def _process(job_id: str) -> None:
    s = get_settings()
    db = SessionLocal()
    try:
        job = db.get(Job, job_id)
        if not job or job.status != "queued":
            return
        job.status = "running"
        job.started_at = datetime.now(timezone.utc)
        db.commit()
        db.refresh(job)
        # SQLite strips tz info on round-trip; track elapsed locally so
        # we don't have to fight naive vs. aware datetimes downstream.
        wall_start = time.monotonic()
        await _notify(job)

        upload_dir = Path(job.upload_dir)
        # Single uploaded file lives directly inside upload_dir.
        source_files = [p for p in upload_dir.iterdir() if p.is_file()]
        if not source_files:
            raise RuntimeError("No source file found in upload dir")
        source = source_files[0] if len(source_files) == 1 else upload_dir

        # The CLI streams two kinds of progress markers (see runner.py):
        # `band` sets the [lo, hi] slice of the bar the current phase
        # owns; `tick` reports fractional progress inside that slice.
        # We keep progress_pct monotonic and below 100 — the final 100
        # is set once the job actually finishes.
        band = [0, 99]

        def _advance(pct: float) -> None:
            job.progress_pct = max(job.progress_pct, min(99, int(pct)))

        async def on_band(lo: int, hi: int) -> None:
            band[0], band[1] = lo, hi
            _advance(lo)
            db.commit()
            await _notify(job)

        async def on_tick(done: int, total: int) -> None:
            lo, hi = band
            frac = min(1.0, done / max(total, 1))
            _advance(lo + (hi - lo) * frac)
            db.commit()
            await _notify(job)

        async def on_page(num: int) -> None:
            job.current_page = max(job.current_page, num)
            db.commit()
            await _notify(job)

        async def on_line(line: str) -> None:
            await broker.broadcast(
                {"type": "log", "id": job.id, "line": line, "owner_id": job.owner_id},
                owner_id=job.owner_id,
            )

        code, tail = await run_convert(
            source=source,
            work_dir=Path(job.output_dir),
            upload_dir=Path(job.upload_dir),
            mode=job.mode,
            on_band=on_band,
            on_tick=on_tick,
            on_page=on_page,
            on_line=on_line,
        )

        job.log_tail = tail
        job.finished_at = datetime.now(timezone.utc)
        job.duration_seconds = max(int(time.monotonic() - wall_start), 1)
        pptx = Path(job.output_dir) / "slides.pptx"
        if code == 0 and pptx.exists():
            job.status = "done"
            job.progress_pct = 100
            job.current_page = job.page_count
        else:
            job.status = "failed"
            job.error_msg = (
                f"convert.py exited with code {code}"
                if not pptx.exists()
                else f"slides.pptx missing (exit {code})"
            )
        db.commit()
        await _notify(job)

        # Nudge ETA cache by reading once — keeps the rolling avg warm.
        per_page_seconds(db, job.mode)
    except Exception as exc:
        try:
            job = db.get(Job, job_id)
            if job:
                job.status = "failed"
                job.error_msg = f"{type(exc).__name__}: {exc}"
                job.finished_at = datetime.now(timezone.utc)
                db.commit()
                await _notify(job)
        except Exception:
            pass
    finally:
        db.close()


async def worker_loop() -> None:
    while True:
        job_id = await _queue.get()
        # If paused (e.g. during git pull), wait until resumed.
        while _paused:
            await asyncio.sleep(1)
        try:
            await _process(job_id)
        except Exception:
            pass
        finally:
            _queue.task_done()


async def boot_recover() -> None:
    """Recover state from a previous process: mark stale running as failed,
    re-enqueue anything still queued."""
    db = SessionLocal()
    try:
        for j in db.query(Job).filter(Job.status == "running").all():
            j.status = "failed"
            j.error_msg = "interrupted by restart"
            j.finished_at = datetime.now(timezone.utc)
        db.commit()
        for j in db.query(Job).filter(Job.status == "queued").order_by(Job.created_at).all():
            await _queue.put(j.id)
    finally:
        db.close()


def queue_size() -> int:
    return _queue.qsize()


def cleanup_job_dirs(job: Job) -> None:
    for p in (Path(job.upload_dir), Path(job.output_dir)):
        if p.exists():
            shutil.rmtree(p, ignore_errors=True)
