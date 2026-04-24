"""
Orchestrator: spawns N worker processes (one per serial port), aggregates progress,
handles SIGINT, and merges per-worker output CSVs into the final output.
"""

from __future__ import annotations

import csv
import multiprocessing as mp
import os
import signal
import sys
from pathlib import Path

from rich.progress import (
    BarColumn,
    Progress,
    TaskProgressColumn,
    TextColumn,
    TimeRemainingColumn,
)

from bmh05108_batch.worker import worker_main


def _count_rows(path: str) -> int:
    """Count data rows in a CSV (excluding header). Fast pure-Python implementation."""
    count = 0
    with open(path, newline="", encoding="utf-8") as fh:
        reader = csv.reader(fh)
        next(reader, None)  # skip header
        for _ in reader:
            count += 1
    return count


def _load_chunk_checkpoint(checkpoint_path: str) -> int:
    """Return last_completed_row_id from a worker checkpoint, or -1."""
    import json

    path = Path(checkpoint_path)
    if not path.exists():
        return -1
    try:
        with path.open() as fh:
            data = json.load(fh)
        return int(data.get("last_completed_row_id", -1))
    except Exception:
        return -1


def _merge_outputs(part_paths: list[str], output_path: str) -> None:
    """Merge per-worker CSV files into the final output, sorted by row_id."""
    all_rows: list[dict[str, str]] = []
    fieldnames: list[str] = []

    for part_path in part_paths:
        p = Path(part_path)
        if not p.exists() or p.stat().st_size == 0:
            continue
        with p.open(newline="", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            if reader.fieldnames and not fieldnames:
                fieldnames = list(reader.fieldnames)
            elif reader.fieldnames:
                # Union of all fieldnames (some workers may produce extra columns)
                new_fields = [f for f in reader.fieldnames if f not in fieldnames]
                fieldnames.extend(new_fields)
            for row in reader:
                all_rows.append(dict(row))

    if not all_rows:
        return

    # Sort by row_id
    all_rows.sort(key=lambda r: int(r.get("row_id", 0)))

    with open(output_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in all_rows:
            writer.writerow(row)


def run_batch(
    input_path: str,
    output_path: str,
    ports: list[str],
    checkpoint_dir: str,
    product_number: int = 0,
    resume: bool = False,
    inter_command_gap_s: float = 0.100,
    limit: int | None = None,
) -> None:
    """Orchestrate N workers (one per port).

    Steps:
      1. Count total rows and divide into N contiguous chunks.
      2. Optionally load checkpoints to determine how many rows remain per worker.
      3. Spawn N worker processes.
      4. Aggregate progress via Queue → rich progress bar.
      5. On SIGINT: set stop_event, wait for workers to flush, exit.
      6. On completion: merge per-worker CSVs into output_path.
    """
    checkpoint_dir_path = Path(checkpoint_dir)
    checkpoint_dir_path.mkdir(parents=True, exist_ok=True)

    n_workers = len(ports)
    total_rows = _count_rows(input_path)

    if total_rows == 0:
        print("Input CSV is empty — nothing to do.")
        return

    # Divide rows into contiguous chunks
    # We need to read all row_ids to partition them correctly
    row_ids: list[int] = []
    with open(input_path, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            row_ids.append(int(row["row_id"]))

    row_ids.sort()
    if limit is not None:
        row_ids = row_ids[:limit]
    chunk_size = (len(row_ids) + n_workers - 1) // n_workers

    chunks: list[tuple[int, int]] = []
    for i in range(n_workers):
        start_idx = i * chunk_size
        end_idx = min(start_idx + chunk_size, len(row_ids))
        if start_idx >= len(row_ids):
            # More ports than rows — this worker gets nothing
            chunks.append((row_ids[-1] + 1, row_ids[-1] + 1))
        else:
            chunk_start = row_ids[start_idx]
            # chunk_end is exclusive: the row_id AFTER the last owned row
            chunk_end = row_ids[end_idx - 1] + 1
            chunks.append((chunk_start, chunk_end))

    # Build per-worker paths
    part_paths = [str(checkpoint_dir_path / f"output_part_{i}.csv") for i in range(n_workers)]
    checkpoint_paths = [str(checkpoint_dir_path / f"checkpoint_{i}.json") for i in range(n_workers)]

    # Compute remaining rows per worker for progress bar
    remaining_per_worker: list[int] = []
    for i, (chunk_start, chunk_end) in enumerate(chunks):
        if not resume:
            remaining = sum(1 for rid in row_ids if chunk_start <= rid < chunk_end)
        else:
            last_done = _load_chunk_checkpoint(checkpoint_paths[i])
            remaining = sum(
                1 for rid in row_ids if chunk_start <= rid < chunk_end and rid > last_done
            )
        remaining_per_worker.append(remaining)

    total_remaining = sum(remaining_per_worker)

    # Inter-process communication
    stop_event: mp.Event = mp.Event()  # type: ignore[assignment]
    progress_queue: mp.Queue[tuple[int, int]] = mp.Queue()

    # SIGINT handler — request graceful shutdown
    original_sigint = signal.getsignal(signal.SIGINT)

    def _sigint_handler(signum: int, frame: object) -> None:
        print("\n[interrupt] Stopping workers — waiting for flush…", flush=True)
        stop_event.set()

    signal.signal(signal.SIGINT, _sigint_handler)

    # Spawn workers
    processes: list[mp.Process] = []
    for i, port in enumerate(ports):
        chunk_start, chunk_end = chunks[i]
        proc = mp.Process(
            target=worker_main,
            kwargs={
                "worker_id": i,
                "port": port,
                "input_path": input_path,
                "chunk_start": chunk_start,
                "chunk_end": chunk_end,
                "output_path": part_paths[i],
                "checkpoint_path": checkpoint_paths[i],
                "progress_queue": progress_queue,
                "stop_event": stop_event,
                "product_number": product_number,
                "inter_command_gap_s": inter_command_gap_s,
            },
            name=f"worker-{i}",
            daemon=False,
        )
        proc.start()
        processes.append(proc)

    # Progress bar loop
    completed = 0
    with Progress(
        TextColumn("[bold blue]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TimeRemainingColumn(),
        TextColumn("[green]{task.completed}/{task.total} rows"),
        refresh_per_second=4,
    ) as progress:
        task = progress.add_task("Batch", total=total_remaining)

        while any(p.is_alive() for p in processes):
            # Drain progress queue
            while True:
                try:
                    _worker_id, delta = progress_queue.get_nowait()
                    completed += delta
                    progress.update(task, advance=delta)
                except Exception:
                    break

            # Check if all workers finished naturally
            if not any(p.is_alive() for p in processes):
                break

            # Small sleep to avoid busy-waiting
            import time
            time.sleep(0.1)

        # Final drain
        while True:
            try:
                _worker_id, delta = progress_queue.get_nowait()
                completed += delta
                progress.update(task, advance=delta)
            except Exception:
                break

    # Restore original SIGINT handler
    signal.signal(signal.SIGINT, original_sigint)

    # Wait for all workers to terminate
    for proc in processes:
        proc.join(timeout=30)
        if proc.is_alive():
            proc.terminate()
            proc.join(timeout=5)

    if stop_event.is_set():
        print("Batch interrupted by user — partial results saved to checkpoint.")
        return

    # Merge per-worker CSVs into final output
    print(f"Merging {n_workers} output files → {output_path}")
    _merge_outputs(part_paths, output_path)

    # Remove part files on successful merge
    for part in part_paths:
        try:
            Path(part).unlink(missing_ok=True)
        except Exception:
            pass

    print(f"Done. {completed} rows processed. Output: {output_path}")
