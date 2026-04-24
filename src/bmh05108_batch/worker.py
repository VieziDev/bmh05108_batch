"""
Worker process for the BMH05108 batch runner.

Each worker owns one serial port and processes a contiguous chunk of input rows.
Workers are spawned by the orchestrator as multiprocessing.Process instances.

Status values written to output CSV:
  ok                   — run_body270 succeeded
  skipped_out_of_range — validate_row returned one or more violations
  error_device         — device returned error_type != 0
  error_checksum       — ChecksumError after retry
  error_timeout        — TimeoutError after retry
"""

from __future__ import annotations

import json
import logging
import multiprocessing as mp
from pathlib import Path

from bmh05108_batch.body270 import Body270Result
from bmh05108_batch.device import BodyError, Device
from bmh05108_batch.io_csv import CsvReader, CsvWriter, validate_row
from bmh05108_batch.protocol import ChecksumError, ProtocolError, TimeoutError

_CHECKPOINT_INTERVAL = 50  # rows between checkpoint saves


def _setup_logging(worker_id: int, checkpoint_dir: str) -> None:
    """Configure a file handler for this worker process."""
    log_path = Path(checkpoint_dir) / f"worker_{worker_id}.log"
    handler = logging.FileHandler(log_path, encoding="utf-8")
    handler.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s")
    )
    root = logging.getLogger()
    root.setLevel(logging.DEBUG)
    # Avoid duplicate handlers if the function is called more than once in tests
    if not any(isinstance(h, logging.FileHandler) for h in root.handlers):
        root.addHandler(handler)


def _load_checkpoint(checkpoint_path: str) -> int:
    """Return the last completed row_id from the checkpoint file, or -1 if absent."""
    path = Path(checkpoint_path)
    if not path.exists():
        return -1
    try:
        with path.open() as fh:
            data = json.load(fh)
        return int(data.get("last_completed_row_id", -1))
    except Exception:
        return -1


def _save_checkpoint(checkpoint_path: str, last_completed_row_id: int) -> None:
    """Atomically persist the last completed row_id via a .tmp rename."""
    path = Path(checkpoint_path)
    tmp = path.with_suffix(".tmp")
    with tmp.open("w") as fh:
        json.dump({"last_completed_row_id": last_completed_row_id}, fh)
    tmp.replace(path)


def _make_output_row(
    row: dict[str, object],
    status: str,
    result: Body270Result | BodyError | None,
    violations: list[str] | None,
) -> dict[str, object]:
    """Merge input fields, status, and measurement result into a single output row."""
    out: dict[str, object] = dict(row)
    out["status"] = status
    out["out_of_range"] = ",".join(violations) if violations else ""

    if isinstance(result, Body270Result):
        out.update(result.asdict())
    elif isinstance(result, BodyError):
        out["device_error_type"] = f"0x{result.error_type:02X}"
        out["device_error_message"] = result.message

    return out


def worker_main(
    worker_id: int,
    port: str,
    input_path: str,
    chunk_start: int,
    chunk_end: int,
    output_path: str,
    checkpoint_path: str,
    progress_queue: "mp.Queue[tuple[int, int]]",
    stop_event: "mp.Event",
    product_number: int = 0,
    inter_command_gap_s: float = 0.100,
) -> None:
    """Entry point for one worker process.

    Args:
        worker_id:           Integer identifier for logging and checkpointing.
        port:                Serial port path (e.g. '/dev/ttyUSB0').
        input_path:          Path to the shared input CSV.
        chunk_start:         First row_id (inclusive) this worker owns.
        chunk_end:           Last row_id (exclusive) this worker owns.
        output_path:         Path for this worker's per-worker output CSV.
        checkpoint_path:     Path for this worker's JSON checkpoint file.
        progress_queue:      Queue to send (worker_id, rows_completed_delta) tuples.
        stop_event:          Set by the orchestrator to request a clean shutdown.
        product_number:      Forwarded to build_body270_input.
        inter_command_gap_s: Forwarded to Device.
    """
    checkpoint_dir = str(Path(checkpoint_path).parent)
    _setup_logging(worker_id, checkpoint_dir)
    log = logging.getLogger(f"worker.{worker_id}")

    log.info(
        "Worker %d starting — port=%s rows=[%d, %d)",
        worker_id, port, chunk_start, chunk_end,
    )

    # Resume support: skip rows already processed
    last_completed = _load_checkpoint(checkpoint_path)
    start_from = max(chunk_start, last_completed + 1)
    log.info("Resuming from row_id=%d (last_completed=%d)", start_from, last_completed)

    rows_since_checkpoint = 0

    reader = CsvReader(input_path, start_row_id=start_from)

    with Device(port, inter_command_gap_s=inter_command_gap_s) as dev, \
         CsvWriter(output_path) as writer:

        log.info("Serial port %s opened", port)

        for row in reader:
            if stop_event.is_set():
                log.info("Stop event received — flushing and exiting")
                break

            row_id = int(row["row_id"])  # type: ignore[arg-type]

            # Enforce chunk boundaries (reader may yield rows from other chunks
            # if the input_path is shared and start_row_id only skips the beginning)
            if row_id < chunk_start or row_id >= chunk_end:
                continue

            # --- Validate hardware ranges ---
            violations = validate_row(row)
            if violations:
                log.debug("row_id=%d skipped_out_of_range: %s", row_id, violations)
                writer.write_row(_make_output_row(row, "skipped_out_of_range", None, violations))
                _report_and_checkpoint(
                    writer, checkpoint_path, row_id,
                    rows_since_checkpoint, progress_queue, worker_id,
                )
                rows_since_checkpoint = (rows_since_checkpoint + 1) % _CHECKPOINT_INTERVAL
                continue

            # --- Run measurement ---
            row["product_number"] = product_number
            status: str
            result: Body270Result | BodyError | None = None

            try:
                result = dev.run_body270(row)
                if isinstance(result, BodyError):
                    status = "error_device"
                    log.warning(
                        "row_id=%d error_device 0x%02X: %s",
                        row_id, result.error_type, result.message,
                    )
                else:
                    status = "ok"
                    log.debug("row_id=%d ok", row_id)

            except ChecksumError as exc:
                status = "error_checksum"
                log.error("row_id=%d error_checksum: %s", row_id, exc)
            except TimeoutError as exc:
                status = "error_timeout"
                log.error("row_id=%d error_timeout: %s", row_id, exc)
            except ProtocolError as exc:
                # Generic protocol failure — treat as timeout
                status = "error_timeout"
                log.error("row_id=%d protocol_error: %s", row_id, exc)
            except Exception as exc:
                # Never crash the worker on a single-row error
                status = "error_device"
                result = BodyError(error_type=0xFF, message=str(exc))
                log.exception("row_id=%d unexpected exception", row_id)

            writer.write_row(_make_output_row(row, status, result, None))
            _report_and_checkpoint(
                writer, checkpoint_path, row_id,
                rows_since_checkpoint, progress_queue, worker_id,
            )
            rows_since_checkpoint = (rows_since_checkpoint + 1) % _CHECKPOINT_INTERVAL

        # Final flush regardless of checkpoint interval
        writer.flush()
        log.info("Worker %d finished", worker_id)


def _report_and_checkpoint(
    writer: CsvWriter,
    checkpoint_path: str,
    row_id: int,
    rows_since_checkpoint: int,
    progress_queue: "mp.Queue[tuple[int, int]]",
    worker_id: int,
) -> None:
    """Send progress update and optionally write checkpoint."""
    progress_queue.put((worker_id, 1))
    # rows_since_checkpoint is the value BEFORE this row is counted
    next_count = rows_since_checkpoint + 1
    if next_count >= _CHECKPOINT_INTERVAL:
        writer.flush()
        _save_checkpoint(checkpoint_path, row_id)
