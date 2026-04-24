"""
Streaming CSV I/O for the BMH05108 batch runner.

CsvReader: streaming row-by-row reader with schema validation and row_id skipping.
CsvWriter: append-only writer with automatic header creation and periodic fsync.
validate_row: checks all numeric fields against hardware-validated ranges.
"""

from __future__ import annotations

import csv
import os
from collections.abc import Iterator
from typing import TextIO

INPUT_COLUMNS = [
    "row_id",
    "gender",
    "age",
    "height_cm",
    "weight_kg",
    "rh_20k",
    "lh_20k",
    "trunk_20k",
    "rf_20k",
    "lf_20k",
    "rh_100k",
    "lh_100k",
    "trunk_100k",
    "rf_100k",
    "lf_100k",
]

# Hardware-validated numeric ranges (inclusive)
HW_RANGES: dict[str, tuple[float, float]] = {
    "age": (6, 99),
    "height_cm": (90, 220),
    "weight_kg": (10, 200),
    "rh_20k": (100, 600),
    "lh_20k": (100, 600),
    "rf_20k": (100, 600),
    "lf_20k": (100, 600),
    "rh_100k": (100, 600),
    "lh_100k": (100, 600),
    "rf_100k": (100, 600),
    "lf_100k": (100, 600),
    "trunk_20k": (10, 100),
    "trunk_100k": (10, 100),
}


def validate_row(row: dict[str, object]) -> list[str]:
    """Validate a row dict against hardware-validated ranges.

    Returns a list of violation strings (e.g. ["trunk_20k>100", "weight_kg<10"]).
    An empty list means the row is valid and can be sent to the device.
    """
    violations: list[str] = []
    for col, (lo, hi) in HW_RANGES.items():
        raw = row.get(col)
        if raw is None or raw == "":
            violations.append(f"{col}=missing")
            continue
        try:
            val = float(raw)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            violations.append(f"{col}=non-numeric")
            continue

        if val < lo:
            violations.append(f"{col}<{lo}")
        elif val > hi:
            violations.append(f"{col}>{hi}")

    return violations


# ---------------------------------------------------------------------------
# CsvReader
# ---------------------------------------------------------------------------


class CsvReader:
    """Streaming row-by-row CSV reader.

    Validates that the file contains all INPUT_COLUMNS on the first read.
    Rows with row_id < start_row_id are skipped without loading into memory.

    Usage::

        for row in CsvReader("input.csv", start_row_id=500):
            process(row)
    """

    def __init__(self, path: str, start_row_id: int = 0) -> None:
        self._path = path
        self._start_row_id = start_row_id

    def __iter__(self) -> Iterator[dict[str, object]]:
        with open(self._path, newline="", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)

            # Validate schema on first access
            if reader.fieldnames is None:
                raise ValueError(f"CSV file {self._path!r} appears to be empty")

            missing = set(INPUT_COLUMNS) - set(reader.fieldnames)
            if missing:
                raise ValueError(
                    f"Input CSV missing required columns: {sorted(missing)}"
                )

            for raw_row in reader:
                row_id = int(raw_row["row_id"])
                if row_id < self._start_row_id:
                    continue
                # Cast to appropriate types for numeric fields
                parsed: dict[str, object] = {"row_id": row_id}
                for col in INPUT_COLUMNS[1:]:
                    parsed[col] = raw_row.get(col, "")
                yield parsed


# ---------------------------------------------------------------------------
# CsvWriter
# ---------------------------------------------------------------------------


class CsvWriter:
    """Append-only CSV writer with automatic header creation and periodic fsync.

    If the file does not exist, the header row is written first.
    If the file already exists, rows are appended without rewriting the header.

    Usage::

        with CsvWriter("output.csv") as writer:
            writer.write_row({"row_id": 1, "status": "ok", ...})
    """

    def __init__(self, path: str, fsync_every: int = 100) -> None:
        self._path = path
        self._fsync_every = fsync_every
        self._fh: TextIO | None = None
        self._writer: csv.DictWriter[str] | None = None
        self._fieldnames: list[str] | None = None
        self._rows_since_sync = 0
        self._row_count = 0

    def _ensure_open(self, fieldnames: list[str]) -> None:
        """Open or re-open the file and writer if not already initialised."""
        if self._fh is not None:
            return

        file_exists = os.path.exists(self._path)
        mode = "a" if file_exists else "w"
        self._fh = open(self._path, mode, newline="", encoding="utf-8")  # noqa: SIM115
        self._fieldnames = fieldnames
        self._writer = csv.DictWriter(
            self._fh, fieldnames=fieldnames, extrasaction="ignore"
        )
        if not file_exists or os.path.getsize(self._path) == 0:
            self._writer.writeheader()

    def write_row(self, row: dict[str, object]) -> None:
        """Write one row to the output CSV.

        The column set is inferred from the first row written. Subsequent rows
        use the same DictWriter (extras are silently ignored, missing values
        default to empty string).
        """
        fieldnames = list(row.keys())
        self._ensure_open(fieldnames)

        assert self._writer is not None
        self._writer.writerow(row)
        self._row_count += 1
        self._rows_since_sync += 1

        if self._rows_since_sync >= self._fsync_every:
            self.flush()
            self._rows_since_sync = 0

    def flush(self) -> None:
        """Flush and fsync the underlying file."""
        if self._fh is not None:
            self._fh.flush()
            os.fsync(self._fh.fileno())

    def close(self) -> None:
        """Flush and close the file."""
        if self._fh is not None:
            self.flush()
            self._fh.close()
            self._fh = None
            self._writer = None

    def __enter__(self) -> "CsvWriter":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
