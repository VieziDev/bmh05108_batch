# bmh05108-batch — Architecture

## What this does

Orchestrates 2–4 BMH05108 body composition modules over USB CDC, sends up to 100k simulated bioimpedance measurement inputs, collects multi-packet Body270 responses, and writes results to CSV. Designed for unattended multi-hour runs with checkpoint-based resilience.

## Stack decisions

| Layer | Choice | Why |
|---|---|---|
| Runtime | Python 3.12 + uv | uv is the fastest Python package manager; 3.12 for full `type[X]` support |
| Concurrency | `multiprocessing.Process` per port | Each USB CDC port runs in its own OS process — isolates driver failures, avoids GIL, maps naturally to hardware parallelism |
| Serial I/O | `pyserial` | De-facto standard for USB CDC; no async dependency needed |
| CLI | `typer` + `rich` | Type-annotated commands, zero boilerplate; rich for progress bars and inspect histograms |
| Output | CSV (append-only per worker) | Survives partial runs; trivially resumable from checkpoint; no database dependency on device |
| Checkpoints | JSON per worker | Lightweight, human-readable, written atomically via tempfile rename |
| Type checking | `mypy --strict` | Protocol and dataclass correctness cannot be tested by running code alone |
| Linting | `ruff` | Replaces flake8 + isort + pyupgrade in one tool, 100× faster |
| Tests | `pytest` | Unit tests for protocol framing + parser; no serial hardware required |
| Analysis | Jupyter + scikit-learn | Regression starter notebook lives in `analysis/` |

**Not used:**
- `asyncio` — serial port reads are blocking by design; multiprocessing is simpler and more resilient
- BLE — USB CDC gives deterministic latency; BLE adds pairing complexity with no throughput benefit
- SQLite / databases — CSV is sufficient for the regression pipeline and avoids schema migrations

## Repository layout

```
bmh05108_batch/
├── src/bmh05108_batch/
│   ├── protocol.py      # Frame builder, checksum, read_frame, build_body270_input
│   ├── body270.py       # Body270Parser (multi-packet accumulator) + Body270Result dataclass
│   ├── io_csv.py        # CsvReader (streaming, validated) + CsvWriter (append, fsync)
│   ├── device.py        # Device — serial session, get_version(), run_body270()
│   ├── worker.py        # Per-process entry point, checkpoint management
│   ├── orchestrator.py  # Chunk division, process spawn, progress aggregation, merge
│   └── cli.py           # typer app: run / probe / dry-run / inspect
├── tests/
│   ├── test_protocol.py # Frame construction, checksum, build_body270_input range checks
│   └── test_body270.py  # Body270Parser accumulation, field decoding, error cases
├── analysis/
│   └── regression_starter.ipynb
├── data/                # Put samples.csv and output CSVs here (gitignored)
└── pyproject.toml
```

## Protocol summary

### Host → device

```
[0x55][total_len][cmd][data...][checksum]
```

- `total_len` = total packet length (all bytes including 0x55 and checksum)
- `checksum` = two's complement of sum of all preceding bytes → `(~sum(payload) + 1) & 0xFF`
- Valid frame: `sum(all_bytes) & 0xFF == 0`

### Device → host

```
[0xAA][total_len][cmd][error_type][packet_info][data...][checksum]
```

Body270 (0xD0) returns 3 or 4 response packets. `packet_info` byte:
- High nibble: total packets (3 or 4)
- Low nibble: packet index (1-based)

Packet sizes (payload after `packet_info`):
| Packet | Size | Contents |
|---|---|---|
| 1 | 78 bytes | 39 × uint16: 13 metrics × (value, min, max) |
| 2 | 40 bytes | 20 × uint16: 5 segments × 4 groups (fat%, muscle, fat mass, lean) |
| 3 | ≥17 bytes | Mixed uint8/uint16/int16: evaluation fields |
| 4 | 10 bytes | 5 × uint16: segmental muscle ratios (firmware ≥ V1.3 only) |

## How to run

### Setup

```sh
uv sync
```

### Discover connected devices

```sh
bmh05108-batch probe
```

Lists all USB CDC ports and attempts `CMD_GET_VERSION` on each.

### Validate input CSV

```sh
bmh05108-batch inspect data/samples.csv
```

Shows per-column histograms, out-of-range counts, and physics sanity warnings (trunk > limb, 100k > 20k).

### Dry run (no hardware)

```sh
bmh05108-batch dry-run data/samples.csv --rows 1000
```

Validates the first 1000 rows and prints what would be sent, without opening serial ports.

### Full batch run

```sh
bmh05108-batch run \
  data/samples.csv \
  --ports /dev/tty.usbmodem1101 /dev/tty.usbmodem1201 \
  --output data/output.csv \
  --checkpoint-dir data/checkpoints/
```

To resume an interrupted run:

```sh
bmh05108-batch run data/samples.csv ... --resume
```

## Worker resilience model

- Each port → one OS process (`multiprocessing.Process`)
- Each process writes to its own append-only CSV (`output_worker_N.csv`)
- Checkpoint JSON (`checkpoint_N.json`) records the last successfully written `row_id`
- On `--resume`, workers skip rows already recorded in their checkpoint
- On `SIGINT`: stop event is set, workers finish the current row, then exit cleanly
- Outputs are merged in `row_id` order after all workers complete

## Retry policy

| Error | Action |
|---|---|
| `error_type != 0` | No retry — device returned a measurement error; written as `status=error_device` |
| `ChecksumError` / `TimeoutError` | 1 retry; if still failing, written as `status=error_checksum` / `status=error_timeout` |
| Out-of-range input | Skipped before sending; written as `status=skipped_out_of_range` |
