"""
CLI entry point for the BMH05108 batch runner.

Commands:
  run        — Run the full batch over an input CSV using one worker per port.
  probe      — Send 0xE0 (get version) to one module and display firmware info.
  dry-run    — Parse and build Body270 frames for the first N rows (no serial I/O).
  inspect    — Pre-validate the dataset with per-column statistics and range checks.
"""

from __future__ import annotations

import csv
import sys
from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table

app = typer.Typer(
    name="bmh05108-batch",
    help="BMH05108 Body270 industrial batch runner.",
    add_completion=False,
)
console = Console()


# ---------------------------------------------------------------------------
# run
# ---------------------------------------------------------------------------


@app.command("run")
def cmd_run(
    input: Annotated[str, typer.Option("--input", "-i", help="Input CSV path")],
    output: Annotated[str, typer.Option("--output", "-o", help="Output CSV path")],
    ports: Annotated[list[str], typer.Option("--ports", "-p", help="Serial ports (one per worker)")],
    product_number: Annotated[int, typer.Option("--product-number", help="Product number passed to device")] = 0,
    checkpoint_dir: Annotated[str, typer.Option("--checkpoint-dir", help="Directory for checkpoints and part files")] = "data/.checkpoints",
    resume: Annotated[bool, typer.Option("--resume/--no-resume", help="Resume from checkpoints")] = False,
    gap_ms: Annotated[int, typer.Option("--gap-ms", help="Minimum inter-command gap (milliseconds)")] = 100,
) -> None:
    """Run the full batch over the input CSV using one worker per port."""
    from bmh05108_batch.orchestrator import run_batch

    console.print(
        f"[bold]BMH05108 Batch Runner[/bold] — input=[cyan]{input}[/cyan] "
        f"ports=[cyan]{', '.join(ports)}[/cyan] resume=[cyan]{resume}[/cyan]"
    )

    run_batch(
        input_path=input,
        output_path=output,
        ports=ports,
        checkpoint_dir=checkpoint_dir,
        product_number=product_number,
        resume=resume,
        inter_command_gap_s=gap_ms / 1000.0,
    )


# ---------------------------------------------------------------------------
# probe
# ---------------------------------------------------------------------------


@app.command("probe")
def cmd_probe(
    port: Annotated[str, typer.Option("--port", "-p", help="Serial port to probe")],
) -> None:
    """Send 0xE0 (get version) to one module and display firmware info."""
    from bmh05108_batch.device import Device
    from bmh05108_batch.protocol import ProtocolError

    console.print(f"Probing [cyan]{port}[/cyan]…")
    try:
        with Device(port) as dev:
            version = dev.get_version()
        console.print(f"[green]Firmware version:[/green] {version!r}")
    except ProtocolError as exc:
        console.print(f"[red]Protocol error:[/red] {exc}")
        raise typer.Exit(code=1) from exc
    except Exception as exc:
        console.print(f"[red]Error:[/red] {exc}")
        raise typer.Exit(code=1) from exc


# ---------------------------------------------------------------------------
# dry-run
# ---------------------------------------------------------------------------


@app.command("dry-run")
def cmd_dry_run(
    input: Annotated[str, typer.Option("--input", "-i", help="Input CSV path")],
    limit: Annotated[int, typer.Option("--limit", help="Number of rows to process")] = 10,
) -> None:
    """Parse and build Body270 frames for the first N rows without touching the serial port."""
    from bmh05108_batch.io_csv import CsvReader, validate_row
    from bmh05108_batch.protocol import build_body270_input, build_frame, CMD_BODY270

    table = Table(title=f"Dry Run — first {limit} rows of {input}", show_lines=True)
    table.add_column("row_id", style="cyan", no_wrap=True)
    table.add_column("violations", style="yellow")
    table.add_column("frame_len", style="green")
    table.add_column("frame_hex (first 16 bytes…)", style="dim")

    processed = 0
    for row in CsvReader(input):
        if processed >= limit:
            break

        row_id = row["row_id"]
        violations = validate_row(row)
        if violations:
            table.add_row(str(row_id), ", ".join(violations), "—", "—")
        else:
            try:
                data = build_body270_input(
                    gender=int(row["gender"]),  # type: ignore[arg-type]
                    height_cm=int(row["height_cm"]),  # type: ignore[arg-type]
                    age=int(row["age"]),  # type: ignore[arg-type]
                    weight_kg=float(row["weight_kg"]),  # type: ignore[arg-type]
                    impedances={
                        k: float(row[k])  # type: ignore[index]
                        for k in (
                            "rh_20k", "lh_20k", "trunk_20k", "rf_20k", "lf_20k",
                            "rh_100k", "lh_100k", "trunk_100k", "rf_100k", "lf_100k",
                        )
                    },
                )
                frame = build_frame(CMD_BODY270, data)
                preview = frame[:16].hex(" ")
                table.add_row(str(row_id), "", str(len(frame)), preview + "…")
            except (ValueError, KeyError) as exc:
                table.add_row(str(row_id), str(exc), "—", "—")

        processed += 1

    console.print(table)


# ---------------------------------------------------------------------------
# inspect
# ---------------------------------------------------------------------------


@app.command("inspect")
def cmd_inspect(
    input: Annotated[str, typer.Option("--input", "-i", help="Input CSV path")],
) -> None:
    """Pre-validate the dataset with per-column statistics and out-of-range counts.

    Prints:
      - Row count, schema, duplicate row_ids
      - Per-column min/max/mean/p01/p99 + ASCII histogram (20 bins via rich)
      - Count of rows outside hardware ranges (will be skipped during batch)
      - Physics sanity checks (warnings only)
    """
    import statistics

    from bmh05108_batch.io_csv import HW_RANGES, INPUT_COLUMNS, validate_row

    rows: list[dict[str, object]] = []
    with open(input, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        if reader.fieldnames is None:
            console.print("[red]Empty CSV file[/red]")
            raise typer.Exit(code=1)
        missing = set(INPUT_COLUMNS) - set(reader.fieldnames)
        if missing:
            console.print(f"[red]Missing columns:[/red] {sorted(missing)}")
            raise typer.Exit(code=1)
        for raw in reader:
            rows.append(dict(raw))

    row_count = len(rows)
    console.print(f"\n[bold]Input:[/bold] {input}")
    console.print(f"[bold]Rows:[/bold] {row_count:,}")

    # Duplicate row_ids
    row_ids = [int(r["row_id"]) for r in rows]  # type: ignore[arg-type]
    seen: set[int] = set()
    dupes: list[int] = []
    for rid in row_ids:
        if rid in seen:
            dupes.append(rid)
        seen.add(rid)
    if dupes:
        console.print(f"[yellow]Duplicate row_ids ({len(dupes)}):[/yellow] {dupes[:20]}")
    else:
        console.print("[green]No duplicate row_ids[/green]")

    # Out-of-range count
    oor_count = 0
    for row in rows:
        if validate_row(row):
            oor_count += 1
    console.print(f"[yellow]Rows out of hardware range (will be skipped):[/yellow] {oor_count:,}")

    # Per-column statistics
    numeric_cols = [c for c in INPUT_COLUMNS if c != "row_id" and c != "gender"]
    stats_table = Table(title="Per-Column Statistics", show_lines=True)
    stats_table.add_column("column", style="cyan")
    stats_table.add_column("min", justify="right")
    stats_table.add_column("p01", justify="right")
    stats_table.add_column("mean", justify="right")
    stats_table.add_column("p99", justify="right")
    stats_table.add_column("max", justify="right")
    stats_table.add_column("hw_range", style="dim")
    stats_table.add_column("histogram (20 bins)", no_wrap=True)

    for col in numeric_cols:
        vals: list[float] = []
        for row in rows:
            raw = row.get(col, "")
            try:
                vals.append(float(raw))  # type: ignore[arg-type]
            except (TypeError, ValueError):
                pass

        if not vals:
            stats_table.add_row(col, "—", "—", "—", "—", "—", "—", "—")
            continue

        vals.sort()
        n = len(vals)
        col_min = vals[0]
        col_max = vals[-1]
        col_mean = statistics.mean(vals)
        p01 = vals[max(0, int(n * 0.01))]
        p99 = vals[min(n - 1, int(n * 0.99))]

        # ASCII histogram (20 bins)
        bins = 20
        if col_max > col_min:
            bin_width = (col_max - col_min) / bins
            counts = [0] * bins
            for v in vals:
                idx = min(int((v - col_min) / bin_width), bins - 1)
                counts[idx] += 1
            max_count = max(counts) or 1
            bar_chars = "▁▂▃▄▅▆▇█"
            histogram = "".join(
                bar_chars[min(int(c / max_count * (len(bar_chars) - 1)), len(bar_chars) - 1)]
                for c in counts
            )
        else:
            histogram = "█" * bins

        hw = HW_RANGES.get(col)
        hw_str = f"[{hw[0]}, {hw[1]}]" if hw else "—"

        stats_table.add_row(
            col,
            f"{col_min:.1f}",
            f"{p01:.1f}",
            f"{col_mean:.1f}",
            f"{p99:.1f}",
            f"{col_max:.1f}",
            hw_str,
            histogram,
        )

    console.print(stats_table)

    # Physics sanity checks (warnings only)
    console.print("\n[bold]Physics sanity checks:[/bold]")
    warn_trunk_gt_limb = 0
    warn_20k_lt_100k = 0
    warn_bmi_implausible = 0

    for row in rows:
        try:
            rh20 = float(row.get("rh_20k", 0))  # type: ignore[arg-type]
            trunk20 = float(row.get("trunk_20k", 0))  # type: ignore[arg-type]
            rh100 = float(row.get("rh_100k", 0))  # type: ignore[arg-type]
            rh20k_val = float(row.get("rh_20k", 0))  # type: ignore[arg-type]
            h = float(row.get("height_cm", 0))  # type: ignore[arg-type]
            w = float(row.get("weight_kg", 0))  # type: ignore[arg-type]

            if trunk20 > rh20:
                warn_trunk_gt_limb += 1
            if rh20 < rh100:
                warn_20k_lt_100k += 1
            if h > 0 and w > 0:
                bmi = w / (h / 100) ** 2
                if bmi < 10 or bmi > 70:
                    warn_bmi_implausible += 1
        except (TypeError, ValueError):
            pass

    _warn("trunk_20k > rh_20k (trunk should be lower than limb)", warn_trunk_gt_limb)
    _warn("rh_20k < rh_100k (20kHz should be >= 100kHz in typical tissue)", warn_20k_lt_100k)
    _warn("BMI outside [10, 70] (implausible anthropometry)", warn_bmi_implausible)


def _warn(description: str, count: int) -> None:
    if count > 0:
        console.print(f"  [yellow]⚠[/yellow]  {description}: {count:,} rows")
    else:
        console.print(f"  [green]✓[/green]  {description}: 0 rows")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    app()
