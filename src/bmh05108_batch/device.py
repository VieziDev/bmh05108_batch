"""
Serial transport wrapper for the BMH05108 bioimpedance module.

One Device instance per physical serial port. Not thread/process safe — do not share
a Device across threads or processes.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass

import serial

from bmh05108_batch.body270 import Body270Parser, Body270Result
from bmh05108_batch.protocol import (
    CMD_BODY270,
    CMD_GET_VERSION,
    ChecksumError,
    ProtocolError,
    TimeoutError,
    build_body270_input,
    build_frame,
    read_frame,
)

logger = logging.getLogger(__name__)

_COLLECT_TIMEOUT_S = 3.0  # maximum seconds to wait for all response packets


@dataclass
class BodyError:
    """Represents a device-reported parameter error (error_type != 0)."""

    error_type: int
    message: str = ""


class Device:
    """Wraps one physical BMH05108 module connected to a serial port.

    Usage::

        with Device("/dev/ttyUSB0") as dev:
            version = dev.get_version()
            result = dev.run_body270(sample)
    """

    def __init__(self, port: str, inter_command_gap_s: float = 0.100) -> None:
        self._port = port
        self._gap = inter_command_gap_s
        self._serial: serial.Serial | None = None

    # -----------------------------------------------------------------------
    # Lifecycle
    # -----------------------------------------------------------------------

    def open(self) -> None:
        """Open the serial port at 38400 baud, 8N1, 2 s read timeout."""
        self._serial = serial.Serial(
            port=self._port,
            baudrate=38400,
            bytesize=serial.EIGHTBITS,
            parity=serial.PARITY_NONE,
            stopbits=serial.STOPBITS_ONE,
            timeout=2.0,
        )
        logger.debug("Opened serial port %s", self._port)

    def close(self) -> None:
        """Close the serial port if open."""
        if self._serial is not None and self._serial.is_open:
            self._serial.close()
            logger.debug("Closed serial port %s", self._port)
        self._serial = None

    def __enter__(self) -> "Device":
        self.open()
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    # -----------------------------------------------------------------------
    # Public commands
    # -----------------------------------------------------------------------

    def get_version(self) -> str:
        """Send the 0xE0 version request and return the firmware version string.

        Raises ProtocolError on communication failure.
        """
        port = self._require_open()
        frame = build_frame(CMD_GET_VERSION, b"")
        port.reset_input_buffer()
        port.write(frame)
        time.sleep(self._gap)

        _cmd, error_type, data = read_frame(port)
        if error_type != 0:
            raise ProtocolError(f"get_version error_type=0x{error_type:02X}")

        try:
            return data.decode("ascii", errors="replace").rstrip("\x00").strip()
        except Exception:
            return data.hex()

    def run_body270(self, sample: dict[str, object]) -> Body270Result | BodyError:
        """Execute one Body270 measurement and collect the multi-packet response.

        Retry policy:
          - error_type != 0 (device parameter error): no retry — return BodyError.
          - ChecksumError or TimeoutError: one retry, then raise ProtocolError.

        Args:
            sample: dict containing gender, height_cm, age, weight_kg, and all
                    impedance keys (rh_20k, lh_20k, trunk_20k, etc.).

        Returns:
            Body270Result on success, BodyError on device-side rejection.

        Raises:
            ProtocolError: after exhausting retries on communication failure.
            ValueError: if sample fields are out of hardware-validated range.
        """
        data = build_body270_input(
            gender=round(float(sample["gender"])),  # type: ignore[arg-type]
            height_cm=round(float(sample["height_cm"])),  # type: ignore[arg-type]
            age=round(float(sample["age"])),  # type: ignore[arg-type]
            weight_kg=float(sample["weight_kg"]),  # type: ignore[arg-type]
            impedances={
                k: float(sample[k])  # type: ignore[index]
                for k in (
                    "rh_20k", "lh_20k", "trunk_20k", "rf_20k", "lf_20k",
                    "rh_100k", "lh_100k", "trunk_100k", "rf_100k", "lf_100k",
                )
            },
            product_number=round(float(sample.get("product_number", 0))),  # type: ignore[arg-type]
        )

        last_exc: ProtocolError | None = None
        for attempt in range(2):  # 0 = first try, 1 = one retry
            try:
                return self._send_and_collect(data)
            except (ChecksumError, TimeoutError) as exc:
                last_exc = exc
                logger.warning(
                    "Port %s attempt %d failed: %s", self._port, attempt + 1, exc
                )
                if attempt == 0:
                    # Flush and retry
                    port = self._require_open()
                    port.reset_input_buffer()
                    time.sleep(self._gap)
                    continue
                break

        assert last_exc is not None
        raise last_exc

    # -----------------------------------------------------------------------
    # Internal
    # -----------------------------------------------------------------------

    def _send_and_collect(self, data: bytes) -> Body270Result | BodyError:
        """Send a Body270 frame and collect all response packets.

        Raises BodyError (as a special return) when error_type != 0.
        Raises ProtocolError on timeout or checksum failure.
        """
        port = self._require_open()
        frame = build_frame(CMD_BODY270, data)
        port.reset_input_buffer()
        port.write(frame)
        time.sleep(self._gap)

        parser = Body270Parser()
        deadline = time.monotonic() + _COLLECT_TIMEOUT_S

        while not parser.complete:
            if time.monotonic() > deadline:
                raise TimeoutError(
                    f"Timed out collecting Body270 packets after {_COLLECT_TIMEOUT_S}s"
                )

            cmd, error_type, pkt_data = read_frame(port)

            if cmd != CMD_BODY270:
                logger.warning(
                    "Unexpected cmd=0x%02X while collecting Body270 packets — skipping", cmd
                )
                continue

            if error_type != 0:
                return BodyError(
                    error_type=error_type,
                    message=f"Device returned error_type=0x{error_type:02X}",
                )

            parser.feed_packet(pkt_data)

        return parser.parse()

    def _require_open(self) -> serial.Serial:
        """Return the open Serial instance or raise RuntimeError."""
        if self._serial is None or not self._serial.is_open:
            raise RuntimeError(f"Serial port {self._port} is not open — call open() first")
        return self._serial
