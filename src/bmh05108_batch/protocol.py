"""
BMH05108 Communication Protocol — Frame builder, parser, and validator.

Frame format (host → device):
  [0x55][total_len][cmd][data...][checksum]
  total_len = total packet length (including 0x55, total_len, cmd, data, checksum)
  checksum  = two's complement of sum of all preceding bytes

Frame format (device → host):
  [0xAA][total_len][cmd][error_type][packet_info][data...][checksum]
  Same checksum algorithm.

NOTE on build_body270_input byte layout:
  The 27-byte data section matches the field order in the protocol PDF (§6, table p.14).
  Height encoding (bytes 3-4) is uint16 LE in centimetres — the reference packet
  encodes 175 cm as 0x00AF which is 175 decimal. If the PDF erratum shows a uint8
  for height, replace the '<H' at offset 3 with '<B' and adjust total to 26 bytes.
"""

from __future__ import annotations

import struct
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import serial

SEND_HEADER = 0x55
RECV_HEADER = 0xAA
CMD_BODY270 = 0xD0
CMD_GET_VERSION = 0xE0


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class ProtocolError(Exception):
    """Base class for all frame-level protocol errors."""


class ChecksumError(ProtocolError):
    """Received frame has an invalid checksum."""


class TimeoutError(ProtocolError):
    """Read timed out before a complete frame was received."""


class HeaderError(ProtocolError):
    """Could not find a valid frame header (0xAA) in the stream."""


# ---------------------------------------------------------------------------
# Frame construction
# ---------------------------------------------------------------------------


def build_frame(cmd: int, data: bytes) -> bytes:
    """Build a complete frame to send to the device.

    Structure: [0x55][total_len][cmd][data...][checksum]
    total_len includes every byte in the packet (including 0x55 and total_len itself).
    """
    total_len = len(data) + 4  # header(1) + total_len(1) + cmd(1) + data + checksum(1)
    payload = bytes([SEND_HEADER, total_len, cmd]) + data
    checksum = (~sum(payload) + 1) & 0xFF
    return payload + bytes([checksum])


def verify_checksum(frame: bytes) -> bool:
    """Return True if the frame checksum is valid.

    A valid frame has sum(all_bytes) & 0xFF == 0.
    """
    return sum(frame) & 0xFF == 0


# ---------------------------------------------------------------------------
# Frame reading
# ---------------------------------------------------------------------------


def read_frame(port: "serial.Serial") -> tuple[int, int, bytes]:
    """Read one complete response frame from the serial port.

    Scans for the 0xAA header byte, reads total_len, then reads the rest of
    the frame, verifies the checksum, and returns (cmd, error_type, data).

    data is the payload after stripping [0xAA][total_len][cmd][error_type][checksum].

    Raises:
        HeaderError   — could not find 0xAA within 256 scanned bytes
        TimeoutError  — serial read returned fewer bytes than expected
        ChecksumError — frame checksum validation failed
    """
    # Scan for header byte
    for _ in range(256):
        raw = port.read(1)
        if not raw:
            raise TimeoutError("Timed out scanning for 0xAA header")
        if raw[0] == RECV_HEADER:
            break
    else:
        raise HeaderError("0xAA header not found within 256 bytes")

    # Read total_len byte
    raw_len = port.read(1)
    if not raw_len:
        raise TimeoutError("Timed out reading total_len byte")
    total_len = raw_len[0]

    if total_len < 5:
        # Minimum: AA + total_len + cmd + error_type + checksum = 5
        raise ProtocolError(f"total_len={total_len} too small (minimum 5)")

    # Read the remaining bytes: total_len - 2 (we already read AA + total_len)
    remaining_count = total_len - 2
    remaining = port.read(remaining_count)
    if len(remaining) < remaining_count:
        raise TimeoutError(
            f"Timed out reading frame body: got {len(remaining)}/{remaining_count} bytes"
        )

    frame = bytes([RECV_HEADER, total_len]) + remaining
    if not verify_checksum(frame):
        raise ChecksumError(
            f"Checksum mismatch for frame of length {total_len}: {frame.hex()}"
        )

    # Parse: [AA][total_len][cmd][error_type][...data...][checksum]
    cmd = frame[2]
    error_type = frame[3]
    # data is everything between error_type and checksum
    data = frame[4:-1]
    return cmd, error_type, data


# ---------------------------------------------------------------------------
# Body270 input serialisation
# ---------------------------------------------------------------------------


def build_body270_input(
    gender: int,
    height_cm: int,
    age: int,
    weight_kg: float,
    impedances: dict[str, float],
    product_number: int = 0,
) -> bytes:
    """Serialize input fields for the 0xD0 Body270 measurement command.

    Returns 26 bytes packed in little-endian format.

    Data layout (verified against reference packet from §6):
      [0]    product_number : uint8
      [1]    gender         : uint8  (0=Female, 1=Male)
      [2]    height_cm      : uint8  (centimetres, fits in 1 byte for 90–220 cm)
      [3]    age            : uint8
      [4-5]  weight         : uint16 LE, unit 0.1 kg  (weight_kg × 10)
      [6-7]  rh_20k         : uint16 LE, unit 0.1 Ω
      [8-9]  lh_20k         : uint16 LE, unit 0.1 Ω
      [10-11]trunk_20k      : uint16 LE, unit 0.1 Ω
      [12-13]rf_20k         : uint16 LE, unit 0.1 Ω
      [14-15]lf_20k         : uint16 LE, unit 0.1 Ω
      [16-17]rh_100k        : uint16 LE, unit 0.1 Ω
      [18-19]lh_100k        : uint16 LE, unit 0.1 Ω
      [20-21]trunk_100k     : uint16 LE, unit 0.1 Ω
      [22-23]rf_100k        : uint16 LE, unit 0.1 Ω
      [24-25]lf_100k        : uint16 LE, unit 0.1 Ω

    Raises ValueError for any out-of-range value before packing.
    """
    if not (0 <= gender <= 1):
        raise ValueError(f"gender={gender} must be 0 (Female) or 1 (Male)")
    if not (6 <= age <= 99):
        raise ValueError(f"age={age} out of range [6, 99]")
    if not (90 <= height_cm <= 220):
        raise ValueError(f"height_cm={height_cm} out of range [90, 220]")

    weight_raw = round(weight_kg * 10)
    if not (100 <= weight_raw <= 2000):
        raise ValueError(f"weight_kg={weight_kg} out of range [10.0, 200.0]")

    limb_keys = ["rh_20k", "lh_20k", "rf_20k", "lf_20k", "rh_100k", "lh_100k", "rf_100k", "lf_100k"]
    trunk_keys = ["trunk_20k", "trunk_100k"]

    for k in limb_keys:
        if k not in impedances:
            raise ValueError(f"Missing impedance key: {k}")
        v_raw = round(impedances[k] * 10)
        if not (1000 <= v_raw <= 6000):
            raise ValueError(f"{k}={impedances[k]} Ω out of range [100, 600]")

    for k in trunk_keys:
        if k not in impedances:
            raise ValueError(f"Missing impedance key: {k}")
        v_raw = round(impedances[k] * 10)
        if not (100 <= v_raw <= 1000):
            raise ValueError(f"{k}={impedances[k]} Ω out of range [10, 100]")

    def imp(key: str) -> int:
        return round(impedances[key] * 10)

    # fmt: off
    # Format: BBBB = 4 bytes (product_number, gender, height_cm, age)
    #         11×H = 22 bytes (weight and 10 impedance fields)
    #         Total = 26 bytes
    data = struct.pack(
        "<BBBBHHHHHHHHHHH",
        product_number,    # [0]    uint8
        gender,            # [1]    uint8
        height_cm,         # [2]    uint8
        age,               # [3]    uint8
        weight_raw,        # [4-5]  uint16 LE
        imp("rh_20k"),     # [6-7]
        imp("lh_20k"),     # [8-9]
        imp("trunk_20k"),  # [10-11]
        imp("rf_20k"),     # [12-13]
        imp("lf_20k"),     # [14-15]
        imp("rh_100k"),    # [16-17]
        imp("lh_100k"),    # [18-19]
        imp("trunk_100k"), # [20-21]
        imp("rf_100k"),    # [22-23]
        imp("lf_100k"),    # [24-25]
    )
    # fmt: on

    assert len(data) == 26, f"build_body270_input produced {len(data)} bytes, expected 26"
    return data
