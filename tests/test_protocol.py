"""
Unit tests for protocol.py.

Reference packet from BMH05108_Communication_Protocol_en.pdf §6 (example input):
  55 1E D0 01 00 AC 17 6F 02 DB 0F C7 10 0C 01 86 0B D0 0B 06 0E 01 0F DF 00 35 0A 83 0A 7F

Verified manually:
  - byte[0]  = 0x55  : SEND_HEADER
  - byte[1]  = 0x1E  : total_len = 30  (= 4 + 26 data bytes ... or 4 + 27 if layout differs)
  - byte[2]  = 0xD0  : CMD_BODY270
  - byte[-1] = 0x7F  : checksum — sum of all bytes & 0xFF == 0 ✓
"""

import struct

import pytest

from bmh05108_batch.protocol import (
    CMD_BODY270,
    ChecksumError,
    HeaderError,
    ProtocolError,
    TimeoutError,
    build_body270_input,
    build_frame,
    verify_checksum,
)

# ---------------------------------------------------------------------------
# Reference fixture
# ---------------------------------------------------------------------------

REFERENCE_PACKET = bytes.fromhex(
    "551ED0"
    "010 0AC176F02DB0FC7100C01860BD00B060E010FDF00350A830A7F".replace(" ", "")
)


def test_reference_packet_length() -> None:
    assert len(REFERENCE_PACKET) == 30


def test_reference_packet_checksum_passes() -> None:
    assert verify_checksum(REFERENCE_PACKET)


def test_reference_packet_header() -> None:
    assert REFERENCE_PACKET[0] == 0x55


def test_reference_packet_total_len_field() -> None:
    # The total_len byte must equal the total packet length
    assert REFERENCE_PACKET[1] == len(REFERENCE_PACKET)


def test_reference_packet_cmd() -> None:
    assert REFERENCE_PACKET[2] == CMD_BODY270


# ---------------------------------------------------------------------------
# build_frame
# ---------------------------------------------------------------------------


def test_build_frame_starts_with_send_header() -> None:
    frame = build_frame(0xD0, bytes(27))
    assert frame[0] == 0x55


def test_build_frame_total_len_field_equals_packet_length() -> None:
    data = bytes(27)
    frame = build_frame(0xD0, data)
    assert frame[1] == len(frame)


def test_build_frame_checksum_valid() -> None:
    data = bytes(range(27))
    frame = build_frame(0xD0, data)
    assert verify_checksum(frame)


def test_build_frame_cmd_at_position_2() -> None:
    frame = build_frame(0xE0, b"")
    assert frame[2] == 0xE0


def test_build_frame_empty_data() -> None:
    frame = build_frame(0xE0, b"")
    assert len(frame) == 4  # header + total_len + cmd + checksum
    assert verify_checksum(frame)


def test_build_frame_known_data_checksum() -> None:
    # Build a frame with all-zero data and check the checksum is reproducible
    frame1 = build_frame(0xD0, bytes(27))
    frame2 = build_frame(0xD0, bytes(27))
    assert frame1 == frame2  # deterministic


# ---------------------------------------------------------------------------
# verify_checksum
# ---------------------------------------------------------------------------


def test_verify_checksum_rejects_corrupt_frame() -> None:
    frame = bytearray(build_frame(0xD0, bytes(27)))
    frame[-1] ^= 0xFF  # flip checksum byte
    assert not verify_checksum(bytes(frame))


def test_verify_checksum_rejects_bit_flip_in_data() -> None:
    frame = bytearray(build_frame(0xD0, bytes(27)))
    frame[5] ^= 0x01  # flip one data bit
    assert not verify_checksum(bytes(frame))


# ---------------------------------------------------------------------------
# build_body270_input — range validation
# ---------------------------------------------------------------------------

VALID_IMPEDANCES: dict[str, float] = {
    "rh_20k": 360.0,  "lh_20k": 358.0,
    "trunk_20k": 28.0,
    "rf_20k": 265.0,  "lf_20k": 263.0,
    "rh_100k": 305.0, "lh_100k": 303.0,
    "trunk_100k": 24.0,
    "rf_100k": 230.0, "lf_100k": 228.0,
}


def _build_valid() -> bytes:
    return build_body270_input(
        gender=1, height_cm=175, age=30, weight_kg=75.0,
        impedances=VALID_IMPEDANCES,
    )


def test_build_body270_valid_returns_26_bytes() -> None:
    assert len(_build_valid()) == 26


def test_build_body270_full_frame_checksum() -> None:
    data = _build_valid()
    frame = build_frame(CMD_BODY270, data)
    assert verify_checksum(frame)


def test_build_body270_weight_too_high_raises() -> None:
    with pytest.raises(ValueError, match="weight_kg"):
        build_body270_input(
            gender=1, height_cm=175, age=30, weight_kg=201.0,
            impedances=VALID_IMPEDANCES,
        )


def test_build_body270_weight_too_low_raises() -> None:
    with pytest.raises(ValueError, match="weight_kg"):
        build_body270_input(
            gender=1, height_cm=175, age=30, weight_kg=9.0,
            impedances=VALID_IMPEDANCES,
        )


def test_build_body270_limb_impedance_too_high_raises() -> None:
    bad = {**VALID_IMPEDANCES, "rh_20k": 601.0}
    with pytest.raises(ValueError, match="rh_20k"):
        build_body270_input(gender=1, height_cm=175, age=30, weight_kg=75.0, impedances=bad)


def test_build_body270_limb_impedance_too_low_raises() -> None:
    bad = {**VALID_IMPEDANCES, "lf_100k": 99.0}
    with pytest.raises(ValueError, match="lf_100k"):
        build_body270_input(gender=1, height_cm=175, age=30, weight_kg=75.0, impedances=bad)


def test_build_body270_trunk_impedance_too_high_raises() -> None:
    bad = {**VALID_IMPEDANCES, "trunk_20k": 101.0}
    with pytest.raises(ValueError, match="trunk_20k"):
        build_body270_input(gender=1, height_cm=175, age=30, weight_kg=75.0, impedances=bad)


def test_build_body270_trunk_impedance_too_low_raises() -> None:
    bad = {**VALID_IMPEDANCES, "trunk_100k": 9.0}
    with pytest.raises(ValueError, match="trunk_100k"):
        build_body270_input(gender=1, height_cm=175, age=30, weight_kg=75.0, impedances=bad)


def test_build_body270_invalid_gender_raises() -> None:
    with pytest.raises(ValueError, match="gender"):
        build_body270_input(gender=2, height_cm=175, age=30, weight_kg=75.0, impedances=VALID_IMPEDANCES)


def test_build_body270_age_out_of_range_raises() -> None:
    with pytest.raises(ValueError, match="age"):
        build_body270_input(gender=0, height_cm=150, age=100, weight_kg=55.0, impedances=VALID_IMPEDANCES)


def test_build_body270_height_out_of_range_raises() -> None:
    with pytest.raises(ValueError, match="height"):
        build_body270_input(gender=0, height_cm=221, age=30, weight_kg=55.0, impedances=VALID_IMPEDANCES)


# ---------------------------------------------------------------------------
# build_body270_input — encoding spot-checks
# ---------------------------------------------------------------------------


def test_build_body270_gender_encoded_correctly() -> None:
    data_m = build_body270_input(gender=1, height_cm=175, age=30, weight_kg=75.0, impedances=VALID_IMPEDANCES)
    data_f = build_body270_input(gender=0, height_cm=175, age=30, weight_kg=75.0, impedances=VALID_IMPEDANCES)
    # byte[1] is gender
    assert data_m[1] == 1
    assert data_f[1] == 0


def test_build_body270_weight_encoding() -> None:
    data = build_body270_input(gender=1, height_cm=175, age=30, weight_kg=75.0, impedances=VALID_IMPEDANCES)
    # bytes [4-5] = weight in 0.1 kg = 750
    (weight_raw,) = struct.unpack_from("<H", data, 4)
    assert weight_raw == 750


def test_build_body270_rh_20k_encoding() -> None:
    data = build_body270_input(gender=1, height_cm=175, age=30, weight_kg=75.0, impedances=VALID_IMPEDANCES)
    # bytes [6-7] = rh_20k in 0.1 Ω = 3600
    (rh_raw,) = struct.unpack_from("<H", data, 6)
    assert rh_raw == 3600


def test_build_body270_trunk_20k_encoding() -> None:
    data = build_body270_input(gender=1, height_cm=175, age=30, weight_kg=75.0, impedances=VALID_IMPEDANCES)
    # bytes [10-11] = trunk_20k = 28.0 Ω → 280
    (trunk_raw,) = struct.unpack_from("<H", data, 10)
    assert trunk_raw == 280


def test_build_body270_product_number_default_zero() -> None:
    data = _build_valid()
    assert data[0] == 0


def test_build_body270_product_number_custom() -> None:
    data = build_body270_input(
        gender=1, height_cm=175, age=30, weight_kg=75.0,
        impedances=VALID_IMPEDANCES,
        product_number=2,
    )
    assert data[0] == 2
