"""
Unit tests for body270.py — Body270Parser multi-packet accumulator.
"""

from __future__ import annotations

import struct

import pytest

from bmh05108_batch.body270 import Body270Parser, Body270Result
from bmh05108_batch.protocol import ProtocolError

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_packet(index: int, total: int, payload: bytes) -> bytes:
    """Wrap a payload with the packet_info byte (high nibble=total, low=index)."""
    packet_info = ((total & 0xF) << 4) | (index & 0xF)
    return bytes([packet_info]) + payload


def _make_p1(body_weight_raw: int = 750) -> bytes:
    """Build a minimal valid packet 1 payload (78 bytes = 39 uint16).

    Sets body_weight to body_weight_raw (unit 0.1 kg), all other fields = 0.
    """
    vals = [0] * 39
    vals[0] = body_weight_raw  # body_weight value
    return struct.pack("<39H", *vals)


def _make_p2() -> bytes:
    """Build a minimal valid packet 2 payload (40 bytes = 20 uint16)."""
    return struct.pack("<20H", *([0] * 20))


def _make_p3(
    body_score: int = 80,
    physical_age: int = 30,
    body_type: int = 2,
    visceral_fat_level: int = 5,
    bmi_eval_raw: int = 220,
    standard_weight_raw: int = 700,
    weight_control_raw: int = -50,   # signed
    fat_mass_control_raw: int = -30,  # signed
    muscle_mass_control_raw: int = 20,  # signed
    target_fat_raw: int = 200,
    fitness_score: int = 75,
) -> bytes:
    """Build a valid packet 3 payload (20 bytes = 17 fields + 3 padding reserved)."""
    core = struct.pack(
        "<BBBBHHhhhHB",
        body_score,
        physical_age,
        body_type,
        visceral_fat_level,
        bmi_eval_raw,
        standard_weight_raw,
        weight_control_raw,
        fat_mass_control_raw,
        muscle_mass_control_raw,
        target_fat_raw,
        fitness_score,
    )
    return core + bytes(3)  # pad to 20 bytes (device sends reserved bytes)


def _make_p4(ratios: tuple[int, ...] = (1050, 1060, 980, 1040, 1030)) -> bytes:
    """Build packet 4 payload (10 bytes = 5 uint16)."""
    return struct.pack("<5H", *ratios)


# ---------------------------------------------------------------------------
# 3-packet accumulation
# ---------------------------------------------------------------------------


def test_three_packet_accumulation_complete() -> None:
    parser = Body270Parser()
    assert not parser.feed_packet(_make_packet(1, 3, _make_p1()))
    assert not parser.feed_packet(_make_packet(2, 3, _make_p2()))
    assert parser.feed_packet(_make_packet(3, 3, _make_p3()))
    assert parser.complete


def test_three_packet_order_independent() -> None:
    """Packets can arrive out of order."""
    parser = Body270Parser()
    parser.feed_packet(_make_packet(3, 3, _make_p3()))
    parser.feed_packet(_make_packet(1, 3, _make_p1()))
    parser.feed_packet(_make_packet(2, 3, _make_p2()))
    assert parser.complete


def test_three_packet_parse_returns_body270_result() -> None:
    parser = Body270Parser()
    parser.feed_packet(_make_packet(1, 3, _make_p1()))
    parser.feed_packet(_make_packet(2, 3, _make_p2()))
    parser.feed_packet(_make_packet(3, 3, _make_p3()))
    result = parser.parse()
    assert isinstance(result, Body270Result)


def test_three_packet_parse_body_weight() -> None:
    """body_weight_raw=750 → body_weight_kg=75.0 (×0.1)."""
    parser = Body270Parser()
    parser.feed_packet(_make_packet(1, 3, _make_p1(body_weight_raw=750)))
    parser.feed_packet(_make_packet(2, 3, _make_p2()))
    parser.feed_packet(_make_packet(3, 3, _make_p3()))
    result = parser.parse()
    assert result.body_weight_kg == pytest.approx(75.0)


def test_three_packet_packet4_fields_are_none() -> None:
    parser = Body270Parser()
    parser.feed_packet(_make_packet(1, 3, _make_p1()))
    parser.feed_packet(_make_packet(2, 3, _make_p2()))
    parser.feed_packet(_make_packet(3, 3, _make_p3()))
    result = parser.parse()
    assert result.right_hand_muscle_ratio is None
    assert result.trunk_muscle_ratio is None


# ---------------------------------------------------------------------------
# 4-packet accumulation
# ---------------------------------------------------------------------------


def test_four_packet_accumulation_complete() -> None:
    parser = Body270Parser()
    parser.feed_packet(_make_packet(1, 4, _make_p1()))
    parser.feed_packet(_make_packet(2, 4, _make_p2()))
    parser.feed_packet(_make_packet(3, 4, _make_p3()))
    assert parser.feed_packet(_make_packet(4, 4, _make_p4()))
    assert parser.complete


def test_four_packet_parse_muscle_ratios() -> None:
    ratios = (1050, 1060, 980, 1040, 1030)
    parser = Body270Parser()
    parser.feed_packet(_make_packet(1, 4, _make_p1()))
    parser.feed_packet(_make_packet(2, 4, _make_p2()))
    parser.feed_packet(_make_packet(3, 4, _make_p3()))
    parser.feed_packet(_make_packet(4, 4, _make_p4(ratios)))
    result = parser.parse()
    assert result.right_hand_muscle_ratio == pytest.approx(105.0)
    assert result.left_hand_muscle_ratio == pytest.approx(106.0)
    assert result.trunk_muscle_ratio == pytest.approx(98.0)
    assert result.right_foot_muscle_ratio == pytest.approx(104.0)
    assert result.left_foot_muscle_ratio == pytest.approx(103.0)


# ---------------------------------------------------------------------------
# Packet 3 field decoding
# ---------------------------------------------------------------------------


def test_packet3_scalar_fields() -> None:
    p3 = _make_p3(
        body_score=88,
        physical_age=35,
        body_type=3,
        visceral_fat_level=7,
        fitness_score=90,
    )
    parser = Body270Parser()
    parser.feed_packet(_make_packet(1, 3, _make_p1()))
    parser.feed_packet(_make_packet(2, 3, _make_p2()))
    parser.feed_packet(_make_packet(3, 3, p3))
    result = parser.parse()
    assert result.body_score == 88
    assert result.physical_age == 35
    assert result.body_type == 3
    assert result.visceral_fat_level == 7
    assert result.fitness_score == 90


def test_packet3_signed_weight_control() -> None:
    """weight_control_raw is int16 — negative values must decode correctly."""
    p3 = _make_p3(weight_control_raw=-50)
    parser = Body270Parser()
    parser.feed_packet(_make_packet(1, 3, _make_p1()))
    parser.feed_packet(_make_packet(2, 3, _make_p2()))
    parser.feed_packet(_make_packet(3, 3, p3))
    result = parser.parse()
    assert result.weight_control_kg == pytest.approx(-5.0)


def test_packet3_standard_weight_scaling() -> None:
    p3 = _make_p3(standard_weight_raw=700)
    parser = Body270Parser()
    parser.feed_packet(_make_packet(1, 3, _make_p1()))
    parser.feed_packet(_make_packet(2, 3, _make_p2()))
    parser.feed_packet(_make_packet(3, 3, p3))
    result = parser.parse()
    assert result.standard_weight_kg == pytest.approx(70.0)


# ---------------------------------------------------------------------------
# Packet 2 segmental fields
# ---------------------------------------------------------------------------


def test_packet2_right_hand_fat_percent() -> None:
    """First uint16 in packet 2 is right_hand_fat_percent × 0.1."""
    vals = [0] * 20
    vals[0] = 250  # 25.0 %
    p2 = struct.pack("<20H", *vals)
    parser = Body270Parser()
    parser.feed_packet(_make_packet(1, 3, _make_p1()))
    parser.feed_packet(_make_packet(2, 3, p2))
    parser.feed_packet(_make_packet(3, 3, _make_p3()))
    result = parser.parse()
    assert result.right_hand_fat_percent == pytest.approx(25.0)


def test_packet2_trunk_muscle_mass() -> None:
    """trunk_muscle_mass_kg: group=muscle_mass, segment=trunk → index 5+2=7."""
    vals = [0] * 20
    vals[7] = 320  # 32.0 kg
    p2 = struct.pack("<20H", *vals)
    parser = Body270Parser()
    parser.feed_packet(_make_packet(1, 3, _make_p1()))
    parser.feed_packet(_make_packet(2, 3, p2))
    parser.feed_packet(_make_packet(3, 3, _make_p3()))
    result = parser.parse()
    assert result.trunk_muscle_mass_kg == pytest.approx(32.0)


# ---------------------------------------------------------------------------
# Parser state / error cases
# ---------------------------------------------------------------------------


def test_complete_false_before_all_packets() -> None:
    parser = Body270Parser()
    assert not parser.complete
    parser.feed_packet(_make_packet(1, 3, _make_p1()))
    assert not parser.complete
    parser.feed_packet(_make_packet(2, 3, _make_p2()))
    assert not parser.complete


def test_parse_before_complete_raises() -> None:
    parser = Body270Parser()
    parser.feed_packet(_make_packet(1, 3, _make_p1()))
    with pytest.raises(ProtocolError, match="not all packets"):
        parser.parse()


def test_reset_clears_state() -> None:
    parser = Body270Parser()
    parser.feed_packet(_make_packet(1, 3, _make_p1()))
    parser.feed_packet(_make_packet(2, 3, _make_p2()))
    parser.reset()
    assert not parser.complete
    parser.feed_packet(_make_packet(1, 3, _make_p1()))
    parser.feed_packet(_make_packet(2, 3, _make_p2()))
    parser.feed_packet(_make_packet(3, 3, _make_p3()))
    assert parser.complete


def test_empty_data_raises() -> None:
    parser = Body270Parser()
    with pytest.raises(ProtocolError, match="Empty"):
        parser.feed_packet(b"")


def test_invalid_total_packets_raises() -> None:
    parser = Body270Parser()
    bad_packet_info = (5 << 4) | 1  # total=5, invalid
    with pytest.raises(ProtocolError, match="total_packets"):
        parser.feed_packet(bytes([bad_packet_info]) + _make_p1())


def test_inconsistent_total_packets_raises() -> None:
    parser = Body270Parser()
    parser.feed_packet(_make_packet(1, 3, _make_p1()))
    # Now claim total=4 but previous was total=3
    with pytest.raises(ProtocolError, match="Inconsistent"):
        parser.feed_packet(_make_packet(2, 4, _make_p2()))


def test_packet_index_out_of_range_raises() -> None:
    parser = Body270Parser()
    bad_packet_info = (3 << 4) | 0  # index=0, invalid (must be 1-based)
    with pytest.raises(ProtocolError, match="index"):
        parser.feed_packet(bytes([bad_packet_info]) + _make_p1())


def test_packet1_too_short_raises() -> None:
    parser = Body270Parser()
    with pytest.raises(ProtocolError, match="too short"):
        parser.feed_packet(_make_packet(1, 3, bytes(10)))  # need 78, got 10


# ---------------------------------------------------------------------------
# asdict
# ---------------------------------------------------------------------------


def test_asdict_contains_body_weight_kg() -> None:
    parser = Body270Parser()
    parser.feed_packet(_make_packet(1, 3, _make_p1(body_weight_raw=820)))
    parser.feed_packet(_make_packet(2, 3, _make_p2()))
    parser.feed_packet(_make_packet(3, 3, _make_p3()))
    result = parser.parse()
    d = result.asdict()
    assert "body_weight_kg" in d
    assert d["body_weight_kg"] == pytest.approx(82.0)


def test_asdict_all_keys_present() -> None:
    parser = Body270Parser()
    parser.feed_packet(_make_packet(1, 3, _make_p1()))
    parser.feed_packet(_make_packet(2, 3, _make_p2()))
    parser.feed_packet(_make_packet(3, 3, _make_p3()))
    result = parser.parse()
    d = result.asdict()
    # Spot-check key groups
    assert "body_fat_percent" in d
    assert "right_hand_fat_percent" in d
    assert "trunk_muscle_mass_kg" in d
    assert "weight_control_kg" in d
    assert "right_hand_muscle_ratio" in d  # None for 3-packet


def test_asdict_packet4_none_when_not_received() -> None:
    parser = Body270Parser()
    parser.feed_packet(_make_packet(1, 3, _make_p1()))
    parser.feed_packet(_make_packet(2, 3, _make_p2()))
    parser.feed_packet(_make_packet(3, 3, _make_p3()))
    d = parser.parse().asdict()
    assert d["right_hand_muscle_ratio"] is None
    assert d["left_foot_muscle_ratio"] is None
