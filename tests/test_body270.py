"""
Unit tests for body270.py — Body270Parser multi-packet accumulator.

Packet payload sizes per protocol PDF §6:
  Packet 1:  74 bytes (37 × uint16)
  Packet 2:  40 bytes (20 × uint16)
  Packet 3:  52 bytes (mixed)
  Packet 4:  16 bytes (8 × uint16)
  Packet 5:  10 bytes (10 × uint8 + 6 reserve)
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
    """Build a minimal valid packet 1 payload (74 bytes = 37 uint16).

    Sets body_weight to body_weight_raw (unit 0.1 kg), all other fields = 0.
    Layout: 12 metrics × (value, min, max) + subcutaneous_fat_mass value.
    """
    vals = [0] * 37
    vals[0] = body_weight_raw  # body_weight value (index 0)
    return struct.pack("<37H", *vals)


def _make_p2(
    trunk_fat_mass_raw: int = 0,
    trunk_muscle_mass_raw: int = 0,
) -> bytes:
    """Build a minimal valid packet 2 payload (40 bytes = 20 uint16).

    Group order: fat_mass(0-4), fat_percent(5-9), muscle_mass(10-14), muscle_ratio(15-19).
    Segment order within each group: right_hand, left_hand, trunk, right_foot, left_foot.
    """
    vals = [0] * 20
    vals[2] = trunk_fat_mass_raw    # fat_mass group, trunk = index 2
    vals[12] = trunk_muscle_mass_raw  # muscle_mass group, trunk = index 12
    return struct.pack("<20H", *vals)


def _make_p3(
    body_score: int = 80,
    physical_age: int = 30,
    body_type: int = 2,
    skeletal_muscle_mass_index: int = 43,
    whr_raw: int = 79,
    whr_min_raw: int = 80,
    whr_max_raw: int = 90,
    visceral_fat_level: int = 5,
    vfl_min_raw: int = 1,
    vfl_max_raw: int = 9,
    bmi_raw: int = 211,
    bmi_min_raw: int = 185,
    bmi_max_raw: int = 230,
    body_fat_percent_raw: int = 220,
    bfp_min_raw: int = 100,
    bfp_max_raw: int = 200,
    basal_metabolism_raw: int = 1406,
    bm_min_raw: int = 1399,
    bm_max_raw: int = 1628,
    recommended_intake_raw: int = 1827,
    ideal_weight_raw: int = 651,
    target_weight_raw: int = 651,
    weight_control_raw: int = 28,
    muscle_control_raw: int = 73,
    fat_control_raw: int = -45,
    subcutaneous_fat_percent_raw: int = 207,
    sfp_min_raw: int = 86,
    sfp_max_raw: int = 167,
    obesity_percent_raw: int = 956,
    obes_min_raw: int = 900,
    obes_max_raw: int = 1100,
) -> bytes:
    """Build a valid packet 3 payload (52 bytes)."""
    uint8_part = struct.pack(
        "<10B",
        body_score, physical_age, body_type, skeletal_muscle_mass_index,
        whr_raw, whr_min_raw, whr_max_raw,
        visceral_fat_level, vfl_min_raw, vfl_max_raw,
    )
    uint16_part = struct.pack(
        "<15H",
        obesity_percent_raw, obes_min_raw, obes_max_raw,
        bmi_raw, bmi_min_raw, bmi_max_raw,
        body_fat_percent_raw, bfp_min_raw, bfp_max_raw,
        basal_metabolism_raw, bm_min_raw, bm_max_raw,
        recommended_intake_raw, ideal_weight_raw, target_weight_raw,
    )
    signed_part = struct.pack("<3h", weight_control_raw, muscle_control_raw, fat_control_raw)
    tail_part = struct.pack("<3H", subcutaneous_fat_percent_raw, sfp_min_raw, sfp_max_raw)
    result = uint8_part + uint16_part + signed_part + tail_part
    assert len(result) == 52, f"_make_p3 produced {len(result)} bytes"
    return result


def _make_p4(
    walk: int = 124,
    golf: int = 109,
    croquet: int = 118,
    tennis: int = 186,
    squash: int = 311,
    mountain: int = 203,
    swim: int = 217,
    badminton: int = 140,
) -> bytes:
    """Build packet 4 payload (16 bytes = 8 uint16 exercise kcal/30min)."""
    return struct.pack("<8H", walk, golf, croquet, tennis, squash, mountain, swim, badminton)


def _make_p5(
    fat_stds: tuple[int, ...] = (1, 1, 1, 1, 1),
    muscle_stds: tuple[int, ...] = (1, 1, 1, 1, 1),
) -> bytes:
    """Build packet 5 payload (16 bytes = 10 uint8 standards + 6 reserve)."""
    return bytes(fat_stds) + bytes(muscle_stds) + bytes(6)


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
    assert isinstance(parser.parse(), Body270Result)


def test_three_packet_parse_body_weight() -> None:
    parser = Body270Parser()
    parser.feed_packet(_make_packet(1, 3, _make_p1(body_weight_raw=750)))
    parser.feed_packet(_make_packet(2, 3, _make_p2()))
    parser.feed_packet(_make_packet(3, 3, _make_p3()))
    assert parser.parse().body_weight_kg == pytest.approx(75.0)


def test_three_packet_packet4_fields_are_none() -> None:
    parser = Body270Parser()
    parser.feed_packet(_make_packet(1, 3, _make_p1()))
    parser.feed_packet(_make_packet(2, 3, _make_p2()))
    parser.feed_packet(_make_packet(3, 3, _make_p3()))
    result = parser.parse()
    assert result.exercise_walk_kcal is None
    assert result.fat_std_right_hand is None


# ---------------------------------------------------------------------------
# 5-packet accumulation (firmware V1.3+)
# ---------------------------------------------------------------------------


def test_five_packet_accumulation_complete() -> None:
    parser = Body270Parser()
    parser.feed_packet(_make_packet(1, 5, _make_p1()))
    parser.feed_packet(_make_packet(2, 5, _make_p2()))
    parser.feed_packet(_make_packet(3, 5, _make_p3()))
    parser.feed_packet(_make_packet(4, 5, _make_p4()))
    assert parser.feed_packet(_make_packet(5, 5, _make_p5()))
    assert parser.complete


def test_five_packet_parse_exercise() -> None:
    parser = Body270Parser()
    parser.feed_packet(_make_packet(1, 5, _make_p1()))
    parser.feed_packet(_make_packet(2, 5, _make_p2()))
    parser.feed_packet(_make_packet(3, 5, _make_p3()))
    parser.feed_packet(_make_packet(4, 5, _make_p4(walk=124, swim=217)))
    parser.feed_packet(_make_packet(5, 5, _make_p5()))
    result = parser.parse()
    assert result.exercise_walk_kcal == pytest.approx(124.0)
    assert result.exercise_swim_kcal == pytest.approx(217.0)


def test_five_packet_parse_segment_standards() -> None:
    parser = Body270Parser()
    parser.feed_packet(_make_packet(1, 5, _make_p1()))
    parser.feed_packet(_make_packet(2, 5, _make_p2()))
    parser.feed_packet(_make_packet(3, 5, _make_p3()))
    parser.feed_packet(_make_packet(4, 5, _make_p4()))
    parser.feed_packet(_make_packet(5, 5, _make_p5(
        fat_stds=(0, 1, 2, 1, 0),
        muscle_stds=(2, 1, 0, 1, 2),
    )))
    result = parser.parse()
    assert result.fat_std_right_hand == 0
    assert result.fat_std_trunk == 2
    assert result.muscle_std_right_hand == 2
    assert result.muscle_std_trunk == 0


# ---------------------------------------------------------------------------
# Packet 1 field decoding
# ---------------------------------------------------------------------------


def test_packet1_body_weight_scaling() -> None:
    vals = [0] * 37
    vals[0] = 820  # body_weight value
    p1 = struct.pack("<37H", *vals)
    parser = Body270Parser()
    parser.feed_packet(_make_packet(1, 3, p1))
    parser.feed_packet(_make_packet(2, 3, _make_p2()))
    parser.feed_packet(_make_packet(3, 3, _make_p3()))
    assert parser.parse().body_weight_kg == pytest.approx(82.0)


def test_packet1_moisture_with_min_max() -> None:
    vals = [0] * 37
    vals[3] = 352   # moisture value  (index 1*3=3)
    vals[4] = 365   # moisture min
    vals[5] = 447   # moisture max
    p1 = struct.pack("<37H", *vals)
    parser = Body270Parser()
    parser.feed_packet(_make_packet(1, 3, p1))
    parser.feed_packet(_make_packet(2, 3, _make_p2()))
    parser.feed_packet(_make_packet(3, 3, _make_p3()))
    result = parser.parse()
    assert result.moisture_kg == pytest.approx(35.2)
    assert result.moisture_kg_min == pytest.approx(36.5)
    assert result.moisture_kg_max == pytest.approx(44.7)


def test_packet1_subcutaneous_fat_mass_no_minmax() -> None:
    """subcutaneous_fat_mass_kg is the last uint16 (index 36) with no min/max."""
    vals = [0] * 37
    vals[36] = 129  # subcutaneous_fat_mass value
    p1 = struct.pack("<37H", *vals)
    parser = Body270Parser()
    parser.feed_packet(_make_packet(1, 3, p1))
    parser.feed_packet(_make_packet(2, 3, _make_p2()))
    parser.feed_packet(_make_packet(3, 3, _make_p3()))
    result = parser.parse()
    assert result.subcutaneous_fat_mass_kg == pytest.approx(12.9)


# ---------------------------------------------------------------------------
# Packet 2 segmental fields
# ---------------------------------------------------------------------------


def test_packet2_trunk_fat_mass() -> None:
    """trunk_fat_mass_kg: fat_mass group (index 0-4), trunk = index 2."""
    result = _parse_three(p2=_make_p2(trunk_fat_mass_raw=720))
    assert result.trunk_fat_mass_kg == pytest.approx(72.0)


def test_packet2_right_hand_fat_percent() -> None:
    """right_hand_fat_percent: fat_percent group (index 5-9), right_hand = index 5."""
    vals = [0] * 20
    vals[5] = 250  # fat_percent group, right_hand
    p2 = struct.pack("<20H", *vals)
    result = _parse_three(p2=p2)
    assert result.right_hand_fat_percent == pytest.approx(25.0)


def test_packet2_trunk_muscle_mass() -> None:
    """trunk_muscle_mass_kg: muscle_mass group (index 10-14), trunk = index 12."""
    result = _parse_three(p2=_make_p2(trunk_muscle_mass_raw=320))
    assert result.trunk_muscle_mass_kg == pytest.approx(32.0)


def test_packet2_left_foot_muscle_ratio() -> None:
    """left_foot_muscle_ratio: muscle_ratio group (index 15-19), left_foot = index 19."""
    vals = [0] * 20
    vals[19] = 1050  # muscle_ratio group, left_foot
    p2 = struct.pack("<20H", *vals)
    result = _parse_three(p2=p2)
    assert result.left_foot_muscle_ratio == pytest.approx(105.0)


# ---------------------------------------------------------------------------
# Packet 3 field decoding
# ---------------------------------------------------------------------------


def test_packet3_scalar_fields() -> None:
    p3 = _make_p3(body_score=88, physical_age=35, body_type=3, visceral_fat_level=7)
    result = _parse_three(p3=p3)
    assert result.body_score == 88
    assert result.physical_age == 35
    assert result.body_type == 3
    assert result.visceral_fat_level == 7


def test_packet3_waist_hip_ratio_scaling() -> None:
    p3 = _make_p3(whr_raw=79, whr_min_raw=80, whr_max_raw=90)
    result = _parse_three(p3=p3)
    assert result.waist_hip_ratio == pytest.approx(0.79)
    assert result.waist_hip_ratio_min == pytest.approx(0.80)
    assert result.waist_hip_ratio_max == pytest.approx(0.90)


def test_packet3_bmi_scaling() -> None:
    p3 = _make_p3(bmi_raw=211, bmi_min_raw=185, bmi_max_raw=230)
    result = _parse_three(p3=p3)
    assert result.bmi == pytest.approx(21.1)
    assert result.bmi_min == pytest.approx(18.5)
    assert result.bmi_max == pytest.approx(23.0)


def test_packet3_body_fat_percent_scaling() -> None:
    p3 = _make_p3(body_fat_percent_raw=229, bfp_min_raw=100, bfp_max_raw=200)
    result = _parse_three(p3=p3)
    assert result.body_fat_percent == pytest.approx(22.9)
    assert result.body_fat_percent_min == pytest.approx(10.0)
    assert result.body_fat_percent_max == pytest.approx(20.0)


def test_packet3_signed_fat_control() -> None:
    """fat_control_kg is int16 — negative values must decode correctly."""
    p3 = _make_p3(fat_control_raw=-45)
    result = _parse_three(p3=p3)
    assert result.fat_control_kg == pytest.approx(-4.5)


def test_packet3_signed_weight_control() -> None:
    p3 = _make_p3(weight_control_raw=28)
    result = _parse_three(p3=p3)
    assert result.weight_control_kg == pytest.approx(2.8)


def test_packet3_basal_metabolism() -> None:
    p3 = _make_p3(basal_metabolism_raw=1406, bm_min_raw=1399, bm_max_raw=1628)
    result = _parse_three(p3=p3)
    assert result.basal_metabolism_kcal == pytest.approx(1406.0)
    assert result.basal_metabolism_kcal_min == pytest.approx(1399.0)
    assert result.basal_metabolism_kcal_max == pytest.approx(1628.0)


def test_packet3_subcutaneous_fat_percent() -> None:
    p3 = _make_p3(subcutaneous_fat_percent_raw=207, sfp_min_raw=86, sfp_max_raw=167)
    result = _parse_three(p3=p3)
    assert result.subcutaneous_fat_percent == pytest.approx(20.7)
    assert result.subcutaneous_fat_percent_min == pytest.approx(8.6)
    assert result.subcutaneous_fat_percent_max == pytest.approx(16.7)


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
    bad_packet_info = (6 << 4) | 1  # total=6, invalid
    with pytest.raises(ProtocolError, match="total_packets"):
        parser.feed_packet(bytes([bad_packet_info]) + _make_p1())


def test_inconsistent_total_packets_raises() -> None:
    parser = Body270Parser()
    parser.feed_packet(_make_packet(1, 3, _make_p1()))
    with pytest.raises(ProtocolError, match="Inconsistent"):
        parser.feed_packet(_make_packet(2, 4, _make_p2()))


def test_packet_index_out_of_range_raises() -> None:
    parser = Body270Parser()
    bad_packet_info = (3 << 4) | 0  # index=0, invalid
    with pytest.raises(ProtocolError, match="index"):
        parser.feed_packet(bytes([bad_packet_info]) + _make_p1())


def test_packet1_too_short_raises() -> None:
    parser = Body270Parser()
    with pytest.raises(ProtocolError, match="too short"):
        parser.feed_packet(_make_packet(1, 3, bytes(10)))  # need 74, got 10


def test_packet3_too_short_raises() -> None:
    parser = Body270Parser()
    parser.feed_packet(_make_packet(1, 3, _make_p1()))
    parser.feed_packet(_make_packet(2, 3, _make_p2()))
    with pytest.raises(ProtocolError, match="too short"):
        parser.feed_packet(_make_packet(3, 3, bytes(20)))  # need 52, got 20


# ---------------------------------------------------------------------------
# asdict
# ---------------------------------------------------------------------------


def test_asdict_contains_body_weight_kg() -> None:
    result = _parse_three(p1=_make_p1(body_weight_raw=820))
    d = result.asdict()
    assert "body_weight_kg" in d
    assert d["body_weight_kg"] == pytest.approx(82.0)


def test_asdict_all_key_groups_present() -> None:
    result = _parse_three()
    d = result.asdict()
    assert "body_weight_kg" in d
    assert "moisture_kg" in d
    assert "trunk_fat_mass_kg" in d
    assert "right_hand_fat_percent" in d
    assert "trunk_muscle_mass_kg" in d
    assert "bmi" in d
    assert "body_fat_percent" in d
    assert "weight_control_kg" in d
    assert "subcutaneous_fat_percent" in d
    assert "device_error_type" in d


def test_asdict_packet4_none_when_not_received() -> None:
    d = _parse_three().asdict()
    assert d["exercise_walk_kcal"] is None
    assert d["fat_std_right_hand"] is None


# ---------------------------------------------------------------------------
# Internal helper
# ---------------------------------------------------------------------------


def _parse_three(
    p1: bytes | None = None,
    p2: bytes | None = None,
    p3: bytes | None = None,
) -> Body270Result:
    """Feed 3 packets and return the parsed result."""
    parser = Body270Parser()
    parser.feed_packet(_make_packet(1, 3, p1 if p1 is not None else _make_p1()))
    parser.feed_packet(_make_packet(2, 3, p2 if p2 is not None else _make_p2()))
    parser.feed_packet(_make_packet(3, 3, p3 if p3 is not None else _make_p3()))
    return parser.parse()
