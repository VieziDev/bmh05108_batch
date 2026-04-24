"""
BMH05108 Body270 multi-packet response parser.

The 0xD0 command returns 3 or 4 response packets (firmware >= V1.3 adds packet 4).
Each packet's data payload begins with a packet_info byte:
  - high nibble: total_packets (3 or 4)
  - low nibble:  packet_index (1-based)

Packet sizes (payload bytes AFTER packet_info byte):
  Packet 1: 78 bytes  (39 uint16 — 13 metrics × value/min/max)
  Packet 2: 40 bytes  (20 uint16 — segmental fat%, muscle, fat mass, lean mass)
  Packet 3: ~20 bytes (mixed uint8/uint16/int16 — evaluation fields)
  Packet 4:  10 bytes (5 uint16 — segmental muscle ratios, optional)
"""

from __future__ import annotations

import struct
from dataclasses import dataclass, field

from bmh05108_batch.protocol import ProtocolError

# Expected payload sizes (bytes after packet_info byte)
_PACKET1_SIZE = 78   # 39 × uint16
_PACKET2_SIZE = 40   # 20 × uint16
_PACKET3_SIZE = 20   # mixed types
_PACKET4_SIZE = 10   # 5 × uint16


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass
class Body270Result:
    """All fields from a completed Body270 measurement response."""

    # --- Packet 1: whole-body composition (value / min / max per metric) ---
    body_weight_kg: float | None = field(default=None)
    body_weight_kg_min: float | None = field(default=None)
    body_weight_kg_max: float | None = field(default=None)

    body_fat_percent: float | None = field(default=None)
    body_fat_percent_min: float | None = field(default=None)
    body_fat_percent_max: float | None = field(default=None)

    body_fat_mass_kg: float | None = field(default=None)
    body_fat_mass_kg_min: float | None = field(default=None)
    body_fat_mass_kg_max: float | None = field(default=None)

    fat_free_mass_kg: float | None = field(default=None)
    fat_free_mass_kg_min: float | None = field(default=None)
    fat_free_mass_kg_max: float | None = field(default=None)

    soft_lean_mass_kg: float | None = field(default=None)
    soft_lean_mass_kg_min: float | None = field(default=None)
    soft_lean_mass_kg_max: float | None = field(default=None)

    skeletal_muscle_mass_kg: float | None = field(default=None)
    skeletal_muscle_mass_kg_min: float | None = field(default=None)
    skeletal_muscle_mass_kg_max: float | None = field(default=None)

    body_water_kg: float | None = field(default=None)
    body_water_kg_min: float | None = field(default=None)
    body_water_kg_max: float | None = field(default=None)

    protein_kg: float | None = field(default=None)
    protein_kg_min: float | None = field(default=None)
    protein_kg_max: float | None = field(default=None)

    mineral_kg: float | None = field(default=None)
    mineral_kg_min: float | None = field(default=None)
    mineral_kg_max: float | None = field(default=None)

    bmi: float | None = field(default=None)
    bmi_min: float | None = field(default=None)
    bmi_max: float | None = field(default=None)

    basal_metabolism_kcal: float | None = field(default=None)
    basal_metabolism_kcal_min: float | None = field(default=None)
    basal_metabolism_kcal_max: float | None = field(default=None)

    body_cell_mass_kg: float | None = field(default=None)
    body_cell_mass_kg_min: float | None = field(default=None)
    body_cell_mass_kg_max: float | None = field(default=None)

    obesity_degree_percent: float | None = field(default=None)
    obesity_degree_percent_min: float | None = field(default=None)
    obesity_degree_percent_max: float | None = field(default=None)

    # --- Packet 2: segmental fat%, muscle mass, fat mass, lean mass ---
    right_hand_fat_percent: float | None = field(default=None)
    left_hand_fat_percent: float | None = field(default=None)
    trunk_fat_percent: float | None = field(default=None)
    right_foot_fat_percent: float | None = field(default=None)
    left_foot_fat_percent: float | None = field(default=None)

    right_hand_muscle_mass_kg: float | None = field(default=None)
    left_hand_muscle_mass_kg: float | None = field(default=None)
    trunk_muscle_mass_kg: float | None = field(default=None)
    right_foot_muscle_mass_kg: float | None = field(default=None)
    left_foot_muscle_mass_kg: float | None = field(default=None)

    right_hand_fat_mass_kg: float | None = field(default=None)
    left_hand_fat_mass_kg: float | None = field(default=None)
    trunk_fat_mass_kg: float | None = field(default=None)
    right_foot_fat_mass_kg: float | None = field(default=None)
    left_foot_fat_mass_kg: float | None = field(default=None)

    right_hand_lean_mass_kg: float | None = field(default=None)
    left_hand_lean_mass_kg: float | None = field(default=None)
    trunk_lean_mass_kg: float | None = field(default=None)
    right_foot_lean_mass_kg: float | None = field(default=None)
    left_foot_lean_mass_kg: float | None = field(default=None)

    # --- Packet 3: evaluation ---
    body_score: int | None = field(default=None)
    physical_age: int | None = field(default=None)
    body_type: int | None = field(default=None)
    visceral_fat_level: int | None = field(default=None)
    bmi_eval: float | None = field(default=None)
    standard_weight_kg: float | None = field(default=None)
    weight_control_kg: float | None = field(default=None)
    fat_mass_control_kg: float | None = field(default=None)
    muscle_mass_control_kg: float | None = field(default=None)
    target_fat_percent: float | None = field(default=None)
    fitness_score: int | None = field(default=None)

    # --- Packet 4: segmental muscle ratio (optional) ---
    right_hand_muscle_ratio: float | None = field(default=None)
    left_hand_muscle_ratio: float | None = field(default=None)
    trunk_muscle_ratio: float | None = field(default=None)
    right_foot_muscle_ratio: float | None = field(default=None)
    left_foot_muscle_ratio: float | None = field(default=None)

    def asdict(self) -> dict[str, object]:
        """Return all fields as a flat plain dict suitable for CSV serialisation."""
        return {
            k: getattr(self, k)
            for k in self.__dataclass_fields__  # type: ignore[attr-defined]
        }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _unpack_u16_list(data: bytes, offset: int, count: int) -> list[int]:
    """Unpack `count` little-endian uint16 values starting at `offset`."""
    end = offset + count * 2
    return list(struct.unpack_from(f"<{count}H", data, offset))


def _scale(raw: int, factor: float) -> float:
    """Scale a raw integer value by `factor`."""
    return round(raw * factor, 4)


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------


class Body270Parser:
    """Stateful accumulator for Body270 multi-packet responses.

    Call feed_packet() for each received packet payload (after error_type is stripped).
    When complete returns True, call parse() to obtain a Body270Result.
    """

    def __init__(self) -> None:
        self._packets: dict[int, bytes] = {}
        self._total: int | None = None

    def feed_packet(self, data: bytes) -> bool:
        """Feed one packet data payload (data[0] must be the packet_info byte).

        Returns True when all expected packets have been received.
        Raises ProtocolError if packet_info is malformed or payload size is wrong.
        """
        if len(data) < 1:
            raise ProtocolError("Empty packet data — missing packet_info byte")

        packet_info = data[0]
        total = (packet_info >> 4) & 0xF
        index = packet_info & 0xF

        if total not in (3, 4):
            raise ProtocolError(f"Unexpected total_packets={total} (expected 3 or 4)")
        if index < 1 or index > total:
            raise ProtocolError(f"Packet index {index} out of range [1, {total}]")

        payload = data[1:]  # strip packet_info byte

        # Validate expected sizes
        expected: dict[int, int] = {
            1: _PACKET1_SIZE,
            2: _PACKET2_SIZE,
            3: _PACKET3_SIZE,
        }
        if total == 4:
            expected[4] = _PACKET4_SIZE

        if index in expected:
            exp_size = expected[index]
            if len(payload) < exp_size:
                raise ProtocolError(
                    f"Packet {index} payload too short: got {len(payload)}, expected >= {exp_size}"
                )

        if self._total is None:
            self._total = total
        elif self._total != total:
            raise ProtocolError(
                f"Inconsistent total_packets: previously {self._total}, now {total}"
            )

        self._packets[index] = payload
        return self.complete

    @property
    def complete(self) -> bool:
        """True when all expected packets have been received."""
        if self._total is None:
            return False
        return all(i in self._packets for i in range(1, self._total + 1))

    def parse(self) -> Body270Result:
        """Parse all accumulated packets into a Body270Result.

        Must only be called when complete is True.
        """
        if not self.complete:
            raise ProtocolError("Cannot parse — not all packets received")

        result = Body270Result()
        self._parse_packet1(result, self._packets[1])
        self._parse_packet2(result, self._packets[2])
        self._parse_packet3(result, self._packets[3])
        if 4 in self._packets:
            self._parse_packet4(result, self._packets[4])
        return result

    def reset(self) -> None:
        """Reset internal state so the parser can be reused for the next measurement."""
        self._packets = {}
        self._total = None

    # -----------------------------------------------------------------------
    # Packet parsers
    # -----------------------------------------------------------------------

    def _parse_packet1(self, result: Body270Result, payload: bytes) -> None:
        """Parse 39 uint16 whole-body composition values (value / min / max × 13 metrics)."""
        vals = _unpack_u16_list(payload, 0, 39)

        # 13 metrics, each with (value, min, max) → indices 0..38
        # Metrics in order per protocol §6:
        # 0: body_weight (×0.1 kg)
        # 1: body_fat_percent (×0.1 %)
        # 2: body_fat_mass (×0.1 kg)
        # 3: fat_free_mass (×0.1 kg)
        # 4: soft_lean_mass (×0.1 kg)
        # 5: skeletal_muscle_mass (×0.1 kg)
        # 6: body_water (×0.1 kg)
        # 7: protein (×0.1 kg)
        # 8: mineral (×0.1 kg)
        # 9: bmi (×0.1)
        # 10: basal_metabolism (×1 kcal)
        # 11: body_cell_mass (×0.1 kg)
        # 12: obesity_degree (×0.1 %)

        metrics = [
            ("body_weight_kg", 0.1),
            ("body_fat_percent", 0.1),
            ("body_fat_mass_kg", 0.1),
            ("fat_free_mass_kg", 0.1),
            ("soft_lean_mass_kg", 0.1),
            ("skeletal_muscle_mass_kg", 0.1),
            ("body_water_kg", 0.1),
            ("protein_kg", 0.1),
            ("mineral_kg", 0.1),
            ("bmi", 0.1),
            ("basal_metabolism_kcal", 1.0),
            ("body_cell_mass_kg", 0.1),
            ("obesity_degree_percent", 0.1),
        ]

        for i, (name, factor) in enumerate(metrics):
            base = i * 3
            setattr(result, name, _scale(vals[base], factor))
            setattr(result, f"{name}_min", _scale(vals[base + 1], factor))
            setattr(result, f"{name}_max", _scale(vals[base + 2], factor))

    def _parse_packet2(self, result: Body270Result, payload: bytes) -> None:
        """Parse 20 uint16 segmental fat%, muscle mass, fat mass, lean mass."""
        vals = _unpack_u16_list(payload, 0, 20)

        # 5 segments × 4 groups = 20 values, all ×0.1
        # Group order: fat%, muscle_mass, fat_mass, lean_mass
        # Segment order: right_hand, left_hand, trunk, right_foot, left_foot
        segments = ["right_hand", "left_hand", "trunk", "right_foot", "left_foot"]
        groups = [
            ("fat_percent", 0.1),
            ("muscle_mass_kg", 0.1),
            ("fat_mass_kg", 0.1),
            ("lean_mass_kg", 0.1),
        ]

        idx = 0
        for group_name, factor in groups:
            for seg in segments:
                field_name = f"{seg}_{group_name}"
                setattr(result, field_name, _scale(vals[idx], factor))
                idx += 1

    def _parse_packet3(self, result: Body270Result, payload: bytes) -> None:
        """Parse evaluation fields from packet 3 (mixed types)."""
        # Layout (per protocol §6):
        #  [0]    body_score         : uint8
        #  [1]    physical_age       : uint8
        #  [2]    body_type          : uint8
        #  [3]    visceral_fat_level : uint8
        #  [4-5]  bmi_eval           : uint16 LE, ×0.1
        #  [6-7]  standard_weight    : uint16 LE, ×0.1 kg
        #  [8-9]  weight_control     : int16  LE, ×0.1 kg (signed)
        #  [10-11]fat_mass_control   : int16  LE, ×0.1 kg (signed)
        #  [12-13]muscle_mass_control: int16  LE, ×0.1 kg (signed)
        #  [14-15]target_fat_percent : uint16 LE, ×0.1 %
        #  [16]   fitness_score      : uint8
        #  (remaining bytes padding / reserved)

        if len(payload) < 17:
            raise ProtocolError(
                f"Packet 3 payload too short: {len(payload)} bytes, need >= 17"
            )

        result.body_score = payload[0]
        result.physical_age = payload[1]
        result.body_type = payload[2]
        result.visceral_fat_level = payload[3]

        (bmi_eval_raw,) = struct.unpack_from("<H", payload, 4)
        (standard_weight_raw,) = struct.unpack_from("<H", payload, 6)
        (weight_control_raw,) = struct.unpack_from("<h", payload, 8)
        (fat_mass_control_raw,) = struct.unpack_from("<h", payload, 10)
        (muscle_mass_control_raw,) = struct.unpack_from("<h", payload, 12)
        (target_fat_raw,) = struct.unpack_from("<H", payload, 14)

        result.bmi_eval = _scale(bmi_eval_raw, 0.1)
        result.standard_weight_kg = _scale(standard_weight_raw, 0.1)
        result.weight_control_kg = _scale(weight_control_raw, 0.1)
        result.fat_mass_control_kg = _scale(fat_mass_control_raw, 0.1)
        result.muscle_mass_control_kg = _scale(muscle_mass_control_raw, 0.1)
        result.target_fat_percent = _scale(target_fat_raw, 0.1)
        result.fitness_score = payload[16]

    def _parse_packet4(self, result: Body270Result, payload: bytes) -> None:
        """Parse 5 uint16 segmental muscle ratios (optional, firmware >= V1.3)."""
        vals = _unpack_u16_list(payload, 0, 5)
        result.right_hand_muscle_ratio = _scale(vals[0], 0.1)
        result.left_hand_muscle_ratio = _scale(vals[1], 0.1)
        result.trunk_muscle_ratio = _scale(vals[2], 0.1)
        result.right_foot_muscle_ratio = _scale(vals[3], 0.1)
        result.left_foot_muscle_ratio = _scale(vals[4], 0.1)
