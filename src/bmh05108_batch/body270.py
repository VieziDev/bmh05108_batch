"""
BMH05108 Body270 multi-packet response parser.

The 0xD0 command returns 3–5 response packets depending on firmware version:
  - V1.0–V1.2: 3 packets (no exercise consumption, no segment standards)
  - V1.3+:     5 packets

Each packet's data payload begins with a packet_info byte:
  - high nibble: total_packets (3, 4, or 5)
  - low nibble:  packet_index (1-based)

Packet payload sizes (bytes AFTER the packet_info byte, per protocol PDF §6):
  Packet 1:  74 bytes  (37 uint16 — 12 metrics×value/min/max + subcutaneous fat value)
  Packet 2:  40 bytes  (20 uint16 — segmental fat_mass, fat_rate, muscle_mass, muscle_ratio)
  Packet 3:  52 bytes  (mixed uint8/uint16/int16 — evaluation/recommendation fields)
  Packet 4:  16 bytes  (8 uint16 — exercise calorie consumption, optional)
  Packet 5:  10 bytes  (10 uint8 — segment fat/muscle standards, optional, V1.3+)
"""

from __future__ import annotations

import struct
from dataclasses import dataclass, field

from bmh05108_batch.protocol import ProtocolError

# Expected payload sizes (bytes after packet_info byte)
_PACKET1_SIZE = 74   # 37 × uint16
_PACKET2_SIZE = 40   # 20 × uint16
_PACKET3_SIZE = 52   # mixed types
_PACKET4_SIZE = 16   # 8 × uint16
_PACKET5_SIZE = 10   # 10 × uint8 (+ 6 reserve, minimum check is 10)


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

    moisture_kg: float | None = field(default=None)
    moisture_kg_min: float | None = field(default=None)
    moisture_kg_max: float | None = field(default=None)

    body_fat_mass_kg: float | None = field(default=None)
    body_fat_mass_kg_min: float | None = field(default=None)
    body_fat_mass_kg_max: float | None = field(default=None)

    protein_kg: float | None = field(default=None)
    protein_kg_min: float | None = field(default=None)
    protein_kg_max: float | None = field(default=None)

    mineral_kg: float | None = field(default=None)
    mineral_kg_min: float | None = field(default=None)
    mineral_kg_max: float | None = field(default=None)

    lean_body_weight_kg: float | None = field(default=None)
    lean_body_weight_kg_min: float | None = field(default=None)
    lean_body_weight_kg_max: float | None = field(default=None)

    muscle_mass_kg: float | None = field(default=None)
    muscle_mass_kg_min: float | None = field(default=None)
    muscle_mass_kg_max: float | None = field(default=None)

    bone_mass_kg: float | None = field(default=None)
    bone_mass_kg_min: float | None = field(default=None)
    bone_mass_kg_max: float | None = field(default=None)

    skeletal_muscle_mass_kg: float | None = field(default=None)
    skeletal_muscle_mass_kg_min: float | None = field(default=None)
    skeletal_muscle_mass_kg_max: float | None = field(default=None)

    intracellular_water_kg: float | None = field(default=None)
    intracellular_water_kg_min: float | None = field(default=None)
    intracellular_water_kg_max: float | None = field(default=None)

    extracellular_water_kg: float | None = field(default=None)
    extracellular_water_kg_min: float | None = field(default=None)
    extracellular_water_kg_max: float | None = field(default=None)

    body_cell_mass_kg: float | None = field(default=None)
    body_cell_mass_kg_min: float | None = field(default=None)
    body_cell_mass_kg_max: float | None = field(default=None)

    subcutaneous_fat_mass_kg: float | None = field(default=None)

    # --- Packet 2: segmental fat mass, fat rate, muscle mass, muscle ratio ---
    right_hand_fat_mass_kg: float | None = field(default=None)
    left_hand_fat_mass_kg: float | None = field(default=None)
    trunk_fat_mass_kg: float | None = field(default=None)
    right_foot_fat_mass_kg: float | None = field(default=None)
    left_foot_fat_mass_kg: float | None = field(default=None)

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

    right_hand_muscle_ratio: float | None = field(default=None)
    left_hand_muscle_ratio: float | None = field(default=None)
    trunk_muscle_ratio: float | None = field(default=None)
    right_foot_muscle_ratio: float | None = field(default=None)
    left_foot_muscle_ratio: float | None = field(default=None)

    # --- Packet 3: evaluation and recommendations ---
    body_score: int | None = field(default=None)
    physical_age: int | None = field(default=None)
    body_type: int | None = field(default=None)
    skeletal_muscle_mass_index: int | None = field(default=None)

    waist_hip_ratio: float | None = field(default=None)
    waist_hip_ratio_min: float | None = field(default=None)
    waist_hip_ratio_max: float | None = field(default=None)

    visceral_fat_level: int | None = field(default=None)
    visceral_fat_level_min: int | None = field(default=None)
    visceral_fat_level_max: int | None = field(default=None)

    obesity_percent: float | None = field(default=None)
    obesity_percent_min: float | None = field(default=None)
    obesity_percent_max: float | None = field(default=None)

    bmi: float | None = field(default=None)
    bmi_min: float | None = field(default=None)
    bmi_max: float | None = field(default=None)

    body_fat_percent: float | None = field(default=None)
    body_fat_percent_min: float | None = field(default=None)
    body_fat_percent_max: float | None = field(default=None)

    basal_metabolism_kcal: float | None = field(default=None)
    basal_metabolism_kcal_min: float | None = field(default=None)
    basal_metabolism_kcal_max: float | None = field(default=None)

    recommended_intake_kcal: float | None = field(default=None)
    ideal_weight_kg: float | None = field(default=None)
    target_weight_kg: float | None = field(default=None)

    weight_control_kg: float | None = field(default=None)
    muscle_control_kg: float | None = field(default=None)
    fat_control_kg: float | None = field(default=None)

    subcutaneous_fat_percent: float | None = field(default=None)
    subcutaneous_fat_percent_min: float | None = field(default=None)
    subcutaneous_fat_percent_max: float | None = field(default=None)

    # --- Packet 4: exercise calorie consumption (optional) ---
    exercise_walk_kcal: float | None = field(default=None)
    exercise_golf_kcal: float | None = field(default=None)
    exercise_croquet_kcal: float | None = field(default=None)
    exercise_tennis_kcal: float | None = field(default=None)
    exercise_squash_kcal: float | None = field(default=None)
    exercise_mountain_kcal: float | None = field(default=None)
    exercise_swim_kcal: float | None = field(default=None)
    exercise_badminton_kcal: float | None = field(default=None)

    # --- Packet 5: segment fat/muscle standards (optional, V1.3+) ---
    # Values: 0=low standard, 1=standard, 2=super standard
    fat_std_right_hand: int | None = field(default=None)
    fat_std_left_hand: int | None = field(default=None)
    fat_std_trunk: int | None = field(default=None)
    fat_std_right_foot: int | None = field(default=None)
    fat_std_left_foot: int | None = field(default=None)

    muscle_std_right_hand: int | None = field(default=None)
    muscle_std_left_hand: int | None = field(default=None)
    muscle_std_trunk: int | None = field(default=None)
    muscle_std_right_foot: int | None = field(default=None)
    muscle_std_left_foot: int | None = field(default=None)

    # --- Metadata ---
    device_error_type: int = field(default=0)

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
    return list(struct.unpack_from(f"<{count}H", data, offset))


def _scale(raw: int, factor: float) -> float:
    """Scale a raw integer value by `factor`."""
    return round(raw * factor, 4)


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------


class Body270Parser:
    """Stateful accumulator for Body270 multi-packet responses.

    Call feed_packet() for each received packet payload (packet_info byte first).
    When complete is True, call parse() to obtain a Body270Result.
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

        if total not in (3, 4, 5):
            raise ProtocolError(
                f"Unexpected total_packets={total} (expected 3, 4, or 5)"
            )
        if index < 1 or index > total:
            raise ProtocolError(f"Packet index {index} out of range [1, {total}]")

        payload = data[1:]  # strip packet_info byte

        expected: dict[int, int] = {
            1: _PACKET1_SIZE,
            2: _PACKET2_SIZE,
            3: _PACKET3_SIZE,
        }
        if total >= 4:
            expected[4] = _PACKET4_SIZE
        if total >= 5:
            expected[5] = _PACKET5_SIZE

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
        if 5 in self._packets:
            self._parse_packet5(result, self._packets[5])
        return result

    def reset(self) -> None:
        """Reset internal state so the parser can be reused for the next measurement."""
        self._packets = {}
        self._total = None

    # -----------------------------------------------------------------------
    # Packet parsers
    # -----------------------------------------------------------------------

    def _parse_packet1(self, result: Body270Result, payload: bytes) -> None:
        """Parse 37 uint16 whole-body composition values.

        Layout (per protocol §6, packet 1):
          12 metrics × (value, min, max) = 36 uint16
          1 final metric (subcutaneous_fat_mass_kg, value only) = 1 uint16
        """
        vals = _unpack_u16_list(payload, 0, 37)

        metrics = [
            ("body_weight_kg", 0.1),
            ("moisture_kg", 0.1),
            ("body_fat_mass_kg", 0.1),
            ("protein_kg", 0.1),
            ("mineral_kg", 0.1),
            ("lean_body_weight_kg", 0.1),
            ("muscle_mass_kg", 0.1),
            ("bone_mass_kg", 0.1),
            ("skeletal_muscle_mass_kg", 0.1),
            ("intracellular_water_kg", 0.1),
            ("extracellular_water_kg", 0.1),
            ("body_cell_mass_kg", 0.1),
        ]

        for i, (name, factor) in enumerate(metrics):
            base = i * 3
            setattr(result, name, _scale(vals[base], factor))
            setattr(result, f"{name}_min", _scale(vals[base + 1], factor))
            setattr(result, f"{name}_max", _scale(vals[base + 2], factor))

        result.subcutaneous_fat_mass_kg = _scale(vals[36], 0.1)

    def _parse_packet2(self, result: Body270Result, payload: bytes) -> None:
        """Parse 20 uint16 segmental values.

        Layout (per protocol §6, packet 2):
          Group 1 (vals 0–4):   segmental fat mass (×0.1 kg)
          Group 2 (vals 5–9):   segmental fat rate (×0.1 %)
          Group 3 (vals 10–14): segmental muscle mass (×0.1 kg)
          Group 4 (vals 15–19): segmental muscle ratio (×0.1 %, V1.3+)

          Segment order: right_hand, left_hand, trunk, right_foot, left_foot
        """
        vals = _unpack_u16_list(payload, 0, 20)

        segments = ["right_hand", "left_hand", "trunk", "right_foot", "left_foot"]
        groups = [
            ("fat_mass_kg", 0.1),
            ("fat_percent", 0.1),
            ("muscle_mass_kg", 0.1),
            ("muscle_ratio", 0.1),
        ]

        for g_idx, (group_name, factor) in enumerate(groups):
            for s_idx, seg in enumerate(segments):
                setattr(result, f"{seg}_{group_name}", _scale(vals[g_idx * 5 + s_idx], factor))

    def _parse_packet3(self, result: Body270Result, payload: bytes) -> None:
        """Parse evaluation and recommendation fields from packet 3.

        Layout (per protocol §6, packet 3, absolute frame bytes 5–56):
          [0]     body_score              uint8
          [1]     physical_age            uint8
          [2]     body_type               uint8
          [3]     skeletal_muscle_mass_index uint8
          [4]     waist_hip_ratio         uint8  ×0.01
          [5]     waist_hip_ratio_min     uint8  ×0.01
          [6]     waist_hip_ratio_max     uint8  ×0.01
          [7]     visceral_fat_level      uint8
          [8]     visceral_fat_level_min  uint8
          [9]     visceral_fat_level_max  uint8
          [10–11] obesity_percent         uint16 ×0.1
          [12–13] obesity_percent_min     uint16 ×0.1
          [14–15] obesity_percent_max     uint16 ×0.1
          [16–17] bmi                     uint16 ×0.1
          [18–19] bmi_min                 uint16 ×0.1
          [20–21] bmi_max                 uint16 ×0.1
          [22–23] body_fat_percent        uint16 ×0.1
          [24–25] body_fat_percent_min    uint16 ×0.1
          [26–27] body_fat_percent_max    uint16 ×0.1
          [28–29] basal_metabolism_kcal   uint16 ×1
          [30–31] basal_metabolism_min    uint16 ×1
          [32–33] basal_metabolism_max    uint16 ×1
          [34–35] recommended_intake_kcal uint16 ×1
          [36–37] ideal_weight_kg         uint16 ×0.1
          [38–39] target_weight_kg        uint16 ×0.1
          [40–41] weight_control_kg       int16  ×0.1
          [42–43] muscle_control_kg       int16  ×0.1
          [44–45] fat_control_kg          int16  ×0.1
          [46–47] subcutaneous_fat_percent     uint16 ×0.1
          [48–49] subcutaneous_fat_percent_min uint16 ×0.1
          [50–51] subcutaneous_fat_percent_max uint16 ×0.1
        """
        if len(payload) < _PACKET3_SIZE:
            raise ProtocolError(
                f"Packet 3 payload too short: {len(payload)} bytes, need >= {_PACKET3_SIZE}"
            )

        (result.body_score,
         result.physical_age,
         result.body_type,
         result.skeletal_muscle_mass_index,
         whr_raw, whr_min_raw, whr_max_raw,
         vfl_raw, vfl_min_raw, vfl_max_raw) = struct.unpack_from("<10B", payload, 0)

        result.waist_hip_ratio = _scale(whr_raw, 0.01)
        result.waist_hip_ratio_min = _scale(whr_min_raw, 0.01)
        result.waist_hip_ratio_max = _scale(whr_max_raw, 0.01)
        result.visceral_fat_level = vfl_raw
        result.visceral_fat_level_min = vfl_min_raw
        result.visceral_fat_level_max = vfl_max_raw

        (obes_raw, obes_min_raw, obes_max_raw,
         bmi_raw, bmi_min_raw, bmi_max_raw,
         bfp_raw, bfp_min_raw, bfp_max_raw,
         bm_raw, bm_min_raw, bm_max_raw,
         rec_raw, ideal_raw, target_raw) = struct.unpack_from("<15H", payload, 10)

        result.obesity_percent = _scale(obes_raw, 0.1)
        result.obesity_percent_min = _scale(obes_min_raw, 0.1)
        result.obesity_percent_max = _scale(obes_max_raw, 0.1)
        result.bmi = _scale(bmi_raw, 0.1)
        result.bmi_min = _scale(bmi_min_raw, 0.1)
        result.bmi_max = _scale(bmi_max_raw, 0.1)
        result.body_fat_percent = _scale(bfp_raw, 0.1)
        result.body_fat_percent_min = _scale(bfp_min_raw, 0.1)
        result.body_fat_percent_max = _scale(bfp_max_raw, 0.1)
        result.basal_metabolism_kcal = float(bm_raw)
        result.basal_metabolism_kcal_min = float(bm_min_raw)
        result.basal_metabolism_kcal_max = float(bm_max_raw)
        result.recommended_intake_kcal = float(rec_raw)
        result.ideal_weight_kg = _scale(ideal_raw, 0.1)
        result.target_weight_kg = _scale(target_raw, 0.1)

        (wc_raw, mc_raw, fc_raw) = struct.unpack_from("<3h", payload, 40)
        result.weight_control_kg = _scale(wc_raw, 0.1)
        result.muscle_control_kg = _scale(mc_raw, 0.1)
        result.fat_control_kg = _scale(fc_raw, 0.1)

        (sfp_raw, sfp_min_raw, sfp_max_raw) = struct.unpack_from("<3H", payload, 46)
        result.subcutaneous_fat_percent = _scale(sfp_raw, 0.1)
        result.subcutaneous_fat_percent_min = _scale(sfp_min_raw, 0.1)
        result.subcutaneous_fat_percent_max = _scale(sfp_max_raw, 0.1)

    def _parse_packet4(self, result: Body270Result, payload: bytes) -> None:
        """Parse 8 uint16 exercise calorie consumption values (kCal per 30 min)."""
        if len(payload) < _PACKET4_SIZE:
            raise ProtocolError(
                f"Packet 4 payload too short: {len(payload)} bytes, need >= {_PACKET4_SIZE}"
            )
        vals = list(struct.unpack_from("<8H", payload, 0))
        result.exercise_walk_kcal = float(vals[0])
        result.exercise_golf_kcal = float(vals[1])
        result.exercise_croquet_kcal = float(vals[2])
        result.exercise_tennis_kcal = float(vals[3])
        result.exercise_squash_kcal = float(vals[4])
        result.exercise_mountain_kcal = float(vals[5])
        result.exercise_swim_kcal = float(vals[6])
        result.exercise_badminton_kcal = float(vals[7])

    def _parse_packet5(self, result: Body270Result, payload: bytes) -> None:
        """Parse 10 uint8 segment fat/muscle standard ratings (V1.3+).

        Values: 0=low standard, 1=standard, 2=super standard.
        Segment order: right_hand, left_hand, trunk, right_foot, left_foot.
        """
        if len(payload) < _PACKET5_SIZE:
            raise ProtocolError(
                f"Packet 5 payload too short: {len(payload)} bytes, need >= {_PACKET5_SIZE}"
            )
        segs = ["right_hand", "left_hand", "trunk", "right_foot", "left_foot"]
        for i, seg in enumerate(segs):
            setattr(result, f"fat_std_{seg}", payload[i])
            setattr(result, f"muscle_std_{seg}", payload[5 + i])
