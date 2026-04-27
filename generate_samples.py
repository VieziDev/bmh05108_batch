#!/usr/bin/env python3
"""
generate_samples.py — Simulated bioimpedance dataset for BMH05108 Body270 batch runner.

Generation strategy:
  75% realistic — literature-calibrated Normal distributions (Kyle et al. 2001, Clin Nutr,
    at 50kHz scaled to 20kHz ×1.15) broken down by sex × age group, corrected per individual
    for fat% deviation from the group mean and for hydration decline with age.
  25% space-filling — Latin Hypercube Sampling across the full hardware range, so the
    downstream regression sees the algorithm's behavior at the extremes (not just the
    realistic population cluster).

All rows satisfy hardware range constraints and physics invariants (beta dispersion,
trunk < limbs, arm > leg, L/R symmetry) enforced by construction, not rejection sampling.

Usage:
    uv run generate_samples.py [--output samples.csv] [--n 100000] [--seed 42]

Dependencies: numpy, pandas
"""

# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy>=2.0", "pandas>=2.2"]
# ///

import argparse
import time
import sys
import numpy as np
import pandas as pd

# ─── Hardware limits ──────────────────────────────────────────────────────────
AGE_MIN,    AGE_MAX     = 6,     99
HEIGHT_MIN, HEIGHT_MAX  = 90.0,  220.0
WEIGHT_MIN, WEIGHT_MAX  = 10.0,  200.0
LIMB_Z_MIN, LIMB_Z_MAX  = 100.0, 600.0
TRUNK_Z_MIN, TRUNK_Z_MAX = 10.0, 100.0

# Asymmetry clip (derived from invariant thresholds)
ARM_ASYM_CLIP = 0.07    # |rh−lh|/mean < 15%  → |asym| < 0.075
LEG_ASYM_CLIP = 0.038   # |rf−lf|/mean < 8%   → |asym| < 0.040

# Beta dispersion: Z_100k = Z_20k × factor, factor strictly < 1
DISP_ARM_LO,   DISP_ARM_HI   = 0.82, 0.92
DISP_LEG_LO,   DISP_LEG_HI   = 0.83, 0.93
DISP_TRUNK_LO, DISP_TRUNK_HI = 0.80, 0.90

# Structural floors — derived from invariants:
#   _LIMB_BASE_MIN   = LIMB_Z_MIN / DISP_ARM_LO + 0.1 = 122.0
#     → ensures limb_100k ≥ 100 after dispersion, no lower-bound floor collision
#   _LIMB_BASE_MAX_LEG = LIMB_Z_MAX / _ARM_LEG_GAP = 600/1.125 = 533 → use 532
#     → ensures arm_base × (1−ARM_ASYM) > leg_base × (1+LEG_ASYM) when arm_base ≤ LIMB_Z_MAX
#   _TRUNK_BASE_MIN  = TRUNK_Z_MIN / DISP_TRUNK_LO + 0.2 = 12.7
#     → ensures trunk_100k ≥ 10 + ε after dispersion
_LIMB_BASE_MIN    = 122.0
_LIMB_BASE_MAX_LEG = 532.0
_TRUNK_BASE_MIN   = 12.7
_ARM_LEG_GAP      = 1.125   # arm_base ≥ leg_base × 1.125 → arm_20k > leg_20k under worst asym

# LHS fraction of total samples
LHS_FRACTION = 0.25

# ─── Literature-calibrated 20kHz parameters ──────────────────────────────────
# Kyle et al. 2001 (Clin Nutr) — healthy European adults at 50kHz, converted to 20kHz (×1.15).
# Children: +15% vs. adults (higher Z per height, lower muscle fraction — Baumgartner 1991).
# Elderly:  +15% vs. adults (sarcopenia + mild chronic dehydration — Roubenoff 1997).
#
# Array index = age_group × 2 + gender:
#   0=child_F  1=child_M  2=adult_F  3=adult_M  4=elderly_F  5=elderly_M

_ARM_MU    = np.array([505., 415., 440., 360., 505., 415.])   # Ω at 20kHz
_ARM_SIG   = np.array([ 55.,  50.,  50.,  45.,  58.,  52.])
_LEG_MU    = np.array([368., 305., 320., 265., 368., 305.])
_LEG_SIG   = np.array([ 50.,  48.,  45.,  38.,  52.,  50.])
_TRK_MU    = np.array([ 41.,  32.,  36.,  28.,  41.,  32.])
_TRK_SIG   = np.array([ 10.,   8.,   9.,   7.,  10.,   8.])

# Group-median fat% (Deurenberg equation at each group's typical BMI + age).
# Used to compute individual fat-deviation correction.
_FAT_REF = np.array([22., 15., 33., 23., 41., 33.])  # % : child_F…elderly_M

# Impedance correction per 1% fat deviation from group median.
# Literature: ~0.5–0.8% Z per 1% fat (fat is highly resistive; muscle is conductive).
# Trunk effect is weaker (larger cross-section partially compensates higher ρ_fat).
K_FAT_LIMB  = 0.006
K_FAT_TRUNK = 0.003

# After fat% and hydration account for part of variance, the residual sigma shrinks.
# Coefficient from: residual = sqrt(1 − r²) where r ≈ 0.45 (fat% explains ~20% of Z variance)
_RESIDUAL_SIG = 0.89


# ─── Stage 1: Demographics ────────────────────────────────────────────────────

def _sample_demographics(n: int, rng: np.random.Generator):
    # Age group: 0=children 6–17 (10%), 1=adults 18–65 (70%), 2=elderly 66–99 (20%)
    age_group = rng.choice(3, size=n, p=[0.10, 0.70, 0.20])

    age_c = rng.normal(12,  3, size=n).clip(AGE_MIN, 17)
    age_a = rng.normal(38, 12, size=n).clip(18, 65)
    age_e = rng.normal(73,  6, size=n).clip(66, AGE_MAX)
    age = np.where(age_group == 0, age_c,
          np.where(age_group == 2, age_e, age_a)).round().astype(int)

    gender = rng.integers(0, 2, size=n)  # 0=F, 1=M

    # Height: Normal(mu, sigma) by sex × age group
    # [child_F, child_M, adult_F, adult_M, elderly_F, elderly_M]
    h_mu  = np.array([130., 132., 163., 175., 160., 171.])
    h_sig = np.array([ 12.,  12.,   7.,   8.,   7.,   8.])
    idx   = age_group * 2 + gender
    height = (rng.normal(size=n) * h_sig[idx] + h_mu[idx]).clip(HEIGHT_MIN, HEIGHT_MAX)

    # BMI → weight (guarantees physiologically plausible weight/height combos)
    bmi_mu  = np.array([17.5, 25.5, 26.5])
    bmi_sig = np.array([ 2.5,  4.5,  4.0])
    bmi_lo  = np.array([13.0, 14.0, 15.0])
    bmi_hi  = np.array([28.0, 45.0, 40.0])
    bmi_raw = rng.normal(size=n) * bmi_sig[age_group] + bmi_mu[age_group]
    bmi     = np.clip(bmi_raw, bmi_lo[age_group], bmi_hi[age_group])
    weight  = (bmi * (height / 100.0) ** 2).clip(WEIGHT_MIN, WEIGHT_MAX)

    return age, gender, height, weight, bmi, idx   # idx = age_group*2+gender (0–5)


# ─── Stage 2: Body composition ────────────────────────────────────────────────

def _body_composition(age, gender, bmi):
    # Deurenberg et al. 1991 simplified fat% estimate
    fat_pct   = (1.20 * bmi + 0.23 * age - 10.8 * gender - 5.4).clip(5.0, 50.0)
    # Hydration declines slightly with age past 30 → impedance rises
    hydration = (1.0 - 0.003 * np.maximum(0.0, age - 30.0)).clip(0.88, 1.0)
    return fat_pct, hydration


# ─── Stage 3a: Realistic impedances ──────────────────────────────────────────

def _realistic(n: int, rng: np.random.Generator, idx, fat_pct, hydration):
    """
    Generate base 20kHz impedances from literature-calibrated distributions.
    Each individual is shifted from the group mean by their fat% deviation and
    hydration correction; the residual sigma captures geometry, hydration noise,
    and measurement variation not explained by body composition.
    """
    fat_delta      = fat_pct - _FAT_REF[idx]
    corr_limb      = (1.0 + K_FAT_LIMB  * fat_delta).clip(0.80, 1.30)
    corr_trunk     = (1.0 + K_FAT_TRUNK * fat_delta).clip(0.88, 1.15)
    hyd_factor     = 1.0 / hydration   # dehydration → higher Z

    arm_base = (
        _ARM_MU[idx] * corr_limb * hyd_factor
        + rng.normal(0.0, _ARM_SIG[idx] * _RESIDUAL_SIG, size=n)
    )
    leg_base = (
        _LEG_MU[idx] * corr_limb * hyd_factor
        + rng.normal(0.0, _LEG_SIG[idx] * _RESIDUAL_SIG, size=n)
    )
    trunk_base = (
        _TRK_MU[idx] * corr_trunk * hyd_factor
        + rng.normal(0.0, _TRK_SIG[idx] * _RESIDUAL_SIG, size=n)
    )
    return arm_base, leg_base, trunk_base


# ─── Stage 3b: Latin Hypercube impedances ─────────────────────────────────────

def _lhs(n: int, rng: np.random.Generator, lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    """
    Latin Hypercube Sample: returns (n, d) array with uniform marginals over [lo[j], hi[j]].
    Each dimension is independently stratified into n equal-probability strata;
    one sample is drawn per stratum and strata are randomly permuted.
    """
    d    = len(lo)
    cuts = np.arange(n, dtype=float) / n   # left edges of n equal strata in [0, 1]
    out  = np.empty((n, d))
    for j in range(d):
        u       = cuts + rng.uniform(0.0, 1.0 / n, size=n)  # one sample per stratum
        out[:, j] = lo[j] + rng.permutation(u) * (hi[j] - lo[j])
    return out


def _lhs_impedances(n_lhs: int, rng: np.random.Generator):
    """
    Quasi-uniform coverage of the hardware range.
    arm_base sampled over full limb range; leg_base limited to _LIMB_BASE_MAX_LEG
    (the arm-leg gap enforcement will bump arm above legs wherever needed).
    """
    lo = np.array([_LIMB_BASE_MIN,    _LIMB_BASE_MIN,   _TRUNK_BASE_MIN])
    hi = np.array([LIMB_Z_MAX,        _LIMB_BASE_MAX_LEG, TRUNK_Z_MAX])
    s  = _lhs(n_lhs, rng, lo, hi)
    return s[:, 0], s[:, 1], s[:, 2]   # arm_base, leg_base, trunk_base


# ─── Stage 4: Physics constraints (shared by both paths) ─────────────────────

def _apply_physics(n: int, rng: np.random.Generator,
                   arm_base, leg_base, trunk_base):
    """
    1. Clip bases to structural bounds.
    2. Guarantee arm_base ≥ leg_base × _ARM_LEG_GAP (arm Z > leg Z — thinner cross-section).
    3. Apply symmetric L/R asymmetry, clipped to invariant thresholds.
    4. Clip 20kHz to hardware limits BEFORE deriving 100kHz
       (prevents both from landing on the same bound when the raw value exceeds the limit).
    5. Derive 100kHz from clipped 20kHz via beta-dispersion factors < 1.
    6. Clip 100kHz.
    7. Enforce trunk < min(limbs) at both frequencies.
    """
    # Step 1 — structural clips
    leg_base   = leg_base.clip(_LIMB_BASE_MIN, _LIMB_BASE_MAX_LEG)
    arm_base   = arm_base.clip(_LIMB_BASE_MIN, LIMB_Z_MAX)
    trunk_base = trunk_base.clip(_TRUNK_BASE_MIN, TRUNK_Z_MAX)

    # Step 2 — arm > leg gap
    arm_base = np.maximum(arm_base, leg_base * _ARM_LEG_GAP)
    arm_base = arm_base.clip(_LIMB_BASE_MIN, LIMB_Z_MAX)

    # Step 3 — L/R asymmetry
    asym_arm = np.clip(rng.normal(0, 0.04,  size=n), -ARM_ASYM_CLIP, ARM_ASYM_CLIP)
    asym_leg = np.clip(rng.normal(0, 0.025, size=n), -LEG_ASYM_CLIP, LEG_ASYM_CLIP)

    rh_20k    = arm_base * (1.0 + asym_arm)
    lh_20k    = arm_base * (1.0 - asym_arm)
    rf_20k    = leg_base * (1.0 + asym_leg)
    lf_20k    = leg_base * (1.0 - asym_leg)
    trunk_20k = trunk_base.copy()

    # Step 4 — clip 20kHz first
    for arr in (rh_20k, lh_20k, rf_20k, lf_20k):
        np.clip(arr, LIMB_Z_MIN, LIMB_Z_MAX, out=arr)
    np.clip(trunk_20k, TRUNK_Z_MIN, TRUNK_Z_MAX, out=trunk_20k)

    # Step 5 — beta dispersion (from clipped 20kHz so upper bounds can't coincide)
    disp_arm   = rng.uniform(DISP_ARM_LO,   DISP_ARM_HI,   size=n)
    disp_leg   = rng.uniform(DISP_LEG_LO,   DISP_LEG_HI,   size=n)
    disp_trunk = rng.uniform(DISP_TRUNK_LO, DISP_TRUNK_HI, size=n)

    rh_100k    = rh_20k    * disp_arm
    lh_100k    = lh_20k    * disp_arm
    rf_100k    = rf_20k    * disp_leg
    lf_100k    = lf_20k    * disp_leg
    trunk_100k = trunk_20k * disp_trunk

    # Step 6 — clip 100kHz
    for arr in (rh_100k, lh_100k, rf_100k, lf_100k):
        np.clip(arr, LIMB_Z_MIN, LIMB_Z_MAX, out=arr)
    np.clip(trunk_100k, TRUNK_Z_MIN, TRUNK_Z_MAX, out=trunk_100k)

    # Step 7 — trunk < min(limbs)
    min_l20  = np.minimum.reduce([rh_20k, lh_20k, rf_20k, lf_20k])
    min_l100 = np.minimum.reduce([rh_100k, lh_100k, rf_100k, lf_100k])
    np.minimum(trunk_20k,  min_l20  * 0.99, out=trunk_20k)
    np.minimum(trunk_100k, min_l100 * 0.99, out=trunk_100k)
    np.clip(trunk_20k,  TRUNK_Z_MIN, TRUNK_Z_MAX, out=trunk_20k)
    np.clip(trunk_100k, TRUNK_Z_MIN, TRUNK_Z_MAX, out=trunk_100k)

    return (rh_20k, lh_20k, trunk_20k, rf_20k, lf_20k,
            rh_100k, lh_100k, trunk_100k, rf_100k, lf_100k)


# ─── Stats ────────────────────────────────────────────────────────────────────

def _print_stats(df: pd.DataFrame, t_elapsed: float, seed: int,
                 output: str, n_real: int, n_lhs: int):
    W = 76
    print("=" * W)
    print("Generation complete")
    print(f"  Total rows     : {len(df):,}")
    print(f"  Realistic (75%): {n_real:,}  — literature-calibrated, fat%/hydration-corrected")
    print(f"  LHS (25%)      : {n_lhs:,}  — Latin Hypercube over full hardware range")
    print(f"  Seed           : {seed}")
    print(f"  Output         : {output}")
    print(f"  Elapsed        : {t_elapsed:.2f}s")
    print("-" * W)
    print("Per-column statistics (all rows):")
    print(f"  {'Column':<16} {'Min':>8} {'Max':>8} {'Mean':>8} {'Std':>8} {'p5':>8} {'p95':>8}")
    for col in df.columns:
        s   = df[col].to_numpy(dtype=float)
        p5, p95 = np.percentile(s, [5, 95])
        print(f"  {col:<16} {s.min():>8.1f} {s.max():>8.1f} "
              f"{s.mean():>8.1f} {s.std():>8.1f} {p5:>8.1f} {p95:>8.1f}")
    print("-" * W)

    limbs_20k  = df[["rh_20k",  "lh_20k",  "rf_20k",  "lf_20k"]]
    limbs_100k = df[["rh_100k", "lh_100k", "rf_100k", "lf_100k"]]
    mean_arm   = (df["rh_20k"] + df["lh_20k"]) / 2.0
    mean_leg   = (df["rf_20k"] + df["lf_20k"]) / 2.0

    hw_checks = [
        ("age ∉ [6, 99]",           (df["age"] < 6) | (df["age"] > 99)),
        ("height_cm ∉ [90, 220]",   (df["height_cm"] < 90) | (df["height_cm"] > 220)),
        ("weight_kg ∉ [10, 200]",   (df["weight_kg"] < 10) | (df["weight_kg"] > 200)),
        ("limb_20k ∉ [100, 600]",   (limbs_20k < 100).any(axis=1)  | (limbs_20k > 600).any(axis=1)),
        ("limb_100k ∉ [100, 600]",  (limbs_100k < 100).any(axis=1) | (limbs_100k > 600).any(axis=1)),
        ("trunk_20k ∉ [10, 100]",   (df["trunk_20k"]  < 10) | (df["trunk_20k"]  > 100)),
        ("trunk_100k ∉ [10, 100]",  (df["trunk_100k"] < 10) | (df["trunk_100k"] > 100)),
    ]
    phy_checks = [
        ("trunk_20k >= min(limb_20k)",   df["trunk_20k"]  >= limbs_20k.min(axis=1)),
        ("trunk_100k >= min(limb_100k)", df["trunk_100k"] >= limbs_100k.min(axis=1)),
        ("rh_20k <= rh_100k",            df["rh_20k"]  <= df["rh_100k"]),
        ("lh_20k <= lh_100k",            df["lh_20k"]  <= df["lh_100k"]),
        ("rf_20k <= rf_100k",            df["rf_20k"]  <= df["rf_100k"]),
        ("lf_20k <= lf_100k",            df["lf_20k"]  <= df["lf_100k"]),
        ("trunk_20k <= trunk_100k",      df["trunk_20k"] <= df["trunk_100k"]),
        ("|rh−lh|/mean_arm >= 15%",      (df["rh_20k"] - df["lh_20k"]).abs() / mean_arm >= 0.15),
        ("|rf−lf|/mean_leg >= 8%",       (df["rf_20k"] - df["lf_20k"]).abs() / mean_leg >= 0.08),
        ("rh_20k <= rf_20k (arm≤leg)",   df["rh_20k"] <= df["rf_20k"]),
        ("lh_20k <= lf_20k (arm≤leg)",   df["lh_20k"] <= df["lf_20k"]),
    ]

    any_fail = False
    print("Hardware range violations (target: all 0):")
    for name, mask in hw_checks:
        n_v  = int(mask.sum())
        flag = "!! " if n_v else "   "
        if n_v: any_fail = True
        print(f"  {flag}{name:<40}: {n_v}")
    print("Physics invariant violations (target: all 0):")
    for name, mask in phy_checks:
        n_v  = int(mask.sum())
        flag = "!! " if n_v else "   "
        if n_v: any_fail = True
        print(f"  {flag}{name:<40}: {n_v}")
    print("=" * W)
    if any_fail:
        print("ERROR: violations found — review tuning constants.", file=sys.stderr)
        sys.exit(1)


# ─── Entry point ──────────────────────────────────────────────────────────────

def generate(n: int, seed: int, output: str):
    t0  = time.perf_counter()
    rng = np.random.default_rng(seed)

    n_lhs  = int(n * LHS_FRACTION)
    n_real = n - n_lhs

    print(f"Generating {n:,} rows (seed={seed}): "
          f"{n_real:,} realistic + {n_lhs:,} LHS...")

    # ── Realistic portion ──────────────────────────────────────────────────────
    age_r, gender_r, h_r, w_r, bmi_r, idx_r = _sample_demographics(n_real, rng)
    fat_r, hyd_r                             = _body_composition(age_r, gender_r, bmi_r)
    arm_r, leg_r, trk_r                      = _realistic(n_real, rng, idx_r, fat_r, hyd_r)
    imp_r = _apply_physics(n_real, rng, arm_r, leg_r, trk_r)

    # ── LHS portion ────────────────────────────────────────────────────────────
    age_l, gender_l, h_l, w_l, bmi_l, idx_l = _sample_demographics(n_lhs, rng)
    arm_l, leg_l, trk_l                      = _lhs_impedances(n_lhs, rng)
    imp_l = _apply_physics(n_lhs, rng, arm_l, leg_l, trk_l)

    # ── Combine and shuffle ────────────────────────────────────────────────────
    c  = np.concatenate
    p  = rng.permutation(n)

    rh_20, lh_20, trk_20, rf_20, lf_20, rh_100, lh_100, trk_100, rf_100, lf_100 = (
        c([a, b])[p] for a, b in zip(imp_r, imp_l)
    )

    df = pd.DataFrame({
        "row_id":     np.arange(n, dtype=np.int32),
        "gender":     c([gender_r,  gender_l])[p].astype(np.int8),
        "age":        c([age_r,     age_l])[p],
        "height_cm":  c([h_r,       h_l])[p].round(1),
        "weight_kg":  c([w_r,       w_l])[p].round(1),
        "rh_20k":     rh_20.round(1),
        "lh_20k":     lh_20.round(1),
        "trunk_20k":  trk_20.round(1),
        "rf_20k":     rf_20.round(1),
        "lf_20k":     lf_20.round(1),
        "rh_100k":    rh_100.round(1),
        "lh_100k":    lh_100.round(1),
        "trunk_100k": trk_100.round(1),
        "rf_100k":    rf_100.round(1),
        "lf_100k":    lf_100.round(1),
    })

    print(f"Writing {output}...")
    df.to_csv(output, index=False)
    _print_stats(df, time.perf_counter() - t0, seed, output, n_real, n_lhs)


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Generate physiologically calibrated bioimpedance dataset "
            "for BMH05108 Body270 batch runner."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--output", default="samples.csv", metavar="PATH")
    parser.add_argument("--n",      default=100_000, type=int,  metavar="N")
    parser.add_argument("--seed",   default=42,      type=int)
    args = parser.parse_args()
    generate(n=args.n, seed=args.seed, output=args.output)


if __name__ == "__main__":
    main()
