"""
Spine, registration, and profile table generation.

Demographic calibration sources
--------------------------------
Age / gender / education distribution among U.S. adult smokers:
  CDC NHANES 2019–2020 Public-Use Data File, Smoking & Tobacco Use (SMQ).
  Available: https://wwwn.cdc.gov/nchs/nhanes/2019-2020/SMQ_P.XPT
  Key statistics: ~54% male; median age ~40; higher education → lower prevalence.

Smoking burden:
  CDC MMWR 2020 — Tobacco Product Use Among Adults, United States, 2019.
  Cigarettes/day mean ~14 (SD≈10); years smoking mean ~20 (SD≈12).

TTM stage distribution (cessation-motivated recruits):
  Prochaska JO et al. (1985). Predicting change in smoking status for
  self-changers. Addict Behav, 10(4), 395–406.
  Approximate recruitment split: 30% precontemplation, 38% contemplation,
  22% preparation; 10% in action/maintenance at enrollment.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from cessation.config import (
    PROFILE_CLASSES,
    TTM_STAGES,
)

# ── stage probabilities at study enrolment (Prochaska 1985 proportions) ───────
_STAGE_PROBS = np.array([0.30, 0.38, 0.22, 0.07, 0.03])

# ── flag fractions relative to total spine ────────────────────────────────────
_FRAC = {
    "app_active":    0.95,
    "survey":        0.060,   # pre-quit psychometric cohort (~480/8000)
    "followup":      0.150,   # 12-month follow-up cohort (~1200)
    "registration":  0.225,   # completed registration (~1800)
    "sms":           0.550,   # received ≥1 SMS (~4400)
    "quiz":          0.059,   # took ≥1 quiz (~470)
    "reengagement":  0.238,   # in reengagement campaign (~1900)
}


def generate_spine(
    n: int = 8_000,
    seed: int = 42,
    rng: np.random.Generator | None = None,
) -> pd.DataFrame:
    if rng is None:
        rng = np.random.default_rng(seed)
    pids = np.arange(1, n + 1)
    flags = {k: rng.random(n) < v for k, v in _FRAC.items()}
    # survey ⊂ registration ⊂ app_active
    flags["registration"] |= flags["survey"]
    flags["app_active"] |= flags["registration"]
    df = pd.DataFrame({"pid": pids, **flags})
    return df


def generate_registration(
    spine: pd.DataFrame,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Registration data for users with registration=True."""
    sub = spine[spine["registration"]].copy()
    n = len(sub)

    # Demographics (CDC NHANES 2019–2020)
    mod_age    = rng.normal(40.0, 11.0, n).clip(18, 72).astype(int)
    mod_gender = rng.choice([1, 2, 3], size=n, p=[0.54, 0.44, 0.02])
    mod_edu    = rng.choice([1, 2, 3, 4], size=n, p=[0.15, 0.32, 0.30, 0.23])
    mod_cpd    = rng.gamma(shape=2.0, scale=7.0, size=n).clip(1, 60).round(0)
    mod_yrs    = rng.normal(20.0, 12.0, n).clip(1, 50).round(1)

    # TTM stage at registration
    stage_idx = rng.choice(len(TTM_STAGES), size=n, p=_STAGE_PROBS)
    mod_stage  = np.array(TTM_STAGES)[stage_idx]

    # Readiness to quit: 1–5 Likert, correlated with stage
    # precontemplation ~1.5, contemplation ~2.5, preparation ~3.5, action/maint ~4.5
    stage_readiness = np.array([1.5, 2.5, 3.5, 4.5, 4.8])[stage_idx]
    mod_readiness = (rng.normal(stage_readiness, 0.7)).clip(1, 5).round(0).astype(int)

    # Quit date offset: days from registration to declared quit attempt
    # Only applicable for action/maintenance stages at enrolment; ~40% set a date
    has_quit_date = (stage_idx >= 3) | (rng.random(n) < 0.35)
    quit_date_raw = rng.choice(np.arange(0, 91), size=n)
    reg_quit_date_offset = np.where(has_quit_date, quit_date_raw, np.nan)

    df = pd.DataFrame({
        "pid":                  sub["pid"].values,
        "mod_age":              mod_age,
        "mod_gender":           mod_gender,
        "mod_edu":              mod_edu,
        "mod_cpd":              mod_cpd,
        "mod_yrs_smk":         mod_yrs,
        "mod_stage":            mod_stage,
        "mod_readiness":        mod_readiness,
        "reg_quit_date_offset": reg_quit_date_offset,
    })
    return df


def generate_profile(
    spine: pd.DataFrame,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Profile data for all spine rows."""
    n = len(spine)
    prf_class  = rng.choice(PROFILE_CLASSES, size=n, p=[0.25, 0.40, 0.35])
    prf_device = rng.choice(["ios", "android"], size=n, p=[0.60, 0.40])
    prf_first_login = rng.integers(0, 14, size=n)
    prf_pn_status   = rng.choice(["enabled", "disabled", "unknown"], size=n,
                                  p=[0.65, 0.30, 0.05])
    prf_stop_comms  = rng.random(n) < 0.08
    return pd.DataFrame({
        "pid":            spine["pid"].values,
        "prf_class":      prf_class,
        "prf_device":     prf_device,
        "prf_first_login": prf_first_login,
        "prf_pn_status":  prf_pn_status,
        "prf_stop_comms": prf_stop_comms,
    })
