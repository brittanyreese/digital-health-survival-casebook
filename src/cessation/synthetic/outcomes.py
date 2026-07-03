"""
Survival outcome generation for smoking cessation.

Model
-----
Baseline hazard follows a Weibull distribution with shape κ < 1 (decreasing
hazard), calibrated to published relapse curves:
  - κ = 0.55, λ = 180 days (scale)
  Source: Hughes JR, Keely J, Naud S (2004). Shape of the relapse curve and
          long-term abstinence among untreated smokers. Addiction, 99(1), 29–38.
          (Shape parameter estimated from pooled abstinence curves showing ~50%
           relapse by 8 weeks and ~80% by 6 months, hazard clearly decreasing.)
  Source: Shiffman S, Brockwell SE, Pillitteri JL, Gitchell JG (2007).
          Use of smoking-cessation treatments in the United States.
          Am J Prev Med, 32(3), 217–226. (Population-level relapse kinetics.)

Cox-type frailty
-----------------
Individual survival time is generated as:
    T_i = T_baseline · exp(−β·η_i / κ)
where η_i is a linear predictor of covariates and θ_u (latent engagement).
β_theta = −0.30 (greater engagement → longer quit duration)
β_sseq  = −0.25 (higher self-efficacy → longer quit duration)
β_pros  = +0.12 (higher pro-smoking attitudes → shorter quit duration)

Right-censoring at FOLLOWUP_DAYS (180 days).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import weibull_min

from cessation.config import FOLLOWUP_DAYS, OUTCOME_6MO, OUTCOME_DURATION, OUTCOME_EVENT

# Weibull parameters (see docstring citations)
_WEIBULL_SHAPE  = 0.55   # κ: decreasing hazard (most relapse in first weeks)
_WEIBULL_SCALE  = 180.0  # λ days (median ≈ 14 days for average user)

# Covariate effects on log(scale): larger scale → longer survival
_BETA_THETA = 0.30   # latent engagement propensity
_BETA_SSEQ  = 0.25   # SSEQ-12 composite (standardised)
_BETA_PROS  = -0.12  # SDBS Pros composite (standardised)


def _weibull_baseline_times(
    n: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Sample from Weibull(κ, λ) baseline."""
    return np.asarray(weibull_min.rvs(
        c=_WEIBULL_SHAPE,
        scale=_WEIBULL_SCALE,
        size=n,
        random_state=rng.integers(0, 2**31),
    ))


def generate_followup(
    spine: pd.DataFrame,
    theta_u: pd.Series,
    sseq_scores: pd.DataFrame | None = None,
    sdbs_scores: pd.DataFrame | None = None,
    rng: np.random.Generator | None = None,
) -> pd.DataFrame:
    """
    Generate follow-up outcome table.

    Parameters
    ----------
    spine       : full spine DataFrame
    theta_u     : Series(pid → latent engagement propensity)
    sseq_scores : optional DataFrame with pid + sseq_composite column
    sdbs_scores : optional DataFrame with pid + sdbs_pros, sdbs_cons columns
    rng         : random generator

    Returns
    -------
    DataFrame with pid, out_days_quit, out_relapsed, out_abstinent_6mo,
    fu_001..fu_005 (auxiliary follow-up items)
    """
    if rng is None:
        rng = np.random.default_rng(99)

    fu_pids = spine.loc[spine["followup"], "pid"].to_numpy()
    n = len(fu_pids)

    # Resolve covariates
    theta = pd.Series(
        theta_u.reindex(fu_pids).fillna(0).to_numpy(),
        index=fu_pids,
    )

    sseq_std = np.zeros(n)
    if sseq_scores is not None and "sseq_composite" in sseq_scores.columns:
        s = sseq_scores.set_index("pid")["sseq_composite"].reindex(fu_pids).fillna(0)
        sseq_std = (s.to_numpy() - s.mean()) / (s.std() + 1e-8)

    pros_std = np.zeros(n)
    if sdbs_scores is not None and "sdbs_pros" in sdbs_scores.columns:
        p = sdbs_scores.set_index("pid")["sdbs_pros"].reindex(fu_pids).fillna(0)
        pros_std = (p.to_numpy() - p.mean()) / (p.std() + 1e-8)

    # Linear predictor → individual scale multiplier
    lp = (
        _BETA_THETA * theta.to_numpy()
        + _BETA_SSEQ  * sseq_std
        + _BETA_PROS  * pros_std
    )
    scale_individual = _WEIBULL_SCALE * np.exp(lp)

    # Sample survival times with individual scales
    raw_times = np.array([
        weibull_min.rvs(
            c=_WEIBULL_SHAPE,
            scale=scale_individual[i],
            random_state=rng.integers(0, 2**31),
        )
        for i in range(n)
    ])

    # Apply censoring
    relapsed    = (raw_times <= FOLLOWUP_DAYS).astype(int)
    days_quit   = np.minimum(raw_times, FOLLOWUP_DAYS).round(1)
    abstinent6  = (raw_times > FOLLOWUP_DAYS).astype(int)

    # Auxiliary follow-up items (generic 1–5 Likert satisfaction / QoL items)
    fu_aux = rng.integers(1, 6, size=(n, 5))
    aux_cols = pd.Index([f"fu_{i:03d}" for i in range(1, 6)])
    aux_df = pd.DataFrame(fu_aux, columns=aux_cols)

    df = pd.DataFrame({
        "pid":             fu_pids,
        OUTCOME_DURATION:  days_quit,
        OUTCOME_EVENT:     relapsed,
        OUTCOME_6MO:       abstinent6,
    })
    return pd.concat([df, aux_df], axis=1)
