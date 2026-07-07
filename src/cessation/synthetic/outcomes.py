"""
Survival outcome generation for smoking cessation.

Model
-----
Baseline hazard follows a Weibull distribution with shape κ < 1 (decreasing
hazard), calibrated to published relapse curves:
  - κ = 0.55, λ = 180 days (scale)
  Source: Hughes JR, Keely J, Naud S (2004). Shape of the relapse curve and
          long-term abstinence among untreated smokers. Addiction, 99(1), 29–38.
          (Decreasing-hazard shape taken from pooled abstinence curves; Hughes
           reports ~80% relapse by 6 months. This model uses a gentler level:
           κ=0.55, λ=180 give ~63% relapse by 180 days, median relapse time
           ~93 days, ~35% abstinent at 6 months. The shape is calibrated, the
           6-month level is a simplification, not a match to Hughes.)
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
from scipy.stats import gengamma, lognorm, weibull_min

from cessation.config import FOLLOWUP_DAYS, OUTCOME_6MO, OUTCOME_DURATION, OUTCOME_EVENT

# Weibull parameters (see docstring citations)
_WEIBULL_SHAPE  = 0.55   # κ: decreasing hazard (most relapse in first weeks)
_WEIBULL_SCALE  = 180.0  # λ days (median relapse time ≈ 93 days for average user)

# Misspecification-recovery alternatives (analysis/13): baselines whose hazard
# shape the Weibull AFT did not choose, calibrated to the same median relapse
# time as the Weibull default (scaled by the same exp(lp) covariate multiplier),
# so the covariate structure and the central tendency are held fixed and the
# family/shape is what changes. Note median-matching does NOT equalize the
# event fraction: at FOLLOWUP_DAYS censoring the observed relapse rate still
# differs by family (weibull ~62%, lognormal ~77%, gengamma ~91%), so the GoF
# rejection reflects a different family AND the event-rate/information change
# that family induces at a fixed median -- it is a baseline-family adequacy
# check, not an isolate-the-shape experiment.
# Matching the median rather than reusing the Weibull scale parameter
# directly matters: under FOLLOWUP_DAYS censoring, a naive scale-parameter
# reuse concentrates the mismatch in the sparsely-observed tail, where a
# refitted Weibull can locally absorb much of the shape difference and the
# detection test loses power. Median-matching keeps the mismatch inside the
# densely observed region instead.
# Lognormal hazard rises then falls (non-monotone); Weibull's is monotone for
# any shape != 1, so no Weibull fit can match it exactly.
_WEIBULL_MEDIAN = _WEIBULL_SCALE * np.log(2) ** (1.0 / _WEIBULL_SHAPE)  # ≈ 92.4 days
_LOGNORM_SIGMA  = 0.9

# Gamma(shape=5) via gengamma(a=5, c=1): monotone-increasing hazard that
# asymptotes to a constant (1/scale). Weibull's hazard is an unbounded/vanishing
# power law, so no Weibull matches it -- a distinct check that detection isn't
# specific to the non-monotone lognormal alternative. Shape 3 was tried first
# and rejected: detection was
# inconsistent across seeds at this cohort's sample size (~1150 followup
# users), too close to Weibull's fitted approximation in the censored range.
_GENGAMMA_A = 5.0
_GENGAMMA_C = 1.0
_GENGAMMA_MEDIAN_UNIT = float(gengamma.ppf(0.5, a=_GENGAMMA_A, c=_GENGAMMA_C))

BASELINE_FAMILIES = ("weibull", "lognormal", "gengamma")

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
    baseline_family: str = "weibull",
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
    baseline_family : baseline hazard family for raw survival times. Default
        "weibull" is the calibrated model used by every caller except
        analysis/13; "lognormal" and "gengamma" swap in a non-Weibull hazard
        shape (same covariate linear predictor and censoring) to test whether
        the estimator detects misspecification. See BASELINE_FAMILIES.

    Returns
    -------
    DataFrame with pid, out_days_quit, out_relapsed, out_abstinent_6mo,
    fu_001..fu_005 (auxiliary follow-up items)
    """
    if baseline_family not in BASELINE_FAMILIES:
        raise ValueError(f"unknown baseline_family: {baseline_family!r}")
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

    # Sample survival times with individual scales. The same exp(lp) covariate
    # multiplier drives every branch; only the baseline family (and, for the
    # alternates, the scale re-anchoring to match the Weibull median) differs,
    # so a mismatch is isolated to the hazard shape.
    if baseline_family == "weibull":
        raw_times = np.array([
            weibull_min.rvs(
                c=_WEIBULL_SHAPE,
                scale=scale_individual[i],
                random_state=rng.integers(0, 2**31),
            )
            for i in range(n)
        ])
    elif baseline_family == "lognormal":
        median_individual = _WEIBULL_MEDIAN * np.exp(lp)
        raw_times = np.array([
            lognorm.rvs(
                s=_LOGNORM_SIGMA,
                scale=median_individual[i],  # lognormal median == scale param
                random_state=rng.integers(0, 2**31),
            )
            for i in range(n)
        ])
    else:  # "gengamma"
        median_individual = _WEIBULL_MEDIAN * np.exp(lp)
        scale_gengamma = median_individual / _GENGAMMA_MEDIAN_UNIT
        raw_times = np.array([
            gengamma.rvs(
                a=_GENGAMMA_A,
                c=_GENGAMMA_C,
                scale=scale_gengamma[i],
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
