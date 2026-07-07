"""Cox-Snell / Nelson-Aalen adequacy harness for a fitted Weibull AFT.

Shared by analysis/13 (misspecification detection) and analysis/11 (the
post-landmark kappa GoF proof): both ask the same question -- does the fitted
Weibull AFT's own survival function make its residuals look like Exp(1)?

A naive KS test of Cox-Snell residuals restricted to observed events against
Exp(1) is a common shortcut, but it is biased under the censoring in this
cohort: conditioning on "observed" selects subjects whose latent residual
happened to fall below their censoring threshold, which skews the observed
residuals toward small values even when the model is exactly correct. This
repo's censoring is administrative Type-I (everyone right-censored at exactly
FOLLOWUP_DAYS if not observed to fail earlier), so each subject's residual-
scale censoring threshold kappa_i = (FOLLOWUP_DAYS/lambda_i)^rho_i is a
deterministic function of their covariates under the fitted model, known
whether or not they were censored. That makes a parametric bootstrap under
H0 tractable: simulate iid Exp(1) latents, censor each at its own kappa_i,
and compare the resulting null distribution of a binned observed-vs-expected
chi-square statistic (a Grønnesby-Borgan-style one-sample test) to the one
observed in the real residuals. A pointwise sup-deviation statistic was
tried first and discarded: its variance is dominated by the few
smallest-at-risk-set observations at the tail, which swamps real shape
signal from the bulk of the distribution.
"""
from __future__ import annotations

from typing import cast

import numpy as np
import pandas as pd
from lifelines import WeibullAFTFitter


def cox_snell_residuals(
    wf: WeibullAFTFitter,
    df: pd.DataFrame,
    duration_col: str,
    event_col: str,
) -> np.ndarray:
    """r_i = -log(S_hat(t_i|x_i)) = (t_i / lambda_i) ** rho_i.

    Direct from the fitted Weibull AFT's own parametric form (S(t|x) =
    exp(-(t/lambda(x))^rho(x))), rather than a shared time grid, so it is
    exact per subject regardless of censoring. Passing a constant in place of
    the true/observed duration column (see `administrative_censoring_kappa`)
    evaluates the same formula at that constant instead.
    """
    covariates = [c for c in df.columns if c not in (duration_col, event_col)]
    X = df[covariates].copy()
    X.insert(0, "Intercept", 1.0)
    col_idx = {c: i for i, c in enumerate(X.columns)}
    X_arr = X.to_numpy()

    lambda_params = cast(pd.Series, wf.params_.loc["lambda_"])
    rho_params = cast(pd.Series, wf.params_.loc["rho_"])
    lambda_cols = [col_idx[c] for c in lambda_params.index]
    rho_cols = [col_idx[c] for c in rho_params.index]
    log_lambda = X_arr[:, lambda_cols] @ lambda_params.to_numpy()
    log_rho = X_arr[:, rho_cols] @ rho_params.to_numpy()
    lam = np.exp(log_lambda)
    rho = np.exp(log_rho)
    t = df[duration_col].to_numpy()
    return (t / lam) ** rho


def administrative_censoring_kappa(
    wf: WeibullAFTFitter,
    df: pd.DataFrame,
    duration_col: str,
    event_col: str,
    censor_at: float,
) -> np.ndarray:
    """Per-subject residual-scale censoring threshold kappa_i at `censor_at`.

    kappa_i = (censor_at / lambda_i) ** rho_i, computable for every subject
    from their covariates alone -- it does not depend on whether subject i
    was actually censored.
    """
    df_at_censor = df.copy()
    df_at_censor[duration_col] = censor_at
    return cox_snell_residuals(wf, df_at_censor, duration_col, event_col)


def _default_bin_edges(n_bins: int = 10) -> np.ndarray:
    """Exp(1)-equal-probability bin edges over n_bins quantiles; last edge is inf."""
    quantiles = np.linspace(0.0, 1.0, n_bins, endpoint=False)
    edges = -np.log(1.0 - quantiles)
    return np.append(edges, np.inf)


def _binned_observed_expected_stat(
    residuals: np.ndarray,
    event_observed: np.ndarray,
    bin_edges: np.ndarray,
) -> float:
    """Observed-vs-expected discrepancy over Exp(1)-quantile bins of the residuals.

    For bin [a, b): observed = # events with residual in [a, b); expected =
    total exposure time subjects spent in [a, b) under a unit-rate (Exp(1))
    hazard, i.e. sum of clip(min(r_i, b) - a, 0, b - a). Bins average out the
    tail-variance problem that sinks a pointwise sup-deviation statistic.

    Despite the Pearson (observed-expected)^2/expected form, this statistic is
    NOT chi-square-distributed: the "expected" terms are exposure integrals, not
    multinomial cell means, and the fitted parameters are reused. Its p-value
    comes only from the parametric bootstrap in cox_snell_gof_test -- never from
    a chi-square CDF or any degrees-of-freedom argument.
    """
    stat = 0.0
    for a, b in zip(bin_edges[:-1], bin_edges[1:]):
        observed = np.sum((event_observed == 1) & (residuals >= a) & (residuals < b))
        exposure = np.clip(np.minimum(residuals, b) - a, 0.0, b - a)
        expected = exposure.sum()
        stat += (observed - expected) ** 2 / max(expected, 1e-9)
    return float(stat)


def cox_snell_gof_test(
    residuals: np.ndarray,
    event_observed: np.ndarray,
    kappa: np.ndarray,
    *,
    n_bins: int = 10,
    n_boot: int = 2000,
    rng: np.random.Generator | None = None,
) -> tuple[float, float]:
    """Parametric-bootstrap GoF test: do these Cox-Snell residuals look Exp(1)?

    Returns (stat_obs, p_value): stat_obs is the observed binned
    observed-vs-expected chi-square and p_value is its bootstrap tail
    probability under H0 (model correctly specified), using each subject's
    own administrative censoring threshold `kappa` to reproduce the exact
    censoring pattern in the null simulations.

    Caveat: the null holds the fitted parameters fixed -- each replicate draws
    Exp(1) latents censored at the *observed-fit* `kappa` rather than re-fitting
    the AFT to every simulated dataset. It therefore ignores parameter-
    estimation uncertainty and is slightly conservative (a non-rejection is
    weaker evidence of adequacy than a full parametric bootstrap that re-fits
    per replicate would give). The bias is small here (n~1150, 3 params) and
    directionally safe -- it inflates p under H0, so it never manufactures a
    spurious rejection -- and a full re-fit is prohibitive at n_boot=2000.
    """
    if rng is None:
        rng = np.random.default_rng(0)
    bin_edges = _default_bin_edges(n_bins)
    stat_obs = _binned_observed_expected_stat(residuals, event_observed, bin_edges)
    n = len(kappa)
    stat_boot = np.empty(n_boot)
    for b in range(n_boot):
        eps = rng.exponential(size=n)
        r_b = np.minimum(eps, kappa)
        e_b = (eps <= kappa).astype(int)
        stat_boot[b] = _binned_observed_expected_stat(r_b, e_b, bin_edges)
    p_value = float((np.sum(stat_boot >= stat_obs) + 1) / (n_boot + 1))
    return stat_obs, p_value
