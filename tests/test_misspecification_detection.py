"""Does the Weibull AFT / Cox-Snell GoF check actually detect misspecification?

The flagship recovery (04/11/12) fits a Weibull AFT to Weibull-generated data:
the estimator can't fail by misspecification because there is none to fail on.
This test is the contrast the review panel asked for: generate data under a
baseline hazard the estimator did NOT choose (lognormal, then
generalized-gamma), fit the same Weibull AFT anyway, and confirm the
goodness-of-fit check rejects adequacy -- while confirming it does NOT reject
adequacy on matched (Weibull-generated) data. Both directions matter: a check
that always rejects, or never does, proves nothing about detection power.

Run:  uv run pytest tests/test_misspecification_detection.py -q
"""
from __future__ import annotations

from typing import cast

import numpy as np
import pandas as pd
import pytest
from lifelines import WeibullAFTFitter

from cessation import config as C
from cessation.survival_gof import (
    administrative_censoring_kappa,
    cox_snell_gof_test,
    cox_snell_residuals,
)
from cessation.synthetic.events import generate_events
from cessation.synthetic.outcomes import generate_followup
from cessation.synthetic.population import generate_spine

N = 8000        # matches analysis/13 and cli.py's N_TOTAL. Detection power is
                # N-sensitive: at N=4000 the gengamma alternative fails to
                # reject on ~1/3 of seeds. At N=8000 the seed panel below
                # separates cleanly on every arm (see _SEED_PANEL).
N_BOOT = 2000   # matches analysis/13's production setting

# Seed panel for the rate-based power characterization (the test_*_rate tests).
# Detection at this cohort size is asserted as a RATE across independent seeds
# rather than pinned to a single favorable draw. Measured over these 8 seeds at
# N=8000: matched not-rejected 8/8 (min p 0.11), lognormal detected 8/8
# (max p 0.003), gengamma detected 8/8 (max p 0.021). The rate tests assert
# >=7/8 to leave Monte-Carlo margin. They are marked `slow` (each is 8 N=8000
# cohort generations, ~2 min) and excluded from the fast CI gate via
# `-m "not slow"`; the single-seed tests above stay in CI as a cheap smoke check.
_SEED_PANEL = (601, 602, 603, 604, 605, 606, 607, 608)


def _gof_p(baseline_family: str, seed: int) -> float:
    rng = np.random.default_rng(seed)
    spine = generate_spine(N, seed=seed, rng=rng)
    events, theta = generate_events(spine, rng)
    followup = generate_followup(spine, theta, rng=rng, baseline_family=baseline_family)

    base = cast(pd.Series, events[events["day_offset"] < C.BASELINE_WINDOW_DAYS]
                .groupby("pid").size()).reset_index(name="n_events")
    d = followup[["pid", C.OUTCOME_DURATION, C.OUTCOME_EVENT]].merge(
        base, on="pid", how="inner")
    d["log_n_events"] = np.log1p(d["n_events"])
    d = cast(pd.DataFrame, d[d[C.OUTCOME_DURATION] > 0])
    dfit = cast(pd.DataFrame, d[[C.OUTCOME_DURATION, C.OUTCOME_EVENT, "log_n_events"]])

    wf = WeibullAFTFitter()
    wf.fit(dfit, duration_col=C.OUTCOME_DURATION, event_col=C.OUTCOME_EVENT)
    residuals = cox_snell_residuals(wf, dfit, C.OUTCOME_DURATION, C.OUTCOME_EVENT)
    kappa = administrative_censoring_kappa(
        wf, dfit, C.OUTCOME_DURATION, C.OUTCOME_EVENT, C.FOLLOWUP_DAYS,
    )
    event_arr = cast(pd.Series, dfit[C.OUTCOME_EVENT]).to_numpy()
    _, p = cox_snell_gof_test(
        residuals, event_arr, kappa,
        n_boot=N_BOOT, rng=np.random.default_rng(seed + 10_000),
    )
    return p


def test_matched_weibull_is_adequate() -> None:
    """Smoke check (CI): a Weibull AFT fit to Weibull data is not rejected on a
    validated seed. test_matched_weibull_adequacy_rate is the power version."""
    p = _gof_p("weibull", seed=601)
    assert p >= 0.05, f"matched (weibull) GoF rejected adequacy: p={p:.4f}"


def test_lognormal_mismatch_is_detected() -> None:
    """Smoke check (CI): a lognormal baseline is caught on a validated seed.
    test_lognormal_detection_rate is the power version."""
    p = _gof_p("lognormal", seed=602)
    assert p < 0.05, f"lognormal mismatch not detected: p={p:.4f}"


def test_gengamma_mismatch_is_detected() -> None:
    """Smoke check (CI): detection isn't specific to the lognormal alternative.
    test_gengamma_detection_rate is the power version."""
    p = _gof_p("gengamma", seed=603)
    assert p < 0.05, f"gengamma mismatch not detected: p={p:.4f}"


@pytest.mark.slow
def test_matched_weibull_adequacy_rate() -> None:
    """Power: a Weibull AFT on Weibull data is rarely falsely rejected."""
    ps = [_gof_p("weibull", s) for s in _SEED_PANEL]
    n_ok = sum(p >= 0.05 for p in ps)
    assert n_ok >= 7, f"matched (weibull) falsely rejected on >1/8 seeds: {ps}"


@pytest.mark.slow
def test_lognormal_detection_rate() -> None:
    """Power: the lognormal mismatch is detected on nearly every seed."""
    ps = [_gof_p("lognormal", s) for s in _SEED_PANEL]
    n_detected = sum(p < 0.05 for p in ps)
    assert n_detected >= 7, f"lognormal detected on <7/8 seeds: {ps}"


@pytest.mark.slow
def test_gengamma_detection_rate() -> None:
    """Power: the gengamma mismatch is detected on nearly every seed."""
    ps = [_gof_p("gengamma", s) for s in _SEED_PANEL]
    n_detected = sum(p < 0.05 for p in ps)
    assert n_detected >= 7, f"gengamma detected on <7/8 seeds: {ps}"


def test_default_baseline_family_unchanged() -> None:
    """Guard: no baseline_family arg stays byte-equivalent to the pre-change path."""
    rng1 = np.random.default_rng(123)
    spine1 = generate_spine(500, seed=123, rng=rng1)
    _, theta1 = generate_events(spine1, rng1)
    fu_default = generate_followup(spine1, theta1, rng=rng1)

    rng2 = np.random.default_rng(123)
    spine2 = generate_spine(500, seed=123, rng=rng2)
    _, theta2 = generate_events(spine2, rng2)
    fu_explicit = generate_followup(spine2, theta2, rng=rng2, baseline_family="weibull")

    pd.testing.assert_frame_equal(fu_default, fu_explicit)


def test_unknown_baseline_family_rejected() -> None:
    rng = np.random.default_rng(1)
    spine = generate_spine(200, seed=1, rng=rng)
    _, theta = generate_events(spine, rng)
    with pytest.raises(ValueError, match="baseline_family"):
        generate_followup(spine, theta, rng=rng, baseline_family="not-a-family")
