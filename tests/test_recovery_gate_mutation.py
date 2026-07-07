"""Mutation test: does the recovery/guard gate actually have teeth? (R4)

The CI promise ("a change that breaks recovery fails CI") has never been
exercised until now. This deliberately corrupts inputs -- an overlapped
feature/label window, and label-shuffled recovery data -- and asserts the
relevant check raises. Every corruption is wrapped in pytest.raises, so the
outer suite stays green; this test proves the inner gate is not vacuous, not
that recovery itself failed.

Run:  uv run pytest tests/test_recovery_gate_mutation.py -q
"""
from __future__ import annotations

from typing import cast

import numpy as np
import pandas as pd
import pytest
from lifelines import WeibullAFTFitter

from cessation import config as C
from cessation.guards import assert_no_temporal_overlap
from cessation.synthetic.events import generate_events
from cessation.synthetic.outcomes import generate_followup
from cessation.synthetic.population import generate_spine

N = 8000  # matches cli.py N_TOTAL; a smaller N was tried and lost enough power
          # that even unshuffled data sometimes failed the recovery assertion
SEED = 801


def _assert_recovers_engagement_effect(d: pd.DataFrame) -> None:
    """Mirrors test_parameter_recovery.py::test_primary_effect_recovery's assertion."""
    wf = WeibullAFTFitter()
    dfit = cast(pd.DataFrame, d[[C.OUTCOME_DURATION, C.OUTCOME_EVENT, "log_n_events"]])
    wf.fit(dfit, duration_col=C.OUTCOME_DURATION, event_col=C.OUTCOME_EVENT)
    s = wf.summary
    coef = float(s.loc[("lambda_", "log_n_events"), "coef"])
    p = float(s.loc[("lambda_", "log_n_events"), "p"])
    assert coef > 0, f"expected positive engagement coefficient, got {coef}"
    assert p < 0.05, f"expected p < 0.05, got {p}"


def _cohort() -> pd.DataFrame:
    rng = np.random.default_rng(SEED)
    spine = generate_spine(N, seed=SEED, rng=rng)
    events, theta = generate_events(spine, rng)
    followup = generate_followup(spine, theta, rng=rng)
    base = cast(pd.Series, events[events["day_offset"] < C.BASELINE_WINDOW_DAYS]
                .groupby("pid").size()).reset_index(name="n_events")
    d = followup[["pid", C.OUTCOME_DURATION, C.OUTCOME_EVENT]].merge(
        base, on="pid", how="inner")
    d["log_n_events"] = np.log1p(d["n_events"])
    return cast(pd.DataFrame, d[d[C.OUTCOME_DURATION] > 0])


def test_overlap_guard_trips_on_corrupted_window() -> None:
    """Covers R4: an overlapped churn/feature window makes the guard raise."""
    with pytest.raises(ValueError, match="temporal leakage"):
        assert_no_temporal_overlap((0, 30), (10, 180), context="mutation_test")


def test_recovery_assertion_trips_on_label_shuffle() -> None:
    """Covers R4: shuffling log_n_events against duration/event breaks recovery."""
    d = _cohort()
    _assert_recovers_engagement_effect(d)  # correct input: does not raise

    rng = np.random.default_rng(SEED + 1)
    shuffled = d.copy()
    shuffled["log_n_events"] = rng.permutation(shuffled["log_n_events"].to_numpy())
    with pytest.raises(AssertionError):
        _assert_recovers_engagement_effect(shuffled)


def test_negative_control_correct_input_does_not_raise() -> None:
    """The mutation, not the harness, is what fails: correct input raises nothing."""
    assert_no_temporal_overlap((0, 30), (30, 180), context="mutation_test_negative")
    _assert_recovers_engagement_effect(_cohort())
