"""Behavioral recovery checks on the data generator itself.

Unlike test_parameter_recovery.py (which reads committed result CSVs), these
generate a fresh small cohort and assert the generator reproduces its injected
structure: the theta->engagement link, the Weibull relapse shape, the channel
transition matrix, and the SMS opt-out rate. No committed artifacts are read.

Run:  uv run pytest tests/test_generator_recovery.py -q
"""
from __future__ import annotations

import functools
from typing import cast

import numpy as np
import pandas as pd
from scipy.stats import weibull_min

from cessation import config as C
from cessation.synthetic.events import _CHANNEL_TRANS, generate_events
from cessation.synthetic.outcomes import generate_followup
from cessation.synthetic.population import generate_spine
from cessation.synthetic.reengagement import generate_sms

N = 2000
INJECTED_KAPPA = 0.55


@functools.lru_cache(maxsize=1)
def _cohort() -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(C.SEED)
    spine = generate_spine(N, seed=C.SEED, rng=rng)
    events, theta = generate_events(spine, rng)
    followup = generate_followup(spine, theta, rng=rng)
    sms = generate_sms(spine, rng)
    return events, theta, followup, sms


def test_theta_engagement_link() -> None:
    """Latent engagement propensity drives observed baseline event volume."""
    events, theta, _, _ = _cohort()
    base = (events[events["day_offset"] < C.BASELINE_WINDOW_DAYS]
            .groupby("pid").size())
    common = theta.index.intersection(base.index)
    r = float(np.corrcoef(
        theta.loc[common].to_numpy(),
        np.log1p(base.loc[common].to_numpy()))[0, 1])
    assert r > 0.5, f"theta->engagement correlation too weak: {r:.3f}"


def test_channel_transition_row_stochastic() -> None:
    """Injected transition matrix is row-stochastic; all channels are produced."""
    assert np.allclose(_CHANNEL_TRANS.sum(axis=1), 1.0)
    events, _, _, _ = _cohort()
    assert events["event_type"].nunique() >= 5


def test_weibull_shape_recovery() -> None:
    """Relapse times recover the injected decreasing-hazard shape."""
    _, _, followup, _ = _cohort()
    relapse = followup.loc[followup[C.OUTCOME_EVENT] == 1, C.OUTCOME_DURATION]
    shape = float(cast(float, weibull_min.fit(relapse.to_numpy(), floc=0)[0]))
    assert 0.40 <= shape <= 0.80, (
        f"recovered shape {shape:.3f} outside [0.40, 0.80] of injected "
        f"{INJECTED_KAPPA}")


def test_optout_rate_in_band() -> None:
    """Realized SMS opt-out sits in the calibrated ~35% window band."""
    _, _, _, sms = _cohort()
    n = sms["pid"].nunique()
    opt = sms.loc[sms["status"] == "opted_out", "pid"].nunique()
    rate = opt / n
    assert 0.25 <= rate <= 0.48, f"opt-out rate {rate:.3f} outside [0.25, 0.48]"
