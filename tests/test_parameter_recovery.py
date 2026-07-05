"""Parameter-recovery checks against the analysis result CSVs.

Reads results/analysis/ rather than running the full pipeline. In the CI lint
job these are the committed CSVs (fast smoke check); in the CI reproduce job the
same tests run against freshly regenerated outputs, so recovery is enforced on
fresh data. Checks injected magnitudes (Weibull shape kappa, AFT effect band,
CFA fit) plus generator determinism, not only sign and significance.

Run:  uv run pytest -q
"""
from __future__ import annotations

from pathlib import Path
from typing import cast

import pandas as pd

from cessation.synthetic.population import generate_spine

ROOT = Path(__file__).resolve().parents[1]

# Injected generation parameter (src/cessation/synthetic/outcomes.py).
INJECTED_WEIBULL_SHAPE = 0.55


def test_generator_determinism() -> None:
    df1 = generate_spine(n=500, seed=123)
    df2 = generate_spine(n=500, seed=123)
    assert df1.equals(df2)


def test_primary_effect_recovery() -> None:
    aft = pd.read_csv(ROOT / "results/analysis/04_aft_weibull.csv")
    row = aft[(aft["param"] == "lambda_") & (aft["covariate"] == "log_n_events")]
    assert not row.empty, "log_n_events not found in AFT results"
    coef = float(cast(pd.Series, row["coef"]).iloc[0])
    exp_coef = float(cast(pd.Series, row["exp(coef)"]).iloc[0])
    p = float(cast(pd.Series, row["p"]).iloc[0])
    assert coef > 0, f"expected positive engagement coefficient, got {coef}"
    assert p < 0.05, f"expected p < 0.05, got {p}"
    # Magnitude plausibility band, not just sign: gross breakage escapes a sign test.
    assert 1.1 <= exp_coef <= 2.2, f"exp(coef)={exp_coef:.3f} outside plausible band"


def test_weibull_shape_recovery() -> None:
    """Recovered Weibull shape (rho_) should land near the injected kappa=0.55.

    Magnitude check against an injected generator parameter, not a sign/significance
    smoke test: a broken generator or estimator would miss the band even with the
    engagement coefficient still positive.
    """
    aft = pd.read_csv(ROOT / "results/analysis/04_aft_weibull.csv")
    row = aft[(aft["param"] == "rho_") & (aft["covariate"] == "Intercept")]
    assert not row.empty, "rho_ Intercept not found in AFT results"
    kappa = float(cast(pd.Series, row["exp(coef)"]).iloc[0])
    assert 0.45 <= kappa <= 0.70, (
        f"recovered Weibull shape {kappa:.3f} outside recovery band [0.45, 0.70] "
        f"around injected kappa={INJECTED_WEIBULL_SHAPE}"
    )


def test_psychometric_recovery() -> None:
    fit = pd.read_csv(ROOT / "results/analysis/01_sdbs_cfa_fit.csv")
    cfi = float(fit["CFI"].iloc[0])
    assert cfi > 0.95, f"expected CFI > 0.95, got {cfi}"


def test_post_landmark_kappa_is_truncation_artifact() -> None:
    """The post-landmark kappa~=0.86 is a left-truncation drift, not a recovery (R2).

    Reads results/analysis/11_kappa_gof.csv (analysis/11's landmark_kappa_gof):
    an analytical simulation of the *known* generator, left-truncated and
    origin-shifted the same way build_frame() does. Direction: the recovered
    shape should drift well above the injected 0.55, toward the observed ~0.86
    band. Rejection: the Cox-Snell GoF check should reject Weibull adequacy for
    the true left-truncated residual, since it isn't Weibull.
    """
    gof = pd.read_csv(ROOT / "results/analysis/11_kappa_gof.csv")
    row = gof.iloc[0]
    injected = float(row["injected_kappa"])
    recovered = float(row["recovered_kappa"])
    p = float(row["gof_p"])
    assert injected == 0.55
    assert 0.70 <= recovered <= 0.95, (
        f"recovered post-landmark kappa {recovered:.3f} outside the drift band "
        f"[0.70, 0.95] expected for a left-truncated Weibull({injected}) residual"
    )
    assert p < 0.05, (
        f"expected the GoF check to reject Weibull adequacy for the true "
        f"left-truncated residual (p < 0.05), got p={p:.4f}"
    )
