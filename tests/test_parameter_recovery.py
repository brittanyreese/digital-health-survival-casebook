"""Fast parameter-recovery checks against committed results.

Does not run the full pipeline and does not touch
data/synthetic/events_clean.csv (not on disk). Mirrors the CI smoke check
for the primary AFT result and adds a determinism + a CFA fit check.

Run:  uv run pytest -q
"""
from __future__ import annotations

from pathlib import Path
from typing import cast

import pandas as pd

from cessation.synthetic.population import generate_spine

ROOT = Path(__file__).resolve().parents[1]


def test_generator_determinism() -> None:
    df1 = generate_spine(n=500, seed=123)
    df2 = generate_spine(n=500, seed=123)
    assert df1.equals(df2)


def test_primary_effect_recovery() -> None:
    aft = pd.read_csv(ROOT / "results/analysis/04_aft_weibull.csv")
    row = aft[(aft["param"] == "lambda_") & (aft["covariate"] == "log_n_events")]
    assert not row.empty, "log_n_events not found in AFT results"
    coef = float(cast(pd.Series, row["coef"]).iloc[0])
    p = float(cast(pd.Series, row["p"]).iloc[0])
    assert coef > 0, f"expected positive engagement coefficient, got {coef}"
    assert p < 0.05, f"expected p < 0.05, got {p}"


def test_psychometric_recovery() -> None:
    fit = pd.read_csv(ROOT / "results/analysis/01_sdbs_cfa_fit.csv")
    cfi = float(fit["CFI"].iloc[0])
    assert cfi > 0.95, f"expected CFI > 0.95, got {cfi}"
