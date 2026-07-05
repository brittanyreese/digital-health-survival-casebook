"""Analysis 13 -- Misspecification recovery: does the pipeline detect a mismatched DGP?

The flagship recovery (analysis/04, /11, /12) fits a Weibull AFT to
Weibull-generated data: the estimator and the data-generating process share
the same functional form, so "recovers the injected signal" cannot fail by
misspecification -- there is none to fail on. This script is the contrast:
it generates a cohort whose baseline hazard is genuinely non-Weibull
(lognormal, then generalized-gamma as a secondary check) while keeping the
covariate linear predictor and censoring identical, fits the same Weibull AFT
anyway, and shows the pipeline's Cox-Snell goodness-of-fit check rejects
adequacy on the mismatched data while failing to reject it on matched data.

That contrast -- not any single script -- is the evidence: a pipeline that
"detected" misspecification on every input regardless of fit, or missed it on
every input regardless of fit, would prove nothing. Passing on matched data
and failing on mismatched data is what proves the check has power.

Run:  uv run python analysis/13_misspecification_recovery.py
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import cast

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
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
from cessation.viz import add_synthetic_footer

OUT = C.RESULTS
OUT.mkdir(parents=True, exist_ok=True)

N = 8000  # matches cli.py N_TOTAL (the cessation-generate cohort size)
SEED_MATCHED   = 601  # baseline_family="weibull": the well-specified contrast
SEED_MISMATCH  = 602  # baseline_family="lognormal": primary mismatch
SEED_MISMATCH2 = 603  # baseline_family="gengamma": secondary mismatch check
N_BOOT = 2000


def fit_and_gof(
    baseline_family: str, seed: int,
) -> tuple[dict[str, float | str | int | bool], np.ndarray, np.ndarray]:
    """Regenerate a cohort under `baseline_family`, fit the Weibull AFT, run GoF."""
    rng = np.random.default_rng(seed)
    spine = generate_spine(N, seed=seed, rng=rng)
    events, theta = generate_events(spine, rng)
    followup = generate_followup(spine, theta, rng=rng, baseline_family=baseline_family)

    base = cast(pd.Series, events[events["day_offset"] < C.BASELINE_WINDOW_DAYS]
                .groupby("pid").size()).reset_index(name="n_events")
    d = followup[["pid", C.OUTCOME_DURATION, C.OUTCOME_EVENT]].merge(
        base, on="pid", how="inner")
    d["log_n_events"] = np.log1p(d["n_events"])
    d = d[d[C.OUTCOME_DURATION] > 0]

    dfit = cast(pd.DataFrame, d[[C.OUTCOME_DURATION, C.OUTCOME_EVENT, "log_n_events"]])
    wf = WeibullAFTFitter()
    wf.fit(dfit, duration_col=C.OUTCOME_DURATION, event_col=C.OUTCOME_EVENT)

    residuals = cox_snell_residuals(wf, dfit, C.OUTCOME_DURATION, C.OUTCOME_EVENT)
    kappa = administrative_censoring_kappa(
        wf, dfit, C.OUTCOME_DURATION, C.OUTCOME_EVENT, C.FOLLOWUP_DAYS,
    )
    event_arr = cast(pd.Series, dfit[C.OUTCOME_EVENT]).to_numpy()
    stat, p = cox_snell_gof_test(
        residuals, event_arr, kappa,
        n_boot=N_BOOT, rng=np.random.default_rng(seed + 10_000),
    )
    kappa_shape = float(wf.summary.loc[("rho_", "Intercept"), "exp(coef)"])
    return {
        "baseline_family": baseline_family,
        "seed": seed,
        "n": len(dfit),
        "events": int(event_arr.sum()),
        "recovered_kappa": kappa_shape,
        "gof_stat": stat,
        "gof_p": p,
        "adequate": bool(p >= 0.05),
    }, residuals, event_arr


def main() -> None:
    print("=== Misspecification recovery: matched vs mismatched DGP ===")
    rows = []
    residual_sets = {}
    for family, seed in [
        ("weibull", SEED_MATCHED),
        ("lognormal", SEED_MISMATCH),
        ("gengamma", SEED_MISMATCH2),
    ]:
        row, residuals, event_arr = fit_and_gof(family, seed)
        rows.append(row)
        residual_sets[family] = (residuals, event_arr)
        verdict = ("fails to reject (adequate)" if row["adequate"]
                   else "REJECTS (inadequate)")
        print(f"  {family:10s} n={row['n']} events={row['events']} "
              f"recovered_kappa={row['recovered_kappa']:.3f} "
              f"GoF stat={row['gof_stat']:.2f} p={row['gof_p']:.4f}  {verdict}")

    df = pd.DataFrame(rows)
    df.to_csv(OUT / "13_misspecification_gof.csv", index=False)

    matched_ok = bool(df.loc[df["baseline_family"] == "weibull", "adequate"].iloc[0])
    mismatches_rejected = bool(
        (~df.loc[df["baseline_family"] != "weibull", "adequate"]).all()
    )
    print(f"\n  Detection check: matched fits adequately={matched_ok}, "
          f"both mismatches rejected={mismatches_rejected}")
    print("  This shows the pipeline distinguishes a correctly specified fit "
          "from a misspecified one; it does not certify any single fit as "
          "correct in general.")

    fig, axes = plt.subplots(1, 3, figsize=(13, 4), sharey=True)
    r_grid = np.linspace(0, 4, 200)
    for ax, family in zip(axes, ["weibull", "lognormal", "gengamma"]):
        residuals, event_arr = residual_sets[family]
        obs = residuals[event_arr == 1]
        ax.hist(obs, bins=30, density=True, color="#4c78a8", alpha=0.7,
                label="Cox-Snell residuals\n(observed events)")
        ax.plot(r_grid, np.exp(-r_grid), color="black", ls="--", label="Exp(1) density")
        p = df.loc[df["baseline_family"] == family, "gof_p"].iloc[0]
        ax.set_title(f"{family}\nGoF p={p:.4f}")
        ax.set_xlabel("Cox-Snell residual")
    axes[0].set_ylabel("density")
    axes[0].legend(fontsize=7)
    fig.suptitle("Weibull AFT fit to matched vs mismatched baseline hazard")
    fig.tight_layout()
    add_synthetic_footer(fig)
    fig.savefig(OUT / "13_fig_misspecification.png", dpi=120)
    plt.close(fig)
    print("\n  saved 13_misspecification_gof.csv + 13_fig_misspecification.png")


if __name__ == "__main__":
    main()
