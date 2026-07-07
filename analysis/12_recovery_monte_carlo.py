"""Analysis 12 -- Monte-Carlo recovery stability.

Single-seed recovery cannot separate a genuine recovery from a lucky draw. This
regenerates the survival flagship across K independent seeds and reports the
sampling distribution of the recovered engagement->duration effect (Weibull AFT
exp(beta) on baseline log-events) and the recovered Weibull shape kappa, with
the Monte-Carlo interval and whether it covers the injected kappa = 0.55.

Enrollment-anchored, matching script 04 (baseline window day_offset < 30,
univariate log_n_events). Slow (regenerates the full cohort per seed), so it is
run once and its output committed; it is not on the CI hot path.

Run:  uv run python analysis/12_recovery_monte_carlo.py [K]   (default K=25)
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
from cessation.synthetic.events import generate_events
from cessation.synthetic.outcomes import generate_followup
from cessation.synthetic.population import generate_spine
from cessation.viz import add_synthetic_footer

OUT = C.RESULTS
OUT.mkdir(parents=True, exist_ok=True)

INJECTED_KAPPA = 0.55
N = 8000  # matches cli.py N_TOTAL (the cessation-generate cohort size)


def recover_one(seed: int) -> dict[str, float] | None:
    """Regenerate the cohort at `seed` and recover exp(beta) and kappa."""
    rng = np.random.default_rng(seed)
    spine = generate_spine(N, seed=seed, rng=rng)
    events, theta = generate_events(spine, rng)
    followup = generate_followup(spine, theta, rng=rng)

    base = cast(pd.Series, events[events["day_offset"] < C.BASELINE_WINDOW_DAYS]
                .groupby("pid").size()).reset_index(name="n_events")
    d = followup[["pid", C.OUTCOME_DURATION, C.OUTCOME_EVENT]].merge(
        base, on="pid", how="inner")
    d["log_n_events"] = np.log1p(d["n_events"])
    d = d[d[C.OUTCOME_DURATION] > 0]
    if len(d) < 100 or int(d[C.OUTCOME_EVENT].sum()) < 30:
        return None

    aft = WeibullAFTFitter()
    aft.fit(d[[C.OUTCOME_DURATION, C.OUTCOME_EVENT, "log_n_events"]],
            duration_col=C.OUTCOME_DURATION, event_col=C.OUTCOME_EVENT)
    s = aft.summary
    return {
        "seed": seed,
        "exp_beta": float(s.loc[("lambda_", "log_n_events"), "exp(coef)"]),
        "kappa": float(s.loc[("rho_", "Intercept"), "exp(coef)"]),
        "n": len(d),
        "events": int(d[C.OUTCOME_EVENT].sum()),
    }


def main(k: int = 25) -> None:
    print(f"=== Monte-Carlo recovery over {k} seeds (n={N}) ===")
    rows = []
    for seed in range(101, 101 + k):
        r = recover_one(seed)
        if r is None:
            continue
        rows.append(r)
        print(f"  seed {seed}: exp(beta)={r['exp_beta']:.3f} "
              f"kappa={r['kappa']:.3f} (n={r['n']}, events={r['events']})")

    df = pd.DataFrame(rows)
    df.to_csv(OUT / "12_recovery_mc.csv", index=False)

    eb_lo, eb_hi = np.percentile(df["exp_beta"], [2.5, 97.5])
    k_lo, k_hi = np.percentile(df["kappa"], [2.5, 97.5])
    covers = bool(k_lo <= INJECTED_KAPPA <= k_hi)
    print(f"\n  exp(beta): mean={df['exp_beta'].mean():.3f} "
          f"MC 95% [{eb_lo:.3f}, {eb_hi:.3f}]  (all > 1: "
          f"{bool((df['exp_beta'] > 1).all())})")
    print(f"  kappa:     mean={df['kappa'].mean():.3f} "
          f"MC 95% [{k_lo:.3f}, {k_hi:.3f}]  covers injected "
          f"{INJECTED_KAPPA}: {covers}")

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].hist(df["exp_beta"], bins=12, color="#4c78a8", edgecolor="white")
    axes[0].axvline(1.0, color="grey", ls=":", label="null (1.0)")
    axes[0].axvline(df["exp_beta"].mean(), color="black", label="MC mean")
    axes[0].set(xlabel="AFT exp(β), log-events", ylabel="seeds",
                title=f"Engagement→duration recovery ({k} seeds)")
    axes[0].legend(fontsize=8)
    axes[1].hist(df["kappa"], bins=12, color="#72b7b2", edgecolor="white")
    axes[1].axvline(INJECTED_KAPPA, color="red", label=f"injected {INJECTED_KAPPA}")
    axes[1].axvline(df["kappa"].mean(), color="black", label="MC mean")
    axes[1].set(xlabel="Weibull shape κ", ylabel="seeds",
                title="Shape recovery")
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    add_synthetic_footer(fig)
    fig.savefig(OUT / "12_fig_recovery_mc.png", dpi=120)
    plt.close(fig)
    print(f"\n  saved 12_recovery_mc.csv + 12_fig_recovery_mc.png ({len(df)} seeds)")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 25)
