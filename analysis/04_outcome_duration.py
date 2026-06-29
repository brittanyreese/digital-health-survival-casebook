"""Analysis 04 -- Engagement -> quit duration (flagship survival analysis).

Links app engagement features to quit duration in the ~1,200-user follow-up
cohort.  Three models triangulate the effect: Kaplan-Meier (descriptive),
Cox PH (covariate-adjusted hazard), Weibull AFT (primary parametric estimator;
handles right-censoring correctly). OLS on log(duration) is included for
interpretability but treats censored observations as if they failed at the
censoring time, biasing estimates — treat as illustrative only.

Outcome: out_days_quit (days of continuous abstinence, right-censored at 180 d)
         out_relapsed (1 = relapsed within follow-up window)

Run:  uv run python analysis/04_outcome_duration.py
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from lifelines import KaplanMeierFitter, CoxPHFitter, WeibullAFTFitter
from lifelines.statistics import multivariate_logrank_test, proportional_hazard_test

from cessation import config as C
from cessation import data

OUT = C.RESULTS
OUT.mkdir(parents=True, exist_ok=True)

CENSOR_DAYS = C.FOLLOWUP_DAYS


# ── helpers ───────────────────────────────────────────────────────────────────

def _opt_covs(df: pd.DataFrame) -> list[str]:
    cands = ["mod_readiness", "mod_age", "mod_edu", "mod_cpd"]
    return [c for c in cands if c in df.columns and df[c].notna().sum() > 20]


def _power_note(n_events: int, label: str) -> None:
    # Schoenfeld (1981): events needed ≈ 4*(z_α+z_β)²/(ln HR)²
    # For HR=0.80, α=.05, power=.80: ≈ 197 (continuous), 630 (binary)
    if n_events >= 197:
        note = "well-powered for continuous predictor"
    elif n_events >= 100:
        note = "moderately powered; interpret with caution"
    else:
        note = "UNDERPOWERED — directional signal only"
    print(f"  [{label}] n_events={n_events}: {note}")


# ── 1. Build model frame ──────────────────────────────────────────────────────

def build_frame() -> pd.DataFrame:
    followup = data.load_followup()
    events   = data.load_events()
    reg      = data.load_registration()

    # Engagement features (full study window)
    g = events.groupby("pid")
    eng = pd.DataFrame({
        "n_events":      g.size(),
        "n_active_days": g["day_offset"].nunique(),
        "n_craving_tool": g["event_type"].apply(lambda x: (x == "craving_tool").sum()),
        "n_content":     g["event_type"].apply(lambda x: (x == "content").sum()),
    }).reset_index()
    eng["intensity"]        = eng["n_events"] / eng["n_active_days"].clip(lower=1)
    eng["log_n_events"]     = np.log1p(eng["n_events"])
    eng["log_active_days"]  = np.log1p(eng["n_active_days"])
    eng["log_craving_tool"] = np.log1p(eng["n_craving_tool"])
    eng["log_content"]      = np.log1p(eng["n_content"])

    # Segment labels (from analysis/02)
    seg_path = OUT / "segments_assignments.csv"
    if seg_path.exists():
        seg = pd.read_csv(seg_path)[["pid", "segment"]]
        seg["activated"] = (seg["segment"] != "Passive").astype(int)
        eng = eng.merge(seg[["pid", "activated"]], on="pid", how="left")
    else:
        eng["activated"] = 0

    # Merge outcome + engagement + covariates
    df = followup[["pid", C.OUTCOME_DURATION, C.OUTCOME_EVENT, C.OUTCOME_6MO]].merge(
        eng, on="pid", how="inner"
    )
    cov_cols = ["pid"] + [C.MODERATORS[k] for k in ("readiness", "age", "education",
                                                       "cigs_per_day")
                          if C.MODERATORS[k] in reg.columns]
    df = df.merge(reg[cov_cols], on="pid", how="left")

    # Engagement tertile for KM
    try:
        df["eng_tertile"] = pd.qcut(df["log_n_events"], q=3,
                                     labels=["Low", "Mid", "High"],
                                     duplicates="drop")
    except ValueError:
        df["eng_tertile"] = "All"

    return df


# ── 2. Descriptives ───────────────────────────────────────────────────────────

def descriptives(df: pd.DataFrame) -> None:
    print(f"\n=== 2. Descriptives (n={len(df)}) ===")
    n_ev = int(df[C.OUTCOME_EVENT].sum())
    cens_rate = 1 - df[C.OUTCOME_EVENT].mean()
    print(f"  events={n_ev}  censoring rate={cens_rate:.2%}")
    print(f"  median quit days (relapsers): "
          f"{df.loc[df[C.OUTCOME_EVENT]==1, C.OUTCOME_DURATION].median():.0f}")
    print(f"  abstinent at 180d: {df[C.OUTCOME_6MO].mean():.2%}")

    desc = df[[C.OUTCOME_DURATION, "log_n_events", "log_active_days"]].describe()
    print(desc.round(2).to_string())
    desc.to_csv(OUT / "04_descriptives.csv")


# ── 3. Kaplan-Meier by engagement tertile ────────────────────────────────────

def km_by_engagement(df: pd.DataFrame) -> None:
    print("\n=== 3. Kaplan-Meier by engagement tertile ===")
    grps = df.dropna(subset=["eng_tertile"])
    result = multivariate_logrank_test(
        grps[C.OUTCOME_DURATION], grps["eng_tertile"], grps[C.OUTCOME_EVENT]
    )
    print(f"  log-rank p={result.p_value:.4e}")

    fig, ax = plt.subplots(figsize=(8, 5))
    kmf = KaplanMeierFitter()
    for t in ["Low", "Mid", "High"]:
        sub = grps[grps["eng_tertile"] == t]
        if len(sub) < 5:
            continue
        kmf.fit(sub[C.OUTCOME_DURATION], event_observed=sub[C.OUTCOME_EVENT], label=t)
        kmf.plot_survival_function(ax=ax, ci_show=True)
    ax.set_title(f"Quit survival by engagement tertile (log-rank p={result.p_value:.4f})")
    ax.set_xlabel("Days since quit attempt")
    ax.set_ylabel("P(still quit)")
    fig.tight_layout()
    fig.savefig(OUT / "04_fig_km_engagement.png", dpi=120)
    plt.close(fig)
    print("  saved 04_fig_km_engagement.png")


# ── 4. Cox PH ─────────────────────────────────────────────────────────────────

def cox_ph(df: pd.DataFrame) -> pd.DataFrame | None:
    print("\n=== 4. Cox proportional-hazards ===")
    opt  = _opt_covs(df)
    chan = [c for c in ["log_craving_tool", "log_content"] if c in df.columns]
    cols = ["log_n_events", "log_active_days"] + chan + opt + [C.OUTCOME_DURATION, C.OUTCOME_EVENT]
    d = df[cols].dropna()
    n_ev = int(d[C.OUTCOME_EVENT].sum())
    print(f"  n={len(d)}  events={n_ev}")
    if len(d) < 20 or n_ev < 10:
        print("  Insufficient data"); return None
    _power_note(n_ev, "Cox")
    cph = CoxPHFitter(penalizer=0.1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cph.fit(d, duration_col=C.OUTCOME_DURATION, event_col=C.OUTCOME_EVENT)
    cph.print_summary(decimals=3)
    res = cph.summary.copy()
    res.to_csv(OUT / "04_cox.csv")

    # Proportional hazards assumption (Schoenfeld-type score test)
    try:
        ph = proportional_hazard_test(cph, d, time_transform="rank")
        ph.print_summary(decimals=3)
        ph.summary.to_csv(OUT / "04_ph_test.csv")
        if ph.summary["p"].min() < 0.05:
            print("  [warn] PH assumption violated for >=1 covariate "
                  "(Schoenfeld p<0.05). Weibull AFT is the primary model; "
                  "Cox estimates shown for comparison only.")
    except Exception as exc:
        print(f"  PH test failed: {exc}")

    return res


# ── 5. Weibull AFT ────────────────────────────────────────────────────────────

def weibull_aft(df: pd.DataFrame) -> None:
    print("\n=== 5. Weibull AFT ===")
    opt  = _opt_covs(df)
    cols = ["log_n_events"] + opt + [C.OUTCOME_DURATION, C.OUTCOME_EVENT]
    d = df[cols].dropna()
    d = d[d[C.OUTCOME_DURATION] > 0]
    if len(d) < 20:
        print("  Insufficient data"); return
    wf = WeibullAFTFitter()
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            wf.fit(d, duration_col=C.OUTCOME_DURATION, event_col=C.OUTCOME_EVENT)
    except Exception as exc:
        print(f"  Weibull fit failed: {exc}"); return
    wf.print_summary(decimals=3)
    wf.summary.to_csv(OUT / "04_aft_weibull.csv")
    # rho_ (shape parameter) is in the summary CSV under param=rho_, covariate=Intercept


# ── 6. OLS on log(duration) ───────────────────────────────────────────────────

def ols_log_duration(df: pd.DataFrame) -> None:
    print("\n=== 6. OLS on log(quit duration) ===")
    print("\n  [BIAS WARNING] OLS treats all observations as uncensored (censored duration = "
          "observed failure). This attenuates and can reverse the coefficient. "
          "Weibull AFT (above) is the primary model; OLS shown for comparability only.")
    opt  = _opt_covs(df)
    chan = [c for c in ["log_craving_tool", "log_content"] if c in df.columns]
    d = df[[C.OUTCOME_DURATION, "log_n_events"] + chan + opt].dropna()
    d["log_duration"] = np.log(d[C.OUTCOME_DURATION].clip(lower=0.1))
    terms = ["log_n_events"] + chan + opt
    formula = "log_duration ~ " + " + ".join(terms)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ols = smf.ols(formula, data=d).fit()
    print(ols.summary())
    pd.DataFrame({
        "coef":   ols.params,
        "se":     ols.bse,
        "t":      ols.tvalues,
        "p":      ols.pvalues,
        "ci_lo":  ols.conf_int()[0],
        "ci_hi":  ols.conf_int()[1],
    }).to_csv(OUT / "04_ols_log_duration.csv")


# ── 7. Engagement stratification figure ──────────────────────────────────────

def engagement_stratification_figure(df: pd.DataFrame) -> None:
    print("\n=== 7. Engagement dose–response ===")
    seg_path = OUT / "segments_assignments.csv"
    if not seg_path.exists():
        print("  Segment file not found; skipping"); return
    seg = pd.read_csv(seg_path)[["pid", "segment"]]
    d = df.merge(seg, on="pid", how="inner")

    mean_dur = d.groupby("segment")[C.OUTCOME_DURATION].mean().sort_values()
    fig, ax = plt.subplots(figsize=(6, 4))
    mean_dur.plot(kind="barh", ax=ax)
    ax.set_xlabel("Mean days quit")
    ax.set_title("Quit duration by engagement segment")
    fig.tight_layout()
    fig.savefig(OUT / "04_fig_duration_by_segment.png", dpi=120)
    plt.close(fig)
    print("  saved 04_fig_duration_by_segment.png")


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    df = build_frame()
    descriptives(df)
    km_by_engagement(df)
    cox_ph(df)
    print("\n  [PRIMARY MODEL] Weibull AFT — parametric, handles right-censoring")
    weibull_aft(df)
    ols_log_duration(df)
    engagement_stratification_figure(df)
    print(f"\nDone. Results in {OUT}")


if __name__ == "__main__":
    main()
