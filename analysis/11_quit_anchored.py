"""Analysis 11 -- Quit-date-anchored engagement -> outcome (flagship longitudinal).

Landmark design (Anderson & Gill, 1982; van Houwelingen, 2007): the analysis
population is restricted to subjects who survive event-free through the
exposure window.  Subjects relapsing before `WINDOW_DAYS` are excluded;
survival time is measured from the end of the exposure window (origin shift).

Without landmark exclusion, early relapsers accumulate fewer events by
construction — a variant of immortal-time bias: a 5-day quitter can log at
most 5 days of events in a 30-day window, inflating the association between
engagement volume and duration.  Excluding subjects who have not survived to
the window close, and re-zeroing time at day `WINDOW_DAYS`, removes this
structural dependence.

Models: KM by engagement tertile, Cox PH (covariate-adjusted), Weibull AFT
(primary).  Robustness battery: window sensitivity (14d/60d) via AFT with
landmark applied at each window, segment-stratified Cox, power honesty check.

Run:  uv run python analysis/11_quit_anchored.py
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
from lifelines import CoxPHFitter, KaplanMeierFitter, WeibullAFTFitter
from lifelines.statistics import multivariate_logrank_test, proportional_hazard_test

from cessation import config as C
from cessation import data

OUT = C.RESULTS
OUT.mkdir(parents=True, exist_ok=True)

WINDOW_DAYS   = 30        # post-quit engagement window
CENSOR_DAYS   = C.FOLLOWUP_DAYS
WEEKS_PER_MONTH = 30.44 / 4.0


# ── helpers ───────────────────────────────────────────────────────────────────

def _opt_covs(df: pd.DataFrame) -> list[str]:
    cands = ["mod_readiness", "mod_age", "mod_edu", "mod_cpd"]
    return [c for c in cands if c in df.columns and df[c].notna().sum() > 20]


def _power_note(n_events: int, label: str) -> None:
    # Schoenfeld (1981): events ≈ 4*(z_α+z_β)²/(ln HR)²
    # HR=0.80, α=.05, power=.80: ~197 (continuous), ~630 (binary)
    if n_events >= 197:
        msg = "well-powered for continuous predictor (Schoenfeld n≥197)"
    elif n_events >= 100:
        msg = "moderately powered; interpret directionally"
    else:
        msg = "UNDERPOWERED — pattern only, not inference"
    print(f"  [{label}] n_events={n_events}: {msg}")


# ── exposure features ─────────────────────────────────────────────────────────

def engagement_features_quit_anchored(
    events: pd.DataFrame,
    reg: pd.DataFrame,
    window_days: int = WINDOW_DAYS,
) -> pd.DataFrame:
    """Per-pid engagement in [0, window_days) of declared quit date."""
    if "reg_quit_date_offset" not in reg.columns:
        print("  [warn] reg_quit_date_offset not found; no quit-anchored features")
        return pd.DataFrame()

    quit_map = (
        reg[["pid", "reg_quit_date_offset"]]
        .dropna(subset=["reg_quit_date_offset"])
        .set_index("pid")["reg_quit_date_offset"]
        .astype(float)
    )

    ev = events.merge(quit_map.rename("quit_day"), on="pid", how="inner")
    ev["day_since_quit"] = ev["day_offset"] - ev["quit_day"]
    w = ev[(ev["day_since_quit"] >= 0) & (ev["day_since_quit"] < window_days)].copy()

    if w.empty:
        print("  [warn] no events in quit window")
        return pd.DataFrame()

    g = w.groupby("pid")
    feat = pd.DataFrame({
        "n_events":       g.size(),
        "n_active_days":  g["day_since_quit"].apply(lambda x: x.astype(int).nunique()),
        "n_craving_tool": g["event_type"].apply(lambda x: (x == "craving_tool").sum()),
        "n_content":      g["event_type"].apply(lambda x: (x == "content").sum()),
        "n_notif":        g["event_type"].apply(lambda x: (x == "notification").sum()),
    }).reset_index()
    feat["intensity"]        = feat["n_events"] / feat["n_active_days"].clip(lower=1)
    feat["log_n_events"]     = np.log1p(feat["n_events"])
    feat["log_active_days"]  = np.log1p(feat["n_active_days"])
    feat["log_craving_tool"] = np.log1p(feat["n_craving_tool"])
    feat["log_content"]      = np.log1p(feat["n_content"])
    print(f"  {len(feat)} users with ≥1 event in {window_days}d quit window")
    return feat


# ── outcome ───────────────────────────────────────────────────────────────────

def load_outcome() -> pd.DataFrame:
    fu = data.load_followup()
    fu = fu[["pid", C.OUTCOME_DURATION, C.OUTCOME_EVENT]].copy()
    fu = fu[fu[C.OUTCOME_DURATION].notna()].copy()
    return fu


# ── model frame ───────────────────────────────────────────────────────────────

def build_frame(window_days: int = WINDOW_DAYS) -> pd.DataFrame:
    events = data.load_events()
    reg    = data.load_registration()

    outcome = load_outcome()
    feat    = engagement_features_quit_anchored(events, reg, window_days)

    if feat.empty:
        return pd.DataFrame()

    df = outcome.merge(feat, on="pid", how="inner")

    # Landmark exclusion: drop subjects with outcome before window closes.
    # Without this, early relapsers have mechanically lower event counts
    # (they simply had less time to accumulate them), which biases
    # log_n_events toward predicting survival time by construction.
    n_total = len(df)
    df = df[df[C.OUTCOME_DURATION] >= window_days].copy()
    n_excluded = n_total - len(df)
    print(f"  Landmark exclusion (duration < {window_days}d): "
          f"{n_excluded} excluded, {len(df)} remaining")

    # Shift survival time origin to the landmark
    df[C.OUTCOME_DURATION] = df[C.OUTCOME_DURATION] - window_days
    df = df[df[C.OUTCOME_DURATION] > 0].copy()

    # Segment labels
    seg_path = OUT / "segments_assignments.csv"
    if seg_path.exists():
        seg = pd.read_csv(seg_path)[["pid", "segment"]]
        seg["activated"] = (seg["segment"] != "Passive").astype(int)
        df = df.merge(seg[["pid", "activated"]], on="pid", how="left")
    else:
        df["activated"] = 0

    # Covariates
    cov_keys = ["readiness", "age", "education", "cigs_per_day"]
    cov_cols = ["pid"] + [C.MODERATORS[k] for k in cov_keys
                          if C.MODERATORS.get(k) and C.MODERATORS.get(k) in reg.columns]
    df = df.merge(reg[cov_cols], on="pid", how="left")

    # Engagement tertile
    try:
        df["tertile"] = pd.qcut(df["n_events"], q=3,
                                 labels=["Low", "Mid", "High"],
                                 duplicates="drop")
    except ValueError:
        df["tertile"] = "All"

    return df


# ── descriptives ──────────────────────────────────────────────────────────────

def descriptives(df: pd.DataFrame) -> None:
    print(f"\n=== Descriptives (n={len(df)}) ===")
    n_ev = int(df[C.OUTCOME_EVENT].sum())
    print(f"  events={n_ev}  censoring={1-df[C.OUTCOME_EVENT].mean():.2%}")
    print(f"  median quit days (relapsers): "
          f"{df.loc[df[C.OUTCOME_EVENT]==1, C.OUTCOME_DURATION].median():.0f}")
    df[[C.OUTCOME_DURATION, "log_n_events", "n_active_days"]].describe().round(
        2
    ).to_csv(OUT / "11_descriptives.csv")


# ── Kaplan-Meier ──────────────────────────────────────────────────────────────

def km_by_tertile(df: pd.DataFrame, label: str = "primary") -> None:
    print(f"\n=== KM by engagement tertile ({label}) ===")
    grps = df.dropna(subset=["tertile"])
    if len(grps) < 20:
        print("  Insufficient data")
        return
    result = multivariate_logrank_test(
        grps[C.OUTCOME_DURATION], grps["tertile"], grps[C.OUTCOME_EVENT]
    )
    print(f"  log-rank p={result.p_value:.4e}")

    fig, ax = plt.subplots(figsize=(8, 5))
    kmf = KaplanMeierFitter()
    for t in ["Low", "Mid", "High"]:
        sub = grps[grps["tertile"] == t]
        if len(sub) < 5:
            continue
        kmf.fit(sub[C.OUTCOME_DURATION], event_observed=sub[C.OUTCOME_EVENT], label=t)
        kmf.plot_survival_function(ax=ax, ci_show=True)
    ax.set_title(
        f"Quit survival by post-quit engagement tertile ({WINDOW_DAYS}d window)\n"
        f"log-rank p={result.p_value:.4f}"
    )
    ax.set_xlabel("Days since quit attempt")
    ax.set_ylabel("P(still quit)")
    fig.tight_layout()
    fig.savefig(OUT / f"11_fig_km_{label}.png", dpi=120)
    plt.close(fig)
    print(f"  saved 11_fig_km_{label}.png")


# ── Cox PH ────────────────────────────────────────────────────────────────────

def cox_ph(df: pd.DataFrame, label: str = "primary") -> pd.DataFrame | None:
    print(f"\n=== Cox PH ({label}) ===")
    opt  = _opt_covs(df)
    # Parsimonious spec aligned with Weibull AFT: log_n_events + activated.
    # log_active_days excluded (collinear with log_n_events; same reasoning as
    # script 04). Channel features excluded so Cox and AFT share the same
    # covariate set, enabling direct HR vs exp(β) comparison.
    base = [c for c in ["log_n_events", "activated"] if c in df.columns]
    cols = base + opt + [C.OUTCOME_DURATION, C.OUTCOME_EVENT]
    d = df[cols].dropna()
    n_ev = int(d[C.OUTCOME_EVENT].sum())
    print(f"  n={len(d)}  events={n_ev}")
    if len(d) < 20 or n_ev < 10:
        print("  Insufficient data")
        return None
    _power_note(n_ev, f"Cox {label}")
    # penalizer=0.1: L2 ridge for numerical stability at small post-landmark n.
    # Shrinks coefficients toward zero; CIs are penalized, not MLE. Sensitivity
    # to 0.05/0.0 should be checked before publication on real data.
    cph = CoxPHFitter(penalizer=0.1)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            cph.fit(d, duration_col=C.OUTCOME_DURATION, event_col=C.OUTCOME_EVENT)
    except Exception as exc:
        print(f"  Cox fit failed: {exc}")
        return None
    cph.print_summary(decimals=3)
    res = cph.summary.copy()
    res.to_csv(OUT / f"11_cox_{label}.csv")

    # Proportional hazards assumption (Schoenfeld-type score test)
    try:
        ph = proportional_hazard_test(cph, d, time_transform="rank")
        ph.print_summary(decimals=3)
        ph.summary.to_csv(OUT / f"11_ph_test_{label}.csv")
        if ph.summary["p"].min() < 0.05:
            print("  [warn] PH assumption violated for >=1 covariate "
                  "(Schoenfeld p<0.05). Weibull AFT is the primary model; "
                  "Cox estimates shown for comparison only.")
    except Exception as exc:
        print(f"  PH test failed: {exc}")

    return res


# ── AFT ───────────────────────────────────────────────────────────────────────

def aft_model(df: pd.DataFrame, label: str = "primary") -> None:
    print(f"\n=== Weibull AFT ({label}) ===")
    opt  = _opt_covs(df)
    # Include activated if present: AFT does not require proportional hazards,
    # so a covariate that violates Cox PH can still be included in AFT — the
    # model estimates its effect on the time-acceleration factor, not a hazard ratio.
    act  = ["activated"] if "activated" in df.columns else []
    cols = ["log_n_events"] + act + opt + [C.OUTCOME_DURATION, C.OUTCOME_EVENT]
    d = df[cols].dropna()
    if len(d) < 20:
        print("  Insufficient data")
        return
    wf = WeibullAFTFitter()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        wf.fit(d, duration_col=C.OUTCOME_DURATION, event_col=C.OUTCOME_EVENT)
    wf.print_summary(decimals=3)
    wf.summary.to_csv(OUT / f"11_aft_{label}.csv")


# ── OLS ───────────────────────────────────────────────────────────────────────

def ols_log(df: pd.DataFrame, label: str = "primary") -> None:
    print(
        f"\n=== OLS log(quit days) ({label}) — comparison only; "
        "Weibull AFT is primary ==="
    )
    print("  [BIAS WARNING] OLS treats censored observations as observed failures. "
          "Estimates are attenuated and may be sign-reversed. See AFT results above.")
    opt  = _opt_covs(df)
    chan = [c for c in ["log_craving_tool", "log_content"] if c in df.columns]
    d = df[[C.OUTCOME_DURATION, "log_n_events"] + chan + opt].dropna()
    d["log_duration"] = np.log(d[C.OUTCOME_DURATION].clip(lower=0.1))
    terms = ["log_n_events"] + chan
    formula = "log_duration ~ " + " + ".join(terms) + (
        " + " + " + ".join(opt) if opt else ""
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ols = smf.ols(formula, data=d).fit()
    print(ols.summary())
    pd.DataFrame({
        "coef": ols.params, "se": ols.bse, "t": ols.tvalues,
        "p": ols.pvalues, "ci_lo": ols.conf_int()[0], "ci_hi": ols.conf_int()[1],
    }).to_csv(OUT / f"11_ols_{label}.csv")


# ── robustness battery ────────────────────────────────────────────────────────

def robustness_battery(df: pd.DataFrame) -> None:
    """Robustness checks: window sensitivity, segment stratification, power."""
    print("\n=== Robustness battery ===")

    # 1. Window sensitivity: 14d vs 60d — Weibull AFT (handles right-censoring;
    #    OLS on censored data is biased)
    events  = data.load_events()
    reg     = data.load_registration()
    outcome = load_outcome()
    window_rows = []
    for w in [14, 60]:
        feat_w = engagement_features_quit_anchored(events, reg, window_days=w)
        if feat_w.empty:
            continue
        df_w = outcome.merge(feat_w, on="pid", how="inner")
        # Apply landmark at each window so comparison is fair
        df_w = df_w[df_w[C.OUTCOME_DURATION] >= w].copy()
        df_w[C.OUTCOME_DURATION] = df_w[C.OUTCOME_DURATION] - w
        if len(df_w) < 20:
            continue
        df_w = df_w[df_w[C.OUTCOME_DURATION] > 0].copy()
        wf_w = WeibullAFTFitter()
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                wf_w.fit(df_w[[C.OUTCOME_DURATION, C.OUTCOME_EVENT, "log_n_events"]],
                         duration_col=C.OUTCOME_DURATION, event_col=C.OUTCOME_EVENT)
            # Extract log_n_events lambda_ parameter
            idx = ("lambda_", "log_n_events")
            b = float(wf_w.params_[idx]) if idx in wf_w.params_.index else np.nan
            p = (float(wf_w.summary.loc[idx, "p"])
                 if idx in wf_w.summary.index else np.nan)
            print(f"  [window={w}d AFT] exp(β)={np.exp(b):.3f} p={p:.3f}")
            window_rows.append({"window_days": w, "beta_log_events": round(b, 4),
                                "exp_beta": round(np.exp(b), 4),
                                "p_value": round(p, 4), "n": len(df_w)})
        except Exception as exc:
            print(f"  [window={w}d] AFT failed: {exc}")
    if window_rows:
        pd.DataFrame(window_rows).to_csv(OUT / "11_robustness_window.csv", index=False)

    # 2. Segment-stratified Cox
    seg_path = OUT / "segments_assignments.csv"
    if seg_path.exists():
        seg = pd.read_csv(seg_path)
        for seg_label in seg["segment"].unique():
            sub = df.merge(
                seg[seg["segment"] == seg_label][["pid"]], on="pid", how="inner"
            )
            if len(sub) < 20:
                continue
            cox_ph(sub, label=f"segment_{seg_label.replace(' ', '_').lower()}")

    # 3. Power honesty
    n_ev = int(df[C.OUTCOME_EVENT].sum())
    print(f"\n  [power] n_events={n_ev} for primary window={WINDOW_DAYS}d")
    _power_note(n_ev, "primary")


# ── sanity check ──────────────────────────────────────────────────────────────

def sanity_print(df: pd.DataFrame) -> None:
    print("\n=== Sanity print ===")
    print(f"  Model frame: n={len(df)}, cols={df.columns.tolist()}")
    print(f"  Events: {df['n_events'].describe().to_dict()}")
    print(f"  Outcome range: {df[C.OUTCOME_DURATION].min():.1f}–"
          f"{df[C.OUTCOME_DURATION].max():.1f} days")
    print(f"  Event rate: {df[C.OUTCOME_EVENT].mean():.2%}")


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    df = build_frame()
    if df.empty:
        print("Empty model frame — run cessation-generate first")
        return
    sanity_print(df)
    descriptives(df)
    km_by_tertile(df)
    cox_ph(df)
    aft_model(df)
    ols_log(df)
    robustness_battery(df)
    print(f"\nDone. Results in {OUT}")


if __name__ == "__main__":
    main()
