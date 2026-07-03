"""Analysis 10 -- Churn prediction: HistGradientBoosting + SHAP.

Predicts 14-day churn (no app use in final 14 days of study) from
early engagement features (first 30 days).  Demonstrates ML/XAI pipeline:
5-fold stratified CV for unbiased AUC estimate, then a final model trained
on the full dataset for SHAP explanation (standard pattern: CV estimates
performance, full-data model explains feature contributions).

Churn definition: no events in days 165–180 (final 14 days of 180-day window).
Features: first-30-day engagement counts + channel mix + TTM stage.
Churn-label note: churn=1 means zero events in days 165-180. This conflates
two populations: (a) disengaged users who lapsed, and (b) successful quitters
who no longer needed the app. The AUC should be interpreted as predictive of
late-window engagement absence, not exclusively dropout. A prospective
deployment would need to condition on abstinence status to separate these groups.

Run:  uv run python analysis/10_churn_ml.py
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
import shap
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score

from cessation import config as C
from cessation import data

OUT = C.RESULTS
OUT.mkdir(parents=True, exist_ok=True)

EARLY_WINDOW   = 30    # days for feature extraction
CHURN_WINDOW   = [C.FOLLOWUP_DAYS - 14, C.FOLLOWUP_DAYS]   # [166, 180]


# ── feature engineering ───────────────────────────────────────────────────────

def build_features(events: pd.DataFrame, spine: pd.DataFrame,
                   reg: pd.DataFrame) -> pd.DataFrame:
    """First-30-day features + churn label."""
    # Temporal separation guard: label window must not overlap feature window
    assert CHURN_WINDOW[0] >= EARLY_WINDOW, (
        f"Label window starts at day {CHURN_WINDOW[0]}, "
        f"overlapping feature window [0, {EARLY_WINDOW}): temporal leakage"
    )
    early = events[events["day_offset"] < EARLY_WINDOW].copy()
    late  = events[
        (events["day_offset"] >= CHURN_WINDOW[0]) &
        (events["day_offset"] <  CHURN_WINDOW[1])
    ].copy()

    active = spine[spine["app_active"]]["pid"].values
    g_e = early.groupby("pid")
    g_l = late.groupby("pid")

    feat = pd.DataFrame(index=active)
    feat.index.name = "pid"

    feat["n_events_30d"]    = g_e.size().reindex(active, fill_value=0)
    feat["n_active_days_30d"] = g_e["day_offset"].nunique().reindex(active, fill_value=0)
    feat["n_craving_30d"]   = g_e["event_type"].apply(
        lambda x: (x == "craving_tool").sum()
    ).reindex(active, fill_value=0)
    feat["n_content_30d"]   = g_e["event_type"].apply(
        lambda x: (x == "content").sum()
    ).reindex(active, fill_value=0)
    feat["n_notif_30d"]     = g_e["event_type"].apply(
        lambda x: (x == "notification").sum()
    ).reindex(active, fill_value=0)
    feat["intensity_30d"]   = (feat["n_events_30d"] /
                                feat["n_active_days_30d"].clip(lower=1))

    # Churn label: 0 events in late window = churned
    late_active = g_l.size().reindex(active, fill_value=0)
    feat["churned"] = (late_active == 0).astype(int)

    # Covariates from registration
    if reg is not None:
        reg_sub = reg.set_index("pid")
        for col in ["mod_readiness", "mod_age", "mod_edu", "mod_cpd"]:
            if col in reg_sub.columns:
                feat[col] = reg_sub[col].reindex(active)

    feat = feat.reset_index()
    feat = feat[feat["n_events_30d"] > 0]  # users with any early activity
    print(f"  feature frame: n={len(feat)}, churned={feat['churned'].mean():.1%}")
    return feat


# ── cross-validation AUC ──────────────────────────────────────────────────────

def cross_val_auc(X: pd.DataFrame, y: pd.Series) -> None:
    """ROC-AUC and AUPRC via 5-fold stratified CV.

    AUPRC (average precision) is reported alongside ROC-AUC because it is
    more informative under class imbalance: it weights precision at each
    recall threshold, giving heavier penalty to false positives when the
    positive class is a minority.  The no-skill AUPRC baseline equals the
    churn prevalence, not 0.5.
    """
    print("\n=== 2. Cross-validation AUC + AUPRC (5-fold stratified) ===")
    churn_rate = y.mean()
    print(f"  Class imbalance: {churn_rate:.1%} churned  ({1 - churn_rate:.1%} retained)")
    model = HistGradientBoostingClassifier(
        max_iter=200, learning_rate=0.05, max_depth=4, random_state=42
    )
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        aucs  = cross_val_score(model, X, y, cv=cv, scoring="roc_auc")
        auprc = cross_val_score(model, X, y, cv=cv, scoring="average_precision")
    print(f"  ROC-AUC: {aucs.mean():.4f} ± {aucs.std():.4f}  (baseline = 0.50)")
    print(f"  AUPRC:   {auprc.mean():.4f} ± {auprc.std():.4f}  (no-skill baseline = {churn_rate:.3f})")
    pd.DataFrame({
        "fold":            range(1, 6),
        "auc":             aucs,
        "auprc":           auprc,
        "baseline_auc":    0.5,
        "baseline_auprc":  churn_rate,
    }).to_csv(OUT / "10_cv_aucs.csv", index=False)


# ── SHAP explanation ──────────────────────────────────────────────────────────

def shap_explain(X: pd.DataFrame, y: pd.Series, feature_names: list[str]) -> None:
    """Train final model on full data and explain with SHAP.

    CV AUC (above) is the unbiased performance estimate.  This model uses
    all available data so SHAP values reflect the full-sample feature signal.
    """
    print("\n=== 3. SHAP feature importance (final model, full data) ===")
    model = HistGradientBoostingClassifier(
        max_iter=200, learning_rate=0.05, max_depth=4, random_state=42
    )
    model.fit(X, y)

    explainer = shap.Explainer(model, X, feature_names=feature_names)
    # check_additivity=False: GBM + missing values; additivity holds within tolerance per SHAP docs
    shap_values = explainer(X, check_additivity=False)

    # Summary plot
    fig = plt.figure(figsize=(8, 6))
    shap.summary_plot(shap_values, X, feature_names=feature_names,
                      show=False)
    plt.tight_layout()
    plt.savefig(OUT / "10_fig_shap_summary.png", dpi=120)
    plt.close()
    print("  saved 10_fig_shap_summary.png")

    # Mean absolute SHAP per feature
    mean_abs = np.abs(shap_values.values).mean(axis=0)
    importance = pd.DataFrame({
        "feature": feature_names,
        "mean_abs_shap": mean_abs,
    }).sort_values("mean_abs_shap", ascending=False)
    importance.to_csv(OUT / "10_shap_importance.csv", index=False)
    print(importance.to_string(index=False))

    # Waterfall for the median-risk participant (ranked by sum of SHAP values).
    # The median-risk individual shows a more representative feature decomposition
    # than an extreme case, and avoids the arbitrary choice of index [0].
    overall_risk = shap_values.values.sum(axis=1)
    idx_median = int(np.argsort(overall_risk)[len(overall_risk) // 2])
    fig = plt.figure(figsize=(8, 5))
    shap.waterfall_plot(shap_values[idx_median], show=False)
    plt.tight_layout()
    plt.savefig(OUT / "10_fig_shap_waterfall.png", dpi=120)
    plt.close()
    print(f"  saved 10_fig_shap_waterfall.png  (participant at median risk rank {idx_median})")


# ── subgroup AUC (fairness check) ────────────────────────────────────────────

def subgroup_aucs(feat: pd.DataFrame, X: pd.DataFrame, y: pd.Series) -> None:
    """AUC disaggregated by demographic proxy — minimum fairness check."""
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import cross_val_predict

    print("\n=== 4. Subgroup AUC (demographic fairness check) ===")
    model = HistGradientBoostingClassifier(
        max_iter=200, learning_rate=0.05, max_depth=4, random_state=42
    )
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        proba = cross_val_predict(model, X, y, cv=cv, method="predict_proba")[:, 1]

    rows = []
    for col in ["mod_age", "mod_edu"]:
        if col not in feat.columns:
            continue
        med = feat[col].median()
        for label, mask in [("below_median", feat[col] <= med),
                             ("above_median", feat[col] > med)]:
            mask_arr = mask.values
            y_sub = y[mask_arr]
            p_sub = proba[mask_arr]
            if y_sub.nunique() < 2 or len(y_sub) < 20:
                continue
            auc = roc_auc_score(y_sub, p_sub)
            rows.append({"covariate": col, "subgroup": label,
                         "n": int(len(y_sub)), "auc": round(auc, 4)})
    if rows:
        df_sg = pd.DataFrame(rows)
        df_sg.to_csv(OUT / "10_subgroup_aucs.csv", index=False)
        print(df_sg.to_string(index=False))
    print("  Note: subgroup differences reflect synthetic data structure and carry no "
          "real equity information. A prospective fairness audit (including race/ethnicity, "
          "rurality, insurance status) is required before deployment.")


# ── calibration reliability diagram ──────────────────────────────────────────

def calibration_plot(X: pd.DataFrame, y: pd.Series) -> None:
    """Reliability diagram to confirm predicted probabilities are calibrated."""
    from sklearn.calibration import calibration_curve
    from sklearn.model_selection import cross_val_predict

    print("\n=== 5. Calibration reliability diagram ===")
    model = HistGradientBoostingClassifier(
        max_iter=200, learning_rate=0.05, max_depth=4, random_state=42
    )
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        proba = cross_val_predict(model, X, y, cv=cv, method="predict_proba")[:, 1]

    fraction_pos, mean_pred = calibration_curve(y, proba, n_bins=10)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(mean_pred, fraction_pos, "s-", label="churn model")
    ax.plot([0, 1], [0, 1], "k--", label="perfect calibration")
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Fraction of positives")
    ax.set_title("Calibration reliability diagram (5-fold CV predictions)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "10_fig_calibration.png", dpi=120)
    plt.close(fig)
    print("  saved 10_fig_calibration.png")


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    events = data.load_events()
    spine  = data.load_spine()
    try:
        reg = data.load_registration()
    except Exception:
        reg = None

    print("\n=== 1. Feature engineering ===")
    feat = build_features(events, spine, reg)

    feature_cols = [c for c in feat.columns if c not in ("pid", "churned")]
    X = feat[feature_cols].fillna(feat[feature_cols].median())
    y = feat["churned"]

    cross_val_auc(X, y)
    shap_explain(X, y, feature_names=feature_cols)
    subgroup_aucs(feat, X, y)
    calibration_plot(X, y)

    print(f"\nDone. Results in {OUT}")


if __name__ == "__main__":
    main()
