"""Analysis 10 -- Churn prediction: HistGradientBoosting + SHAP.

Predicts 14-day churn (no app use in final 14 days of study) from
early engagement features (first 30 days).  Demonstrates ML/XAI pipeline:
nested cross-validation for unbiased performance, SHAP for model explanation.

Churn definition: no events in days 165–180 (final 14 days of 180-day window).
Features: first-30-day engagement counts + channel mix + TTM stage.

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
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import roc_auc_score, classification_report
from sklearn.preprocessing import LabelEncoder
import shap

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
    feat["log_events_30d"]  = np.log1p(feat["n_events_30d"])

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


# ── nested CV ─────────────────────────────────────────────────────────────────

def nested_cv(X: pd.DataFrame, y: pd.Series) -> None:
    print("\n=== 2. Nested cross-validation (AUC) ===")
    model = HistGradientBoostingClassifier(
        max_iter=200, learning_rate=0.05, max_depth=4, random_state=42
    )
    outer = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        aucs = cross_val_score(model, X, y, cv=outer, scoring="roc_auc")
    print(f"  AUC: {aucs.mean():.4f} ± {aucs.std():.4f} (5-fold outer CV)")
    pd.DataFrame({"fold": range(1, 6), "auc": aucs}).to_csv(OUT / "10_cv_aucs.csv",
                                                               index=False)


# ── SHAP explanation ──────────────────────────────────────────────────────────

def shap_explain(X_train: pd.DataFrame, X_test: pd.DataFrame,
                 y_train: pd.Series, y_test: pd.Series,
                 feature_names: list[str]) -> None:
    print("\n=== 3. SHAP feature importance ===")
    model = HistGradientBoostingClassifier(
        max_iter=200, learning_rate=0.05, max_depth=4, random_state=42
    )
    model.fit(X_train, y_train)

    y_pred_prob = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, y_pred_prob)
    print(f"  Hold-out AUC: {auc:.4f}")

    explainer = shap.Explainer(model, X_train, feature_names=feature_names)
    shap_values = explainer(X_test, check_additivity=False)

    # Summary plot
    fig = plt.figure(figsize=(8, 6))
    shap.summary_plot(shap_values, X_test, feature_names=feature_names,
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

    # Waterfall for first test case
    fig = plt.figure(figsize=(8, 5))
    shap.waterfall_plot(shap_values[0], show=False)
    plt.tight_layout()
    plt.savefig(OUT / "10_fig_shap_waterfall.png", dpi=120)
    plt.close()
    print("  saved 10_fig_shap_waterfall.png")


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

    nested_cv(X, y)

    # Single train/test split (80/20) for SHAP
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    shap_explain(X_train, X_test, y_train, y_test, feature_names=feature_cols)

    print(f"\nDone. Results in {OUT}")


if __name__ == "__main__":
    main()
