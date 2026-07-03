"""Analysis 02 -- Engagement segmentation and retention.

Segments app users by engagement pattern using KMeans clustering, then
tests whether segments differ on retention (time-to-churn) via log-rank
test and Kaplan-Meier curves.

Analyses:
  1. Feature engineering from event log (total events, active days, channel mix)
  2. KMeans (k=2–4, silhouette selection)
  3. Segment profiles (mean engagement by segment)
  4. Retention curves (KM) and log-rank test
  5. Segment × TTM stage cross-tabulation

Run:  uv run python analysis/02_segmentation.py
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lifelines import KaplanMeierFitter
from lifelines.statistics import multivariate_logrank_test
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from cessation import config as C
from cessation import data

OUT = C.RESULTS
OUT.mkdir(parents=True, exist_ok=True)

CHURN_THRESHOLD_DAYS = 14   # no events in final 14 days → churned
STUDY_DAYS = C.FOLLOWUP_DAYS


# ── 1. Feature engineering ────────────────────────────────────────────────────

def build_features(events: pd.DataFrame, spine: pd.DataFrame) -> pd.DataFrame:
    """Per-user engagement features for clustering."""
    g = events.groupby("pid")
    feat = pd.DataFrame({
        "n_events":       g.size(),
        "n_active_days":  g["day_offset"].nunique(),
        "n_craving_tool": g["event_type"].apply(lambda x: (x == "craving_tool").sum()),
        "n_content":      g["event_type"].apply(lambda x: (x == "content").sum()),
        "n_peer_support": g["event_type"].apply(lambda x: (x == "peer_support").sum()),
        "n_notification": g["event_type"].apply(lambda x: (x == "notification").sum()),
        "n_quiz":         g["event_type"].apply(lambda x: (x == "quiz").sum()),
        "max_day":        g["day_offset"].max(),
    }).reset_index()
    feat["intensity"] = feat["n_events"] / feat["n_active_days"].clip(lower=1)
    feat["pct_craving"] = feat["n_craving_tool"] / feat["n_events"].clip(lower=1)
    feat["pct_content"]  = feat["n_content"]      / feat["n_events"].clip(lower=1)
    feat["log_events"]   = np.log1p(feat["n_events"])
    feat["log_days"]     = np.log1p(feat["n_active_days"])

    # Retention outcome: churned = 0 events in last 14 days of study
    last_event_day = g["day_offset"].max().reset_index(name="last_day")
    feat = feat.merge(last_event_day[["pid", "last_day"]], on="pid", how="left")
    feat["churned"]        = (
        feat["last_day"] < STUDY_DAYS - CHURN_THRESHOLD_DAYS
    ).astype(int)
    feat["tenure_days"]    = feat["last_day"].clip(upper=STUDY_DAYS)
    return feat


# ── 2. KMeans clustering ──────────────────────────────────────────────────────

def cluster(feat: pd.DataFrame, k_range: range = range(2, 5)) -> pd.DataFrame:
    print("\n=== 2. KMeans segmentation ===")
    cluster_features = ["log_events", "log_days", "intensity", "pct_craving",
                        "pct_content"]
    X = feat[cluster_features].dropna()
    scaler = StandardScaler()
    X_sc = scaler.fit_transform(X)

    scores = {}
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=20)
        labs = km.fit_predict(X_sc)
        scores[k] = silhouette_score(X_sc, labs)
        print(f"  k={k}  silhouette={scores[k]:.4f}")

    best_k = max(scores, key=scores.get)
    print(f"  → best k={best_k}")

    km = KMeans(n_clusters=best_k, random_state=42, n_init=20)
    labels = km.fit_predict(X_sc)
    feat_out = feat.loc[X.index].copy()
    feat_out["cluster_raw"] = labels

    # Sort clusters by mean events (descending) → 0=highest engager
    cluster_mean = feat_out.groupby("cluster_raw")["n_events"].mean()
    rank = cluster_mean.rank(ascending=False).astype(int) - 1
    feat_out["cluster"] = feat_out["cluster_raw"].map(rank)

    # Label
    label_map = {0: "High engagers", 1: "Moderate", 2: "Passive", 3: "One-and-done"}
    feat_out["segment"] = feat_out["cluster"].map(
        lambda c: label_map.get(c, f"Cluster {c}")
    )
    # Save silhouette scores
    pd.DataFrame({"k": list(scores.keys()),
                  "silhouette": list(scores.values())}).to_csv(
        OUT / "02_silhouette.csv", index=False
    )
    return feat_out


# ── 3. Segment profiles ───────────────────────────────────────────────────────

def segment_profiles(feat: pd.DataFrame) -> None:
    print("\n=== 3. Segment profiles ===")
    cols = ["n_events", "n_active_days", "intensity",
            "pct_craving", "pct_content", "churned"]
    profile = feat.groupby("segment")[cols].agg(["mean", "std"])
    print(profile.round(2))
    profile.round(3).to_csv(OUT / "02_segment_profiles.csv")


# ── 4. Retention KM + log-rank ────────────────────────────────────────────────

def retention_km(feat: pd.DataFrame) -> None:
    print("\n=== 4. Retention (KM by segment) ===")
    segments = feat["segment"].unique()
    fig, ax = plt.subplots(figsize=(8, 5))
    kmf = KaplanMeierFitter()
    durations = []
    events_lr = []
    groups_lr = []
    for seg in sorted(segments):
        sub = feat[feat["segment"] == seg]
        kmf.fit(sub["tenure_days"], event_observed=sub["churned"], label=seg)
        kmf.plot_survival_function(ax=ax, ci_show=True)
        durations.extend(sub["tenure_days"].tolist())
        events_lr.extend(sub["churned"].tolist())
        groups_lr.extend([seg] * len(sub))

    # Log-rank
    result = multivariate_logrank_test(durations, groups_lr, events_lr)
    p_val  = result.p_value
    print(f"  log-rank p={p_val:.4e}")
    ax.set_title(f"Retention by engagement segment (log-rank p={p_val:.4f})")
    ax.set_xlabel("Days in study")
    ax.set_ylabel("Proportion still active")
    fig.tight_layout()
    fig.savefig(OUT / "02_fig_retention_km.png", dpi=120)
    plt.close(fig)
    print("  saved 02_fig_retention_km.png")

    # Export KM table
    km_rows = []
    for seg in sorted(segments):
        sub = feat[feat["segment"] == seg]
        kmf.fit(sub["tenure_days"], event_observed=sub["churned"], label=seg)
        tbl = kmf.survival_function_.copy()
        tbl["segment"] = seg
        km_rows.append(tbl)
    pd.concat(km_rows).reset_index().to_csv(OUT / "02_km_curves.csv", index=False)


# ── 5. Stage × segment cross-tab ─────────────────────────────────────────────

def stage_by_segment(feat: pd.DataFrame) -> None:
    print("\n=== 5. TTM stage × segment ===")
    try:
        reg = data.load_registration()
        merged = feat.merge(reg[["pid", "mod_stage"]], on="pid", how="left")
        if merged["mod_stage"].notna().sum() < 10:
            print("  Insufficient stage data")
            return
        tab = pd.crosstab(merged["mod_stage"], merged["segment"], normalize="index")
        print(tab.round(3).to_string())
        tab.round(3).to_csv(OUT / "02_stage_x_segment.csv")
    except Exception as exc:
        print(f"  Stage cross-tab skipped: {exc}")


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    events = data.load_events()
    spine  = data.load_spine()
    print(f"Events n={len(events):,} | active pids={events['pid'].nunique():,}")

    feat = build_features(events, spine)
    feat = cluster(feat)
    segment_profiles(feat)
    retention_km(feat)
    stage_by_segment(feat)

    # Save assignments for downstream analyses
    feat[["pid", "segment", "cluster"]].to_csv(
        OUT / "segments_assignments.csv", index=False
    )
    print(f"\nDone. Segment assignments → {OUT/'segments_assignments.csv'}")


if __name__ == "__main__":
    main()
