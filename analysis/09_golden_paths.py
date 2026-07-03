"""Analysis 09 -- Golden paths: sessionization + Markov chain analysis.

Sessions are defined by a 30-minute inactivity gap between events.  For each
session, the sequence of channel types is extracted and modelled as a first-order
Markov chain.  Top-k transition paths are identified per engagement segment.

Analyses:
  1. Sessionization (30-min gap rule)
  2. Session-level descriptives (duration, depth, channel diversity)
  3. Markov transition matrix estimation (MLE)
  4. Stationary distribution comparison across segments
  5. Top-5 most common 3-step paths per segment

Run:  uv run python analysis/09_golden_paths.py
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
from scipy.stats import spearmanr

from cessation import config as C
from cessation import data

OUT = C.RESULTS
OUT.mkdir(parents=True, exist_ok=True)

SESSION_GAP_MINS = 30
CHANNELS = C.EVENT_CHANNELS
CHANNEL_IDX = {c: i for i, c in enumerate(CHANNELS)}


# ── 1. Sessionize ─────────────────────────────────────────────────────────────

def sessionize(events: pd.DataFrame) -> pd.DataFrame:
    """Assign session IDs using 30-minute inactivity gap."""
    print("\n=== 1. Sessionization ===")
    ev = events.sort_values(["pid", "day_offset", "hour"]).copy()
    # Approximate timestamp in minutes
    ev["t_min"] = ev["day_offset"] * 1440 + ev["hour"] * 60
    ev["gap"] = ev.groupby("pid")["t_min"].diff().fillna(9999)
    ev["new_session"] = (ev["gap"] > SESSION_GAP_MINS).astype(int)
    ev["session_id"] = ev.groupby("pid")["new_session"].cumsum()
    n_sessions = ev.groupby("pid")["session_id"].nunique().sum()
    print(f"  {n_sessions:,} sessions across {ev['pid'].nunique():,} users")
    print(f"  mean events/session: {len(ev)/n_sessions:.1f}")
    return ev


# ── 2. Session descriptives ────────────────────────────────────────────────────

def session_descriptives(ev: pd.DataFrame) -> None:
    print("\n=== 2. Session descriptives ===")
    g = ev.groupby(["pid", "session_id"])
    sess = pd.DataFrame({
        "depth":     g.size(),
        "n_channels": g["event_type"].nunique(),
        "duration_approx": g["t_min"].apply(lambda x: x.max() - x.min()),
    }).reset_index()
    print(f"  median depth (events/session): {sess['depth'].median():.0f}")
    print(f"  median channel diversity: {sess['n_channels'].median():.1f}")
    sess.describe().round(2).to_csv(OUT / "09_session_stats.csv")


# ── 3. Markov transition matrix ───────────────────────────────────────────────

def markov_transitions(ev: pd.DataFrame, label: str = "all") -> np.ndarray:
    """MLE Markov transition matrix from ordered event sequences."""
    n = len(CHANNELS)
    counts = np.zeros((n, n), dtype=int)
    for _, group in ev.sort_values(["day_offset", "hour"]).groupby(
        ["pid", "session_id"]
    ):
        seq = [CHANNEL_IDX[c] for c in group["event_type"] if c in CHANNEL_IDX]
        for i, j in zip(seq[:-1], seq[1:]):
            counts[i, j] += 1
    # Normalize rows
    row_sums = counts.sum(axis=1, keepdims=True).clip(min=1)
    trans = counts / row_sums
    df_trans = pd.DataFrame(trans, index=CHANNELS, columns=CHANNELS)
    df_trans.round(4).to_csv(OUT / f"09_markov_trans_{label}.csv")
    return trans


def plot_transitions(trans: np.ndarray, title: str, fname: str) -> None:
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(trans, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(CHANNELS)))
    ax.set_yticks(range(len(CHANNELS)))
    ax.set_xticklabels(CHANNELS, rotation=45, ha="right")
    ax.set_yticklabels(CHANNELS)
    for i in range(len(CHANNELS)):
        for j in range(len(CHANNELS)):
            ax.text(j, i, f"{trans[i,j]:.2f}", ha="center", va="center",
                    fontsize=8, color="black" if trans[i, j] < 0.7 else "white")
    plt.colorbar(im, ax=ax, label="Transition probability")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(OUT / fname, dpi=120)
    plt.close(fig)


# ── 4. Stationary distribution ────────────────────────────────────────────────

def stationary_dist(trans: np.ndarray) -> np.ndarray:
    """Compute stationary distribution via eigenvector method."""
    vals, vecs = np.linalg.eig(trans.T)
    # Eigenvector for eigenvalue closest to 1
    idx = np.argmin(np.abs(vals - 1.0))
    stat = np.real(vecs[:, idx])
    stat = np.abs(stat) / np.abs(stat).sum()
    return stat


def compare_stationary(ev_all: pd.DataFrame, seg: pd.DataFrame) -> None:
    print("\n=== 4. Stationary distribution by segment ===")
    rows = []
    for segment in seg["segment"].unique():
        pids = seg[seg["segment"] == segment]["pid"].values
        ev_s = ev_all[ev_all["pid"].isin(pids)]
        if len(ev_s) < 100:
            continue
        trans = markov_transitions(ev_s, label=segment.replace(" ", "_").lower())
        stat = stationary_dist(trans)
        row = {"segment": segment}
        row.update({c: round(float(stat[i]), 4) for i, c in enumerate(CHANNELS)})
        rows.append(row)
        print(f"  {segment}: " + " ".join(f"{c}={stat[i]:.3f}"
                                            for i, c in enumerate(CHANNELS)))
    pd.DataFrame(rows).to_csv(OUT / "09_stationary_by_segment.csv", index=False)


# ── 5a. Channel time-share vs clinical outcome ───────────────────────────────

def channel_outcome_corr(ev: pd.DataFrame, seg: pd.DataFrame) -> None:
    """Correlate per-user channel time-share with quit duration (Spearman).

    Channel proportions are computed over the enrollment-anchored 30-day
    baseline window (day_offset < 30), consistent with the exposure window
    used in scripts 04 and 10.  Using the full 180-day window would conflate
    pre- and post-relapse channel use, introducing the same temporal
    contamination that motivates the baseline window restriction in the
    survival and churn models.

    Note: associations reflect the parameter structure injected by the
    generator, not real causal mechanisms.  Volume is not controlled; these
    are unadjusted bivariate correlations.  Real-data replication required
    before any product decision.
    """
    print("\n=== 5a. Channel time-share vs quit duration (30-day baseline window) ===")
    try:
        followup = data.load_followup()
    except Exception as exc:
        print(f"  followup not available: {exc}")
        return

    # enrollment-anchored baseline, consistent with 04/10
    ev_window = ev[ev["day_offset"] < 30]
    ev_ch = ev_window.groupby(["pid", "event_type"]).size().unstack(fill_value=0)
    ev_ch = ev_ch.div(ev_ch.sum(axis=1), axis=0).reset_index()

    d = ev_ch.merge(followup[["pid", C.OUTCOME_DURATION, C.OUTCOME_EVENT]],
                    on="pid", how="inner")
    if seg is not None:
        d = d.merge(seg[["pid", "segment"]], on="pid", how="left")

    # Spearman on relapsers only (completed outcome, not censored)
    d_rel = d[d[C.OUTCOME_EVENT] == 1]
    if len(d_rel) < 20:
        print("  Too few completed outcomes for correlation")
        return

    rows = []
    for ch in CHANNELS:
        if ch not in d_rel.columns:
            continue
        r, p = spearmanr(d_rel[ch], d_rel[C.OUTCOME_DURATION])
        rows.append({"channel": ch, "spearman_r": round(float(r), 3),
                     "p": round(float(p), 4), "n": len(d_rel)})

    df_out = pd.DataFrame(rows).sort_values("spearman_r", ascending=False)
    print(df_out.to_string(index=False))
    df_out.to_csv(OUT / "09_channel_outcome_corr.csv", index=False)
    print("  saved 09_channel_outcome_corr.csv")


# ── 5b. Top-k 3-step paths ────────────────────────────────────────────────────

def top_paths(ev: pd.DataFrame, k: int = 5) -> None:
    print(f"\n=== 5b. Top-{k} 3-step paths ===")
    from collections import Counter
    path_counts: Counter = Counter()
    for _, group in ev.sort_values(["day_offset", "hour"]).groupby(
        ["pid", "session_id"]
    ):
        seq = [c for c in group["event_type"] if c in CHANNEL_IDX]
        for i in range(len(seq) - 2):
            path_counts[(seq[i], seq[i+1], seq[i+2])] += 1

    top = path_counts.most_common(k)
    rows = [{"rank": i+1, "path": " → ".join(p), "count": c}
            for i, (p, c) in enumerate(top)]
    df = pd.DataFrame(rows)
    print(df.to_string(index=False))
    df.to_csv(OUT / "09_top_paths.csv", index=False)


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    events = data.load_events()
    ev     = sessionize(events)
    session_descriptives(ev)

    trans_all = markov_transitions(ev)
    plot_transitions(trans_all, "Markov channel transitions (all users)",
                     "09_fig_markov_all.png")
    print("  saved 09_fig_markov_all.png")

    seg_path = OUT / "segments_assignments.csv"
    seg = None
    if seg_path.exists():
        seg = pd.read_csv(seg_path)
        compare_stationary(ev, seg)

    channel_outcome_corr(ev, seg)
    top_paths(ev)
    print(f"\nDone. Results in {OUT}")


if __name__ == "__main__":
    main()
