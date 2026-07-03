"""
Synthetic event log generation.

Engagement model
----------------
Each user has a latent propensity θ_u ~ N(0, 1).  θ_u controls:
  - Session arrival rate: λ_u = exp(α + β · θ_u)  [Poisson]
  - Within-session channel transitions (Markov chain)

Channel types: craving_tool | content | peer_support | notification | quiz

Session arrival calibration
----------------------------
mHealth engagement literature (e.g., Aguilera & Muñoz 2011; BinDhim et al.
2018 on smoking-cessation apps) reports median ~15–30 sessions in first 90 days
for active users, with heavy right tail.  We use λ_baseline = 0.25 sessions/day
(~22 sessions/90 d) with β=1.0 so that θ=+1 → ~0.68 sessions/day (62/90d)
and θ=-1 → ~0.09 sessions/day (8/90d).

Channel Markov transition matrix
----------------------------------
Initial channel probabilities derived from generic mHealth app engagement
literature: notification most common entry point (~40%); craving-tool use
next (~25%); content browsing (~20%); peer-support (~10%); quiz (~5%).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from cessation.config import EVENT_CHANNELS, FOLLOWUP_DAYS

# ── Poisson session model ─────────────────────────────────────────────────────
_LOG_LAMBDA_BASE = np.log(0.25)  # baseline 0.25 sessions/day
_THETA_BETA      = 1.0           # propensity effect on log-rate

# ── initial channel distribution ──────────────────────────────────────────────
_CHANNEL_INIT = np.array([0.25, 0.20, 0.10, 0.40, 0.05])  # matches EVENT_CHANNELS order

# ── Markov transition matrix (rows = from, cols = to) ─────────────────────────
# craving_tool → likely to browse content or come back again
# notification → often leads to content or craving_tool
_CHANNEL_TRANS = np.array([
    #  crav  cont  peer  notif quiz
    [0.40, 0.30, 0.10, 0.15, 0.05],  # craving_tool
    [0.20, 0.45, 0.15, 0.15, 0.05],  # content
    [0.15, 0.25, 0.45, 0.10, 0.05],  # peer_support
    [0.30, 0.25, 0.10, 0.30, 0.05],  # notification
    [0.20, 0.30, 0.10, 0.15, 0.25],  # quiz
])
assert np.allclose(_CHANNEL_TRANS.sum(axis=1), 1.0), "Transition rows must sum to 1"

# ── events per session distribution ──────────────────────────────────────────
_EVENTS_PER_SESSION_MEAN = 4.5
_EVENTS_PER_SESSION_SD   = 3.0


def _session_days(n_days: int, rate: float, rng: np.random.Generator) -> np.ndarray:
    """Sample session arrival days via Poisson process over n_days."""
    expected = int(n_days * rate) + 1
    gaps = rng.exponential(1.0 / rate, size=expected * 3)
    arrival_times = np.cumsum(gaps)
    days = arrival_times[arrival_times < n_days].astype(int)
    return days


def generate_events(
    spine: pd.DataFrame,
    rng: np.random.Generator,
    n_days: int = FOLLOWUP_DAYS,
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Generate event log for all app_active users.

    Returns
    -------
    events : DataFrame with columns pid, event_type, day_offset, hour,
             device_type, detail, day_since_quit
    theta_u : Series indexed by pid (latent engagement propensity)
    """
    active_pids = spine.loc[spine["app_active"], "pid"].values
    n_active = len(active_pids)

    # Latent propensity
    theta = rng.standard_normal(n_active)
    theta_series = pd.Series(theta, index=active_pids, name="theta_u")

    records = []
    device_choices = ["ios", "android"]
    channel_arr = np.array(EVENT_CHANNELS)

    for i, pid in enumerate(active_pids):
        rate = np.exp(_LOG_LAMBDA_BASE + _THETA_BETA * theta[i])
        if rate < 0.01:
            continue
        session_days_arr = _session_days(n_days, rate, rng)
        if len(session_days_arr) == 0:
            continue

        # Assign device for this user
        device = rng.choice(device_choices, p=[0.60, 0.40])

        for day in session_days_arr:
            hour = int(rng.integers(7, 23))
            # First event in session
            ch_idx = rng.choice(len(EVENT_CHANNELS), p=_CHANNEL_INIT)
            n_ev = max(1, int(rng.normal(_EVENTS_PER_SESSION_MEAN,
                                          _EVENTS_PER_SESSION_SD)))
            for _ in range(n_ev):
                ev_type = channel_arr[ch_idx]
                records.append({
                    "pid":        pid,
                    "event_type": ev_type,
                    "day_offset": int(day),
                    "hour":       hour,
                    "device_type": device,
                    "detail":     f"{ev_type}_action",
                })
                # Markov transition
                ch_idx = rng.choice(len(EVENT_CHANNELS),
                                    p=_CHANNEL_TRANS[ch_idx])

    events = pd.DataFrame(records)

    # day_since_quit: placeholder (populated in outcomes module via merge)
    events["day_since_quit"] = np.nan

    return events, theta_series


def compute_engagement_features(
    events: pd.DataFrame,
    window_days: int | None = None,
    reference_day_col: str | None = None,
) -> pd.DataFrame:
    """
    Per-pid engagement features from event log.

    Parameters
    ----------
    events        : full event DataFrame
    window_days   : if not None, restrict to events with day_offset < window_days
    reference_day_col : if provided, events must have this column; filter
                        events in [0, window_days) relative to reference day

    Returns
    -------
    DataFrame with columns pid, n_events, n_active_days, intensity,
    log_n_events, log_active_days, n_craving_tool, n_content,
    n_peer_support, n_notification, n_quiz
    """
    ev = events.copy()
    if window_days is not None:
        ev = ev[ev["day_offset"] < window_days]

    if ev.empty:
        return pd.DataFrame(columns=pd.Index([
            "pid", "n_events", "n_active_days", "intensity",
            "log_n_events", "log_active_days",
            "n_craving_tool", "n_content", "n_peer_support",
            "n_notification", "n_quiz",
        ]))

    g = ev.groupby("pid")
    feat = pd.DataFrame({
        "n_events":       g.size(),
        "n_active_days":  g["day_offset"].nunique(),
        "n_craving_tool": g["event_type"].apply(lambda x: (x == "craving_tool").sum()),
        "n_content":      g["event_type"].apply(lambda x: (x == "content").sum()),
        "n_peer_support": g["event_type"].apply(lambda x: (x == "peer_support").sum()),
        "n_notification": g["event_type"].apply(lambda x: (x == "notification").sum()),
        "n_quiz":         g["event_type"].apply(lambda x: (x == "quiz").sum()),
    }).reset_index()
    feat["intensity"]      = feat["n_events"] / feat["n_active_days"].clip(lower=1)
    feat["log_n_events"]   = np.log1p(feat["n_events"])
    feat["log_active_days"] = np.log1p(feat["n_active_days"])
    return feat
