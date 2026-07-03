"""Analysis 06 -- SMS re-engagement quasi-experiment.

Evaluates whether delivered SMS notifications increase app return within
a 14-day window, using a delivered-vs-opted-out contrast.  Treats opt-out
as a quasi-control condition (users who opted out before the SMS date serve
as a comparison group for those who received the message).

Analyses:
  1. SMS delivery and opt-out descriptives
  2. Event study: app return rate in 7–14 days post-SMS
  3. Kaplan-Meier: time-to-return by delivery status
  4. Logistic regression: P(return | delivered) + covariates
  5. Subgroup analysis by engagement segment

Causal interpretation: Opt-out is endogenous (lower-motivation users opt out
earlier), so the contrast is observational with likely upward bias on the
estimated effect.  Treat as effect-direction evidence, not causal estimate.

Run:  uv run python analysis/06_sms_reengagement.py
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
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test

from cessation import config as C
from cessation import data

OUT = C.RESULTS
OUT.mkdir(parents=True, exist_ok=True)

RETURN_WINDOW = 14   # days post-SMS for re-engagement window


# ── 1. Descriptives ───────────────────────────────────────────────────────────

def sms_descriptives(sms: pd.DataFrame) -> None:
    print("\n=== 1. SMS descriptives ===")
    n_users   = sms["pid"].nunique()
    n_total   = len(sms)
    opt_users = sms[sms["status"] == "opted_out"]["pid"].nunique()
    opt_rate  = opt_users / n_users
    print(f"  {n_total:,} SMS to {n_users:,} users")
    print(f"  Opt-out rate: {opt_users:,}/{n_users:,} = {opt_rate:.2%}")
    print("  (Calibration target: 48% — SmokefreeVET PMC5144826)")

    status_dist = sms.groupby("status")["pid"].nunique()
    print(f"\n  Status breakdown:\n{status_dist.to_string()}")
    status_dist.to_csv(OUT / "06_sms_status.csv")


# ── 2. Event study ────────────────────────────────────────────────────────────

def event_study(sms: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
    print(f"\n=== 2. Event study: return within {RETURN_WINDOW}d post-SMS ===")
    # First SMS per user with delivery status
    first = (sms.sort_values("sms_seq")
               .groupby("pid")
               .first()
               .reset_index()[["pid", "day_offset", "status"]])
    first.rename(columns={"day_offset": "sms_day"}, inplace=True)

    # App events after first SMS
    ev_after = events.merge(first[["pid", "sms_day"]], on="pid")
    ev_after["days_after_sms"] = ev_after["day_offset"] - ev_after["sms_day"]
    returned_window = ev_after[
        (ev_after["days_after_sms"] > 0) &
        (ev_after["days_after_sms"] <= RETURN_WINDOW)
    ]["pid"].unique()

    first["returned"] = first["pid"].isin(returned_window).astype(int)
    first["days_to_return"] = np.nan

    for pid in returned_window:
        sub = ev_after[(ev_after["pid"] == pid) & (ev_after["days_after_sms"] > 0)]
        if not sub.empty:
            first.loc[first["pid"] == pid, "days_to_return"] = sub[
                "days_after_sms"
            ].min()

    first["is_delivered"] = (first["status"] == "delivered").astype(int)

    # Return rates
    for status, label in [("delivered", "Delivered"), ("opted_out", "Opted-out")]:
        sub = first[first["status"] == status]
        rate = sub["returned"].mean()
        print(f"  {label}: {rate:.2%} returned within {RETURN_WINDOW}d (n={len(sub)})")

    first.to_csv(OUT / "06_event_study_frame.csv", index=False)
    return first


# ── 3. KM: time-to-return ─────────────────────────────────────────────────────

def km_time_to_return(frame: pd.DataFrame) -> None:
    print("\n=== 3. KM: time-to-return by delivery status ===")
    frame = frame.copy()
    frame["days_to_return"] = frame["days_to_return"].fillna(RETURN_WINDOW + 1)
    frame["returned_binary"] = (frame["days_to_return"] <= RETURN_WINDOW).astype(int)

    delivered = frame[frame["status"] == "delivered"]
    opted_out = frame[frame["status"] == "opted_out"]

    lr = logrank_test(
        delivered["days_to_return"], opted_out["days_to_return"],
        event_observed_A=delivered["returned_binary"],
        event_observed_B=opted_out["returned_binary"],
    )
    print(f"  log-rank p={lr.p_value:.4e}")

    fig, ax = plt.subplots(figsize=(7, 5))
    kmf = KaplanMeierFitter()
    for sub, lbl in [(delivered, "Delivered"), (opted_out, "Opted-out")]:
        kmf.fit(sub["days_to_return"], event_observed=sub["returned_binary"], label=lbl)
        kmf.plot_survival_function(ax=ax, ci_show=True)

    ax.set_xlim(0, RETURN_WINDOW + 2)
    ax.set_title(f"Time to app return post-SMS (log-rank p={lr.p_value:.4f})")
    ax.set_xlabel("Days since SMS")
    ax.set_ylabel("P(not yet returned)")
    fig.tight_layout()
    fig.savefig(OUT / "06_fig_km_return.png", dpi=120)
    plt.close(fig)
    print("  saved 06_fig_km_return.png")


# ── 4. Logistic regression ────────────────────────────────────────────────────

def logistic_return(frame: pd.DataFrame) -> None:
    print("\n=== 4. Logistic regression: P(return | delivered) ===")
    try:
        reg = data.load_registration()
        d = frame.merge(
            reg[["pid", "mod_readiness", "mod_age"]],
            on="pid", how="left"
        )
    except Exception:
        d = frame.copy()

    d["returned_binary"] = (d["returned"] == 1).astype(int)
    terms = ["is_delivered"]
    for c in ["mod_readiness", "mod_age"]:
        if c in d.columns and d[c].notna().sum() > 10:
            terms.append(c)
    formula = "returned_binary ~ " + " + ".join(terms)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            logit = smf.logit(formula, data=d.dropna(subset=terms)).fit(disp=False)
        except Exception as exc:
            print(f"  Logistic fit failed: {exc}")
            return

    print(logit.summary())
    pd.DataFrame({
        "coef": logit.params,
        "or":   np.exp(logit.params),
        "p":    logit.pvalues,
        "ci_lo_or": np.exp(logit.conf_int()[0]),
        "ci_hi_or": np.exp(logit.conf_int()[1]),
    }).to_csv(OUT / "06_logistic_return.csv")


# ── 5. Subgroup by segment ────────────────────────────────────────────────────

def segment_subgroup(frame: pd.DataFrame) -> None:
    print("\n=== 5. Return rate by engagement segment ===")
    seg_path = OUT / "segments_assignments.csv"
    if not seg_path.exists():
        print("  Segment file not found; skipping")
        return
    seg = pd.read_csv(seg_path)[["pid", "segment"]]
    d = frame.merge(seg, on="pid", how="inner")
    tab = d.groupby(["segment", "status"])["returned"].mean().unstack(fill_value=0)
    print(tab.round(3).to_string())
    tab.round(3).to_csv(OUT / "06_segment_return_rates.csv")


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    sms    = data.load_sms()
    events = data.load_events()
    sms_descriptives(sms)
    frame = event_study(sms, events)
    km_time_to_return(frame)
    logistic_return(frame)
    segment_subgroup(frame)
    print(f"\nDone. Results in {OUT}")


if __name__ == "__main__":
    main()
