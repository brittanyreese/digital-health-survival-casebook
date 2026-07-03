"""Analysis 06 -- SMS re-engagement: negative-control demonstration.

Runs the full delivered-vs-opted-out analysis stack (event study, Kaplan-Meier,
logistic regression) on the SMS campaign and reports app return within a 14-day
window.

This is a NEGATIVE CONTROL by construction, not a treatment-effect estimate.
The synthetic generator injects no causal effect of SMS delivery on app return:
opt-out timing is sampled independently of engagement propensity (theta_u), and
delivery status is a deterministic function of that random opt-out day
(reengagement.py:_sample_opt_out_day). The 14-day return outcome measured here
comes from the generic event log, which is generated without reference to SMS
delivery. So a correct pipeline should return a NULL delivered-vs-opted-out
contrast, and it does (delivered and opted-out return rates are within noise,
logistic is_delivered p >> 0.05). The value of the script is that the event-study
/ KM / logistic machinery is wired correctly and does not manufacture an effect
where the DGP has none.

What a real study would need: opt-out driven by (unobserved) motivation, an
actual SMS-to-return effect, and a design that separates the two. On real data
opt-out would be endogenous and the naive contrast would be confounded; none of
that endogeneity exists in this synthetic build.

Section 6 adds a complementary POSITIVE control. Unlike SMS delivery, the
generator does make 14-day return rise with latent engagement propensity
(reengagement.py: p_return = P_RETURN_14D + THETA_BETA * theta_u, injected on
reengagement_clean.reengaged). A correct pipeline should therefore recover a
positive engagement->return association there, and it does. Together the two
sections show the machinery reports an effect where the DGP has one and a null
where it does not.

Analyses:
  1. SMS delivery and opt-out descriptives
  2. Event study: app return rate within 14 days post-SMS
  3. Kaplan-Meier: time-to-return by delivery status
  4. Logistic regression: P(return | delivered) + covariates
  5. Subgroup analysis by engagement segment
  6. Reengagement return vs baseline engagement (positive control)

Run:  uv run python analysis/06_sms_reengagement.py
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path
from typing import cast

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
from statsmodels.tools.sm_exceptions import PerfectSeparationError

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
    opt_users = sms.loc[sms["status"] == "opted_out", "pid"].nunique()
    opt_rate  = opt_users / n_users
    print(f"  {n_total:,} SMS to {n_users:,} users")
    print(f"  Opt-out rate: {opt_users:,}/{n_users:,} = {opt_rate:.2%}")
    print("  (Calibration target: 48%; SmokefreeVET PMC5144826)")

    status_dist = sms.groupby("status")["pid"].nunique()
    print(f"\n  Status breakdown:\n{status_dist.to_string()}")
    status_dist.to_csv(OUT / "06_sms_status.csv")


# ── 2. Event study ────────────────────────────────────────────────────────────

def event_study(sms: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
    print(f"\n=== 2. Event study: return within {RETURN_WINDOW}d post-SMS ===")
    # First SMS per user with delivery status
    first = cast(pd.DataFrame, sms.sort_values("sms_seq")
               .groupby("pid")
               .first()
               .reset_index()[["pid", "day_offset", "status"]])
    first.rename(columns={"day_offset": "sms_day"}, inplace=True)

    # App events after first SMS
    ev_after = events.merge(first[["pid", "sms_day"]], on="pid")
    ev_after["days_after_sms"] = ev_after["day_offset"] - ev_after["sms_day"]
    returned_window = ev_after.loc[
        (ev_after["days_after_sms"] > 0) &
        (ev_after["days_after_sms"] <= RETURN_WINDOW),
        "pid"
    ].unique()

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


# ── 6. Reengagement return vs engagement propensity (positive control) ─────────

def reengagement_recovery(events: pd.DataFrame) -> None:
    """Recover the injected engagement->reengagement effect from reengagement_clean.

    Complements the SMS negative control above. The generator makes 14-day
    return after a delivered SMS rise with latent engagement propensity theta_u
    (reengagement.py: p_return = P_RETURN_14D + THETA_BETA * theta). theta is
    unobserved, so this regresses the observed reengaged outcome on a baseline
    engagement proxy (log events in day_offset < BASELINE_WINDOW_DAYS, an
    analogous log-event proxy to the ones scripts 04/11 use for theta). A
    positive odds ratio recovers the
    injected structure. As everywhere in this casebook the association reflects
    the injected parameter, not a causal SMS or engagement effect.

    Restricted to users with a delivered SMS (sms_seq > 0): the theta-driven
    outcome only applies to them, while no-SMS users have reengaged forced to 0.
    """
    print("\n=== 6. Reengagement return vs baseline engagement (positive control) ===")
    try:
        reng = data.load_reengagement()
    except (FileNotFoundError, OSError) as exc:
        print(f"  reengagement_clean not available ({exc}); skipping")
        return

    reng = reng[reng["sms_seq"] > 0]
    if len(reng) < 30 or reng["reengaged"].sum() < 10:
        print("  Insufficient reengagement outcomes; skipping")
        return

    baseline = events[events["day_offset"] < C.BASELINE_WINDOW_DAYS]
    eng = cast(pd.Series, baseline.groupby("pid").size()).reset_index(name="n_events")
    d = reng.merge(eng, on="pid", how="left")
    d["n_events"] = d["n_events"].fillna(0)
    d["log_events"] = np.log1p(d["n_events"])

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            logit = smf.logit("reengaged ~ log_events", data=d).fit(disp=False)
        except (np.linalg.LinAlgError, ValueError, PerfectSeparationError) as exc:
            print(f"  Logistic fit failed: {exc}")
            return

    or_ = float(np.exp(logit.params["log_events"]))
    p = float(logit.pvalues["log_events"])
    print(f"  reengaged ~ log_events: OR={or_:.3f} p={p:.4f} "
          f"n={len(d)} events={int(d['reengaged'].sum())}")
    pd.DataFrame({
        "coef": logit.params,
        "or":   np.exp(logit.params),
        "p":    logit.pvalues,
        "ci_lo_or": np.exp(logit.conf_int()[0]),
        "ci_hi_or": np.exp(logit.conf_int()[1]),
    }).to_csv(OUT / "06_reengagement_logit.csv")
    print("  saved 06_reengagement_logit.csv")


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    sms    = data.load_sms()
    events = data.load_events()
    sms_descriptives(sms)
    frame = event_study(sms, events)
    km_time_to_return(frame)
    logistic_return(frame)
    segment_subgroup(frame)
    reengagement_recovery(events)
    print(f"\nDone. Results in {OUT}")


if __name__ == "__main__":
    main()
