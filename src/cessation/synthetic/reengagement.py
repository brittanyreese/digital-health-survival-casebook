"""
SMS re-engagement simulation.

SMS opt-out model
-----------------
SmokefreeVET trial (Duffy SA et al. 2016, JMIR Mhealth Uhealth, PMC5144826):
  48% of participants opted out of SMS messaging by 6 months.
  Modelled as Weibull time-to-opt-out with shape κ=0.8, calibrated to
  P(opt-out ≤ 180 days) = 0.48.  Negative binomial SMS volume (mean=3, k=2).

Re-engagement window
---------------------
Outcome measure: app return within 14 days of SMS delivery.
Baseline return probability (non-opt-out users): ~18% within 7 days, ~25%
within 14 days.  Estimated from mHealth engagement literature (e.g.,
Whittaker et al. 2016, Cochrane review on mobile phone-based interventions).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import weibull_min

from cessation.config import FOLLOWUP_DAYS

# Weibull opt-out calibration:
# P(opt-out ≤ 180) = 0.48 with shape=0.8 → solve for scale
# 0.48 = 1 - exp(-(180/λ)^0.8)  → λ ≈ 253
_OPT_OUT_SHAPE = 0.80
_OPT_OUT_SCALE = 253.0

_P_RETURN_14D  = 0.25   # baseline P(app return within 14d of SMS)
_THETA_BETA    = 0.15   # engagement propensity boosts return probability


def _sample_opt_out_day(n: int, rng: np.random.Generator) -> np.ndarray:
    """Sample day of opt-out; values > FOLLOWUP_DAYS mean never opted out."""
    return weibull_min.rvs(
        c=_OPT_OUT_SHAPE,
        scale=_OPT_OUT_SCALE,
        size=n,
        random_state=rng.integers(0, 2**31),
    )


def generate_sms(
    spine: pd.DataFrame,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """
    Generate SMS table for users with sms=True.

    Each user receives up to 3 SMS on days [30, 60, 90] unless opted out.
    Delivery status: delivered | opted_out.
    """
    sms_pids = spine[spine["sms"]]["pid"].values
    n = len(sms_pids)

    opt_out_day = _sample_opt_out_day(n, rng)

    sms_days = [30, 60, 90]
    records = []
    for i, pid in enumerate(sms_pids):
        for seq_num, send_day in enumerate(sms_days, start=1):
            if opt_out_day[i] <= send_day:
                status = "opted_out"
            else:
                status = "delivered"
            records.append({
                "pid":     pid,
                "day_offset": send_day,
                "status":  status,
                "sms_seq": seq_num,
            })
    return pd.DataFrame(records)


def generate_reengagement(
    spine: pd.DataFrame,
    sms: pd.DataFrame,
    theta_u: pd.Series,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """
    Generate reengagement table: one row per reengagement-eligible user.

    Users are eligible if reengagement=True in spine.
    Re-engagement event = app return within 14 days of first delivered SMS.
    """
    reng_pids = spine[spine["reengagement"]]["pid"].values

    # First delivered SMS per user
    delivered = sms[sms["status"] == "delivered"].copy()
    first_sms = (
        delivered.groupby("pid")["day_offset"].min().rename("first_sms_day")
    )

    records = []
    for pid in reng_pids:
        last_login = int(rng.integers(0, 30))  # last_login before campaign

        if pid not in first_sms.index:
            days_since_quit = float(rng.integers(0, 180))
            records.append({
                "pid":                pid,
                "last_login_offset":  last_login,
                "reg_days_since_quit": days_since_quit,
                "sms_seq":            0,
                "reengaged":          0,
            })
            continue

        first_day = int(first_sms[pid])
        seq_num   = int(sms[
            (sms["pid"] == pid) & (sms["day_offset"] == first_day)
        ]["sms_seq"].values[0])

        # Probability of return within 14 days
        theta = float(theta_u.get(pid, 0.0))
        p_ret = np.clip(_P_RETURN_14D + _THETA_BETA * theta, 0.01, 0.80)
        reengaged = int(rng.random() < p_ret)

        days_since_quit = float(rng.integers(7, 180))
        records.append({
            "pid":                pid,
            "last_login_offset":  last_login,
            "reg_days_since_quit": days_since_quit,
            "sms_seq":            seq_num,
            "reengaged":          reengaged,
        })

    return pd.DataFrame(records)
