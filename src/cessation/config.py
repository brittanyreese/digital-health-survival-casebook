"""Project-wide paths and column-name constants."""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_SYNTHETIC = ROOT / "data" / "synthetic"
RESULTS = ROOT / "results" / "analysis"

PROCESSED = DATA_SYNTHETIC  # alias used by data.py loaders

DATA_SYNTHETIC.mkdir(parents=True, exist_ok=True)
RESULTS.mkdir(parents=True, exist_ok=True)

# ── outcome column names ──────────────────────────────────────────────────────
OUTCOME_EVENT    = "out_relapsed"       # 1 = relapsed within follow-up window
OUTCOME_DURATION = "out_days_quit"      # continuous days of abstinence
OUTCOME_6MO      = "out_abstinent_6mo"  # binary: still quit at 180 days

# ── moderator / covariate column names ───────────────────────────────────────
MODERATORS = {
    "readiness":      "mod_readiness",
    "age":            "mod_age",
    "education":      "mod_edu",
    "gender":         "mod_gender",
    "cigs_per_day":   "mod_cpd",
    "years_smoking":  "mod_yrs_smk",
    "stage":          "mod_stage",
}

# ── reengagement column names ─────────────────────────────────────────────────
REENG_COLS = {
    "last_login":      "last_login_offset",
    "days_since_quit": "reg_days_since_quit",
    "sms_seq":         "sms_seq",
}

# ── event channel labels ──────────────────────────────────────────────────────
EVENT_CHANNELS = ["craving_tool", "content", "peer_support", "notification", "quiz"]

# ── profile categories ────────────────────────────────────────────────────────
PROFILE_CLASSES = ["high_engager", "moderate", "passive"]

# ── random seed ───────────────────────────────────────────────────────────────
SEED = 42  # single source of truth for generation + analysis reproducibility

# ── follow-up + exposure windows ──────────────────────────────────────────────
FOLLOWUP_DAYS   = 180
CENSOR_DAYS     = FOLLOWUP_DAYS
# Pre-outcome baseline exposure window (days from enrollment). Shared by the
# churn features (script 10) and the channel-outcome correlation (script 09) so
# the two cannot silently drift apart.
BASELINE_WINDOW_DAYS = 30

# ── TTM stages ───────────────────────────────────────────────────────────────
TTM_STAGES = [
    "precontemplation",
    "contemplation",
    "preparation",
    "action",
    "maintenance",
]
