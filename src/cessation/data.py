"""Thin loaders for synthetic data tables."""
from __future__ import annotations

import pandas as pd

from cessation import config as C


def load_spine() -> pd.DataFrame:
    return pd.read_csv(C.PROCESSED / "spine.csv", low_memory=False)


def load_registration() -> pd.DataFrame:
    return pd.read_csv(C.PROCESSED / "registration_clean.csv", low_memory=False)


def load_survey() -> pd.DataFrame:
    return pd.read_csv(C.PROCESSED / "survey_pilot_clean.csv", low_memory=False)


def load_followup() -> pd.DataFrame:
    return pd.read_csv(C.PROCESSED / "followup_clean.csv", low_memory=False)


def load_events() -> pd.DataFrame:
    return pd.read_csv(C.PROCESSED / "events_clean.csv", low_memory=False)


def load_sms() -> pd.DataFrame:
    return pd.read_csv(C.PROCESSED / "sms_clean.csv", low_memory=False)


def load_reengagement() -> pd.DataFrame:
    return pd.read_csv(C.PROCESSED / "reengagement_clean.csv", low_memory=False)


def load_profile() -> pd.DataFrame:
    return pd.read_csv(C.PROCESSED / "profile_clean.csv", low_memory=False)
