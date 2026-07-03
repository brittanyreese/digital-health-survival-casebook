"""
CLI entry points for synthetic data generation and validation.

Entry points (defined in pyproject.toml):
  cessation-generate   → generate all synthetic tables
  cessation-validate   → run distributional validation suite
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "src"))

from cessation import config as C
from cessation.synthetic import (
    instruments as inst,
)
from cessation.synthetic import validate as val
from cessation.synthetic.copula import (
    generate_one_factor_items,
    generate_two_factor_items,
)
from cessation.synthetic.events import generate_events
from cessation.synthetic.outcomes import generate_followup
from cessation.synthetic.population import (
    generate_profile,
    generate_registration,
    generate_spine,
)
from cessation.synthetic.reengagement import generate_reengagement, generate_sms

# ── constants ─────────────────────────────────────────────────────────────────
N_TOTAL   = 8_000
N_POINTS  = 5      # Likert scale points
SEED      = 42


def _banner(msg: str) -> None:
    print(f"\n{'─' * 60}\n  {msg}\n{'─' * 60}")


# ── survey generation ─────────────────────────────────────────────────────────
def _generate_survey(
    spine: pd.DataFrame,
    reg: pd.DataFrame,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Generate survey_pilot_clean.csv (SDBS + SSEQ-12 + MARS + moderators)."""
    survey_pids = spine.loc[spine["survey"], "pid"].to_numpy()
    n = len(survey_pids)

    # Stage lookup for each pid
    stage_map = (
        reg.set_index("pid")["mod_stage"].reindex(survey_pids).fillna("contemplation")
    )

    # ── SDBS T1 (20 items: 1-10=Pros, 11-20=Cons) ──────────────────────────
    # Scale factor for converting published total score to factor mean:
    # factor_mean ≈ (target_per_item - 3) / (loading/2)
    pros_records, cons_records = [], []
    for stage in C.TTM_STAGES:
        mask = (stage_map == stage).values
        n_s = mask.sum()
        if n_s == 0:
            continue
        cons_m, cons_sd, pros_m, pros_sd = inst.SDBS_STAGE_PARAMS[stage]
        # Convert total score mean → per-item mean → factor mean (z-units)
        pros_fac = (pros_m / 10 - 3.0) / (inst.SDBS_LOADING_PROS / 2.0)
        cons_fac = (cons_m / 10 - 3.0) / (inst.SDBS_LOADING_CONS / 2.0)

        pros_items = generate_one_factor_items(
            n_s, 10, inst.SDBS_LOADING_PROS, pros_fac, rng=rng
        )
        cons_items = generate_one_factor_items(
            n_s, 10, inst.SDBS_LOADING_CONS, cons_fac, rng=rng
        )
        pros_records.append((np.where(mask)[0], pros_items))
        cons_records.append((np.where(mask)[0], cons_items))

    sdbs_t1 = np.zeros((n, 20), dtype=int)
    for idx, items in pros_records:
        sdbs_t1[np.ix_(idx, range(10))] = items
    for idx, items in cons_records:
        sdbs_t1[np.ix_(idx, range(10, 20))] = items

    # ── SDBS T2 (70% have T2 data) ──────────────────────────────────────────
    has_t2 = rng.random(n) < 0.70
    sdbs_t2 = sdbs_t1.copy()  # start from T1, add noise for change
    noise = rng.normal(0, 0.3, size=(n, 20)).clip(-1, 1)
    sdbs_t2 = np.clip(sdbs_t2 + noise.round(0).astype(int), 1, 5)
    sdbs_t2[~has_t2] = 0  # 0 = missing (not responded)

    # ── SSEQ-12 (12 items: 1-6=Internal, 7-12=External) ─────────────────────
    factor_cov = np.array([
        [1.0,                    inst.SSEQ_INTERFACTOR_R],
        [inst.SSEQ_INTERFACTOR_R, 1.0],
    ])
    sseq_all = np.zeros((n, 12), dtype=int)
    for stage in C.TTM_STAGES:
        mask = (stage_map == stage).values
        n_s = mask.sum()
        if n_s == 0:
            continue
        sseq_m, sseq_sd = inst.SSEQ_STAGE_PARAMS[stage]
        # Convert 12-item sum score (range 12–60) → per-item → factor mean
        fac_mean = (sseq_m / 12 - 3.0) / 0.40  # avg loading ~0.80 → slope ≈ 0.40
        fac_means = np.array([fac_mean, fac_mean])  # both factors similar
        items = generate_two_factor_items(
            n_s,
            loadings_f1=inst.SSEQ_LOADINGS_INTERNAL,
            loadings_f2=inst.SSEQ_LOADINGS_EXTERNAL,
            factor_means=fac_means,
            factor_cov=factor_cov,
            rng=rng,
        )
        sseq_all[mask] = items

    # ── MARS (23 items, 4 subscales) ─────────────────────────────────────────
    mars_all = np.zeros((n, 23), dtype=int)
    col_start = 0
    for subscale, n_items in inst.MARS_ITEMS.items():
        m, sd = inst.MARS_SUBSCALE_PARAMS[subscale]
        fac_mean = (m - 3.0) / 0.35
        items = generate_one_factor_items(n, n_items, 0.70, fac_mean, rng=rng)
        mars_all[:, col_start:col_start + n_items] = items
        col_start += n_items
    # Overall quality item (item 23)
    mars_all[:, 22] = rng.integers(2, 6, size=n)

    # ── Assemble DataFrame ────────────────────────────────────────────────────
    sdbs_t1_cols = {f"t1_sdbs_{i:02d}": sdbs_t1[:, i - 1] for i in range(1, 21)}
    sdbs_t2_cols = {f"t2_sdbs_{i:02d}": sdbs_t2[:, i - 1] for i in range(1, 21)}
    sseq_cols    = {f"sseq_{i:02d}": sseq_all[:, i - 1] for i in range(1, 13)}
    mars_cols    = {f"mars_{i:02d}": mars_all[:, i - 1] for i in range(1, 24)}

    df = pd.DataFrame({
        "pid":       survey_pids,
        "mod_stage": stage_map.values,
        **sdbs_t1_cols,
        **sdbs_t2_cols,
        **sseq_cols,
        **mars_cols,
    })

    # Merge reg moderators
    reg_sub = reg.loc[
        reg["pid"].isin(survey_pids),
        ["pid", "mod_readiness", "mod_age", "mod_edu", "mod_gender",
         "mod_cpd", "mod_yrs_smk"],
    ]
    df = df.merge(reg_sub, on="pid", how="left")

    # Mask T2 missing as NaN
    t2_cols = [c for c in df.columns if c.startswith("t2_")]
    for c in t2_cols:
        df.loc[df[c] == 0, c] = np.nan

    return df


# ── main generate entry point ─────────────────────────────────────────────────
def generate() -> None:
    t0 = time.time()
    _banner(f"cessation-generate  (N={N_TOTAL:,}  seed={SEED})")
    rng = np.random.default_rng(SEED)

    _banner("Phase 1 — Spine")
    spine = generate_spine(N_TOTAL, seed=SEED, rng=rng)
    spine.to_csv(C.PROCESSED / "spine.csv", index=False)
    print(f"  spine: {len(spine):,} rows")

    _banner("Phase 2 — Registration")
    reg = generate_registration(spine, rng)
    reg.to_csv(C.PROCESSED / "registration_clean.csv", index=False)
    print(f"  registration: {len(reg):,} rows")

    _banner("Phase 3 — Profile")
    profile = generate_profile(spine, rng)
    profile.to_csv(C.PROCESSED / "profile_clean.csv", index=False)
    print(f"  profile: {len(profile):,} rows")

    _banner("Phase 4 — Survey (SDBS + SSEQ-12 + MARS)")
    survey = _generate_survey(spine, reg, rng)
    survey.to_csv(C.PROCESSED / "survey_pilot_clean.csv", index=False)
    print(f"  survey: {len(survey):,} rows × {len(survey.columns)} cols")

    # Composite scores for survival model
    sdbs_pros_cols = [f"t1_sdbs_{i:02d}" for i in range(1, 11)]
    sdbs_cons_cols = [f"t1_sdbs_{i:02d}" for i in range(11, 21)]
    sseq_cols      = [f"sseq_{i:02d}" for i in range(1, 13)]
    sdbs_scores = cast(pd.DataFrame, survey[["pid"]].copy())
    sdbs_scores["sdbs_pros"] = survey[sdbs_pros_cols].sum(axis=1)
    sdbs_scores["sdbs_cons"] = survey[sdbs_cons_cols].sum(axis=1)
    sseq_scores = cast(pd.DataFrame, survey[["pid"]].copy())
    sseq_scores["sseq_composite"] = survey[sseq_cols].sum(axis=1)

    _banner("Phase 5 — Events")
    events, theta_u = generate_events(spine, rng)
    events.to_csv(C.PROCESSED / "events_clean.csv", index=False)
    print(f"  events: {len(events):,} rows")

    _banner("Phase 6 — Follow-up outcomes")
    followup = generate_followup(
        spine, theta_u,
        sseq_scores=sseq_scores,
        sdbs_scores=sdbs_scores,
        rng=rng,
    )
    followup.to_csv(C.PROCESSED / "followup_clean.csv", index=False)
    print(f"  followup: {len(followup):,} rows")
    relapse_rate = followup["out_relapsed"].mean()
    median_days  = followup.loc[followup["out_relapsed"] == 1, "out_days_quit"].median()
    print(
        f"  relapse rate={relapse_rate:.2%}  "
        f"median days quit (relapsers)={median_days:.0f}"
    )

    _banner("Phase 7 — SMS")
    sms = generate_sms(spine, rng)
    sms.to_csv(C.PROCESSED / "sms_clean.csv", index=False)
    print(f"  sms: {len(sms):,} rows")

    _banner("Phase 8 — Reengagement")
    reeng = generate_reengagement(spine, sms, theta_u, rng)
    reeng.to_csv(C.PROCESSED / "reengagement_clean.csv", index=False)
    print(f"  reengagement: {len(reeng):,} rows")

    # ── metadata ──────────────────────────────────────────────────────────────
    meta = {
        "seed":        SEED,
        "n_total":     N_TOTAL,
        "generated":   time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "elapsed_sec": round(time.time() - t0, 1),
        "citations": {
            "SDBS": "Velicer et al. (1985) J Pers Soc Psychol 48(5):1279–1289",
            "SSEQ": "Etter et al. (2000) Addiction 95(6):901–913",
            "TTM_stage": "Prochaska et al. (1985) Addict Behav 10(4):395–406",
            "SSEQ_stage": "DiClemente et al. (1985) Cogn Ther Res 9(2):181–200",
            "Weibull_relapse": (
                "Etter & Stapleton (2006) Tob Control 15:280–285; "
                "Hughes, Keely & Naud (2004) Addiction 99(1):29–38"
            ),
            "SMS_optout": "Christofferson et al. (2016) Addictive Behaviors 62:47-53 PMC5144826",
            "demographics": "CDC NHANES 2019–2020 SMQ; CDC MMWR 2020",
        },
        "tables": {
            "spine":                len(spine),
            "registration_clean":   len(reg),
            "profile_clean":        len(profile),
            "survey_pilot_clean":   len(survey),
            "events_clean":         len(events),
            "followup_clean":       len(followup),
            "sms_clean":            len(sms),
            "reengagement_clean":   len(reeng),
        },
    }
    meta_path = C.PROCESSED / "generation_metadata.json"
    meta_path.write_text(json.dumps(meta, indent=2))
    print(f"\n  metadata → {meta_path}")
    _banner(f"Done in {time.time() - t0:.0f}s")


# ── validate entry point ──────────────────────────────────────────────────────
def validate() -> None:
    _banner("cessation-validate")

    spine    = pd.read_csv(C.PROCESSED / "spine.csv")
    survey   = pd.read_csv(C.PROCESSED / "survey_pilot_clean.csv")
    followup = pd.read_csv(C.PROCESSED / "followup_clean.csv")
    sms      = pd.read_csv(C.PROCESSED / "sms_clean.csv")
    events   = pd.read_csv(C.PROCESSED / "events_clean.csv")

    # Reconstruct theta_u from events as proxy
    eng = cast(pd.Series, events.groupby("pid").size()).rename("n_events")
    theta_proxy = (np.log1p(eng) - np.log1p(eng).mean()) / np.log1p(eng).std()
    theta_proxy.name = "theta_u"

    results = val.run_all(spine, survey, followup, sms, events, theta_proxy)
    val.print_report(results)

    n_fail = sum(not r.passed for r in results)
    sys.exit(1 if n_fail > 0 else 0)
