"""
Literature-calibrated instrument parameters for synthetic generation.

Sources
-------
SDBS  : Velicer WF, DiClemente CC, Prochaska JO, Brandenburg N (1985).
        Decisional balance measure for assessing and predicting smoking status.
        J Pers Soc Psychol, 48(5), 1279–1289.
SSEQ  : Etter JF, Bergman MM, Humair JP, Perneger TV (2000).
        Development and validation of a scale measuring self-efficacy of
        current and former smokers. Addiction, 95(6), 901–913.
Stage : Prochaska JO, DiClemente CC, Velicer WF, Ginpil S, Norcross JC (1985).
        Predicting change in smoking status for self-changers. Addict Behav,
        10(4), 395–406. (Stage distribution: contemplation ~40% of cessation
        sample; preparation ~20%; precontemplation ~30%.)
DiCl  : DiClemente CC, Prochaska JO, Gibertini M (1985). Self-efficacy and
        the stages of self-change of smoking. Cogn Ther Res, 9(2), 181–200.
        (Stage-conditional self-efficacy norms, Table III; cross-scale r
        described as "small but significant".)
"""
from __future__ import annotations

import numpy as np

# ── TTM stage order ────────────────────────────────────────────────────────────
STAGES = ["precontemplation", "contemplation", "preparation", "action", "maintenance"]

# ── SDBS (Velicer 1985) ────────────────────────────────────────────────────────
# 20 items: 1–10 = Pros, 11–20 = Cons; 5-pt (1=strongly disagree–5=strongly agree)
# Subscale range: 10–50.
# Alpha: Pros = .87, Cons = .90 (Velicer 1985, p.1283)
SDBS_ALPHA = {"pros": 0.87, "cons": 0.90}

# Mean inter-item r implied by alpha = n*r/(1+(n-1)*r), n=10
# Pros: r = .87/(10 - .87*9) = .87/2.17 ≈ .401 → loading λ_avg ≈ sqrt(.401) ≈ .633
# Cons: r = .90/(10 - .90*9) = .90/1.90 ≈ .474 → loading λ_avg ≈ sqrt(.474) ≈ .688
SDBS_LOADING_PROS = 0.633
SDBS_LOADING_CONS = 0.688

# Stage-conditional subscale TOTAL means and SDs (Velicer 1985, Table 2, N=1000)
# Format: (cons_mean, cons_sd, pros_mean, pros_sd)
SDBS_STAGE_PARAMS: dict[str, tuple[float, float, float, float]] = {
    "precontemplation": (21.35, 6.69, 23.01, 7.57),  # Immotives
    "contemplation":    (27.03, 6.99, 23.72, 6.32),  # Contemplators
    "preparation":      (27.06, 7.41, 24.18, 6.22),  # Relapsers (proxy)
    "action":           (24.83, 7.01, 20.01, 6.82),  # Recent quitters
    "maintenance":      (21.76, 8.93, 17.33, 8.46),  # Long-term quitters
}

# ── SSEQ-12 (Etter 2000) ──────────────────────────────────────────────────────
# 12 items: 1–6 = Internal stimuli, 7–12 = External stimuli
# 5-pt (1=not at all sure–5=absolutely sure)
# Alpha: Internal = .95, External = .94 (Etter 2000, Table 2)
SSEQ_ALPHA = {"internal": 0.95, "external": 0.94}

# Factor loadings (×100 in Etter 2000, Table 1; two-factor oblique solution)
# Internal stimuli items (6):
SSEQ_LOADINGS_INTERNAL = np.array([0.85, 0.83, 0.82, 0.81, 0.78, 0.60])
# External stimuli items (6):
SSEQ_LOADINGS_EXTERNAL = np.array([0.87, 0.82, 0.79, 0.72, 0.68, 0.63])

# Inter-factor correlation: r = 0.79 (Etter 2000, confirmed from full text)
SSEQ_INTERFACTOR_R = 0.79

# Stage-conditional self-efficacy scores for calibration.
# Source: DiClemente 1985, Table III (31-item version; rescaled proportionally
# from 31-item range 31–155 to 12-item SSEQ range 12–60 using factor 12/31).
# Per-factor means estimated from total confidence score (Table III col 2).
_DICL_TOTAL = {
    "precontemplation": (60.0, 23.0),   # Immotives
    "contemplation":    (67.0, 20.0),   # Contemplators
    "preparation":      (69.0, 23.0),   # Relapsers (proxy)
    "action":           (117.0, 28.0),  # Recent quitters
    "maintenance":      (131.0, 32.0),  # Long-term quitters
}
# Rescale DiClemente 31-item scores to 12-item SSEQ-12 range (12–60)
# Scale factor = 12 / 31 ≈ 0.387
_SSEQ_SCALE = 12.0 / 31.0
SSEQ_STAGE_PARAMS: dict[str, tuple[float, float]] = {
    s: (m * _SSEQ_SCALE, sd * _SSEQ_SCALE) for s, (m, sd) in _DICL_TOTAL.items()
}

# ── Cross-scale correlations ───────────────────────────────────────────────────
# Not explicitly tabulated in any retrieved paper.
# DiClemente 1985 (p.193): "decision-making variables demonstrated a small but
# significant relationship with efficacy."  Direction: Pros↑ → efficacy↓;
# Cons↑ → efficacy↑ (for active-change smokers, pre-saliency effect).
# Approximated as r ≈ ±.20 (consistent with "small" per Cohen 1988 conventions).
CROSS_SCALE_R: dict[tuple[str, str], float] = {
    ("pros", "sseq_internal"):  -0.20,
    ("pros", "sseq_external"):  -0.18,
    ("cons", "sseq_internal"):  +0.20,
    ("cons", "sseq_external"):  +0.15,
}

# ── MARS (Stoyanov et al. 2015) ────────────────────────────────────────────────
# 17 items: 16 across 4 subscales + 1 single global app-quality rating, 5-pt
# (1=inadequate–5=excellent). Mean ratings from Stoyanov SR et al. (2015) JMIR
# Mhealth Uhealth, 3(1):e27. Subscale reliability is analyzed in analysis/01.
# Subscales are generated as independent blocks (no injected cross-subscale
# correlation), so the analysis recovers per-subscale reliability, not a
# correlated 4-factor structure.
MARS_SUBSCALE_PARAMS: dict[str, tuple[float, float]] = {
    "engagement":     (3.10, 0.70),
    "functionality":  (4.10, 0.60),
    "aesthetics":     (3.40, 0.70),
    "information":    (3.70, 0.60),
}
# Items per subscale
MARS_ITEMS = {"engagement": 5, "functionality": 4, "aesthetics": 3, "information": 4}
MARS_QUALITY_ITEMS = 1   # single overall quality item (item 23)