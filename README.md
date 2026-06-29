# cessation-dht

**Synthetic smoking-cessation digital-health cohort — methods demonstration**

This repository contains a fully synthetic longitudinal cohort generated from
published literature parameters, paired with a complete quantitative
research pipeline.  It is an explicit demonstration of methods, not a real
study.  All generation parameters are cited to peer-reviewed sources.

## What this demonstrates

| Skill | Scripts |
|---|---|
| Psychometric validation (CFA, IRT-GRM, metric invariance, misspecification checks) | `analysis/01_psychometrics.py` |
| Behavioral segmentation (KMeans, silhouette selection, log-rank retention) | `analysis/02_segmentation.py` |
| Survival analysis (Cox PH, Weibull AFT primary, right-censoring handling) | `analysis/04_outcome_duration.py` |
| SMS re-engagement analysis (event study) | `analysis/06_sms_reengagement.py` |
| Behavioral analytics (Markov chain, golden paths) | `analysis/09_golden_paths.py` |
| ML / XAI (HistGradientBoosting + SHAP, calibration, subgroup AUC) | `analysis/10_churn_ml.py` |
| Landmark analysis design (quit-anchored longitudinal, AFT window sensitivity) | `analysis/11_quit_anchored.py` |

Scripts are numbered by analysis order.  Scripts 03, 05, 07, and 08 are
reserved for data-cleaning, imputation, sensitivity modeling, and feature
selection pipelines not included in this showcase.

## Key results (synthetic data)

All figures reflect parameter recovery from calibrated synthetic data.  They
confirm the analytical pipeline correctly recovers injected parameters, not
that real app users would show these associations.  See "Synthetic data
disclosure" below.

| Analysis | Finding |
|---|---|
| Weibull AFT (primary) — Script 04 | exp(β) = 1.55 (95% CI 1.16–2.06) per log-unit engagement, p = 0.003 |
| Weibull AFT (primary) — Script 11 | exp(β) = 1.89 (95% CI 1.23–2.90) in quit-anchored window, p = 0.004 |
| Cox PH — Script 04 | All Schoenfeld p > 0.26; proportional hazards holds |
| Cox PH — Script 11 | `activated` Schoenfeld p = 0.037 (violation); Weibull AFT used as primary |
| AFT window sensitivity — Script 11 | exp(β) = 2.05 (p = 0.007) at 14d; exp(β) = 2.00 (p = 0.001) at 60d |
| Churn prediction AUC — Script 10 | 0.843 ± 0.006 (5-fold stratified CV); majority-class baseline = 0.50 |
| CFA fit — Script 01 | SDBS 2-factor: CFI = 0.987, RMSEA = 0.021 (parameter recovery) |

All analyses are exploratory within a single synthetic dataset.  No correction
for multiple comparisons is applied across scripts.  A pre-registered,
FDR-controlled replication on real data would be required before any clinical
interpretation.

## Synthetic data generation

Data are generated from a Gaussian-copula CFA model calibrated to:

- **SDBS** (Smoking Decisional Balance Scale, 20 items): Velicer et al. (1985).
  *J Pers Soc Psychol*, 48(5), 1279–1289.
- **SSEQ-12** (Smoking Self-Efficacy Questionnaire): Etter et al. (2000).
  *Addiction*, 95(6), 901–913.
- **TTM stage distribution**: Prochaska et al. (1985). *Addict Behav*, 10(4), 395–406.
- **Stage-conditional self-efficacy**: DiClemente et al. (1985). *Cogn Ther Res*, 9(2), 181–200.
- **Relapse kinetics (Weibull shape κ=0.55)**: Etter & Stapleton (2006).
  *Tob Control*, 15, 280–285; Shiffman et al. (2007). *Am J Prev Med*, 32(3), 217–226.
- **SMS opt-out (48% by 6 months)**: Duffy et al. (2016). *JMIR Mhealth Uhealth*,
  PMC5144826.
- **Demographics**: CDC NHANES 2019–2020 (SMQ); CDC MMWR 2020.

## Reproduce

```bash
# Install uv (https://docs.astral.sh/uv/getting-started/installation/)
curl -LsSf https://astral.sh/uv/install.sh | sh
uv sync

# Generate all synthetic tables (~30 seconds)
cessation-generate

# Validate distributional properties
cessation-validate

# Run analyses
uv run python analysis/01_psychometrics.py
uv run python analysis/02_segmentation.py
uv run python analysis/04_outcome_duration.py
uv run python analysis/06_sms_reengagement.py
uv run python analysis/09_golden_paths.py
uv run python analysis/10_churn_ml.py
uv run python analysis/11_quit_anchored.py
```

Results (figures + CSV tables) appear in `results/analysis/`.

## Design

See `docs/` for the full design specification.

## Synthetic data disclosure

The data in `data/synthetic/` are entirely generated from published parameters.
No real participant data are present.  `data/synthetic/generation_metadata.json`
records the random seed, N per table, and all citation keys used.

Because the synthetic data are generated from the same CFA-compatible
correlation structure that analysis/01 then confirms, the CFA fit indices
(CFI, RMSEA, SRMR) and IRT parameters reflect parameter recovery fidelity,
not real-world construct validity.  Script 01 includes a misspecification
check with three cases: a trivially wrong 1-factor model, the correctly
specified 2-factor model, and a harder bifactor DGP where a 2-factor model
is partially misspecified.  Results confirm the pipeline detects easy
misfit (CFI < 0.93 on the 1-factor case) and reports the harder bifactor
case honestly.

Survival analyses (scripts 04, 11) use Weibull AFT as the primary
estimator because it handles right-censoring correctly and does not require
the proportional hazards assumption.  Cox PH is reported for comparison;
where the Schoenfeld test detects a PH violation (script 11: `activated`,
p = 0.037), Cox estimates are treated as descriptive only.  OLS on
log(duration) is shown for interpretability but is biased for
right-censored outcomes and should not be taken as a primary result.

The landmark design in analysis/11 demonstrates the structural logic of
quit-date anchoring: exposure window closes before outcome accrual begins.
On jointly-generated synthetic data this exercises the code structure of the
design, not a causal mechanism.  Causal interpretation requires real data
with the same design applied prospectively.

Published benchmarks for context: 6-month abstinence rates in mHealth
cessation trials range from 15–30% (vs. 3–5% control; Whittaker et al.
2019, Cochrane).  Churn prediction AUC in comparable digital health
analytics studies typically falls in the 0.65–0.78 range.  The synthetic
AUC of 0.84 reflects the signal-to-noise ratio embedded in the generation
parameters, not a claim about deployment performance.
