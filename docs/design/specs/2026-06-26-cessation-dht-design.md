# Digital-health survival casebook: synthetic cohort design

**Date:** 2026-06-26
**Status:** approved

## Purpose

Public portfolio repository demonstrating quant UXR / outcomes research methods on a fully synthetic smoking-cessation cohort for the Acme quit-support program, an app from the fictional vendor Acme Health. All organizations, programs, and people are fictitious, and any resemblance to real studies or products is coincidental. All generation parameters are calibrated to published literature with inline citations. No real participant data is used or implied.

## Data generation architecture

Gaussian copula pipeline in five ordered phases:

1. **Population** (`population.py`): spine (8,000 users), registration (1,800), profile (8,000). Demographic calibration: CDC NHANES 2017–March 2020 pre-pandemic, CDC MMWR 2020.
2. **Survey** (`cli.py:_generate_survey`): SDBS (20 items), SSEQ-12 (12 items), MARS (17 items: 16 subscale + 1 global rating; reliability analyzed in script 01) for 480-user pre-quit cohort. CFA model with published factor loadings; stage-conditional means from Velicer 1985 / DiClemente 1985.
3. **Events** (`events.py`): 1.2M events via Poisson session arrivals (λ_u = exp(α + β·θ_u)) + first-order Markov channel transitions. 5 channels: craving_tool, content, peer_support, notification, quiz.
4. **Outcomes** (`outcomes.py`): Weibull(κ=0.55, λ=180d) baseline survival; Cox-type frailty: HR ~ exp(β_θ·θ_u + β_SSEQ·sseq_std + β_pros·pros_std). Right-censored at 180 days. 1,200-user follow-up cohort.
5. **SMS / reengagement** (`reengagement.py`): opt-out modelled as Weibull(κ=0.80, λ=253d) calibrated to 48% opt-out by 6 months (SmokefreeVET, PMC5144826). Re-engagement: P(return|14d) based on θ_u.

## Instruments and literature sources

| Instrument | Items | Scale | Source |
|---|---|---|---|
| SDBS | 20 (10 Pros + 10 Cons) | 5-pt (1–5) | Velicer et al. 1985, J Pers Soc Psychol 48(5):1279 |
| SSEQ-12 | 12 (6 Int + 6 Ext) | 5-pt (1–5) | Etter et al. 2000, Addiction 95(6):901 |
| MARS | 17 (16 across 4 subscales + 1 global rating) | 5-pt (1–5) | Stoyanov et al. 2015, JMIR Mhealth |

**Confirmed parameters (from PDF extraction):**
- SDBS α: Pros=.87, Cons=.90 (Velicer 1985, p.1283)
- SDBS stage means: Velicer 1985 Table 2 (all 5 stages, Cons/Pros mean+SD)
- SSEQ-12 α: Internal=.95, External=.94 (Etter 2000, Table 2)
- SSEQ-12 inter-factor r = 0.79 (Etter 2000, confirmed full text)
- SSEQ-12 factor loadings: I=[.85,.83,.82,.81,.78,.60]; II=[.87,.82,.79,.72,.68,.63]
- SSEQ stage-conditional scores: DiClemente 1985 Table III (31-item version, rescaled)
- Cross-scale r(SDBS, SSEQ): NOT injected. SDBS Pros, SDBS Cons, and the SSEQ subscales are generated as independent factor blocks (`cli.py:_generate_survey`), so no controlled Pros/Cons or SDBS/SSEQ cross-correlation is imposed; any residual association is a stage-mixture artifact. Only the within-SSEQ inter-factor r=0.79 is injected (via `factor_cov` in `generate_two_factor_items`). The DiClemente 1985 narrative estimate (≈±.20, "small but significant"; no table published) is a calibration reference, not a generated parameter.
- Weibull shape: κ≈0.55 (Hughes, Keely & Naud 2004)

## Analysis scripts (7)

| Script | Methods | Area |
|---|---|---|
| 01_psychometrics.py | Reliability, EFA/CFA, IRT-GRM, cross-group factor congruence | Psychometric validation |
| 02_segmentation.py | KMeans (silhouette), log-rank, KM retention | Behavioral segmentation |
| 04_outcome_duration.py | Cox PH, Weibull AFT, OLS log(T), Schoenfeld | Survival analysis |
| 06_sms_reengagement.py | Event study, KM, logistic regression | Quasi-experimental design |
| 09_golden_paths.py | Sessionization, Markov chain, stationary dist | Behavioral analytics |
| 10_churn_ml.py | HistGradientBoosting, nested CV, SHAP | ML / XAI |
| 11_quit_anchored.py | Landmark design, Cox/AFT/OLS, 7-attack robustness | Causal framing |

Numbering follows the full pipeline order. The unlisted numbers (03, 05, 07, 08) are data-preparation and shared-utility stages that live in the generation layer, not standalone analyses.

## Power analysis

Schoenfeld (1983) formula: events = 4(z_α+z_β)²/(ln HR)²
- HR=0.80, α=.05, power=.80: 197 events (continuous predictor)
- HR=0.80, α=.05, power=.80: 630 events (binary predictor)
- Generated 780 events at the synthetic relapse rate → covers both.

## Repository structure

```
digital-health-survival-casebook/
├── pyproject.toml          # hatchling, uv, ruff, pyright; CLI entry points
├── src/cessation/
│   ├── config.py           # paths + column-name constants
│   ├── data.py             # thin CSV loaders
│   └── synthetic/          # generation pipeline (8 modules)
├── analysis/               # 7 analysis scripts
├── data/synthetic/         # committed CSVs + generation_metadata.json
└── results/analysis/       # committed figures + result CSVs
```

## CLI

```bash
cessation-generate   # generate all 8 synthetic tables → data/synthetic/
cessation-validate   # run distributional validation suite
```

## Privacy

The dataset is fully synthetic; see Purpose. No real participant data exists in this repository.

## Toolchain

- Python 3.12, uv, hatchling
- Lint: ruff; type-check: pyright (standard mode)
- Analysis: pandas, numpy, scipy, lifelines, statsmodels, semopy, factor-analyzer, girth, shap, scikit-learn
