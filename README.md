# cessation-dht

**Synthetic smoking-cessation digital-health cohort — methods demonstration**

This repository contains a fully synthetic longitudinal cohort generated from
published literature parameters, paired with a complete quantitative
research pipeline.  It is an explicit demonstration of methods, not a real
study.  All generation parameters are cited to peer-reviewed sources.

## What this demonstrates

| Skill | Scripts |
|---|---|
| Psychometric validation (CFA, IRT-GRM, metric invariance) | `analysis/01_psychometrics.py` |
| Behavioral segmentation (KMeans, log-rank retention) | `analysis/02_segmentation.py` |
| Survival analysis (Cox PH, Weibull AFT, OLS log-T) | `analysis/04_outcome_duration.py` |
| Quasi-experimental design (SMS event study) | `analysis/06_sms_reengagement.py` |
| Behavioral analytics (Markov chain, golden paths) | `analysis/09_golden_paths.py` |
| ML / XAI (HistGradientBoosting + SHAP) | `analysis/10_churn_ml.py` |
| Landmark causal design (quit-anchored longitudinal) | `analysis/11_quit_anchored.py` |

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
# Install
pip install uv
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

Full design specification: `docs/superpowers/specs/2026-06-26-cessation-dht-design.md`

## Synthetic data disclosure

The data in `data/synthetic/` are entirely generated from published parameters.
No real participant data are present.  `data/synthetic/generation_metadata.json`
records the random seed, N per table, and all citation keys used.
