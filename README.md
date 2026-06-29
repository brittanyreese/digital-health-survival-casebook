# cessation-dht

**Synthetic smoking-cessation digital-health cohort — methods demonstration**

**This is entirely synthetic data and an explicit demonstration of analytical
methods.  No real participants are involved.  Results confirm the pipeline
recovers parameters that were injected at generation time; they do not
constitute evidence about real users.**

Cohort: N = 8,000 app users (synthetic); ~480-user psychometric subsample;
~1,200-user follow-up cohort for survival analysis.  Random seed: 42.
All generation parameters are cited to peer-reviewed sources.

## Context and motivation

Engagement with mHealth cessation apps predicts abstinence outcomes, but most
published analyses stop at descriptive usage metrics (session counts, days
active) rather than connecting engagement trajectories to clinical endpoints
through statistically appropriate models.  Perski et al. (2017, *Health
Psychology Review*) documented that engagement is multidimensional and poorly
operationalized across digital health interventions.  Baumel et al. (2019,
*JMIR*) found that most digital mental health studies report aggregate usage
averages without trajectory-based outcome analyses.  The Cochrane review by
Whittaker et al. (2019) on mobile phone-based cessation interventions
identified that adherence metrics were rarely analyzed with survival-appropriate
estimators.

This repository demonstrates the full pipeline that would close that gap on
real data: psychometric validation of the underlying constructs, engagement
feature engineering with temporal validity constraints, survival analysis with
correct censoring handling, and ML-based early-warning prediction.  It uses
a synthetic cohort because the pipeline, not the data, is the contribution.

## What this demonstrates

| Skill | Script |
|---|---|
| Psychometric validation: CFA, IRT-GRM, metric invariance, misspecification checks | `analysis/01_psychometrics.py` |
| Behavioral segmentation: KMeans, silhouette selection, log-rank retention | `analysis/02_segmentation.py` |
| Survival analysis: Cox PH + Schoenfeld test, Weibull AFT primary, right-censoring | `analysis/04_outcome_duration.py` |
| SMS re-engagement quasi-experiment: event study, logistic regression | `analysis/06_sms_reengagement.py` |
| Behavioral analytics: Markov channel transitions, stationary distributions, channel-outcome correlation | `analysis/09_golden_paths.py` |
| ML / XAI: HistGradientBoosting, ROC-AUC + AUPRC, SHAP, calibration, subgroup AUC | `analysis/10_churn_ml.py` |
| Landmark analysis: quit-date anchoring, AFT window sensitivity, Schoenfeld test | `analysis/11_quit_anchored.py` |

Script numbers follow the full analysis pipeline order.  Data cleaning,
imputation, and feature selection stages are handled within the synthetic
generation layer for this demonstration and do not appear as separate scripts.

## Analytical design decisions

Three choices in this pipeline address common methodological pitfalls
in mHealth analytics:

**Exposure window restriction (scripts 04, 10, 11).** Total engagement
over the full follow-up period is a biased predictor of quit duration:
longer survivors accumulate more events by construction (immortal time
bias).  All survival and ML analyses restrict engagement features to a
pre-outcome baseline window (days 0-29 from enrollment in scripts 04 and 10;
days 0-29 post-quit-date in script 11's landmark design).

**Proportional hazards testing (scripts 04, 11).** Cox PH is fitted and
the Schoenfeld residual test is reported before choosing the primary estimator.
Where no violation is detected, Cox results are shown alongside Weibull AFT
for triangulation.  Where a violation occurs (script 11: `activated` covariate,
p = 0.037), Cox estimates are treated as descriptive only and Weibull AFT
is the primary result.

**Feature-outcome temporal separation (script 10).** Churn features come
from days 0-29; the churn label is defined over days 166-180.  A runtime
assertion confirms no post-window signal enters the feature set, approximating
the temporal constraint a prospective deployment model would face.

## Key results (synthetic data — parameter recovery only)

All values reflect how well the pipeline recovers parameters that were
injected into the synthetic data generator.  They do not generalize to
real users.  Each result is annotated with its implication for a real-data
study design.

| Analysis | Synthetic result | Real-data implication |
|---|---|---|
| Weibull AFT — Script 04 | exp(β) = 1.48 (95% CI 1.08–2.02), p = 0.015, per log-unit 30-day engagement | Engagement quartile differences would translate to roughly 1.5x longer abstinence windows — a detectable effect at N ≈ 300 with 80% power |
| Weibull AFT — Script 11 | exp(β) = 1.89 (95% CI 1.23–2.90), p = 0.004, quit-anchored 30-day window | Quit-anchored design removes post-hoc engagement bias; effect size 28% larger than enrollment-anchored window |
| Weibull shape κ — Scripts 04, 11 | κ = 0.54 (recovers injected κ = 0.55) | Decreasing-hazard profile: relapse risk peaks immediately post-quit, supporting early-window intervention timing |
| Cox PH test — Script 04 | All Schoenfeld p > 0.34; PH holds | Cox and Weibull estimates consistent; no PH-driven model selection required |
| Cox PH test — Script 11 | `activated` Schoenfeld p = 0.037; PH violated | Engagement-activation effect is time-varying; Weibull AFT is the correct primary estimator |
| AFT window sensitivity — Script 11 | exp(β) = 2.05 (p = 0.007) at 14d; exp(β) = 2.00 (p = 0.001) at 60d | Consistent effect across window widths; 30-day window is a reasonable default |
| CFA misspecification check — Script 01 | CFI < 0.93 on 1-factor data; CFI < 0.95 on bifactor data; CFI = 0.987 on correct 2-factor data | Pipeline discriminates trivial and subtle misspecification — the relevant validation on synthetic data |
| CFA fit — Script 01 | SDBS 2-factor: CFI = 0.987, RMSEA = 0.021 | Reflects parameter recovery fidelity; a real-data CFA on this instrument would face measurement noise, partial invariance, and potentially different factor structure across populations |
| Churn ROC-AUC — Script 10 | 0.844 ± 0.006 (5-fold stratified CV) | Above published DHT benchmark range of 0.65–0.78; elevated because synthetic signal-to-noise is clean; real AUC would likely fall in that range |
| Churn AUPRC — Script 10 | 0.334 ± 0.029 (no-skill baseline = 0.103) | 3.2x no-skill lift at 10.3% churn prevalence; AUPRC is the informative metric under class imbalance |
| Subgroup AUC — Script 10 | Age: below-median 0.831, above-median 0.855; Education: below-median 0.837, above-median 0.858 | Modest performance gap by age and education on synthetic data; real fairness audit would require race/ethnicity, rurality, and insurance status |
| Channel-outcome correlation — Script 09 | Content: r = 0.07 (p = 0.056); all others p > 0.39 | Channel mix does not predict quit duration beyond overall engagement volume on this cohort; individual channel effects require larger N and randomized exposure |

All analyses are exploratory within a single synthetic dataset.  No correction
for multiple comparisons is applied across scripts.  A pre-registered,
FDR-controlled replication on real data would be required before any clinical
interpretation.

## Synthetic data generation

Data are generated from a Gaussian-copula CFA model calibrated to:

- SDBS (Smoking Decisional Balance Scale, 20 items): Velicer et al. (1985).
  *J Pers Soc Psychol*, 48(5), 1279–1289.
- SSEQ-12 (Smoking Self-Efficacy Questionnaire): Etter et al. (2000).
  *Addiction*, 95(6), 901–913.
- TTM stage distribution: Prochaska et al. (1985). *Addict Behav*, 10(4), 395–406.
- Stage-conditional self-efficacy: DiClemente et al. (1985). *Cogn Ther Res*, 9(2), 181–200.
- Relapse kinetics (Weibull shape κ = 0.55): Etter & Stapleton (2006).
  *Tob Control*, 15, 280–285; Shiffman et al. (2007). *Am J Prev Med*, 32(3), 217–226.
- SMS opt-out (48% by 6 months): Duffy et al. (2016). *JMIR Mhealth Uhealth*,
  PMC5144826.
- Demographics: CDC NHANES 2019–2020 (SMQ); CDC MMWR 2020.

## Reproduce

```bash
# Install uv (https://docs.astral.sh/uv/getting-started/installation/)
curl -LsSf https://astral.sh/uv/install.sh | sh
uv sync

# Generate all synthetic tables (~30 seconds)
cessation-generate

# Validate distributional properties against calibration targets
cessation-validate

# Run analyses in order (02 must precede 04, 06, 09, 10, 11)
uv run python analysis/01_psychometrics.py
uv run python analysis/02_segmentation.py
uv run python analysis/04_outcome_duration.py
uv run python analysis/06_sms_reengagement.py
uv run python analysis/09_golden_paths.py
uv run python analysis/10_churn_ml.py
uv run python analysis/11_quit_anchored.py
```

Results (figures + CSV tables) appear in `results/analysis/`.
`data/synthetic/generation_metadata.json` records the seed, table row counts,
and all citation keys used at generation time.

## Synthetic data disclosure

The data in `data/synthetic/` are entirely generated from published parameters.
No real participant data are present.

**Psychometric analyses.** Because the synthetic data are generated from the
same CFA-compatible correlation structure that analysis/01 then confirms, the
CFA fit indices (CFI, RMSEA, SRMR) and IRT parameters reflect parameter
recovery fidelity, not real-world construct validity.  The most meaningful
psychometric result is therefore the misspecification check in section 7 of
script 01: fitting the 2-factor SDBS model on three distinct data-generating
processes (1-factor wrong DGP, 2-factor correct DGP, bifactor partially
wrong DGP) confirms the pipeline detects trivial misfit (CFI < 0.93 on the
1-factor case) and reports the harder bifactor case honestly (CFI < 0.95).
That discriminative power is what a real-data analysis would rely on.

**Survival analyses.** Scripts 04 and 11 use Weibull AFT as the primary
estimator because it handles right-censoring correctly and does not require
the proportional hazards assumption.  Cox PH is reported for comparison;
where the Schoenfeld test detects a PH violation (script 11: `activated`,
p = 0.037), Cox estimates are treated as descriptive only.  OLS on
log(duration) is shown for interpretability only — it treats censored
observations as uncensored failures and is biased.

**Landmark design.** The landmark analysis in script 11 demonstrates the
structural logic of quit-date anchoring: the exposure window closes before
outcome accrual begins, which rules out reverse causality by design.  On
jointly-generated synthetic data this exercises the code structure of that
design, not a causal mechanism.  Causal interpretation requires prospective
real data collected under the same protocol.

**ML pipeline.** Churn features are extracted from days 0-29; the churn
label is defined over days 166-180; a runtime assertion enforces the
temporal separation.  SHAP values are computed on a full-data model trained
after CV performance estimation — the standard pattern for explanation (CV
estimates generalization, full-data model maximizes explanation signal).
The synthetic AUC of 0.844 reflects the signal-to-noise ratio embedded in
the generation parameters; published DHT churn benchmarks fall in the
0.65–0.78 range.

**Benchmarks.** Six-month abstinence rates in mHealth cessation trials
range from 15–30% vs. 3–5% control (Whittaker et al., 2019, Cochrane).
Churn prediction AUC in comparable digital health analytics studies
typically falls in the 0.65–0.78 range.  These figures provide context for
interpreting what real-data results from this pipeline would look like.
