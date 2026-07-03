# Digital health survival casebook

**Simulated smoking-cessation digital-health cohort — methods demonstration**

**This is entirely synthetic data and an explicit demonstration of analytical
methods.  No real participants are involved.  Results confirm the pipeline
recovers parameters that were injected at generation time; they do not
constitute evidence about real users.**

Cohort: N = 8,000 app users (synthetic); ~480-user psychometric subsample;
~1,200-user follow-up cohort for survival analysis.  Random seed: 42.
All generation parameters are cited to peer-reviewed literature and public
federal data sources (NHANES, MMWR).

## Context and motivation

Engagement with mHealth cessation apps predicts abstinence outcomes, but most
published analyses stop at descriptive usage metrics (session counts, days
active) rather than connecting engagement trajectories to clinical endpoints
through statistically appropriate models.  Perski et al. (2017, *Translational
Behavioral Medicine*) documented that engagement is multidimensional and poorly
operationalized across digital health interventions.  Baumel et al. (2019,
*JMIR*) found that most digital mental health studies report aggregate usage
averages without trajectory-based outcome analyses.  The Cochrane review by
Whittaker et al. (2019, "Mobile phone text messaging and app-based
interventions for smoking cessation") identified that adherence metrics were
rarely analyzed with survival-appropriate estimators.

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

**Exposure window restriction and landmark design (scripts 04, 10, 11).**
Total engagement over the full follow-up period is a biased predictor of
quit duration: longer survivors accumulate more events by construction
(immortal time bias).  Scripts 04 and 10 restrict engagement features to a
pre-outcome baseline window (days 0-29 from enrollment).  Script 11
implements a proper landmark analysis (Andersen & Gill, 1982; van
Houwelingen, 2007): subjects who relapse before the window closes are
excluded entirely (their event counts
are mechanically capped by how long they survived), and the survival time
origin is shifted to the window close date (day 30 post-quit).  This removes
the structural dependence between event count and survival time in the
at-risk sample.

**Proportional hazards testing (scripts 04, 11).** Cox PH is fitted and
the Schoenfeld residual test is reported alongside the primary estimator.
In both scripts, no PH violation is detected (script 04: all p > 0.35;
script 11: all 6 tests p > 0.24).  Weibull AFT is pre-specified as primary
regardless, because it does not require the proportional hazards assumption.
Cox uses the same parsimonious covariate set as AFT within each script:
script 04 uses log_n_events + demographic moderators (no `activated`);
script 11 adds `activated`.  log_active_days and channel covariates are
excluded as collinear in both, enabling direct HR vs. exp(β) comparison.

**Feature-outcome temporal separation (script 10).** Churn features come
from days 0-29; the churn label is defined over days 166-180.  A runtime
assertion confirms the feature and label windows do not overlap (window-boundary
check on the two constants).  The column-level filter `day_offset < EARLY_WINDOW`
enforces the separation at the feature-engineering step.

## Key results (synthetic data — parameter recovery only)

All values reflect how well the pipeline recovers parameters that were
injected into the synthetic data generator.  They do not generalize to
real users.  Each result is annotated with its implication for a real-data
study design.

| Analysis | Synthetic result | Real-data implication |
|---|---|---|
| Weibull AFT — Script 04 | exp(β) = 1.48 (95% CI 1.08–2.02), p = 0.015, per log-unit 30-day engagement | Engagement quartile differences would translate to roughly 1.5x longer abstinence windows — a detectable effect at N ≈ 300 with 80% power |
| Weibull AFT — Script 11 | exp(β) = 2.08 (95% CI 1.21–3.57), p = 0.008; activated exp(β) = 0.31 (p = 0.044); n=74 post-landmark (39 early relapsers excluded); 27 events (UNDERPOWERED) | Landmark exclusion removes the immortal-time variant; engagement effect survives; activated result unreliable at 27 events — treat as directional only |
| Weibull shape κ — Scripts 04, 11 | κ = 0.54 (script 04, enrollment-anchored); κ = 0.94 (script 11, post-landmark; 95% CI 0.66–1.33); injected κ = 0.55 | Script 04 recovers the injected decreasing-hazard shape; script 11's near-flat hazard (κ ≈ 1) is correct for the post-landmark cohort — early relapse risk was conditioned away by landmark exclusion; κ is estimated with high uncertainty at 27 events (SE = 0.18) |
| Cox PH test — Script 04 | All Schoenfeld p > 0.35; PH holds; Cox HR = 0.822 (p = 0.023, MLE; penalizer=0), directionally consistent with AFT | Aligned covariate sets enable direct Cox/AFT comparison; both significant in same direction |
| Cox PH test — Script 11 | All 6 Schoenfeld p > 0.24 (minimum: mod_cpd p = 0.243); no PH violation detected | Weibull AFT is pre-specified as primary regardless; Cox uses the same parsimonious covariate set as AFT (log_n_events + activated + demographic moderators) for direct comparison |
| AFT window sensitivity — Script 11 | exp(β) = 1.97 (p = 0.021, n=87) at 14d; exp(β) = 1.85 (p = 0.029, n=66) at 60d | Effect attenuates slightly at wider windows after landmark; monotonic strengthening seen pre-landmark was the bias signature |
| CFA misspecification check — Script 01 | CFI = 0.469 on 1-factor case (trivial misfit detected); CFI = 0.997 on correct 2-factor case; CFI = 1.003 on bifactor case (degenerate — check fails; script flags `detects_hard=False`) | Pipeline detects trivial misfit but cannot distinguish bifactor from correlated-factor structure; the script's own warning correctly bounds this scope |
| CFA fit — Script 01 | SDBS 2-factor: CFI = 0.987, RMSEA = 0.021 | Reflects parameter recovery fidelity; a real-data CFA on this instrument would face measurement noise, partial invariance, and potentially different factor structure across populations |
| Churn ROC-AUC — Script 10 | 0.844 ± 0.006 (5-fold stratified CV) | Above published DHT benchmark range of 0.65–0.78; elevated because synthetic signal-to-noise is clean; real AUC would likely fall in that range |
| Churn AUPRC — Script 10 | 0.334 ± 0.029 (no-skill baseline = 0.103) | 3.2x no-skill lift at 10.3% churn prevalence; AUPRC is the informative metric under class imbalance |
| Subgroup AUC — Script 10 | Age: below-median 0.831, above-median 0.855; Education: below-median 0.837, above-median 0.858 | Modest performance gap by age and education on synthetic data; real fairness audit would require race/ethnicity, rurality, and insurance status |
| Channel-outcome correlation — Script 09 | All channels p > 0.28 in 30-day baseline window (content r = 0.04; all others r ≤ 0.01); restricted to observed relapsers (n = 694; censored abstainers excluded) | Channel mix in the first 30 days shows no association with quit duration (unadjusted bivariate; volume not controlled); individual channel effects require larger N and randomized exposure |

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
recovery fidelity, not real-world construct validity.  The misspecification
check in section 7 of script 01 fits the 2-factor SDBS model on three
data-generating processes: 1-factor case CFI = 0.469 (trivial misfit
detected), correct 2-factor case CFI = 0.997 (good fit confirmed), bifactor
case CFI = 1.003 (degenerate — the 2-factor model absorbs enough bifactor
variance to appear to fit, and the check fails to detect the subtle
misspecification).  The script correctly flags this via a
`detects_hard=False` warning (analysis/01_psychometrics.py:417).  The honest
scope: the pipeline discriminates trivial misspecification but cannot
distinguish bifactor from correlated-factor structure on these parameter
values.  A real-data application would need to fit the bifactor model
explicitly and compare it to the 2-factor model via BIC or LRT.

**Survival analyses.** Scripts 04 and 11 use Weibull AFT as the primary
estimator because it handles right-censoring correctly and does not require
the proportional hazards assumption.  Cox PH uses the same parsimonious covariate set as AFT within each script
(script 04: log_n_events + demographic moderators; script 11: + activated;
log_active_days and channel covariates excluded as collinear) so that Cox HR
and AFT exp(β) can be directly compared.  Both reach significance in the same
direction in script 04 (HR=0.822 p=0.023, MLE penalizer=0; exp(β)=1.48
p=0.015).  Script 11 Cox uses penalizer=0.1 (L2 ridge) for numerical stability
at post-landmark n=74 (27 events); those HR estimates are shrunk and CIs are
narrower than MLE — treat as directional only.  The Schoenfeld residual test
is reported for all Cox models; no violation is found in either script
(script 04: all p > 0.35; script 11: all 6 tests p > 0.24, minimum mod_cpd
p = 0.243).  AFT is pre-specified as primary regardless of the Schoenfeld
outcome.  OLS on log(duration) is shown for interpretability only — it treats
censored observations as uncensored failures and is biased.

**Landmark design.** Script 11 implements a proper landmark analysis
(Andersen & Gill, 1982; van Houwelingen, 2007): subjects who relapse before
the window closes (day 30 post-quit) are excluded (n=39 in this cohort), and
survival time is measured from the window close date.  Without this step,
subjects with early relapses have mechanically fewer events — their event count is bounded by how
long they survived, creating a structural positive correlation between
engagement and survival time even under the null.  The robustness battery at
14d and 60d windows applies the same landmark at each width.  The window
AFTs use a reduced specification (log_n_events only, no covariates) to avoid
singularity in smaller sub-samples; their exp(β) values are unadjusted
estimates and are not directly comparable to the primary model's 2.08.
Direction and significance are consistent.  The effect attenuates slightly
at wider windows (rather than monotonically strengthening, as it did
pre-landmark — the monotonic pattern was the bias signature).  On
synthetic data this exercises the code logic of the design; causal
interpretation requires prospective real data.

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

## AI assistance

Development used AI coding assistants for implementation and review
scaffolding. Study design, method choices, analysis decisions, and all
reported results were specified, verified, and are owned by the author.
