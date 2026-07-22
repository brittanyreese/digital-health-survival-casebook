# Digital-health survival casebook

This is a methods demonstration on fully synthetic data: it shows how these analyses are designed, validated, and stress-tested, and it makes no real-world clinical claims.

[![Reproducibility CI](https://github.com/brittanyreese/digital-health-survival-casebook/actions/workflows/ci.yml/badge.svg)](https://github.com/brittanyreese/digital-health-survival-casebook/actions/workflows/ci.yml) ![Python](https://img.shields.io/badge/python-3.12+-blue) ![License](https://img.shields.io/badge/license-MIT-green) [![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)

A reproducible methods casebook for **digital-health survival analysis**, built on a fully synthetic smoking-cessation cohort (8,000 app users, a ~480-user psychometric subsample, a ~1,200-user follow-up cohort, seed 42). One seeded generator draws its parameters from published literature and public federal data (NHANES, MMWR) and writes every table. The analysis pipeline recovers those injected parameters. Because the ground truth is known, each result is scored against it rather than argued from authority. The cohort and its fictional vendor are inventions with no real-study counterpart, and the results recover what the generator put in. They say nothing about real users or whether any real app works.

It exists to make good method inspectable. mHealth research rarely connects engagement to clinical endpoints with statistically appropriate models (see [Context and motivation](#context-and-motivation)); this repository shows how, with leakage-free code and stated bounds.

**What it demonstrates:** correct censoring and immortal-time handling (landmark design), feature/label temporal separation via a fixed baseline window and a boundary assertion on the window constants, negative and positive controls, honest reporting of what parameter recovery does and does not prove, and reproducibility a CI job re-checks on freshly generated data.

```mermaid
flowchart LR
    G["Seeded generator<br/>Gaussian-copula CFA<br/>published parameters"] --> D["Synthetic tables<br/>8,000 users, seed 42"]
    D --> P["Psychometrics 01<br/>CFA / IRT-GRM / invariance"]
    D --> S["Survival 04, 11<br/>Cox PH + Weibull AFT<br/>landmark design"]
    D --> B["Behavioral 02, 06, 09<br/>segmentation, SMS event study, Markov"]
    D --> M["Churn ML 10<br/>gradient boosting + SHAP"]
    P --> R["Parameter recovery<br/>recovered vs injected"]
    S --> R
    B --> R
    M --> R
```

> Every number in this repository is scored against a known injected truth: how well the pipeline recovers the generator's parameters. Real users never enter it, and no result speaks to whether any real app works.

## Context and motivation

Most published mHealth-cessation analyses stop at descriptive usage metrics (session counts, days active) and rarely connect engagement trajectories to clinical endpoints with survival-appropriate models. Engagement is multidimensional and poorly operationalized (Perski et al. 2017, _Transl Behav Med_), sustained objective app usage is low and decays quickly (Baumel et al. 2019, _JMIR_), and adherence is seldom analyzed with the right estimators (Whittaker et al. 2019, Cochrane). This casebook builds the pipeline those gaps call for and runs it end to end on the synthetic cohort, so the method is the contribution.

## Analyses

| Analysis | Script |
| --- | --- |
| Psychometric validation: CFA, IRT-GRM, and cross-group factor congruence on the SDBS/SSEQ respondent scales; misspecification checks; MARS subscale reliability (MARS is an expert app-quality rubric completed by evaluators, distinct from the respondent scales SDBS/SSEQ) | `analysis/01_psychometrics.py` |
| Behavioral segmentation: KMeans, silhouette selection, log-rank retention | `analysis/02_segmentation.py` |
| Survival analysis: Cox PH + Schoenfeld test, Weibull AFT primary, right-censoring | `analysis/04_outcome_duration.py` |
| SMS re-engagement negative control: event study, logistic regression | `analysis/06_sms_reengagement.py` |
| Behavioral analytics: Markov channel transitions, stationary distributions, channel-outcome correlation | `analysis/09_golden_paths.py` |
| ML / XAI: HistGradientBoosting, ROC-AUC + AUPRC, SHAP, calibration, subgroup AUC | `analysis/10_churn_ml.py` |
| Landmark analysis: quit-date anchoring, AFT window sensitivity, Schoenfeld test | `analysis/11_quit_anchored.py` |
| Recovery stability: Monte-Carlo recovery intervals across independent seeds | `analysis/12_recovery_monte_carlo.py` |
| Baseline-family misspecification. Fits the Weibull AFT to a non-Weibull (lognormal / generalized-gamma) baseline, and a Cox-Snell goodness-of-fit check rejects it while matched Weibull data passes. Tests baseline-hazard family adequacy only. Covariate and link misspecification stay out of scope. The alternatives also differ in observed event rate at a fixed median (see disclosure) | `analysis/13_misspecification_recovery.py` |

## Analytical design decisions

These choices address common mHealth-analytics pitfalls:

- **Immortal-time / landmark design (04, 10, 11):** Total engagement over the full follow-up is a biased predictor of duration: longer survivors accumulate more events by construction. Scripts 04 and 10 restrict engagement features to a pre-outcome baseline window (days 0-29); script 11 uses a landmark analysis (van Houwelingen 2007), excluding pre-window relapsers and shifting the time origin to day 30.
- **Proportional-hazards testing (04, 11):** Cox PH is fitted with a Schoenfeld residual test alongside the AFT primary. No violation is detected (script 04 minimum p = 0.32, script 11 minimum p = 0.086). Weibull AFT is pre-specified as primary regardless, since it needs no PH assumption.
- **Feature/label temporal separation (04, 09, 10):** Each script derives its feature window from the shared `BASELINE_WINDOW_DAYS` constant; a single shared assertion (`assert_no_temporal_overlap`, `src/cessation/guards.py`) checks that the configured feature and label window bounds do not overlap in all three, catching a mis-set window constant. It validates the configured window bounds, a coarser guarantee than a per-row data scan. For the survival scripts (04, 09) the raw relapse clock starts at enrollment, so the label window overlaps the feature window by construction; script 11's landmark design supplies the real immortal-time control there (origin shifted to day 30), and this assertion only checks the window bounds. The assertion is a real label/feature separation only in script 10, where the churn label is defined strictly on the day-166-180 window.
- **Engagement operationalization (02, 09):** Perski et al. (2017) frame engagement as multidimensional across several usage signals; script 02's clustering feature set spans five such dimensions (event volume, active days, intensity, two channel-mix shares). Baumel et al. (2019) documented low sustained objective app engagement, motivating measurement from behavioral logs over self-report, so engagement here is measured entirely from the event log across every behavioral script (02, 04, 09, 10, 11): the SDBS/SSEQ survey scores enter the outcome models only as separate psychological covariates, kept distinct from the engagement measures.

## Key results (synthetic data: parameter recovery only)

All values reflect how well the pipeline recovers the injected parameters. They do not generalize to real users. Each result is annotated with its implication for a real-data study design.

| Analysis | Synthetic result | Real-data implication |
| --- | --- | --- |
| Weibull AFT (script 04) | exp(β) = 1.21 (95% CI 1.08-1.36), p = 0.0014, per log-unit 30-day engagement | The pipeline recovers the injected engagement→duration signal. The generator injects a latent-propensity coefficient (β on θ, not on log-events), so this exp(β) is the association θ induces through observed engagement, on a different scale from any single injected constant. The recoverable facts are the direction and its significance; the exact magnitude is a proxy-scale estimate that stands in for the effect without being it |
| Weibull AFT (script 11) | exp(β) = 1.27 (95% CI 1.06-1.53), p = 0.011. Activated exp(β) = 0.81 (95% CI 0.53-1.25, p = 0.35, n.s.); n=336 post-landmark (139 early relapsers excluded at <30d); 172 events | Landmark exclusion removes the immortal-time variant. The engagement effect survives at moderate power. The funnel-nesting fix (registration ⊇ followup) lifted this cohort from an underpowered n=74 (27 events) to 336 (172 events); the activated segment contrast is non-significant, but it is partly collinear with log_n_events (same engagement signal), so the non-significance stays inconclusive: collinearity can mask a segment effect |
| Weibull shape κ (scripts 04, 11) | κ = 0.59 (script 04, enrollment-anchored, 95% CI 0.55-0.63). κ = 0.86 (script 11, post-landmark, 95% CI 0.75-0.98). Injected κ = 0.55 | Script 04's 0.59 recovers the injected decreasing-hazard shape (CI includes 0.55). Script 11's 0.86 is a different quantity from κ. Left-truncating and origin-shifting a Weibull(0.55) leaves a residual-time distribution (T - 30 given T > 30) that is no longer Weibull and whose best-fit shape drifts toward 1. The 0.86 describes that post-landmark conditional hazard. The generative parameter stays 0.55, and at n=336 the estimate is tight enough that its CI now excludes 1. An analytical simulation of the known generator (`results/analysis/11_kappa_gof.csv`) reproduces this drift (recovered κ ≈ 0.83, close to the observed 0.86) and a Cox-Snell goodness-of-fit test rejects Weibull adequacy for the residual (p = 0.002), confirming the post-landmark shape is a left-truncation artifact |
| Cox PH test (script 04) | All Schoenfeld p > 0.32, so PH holds. Cox HR = 0.897 (95% CI 0.84-0.96, p = 0.0017, MLE, penalizer=0), directionally consistent with AFT | Aligned covariate sets enable direct Cox/AFT comparison; both significant in same direction |
| Cox PH test (script 11) | All 6 Schoenfeld p > 0.08 (minimum: mod_readiness p = 0.086). PH holds at α = 0.05 (no violation detected) | Weibull AFT is pre-specified as primary regardless; Cox uses the same parsimonious covariate set as AFT (log_n_events + activated + demographic moderators) for direct comparison |
| AFT window sensitivity (script 11) | exp(β) = 1.09 (p = 0.37, n=336) at 14d. exp(β) = 1.05 (p = 0.56, n=279) at 60d | Direction is stable (exp(β) > 1 at both widths) but significance is not retained in the reduced-specification windows; the effect does not strengthen monotonically toward wider windows as it did pre-landmark, which was the bias signature |
| CFA misspecification check (script 01) | CFI = 0.469 on 1-factor case (trivial misfit detected). CFI = 0.997 on correct 2-factor case. CFI = 1.003 on the "bifactor" case (`detects_hard=False`) | Pipeline detects trivial misfit. The "bifactor" case tests something narrower than its name implies. With uniform loadings on the two item blocks, it is observationally equivalent to a correlated-2-factor model (rank-1 blocks), so the good 2-factor fit is the correct answer, and the case has no hidden misspecification to catch. A real bifactor stress-test needs specific factors that cross-cut the item blocks |
| CFA fit (script 01) | SDBS 2-factor: CFI = 0.981, RMSEA = 0.026 | Reflects parameter recovery fidelity; a real-data CFA on this instrument would face measurement noise and partial invariance, and it might show a different factor structure across populations |
| Churn ROC-AUC (script 10) | 0.839 ± 0.009 (5-fold stratified CV) | Above the 0.65-0.78 range typically reported for churn/dropout models in digital health (an illustrative band drawn from several reported studies); elevated because synthetic signal-to-noise is clean; real AUC would likely fall in that range |
| Churn AUPRC (script 10) | 0.342 ± 0.028 (no-skill baseline = 0.106) | 3.2x no-skill lift at 10.6% churn prevalence; AUPRC is the informative metric under class imbalance |
| Subgroup AUC (script 10) | Age: below-median 0.870, above-median 0.816. Education: below-median 0.845, above-median 0.837 | Modest performance gap by age and education on synthetic data; real fairness audit would require race/ethnicity, rurality, and insurance status |
| Channel-outcome association (script 09) | No channel significant (univariate Cox HR per unit 30-day time-share: quiz 2.05 p=0.16, content 1.34 p=0.31, others ≤ 1.0, all p > 0.15); full cohort n = 1162 (relapsers + censored) | Channel mix in the first 30 days shows no association with quit hazard (unadjusted univariate Cox; time-shares are compositional so HRs are not mutually independent; volume not controlled); channel effects require larger N and randomized exposure |
| Reengagement return (script 06) | reengaged ~ baseline engagement: OR = 1.34 per log-unit (95% CI 1.18-1.53), p < 0.001; n=853 delivered-SMS users, 215 returns | Positive control complementing the SMS negative control (delivered-vs-opted-out is null by construction). The generator injects a θ→14-day-return effect, and the pipeline recovers a positive engagement→return association through the observed engagement proxy. Direction and significance are the recoverable facts. The magnitude is a proxy-scale association, correlational only |
| Monte-Carlo recovery (script 12, 25 seeds) | AFT exp(β) = 1.28 (MC 95% [1.17, 1.41]), positive in all 25 seeds. Weibull marginal shape κ = 0.59 (MC 95% [0.56, 0.64]) | The engagement→duration recovery reproduces across all 25 independent draws. The κ interval sits just above the baseline injected 0.55 because individual frailty (the exp(lp) scale mixture) lifts the marginal shape, matching script 04's 0.59. A real replication would test exactly this stability. Here it is guaranteed by construction, so the value shows recovery reproduces across samples |

All analyses are exploratory within a single synthetic dataset. No correction for multiple comparisons is applied across scripts. A pre-registered, FDR-controlled replication on real data would be required before any clinical interpretation.

## Synthetic data generation

Data are generated from a Gaussian-copula CFA model calibrated to:

- SDBS (Smoking Decisional Balance Scale, 20 items): Velicer et al. (1985). _J Pers Soc Psychol_, 48(5), 1279-1289.
- SSEQ-12 (Smoking Self-Efficacy Questionnaire): Etter et al. (2000). _Addiction_, 95(6), 901-913.
- TTM stage distribution: Prochaska et al. (1985). _Addict Behav_, 10(4), 395-406.
- Stage-conditional self-efficacy: DiClemente et al. (1985). _Cogn Ther Res_, 9(2), 181-200.
- Relapse kinetics (Weibull shape κ = 0.55): Hughes JR, Keely J, Naud S (2004). Shape of the relapse curve and long-term abstinence among untreated smokers. _Addiction_, 99(1), 29-38.
- SMS opt-out (48% by 6 months): Christofferson DE, Hertzberg JS, Beckham JC, Dennis PA, Hamlett-Berry K (2016). Engagement and abstinence among users of a smoking cessation text message program for veterans. _Addictive Behaviors_, 62, 47-53. PMC5144826.
- Demographics: CDC NHANES 2017-March 2020 pre-pandemic cycle (SMQ, `P_SMQ.xpt`); Cornelius ME, Wang TW, Jamal A, Loretan CG, Neff LJ (2020). Tobacco Product Use Among Adults, United States, 2019. _MMWR Morb Mortal Wkly Rep_, 69(46), 1736-1742.

## Reproduce

```bash
# Install uv: brew install uv, pipx install uv, or see
# https://docs.astral.sh/uv/getting-started/installation/
uv sync

# Generate all synthetic tables (~30 seconds)
uv run cessation-generate

# Validate distributional properties against calibration targets
uv run cessation-validate

# Run analyses in order (02 must precede 04, 06, 09, 10, 11)
uv run python analysis/01_psychometrics.py
uv run python analysis/02_segmentation.py
uv run python analysis/04_outcome_duration.py
uv run python analysis/06_sms_reengagement.py
uv run python analysis/09_golden_paths.py
uv run python analysis/10_churn_ml.py
uv run python analysis/11_quit_anchored.py
uv run python analysis/13_misspecification_recovery.py
```

Script 13 regenerates its own cohort under a non-Weibull baseline (and a matched Weibull contrast) rather than reading the tables above. It is on the CI hot path.

Recovery holds across independent seeds: `uv run python analysis/12_recovery_monte_carlo.py` regenerates the survival flagship over 25 independent seeds and reports Monte-Carlo recovery intervals (slow, ~7 minutes; run once, output committed, not on the CI hot path).

Results (figures + CSV tables) appear in `results/analysis/`. `data/synthetic/generation_metadata.json` records the seed, package and platform versions, table row counts, and all citation keys used at generation time.

Under seed 42 the result CSV tables are byte-identical across runs. Figures are not held to that standard: one SHAP summary plot (`results/analysis/10_fig_shap_summary.png`) varies by about 0.35% between runs, from nondeterminism in the SHAP and matplotlib rendering path. The committed tables were generated on macOS/ARM; regeneration on another platform, like the Linux CI, can differ in trailing digits.

## Synthetic data disclosure

`data/synthetic/` is entirely generated from published parameters. No real participant data are present. Because the data are generated from the same structure the analyses then recover, fit indices, IRT parameters, and effect sizes measure how faithfully the pipeline recovers its own inputs, a property internal to the design that says nothing about real-world validity. Specific caveats:

- **CFA / IRT (01):** fit is good by construction. The section-7 misspecification check confirms the pipeline detects a wrong (1-factor) model (CFI 0.469 vs 0.997 for the correct 2-factor). Its "bifactor" case (CFI 1.003) is observationally equivalent to a correlated-2-factor model, so the good fit there is the correct answer, and the case has no hidden misspecification to catch. EFA retains only 1 factor for the SSEQ (injected inter-factor r = 0.79), so the 2-factor SSEQ CFA is an imposed structure the data do not independently support. The generator draws SDBS Pros, SDBS Cons, and the SSEQ subscales as independent factor blocks. No SDBS/SSEQ cross-scale correlation is injected, so this pipeline cannot speak to how decisional balance relates to self-efficacy, only to each scale's own internal structure.
- **Survival (04, 11):** Cox and AFT agree in direction and significance (04: HR 0.897, AFT exp(β) 1.21). Script 11's ridge-shrunk HRs are directional only, and OLS on log-duration is biased (treats censored as failures), shown for interpretability only. Demographic moderators are generated independently of the outcome, so their ~1.0 coefficients are expected and do not shift the engagement estimate.
- **Cohort (11):** the funnel-nesting fix (registration ⊇ followup) lifted the landmark cohort from n=74 to 336. Registration is drawn independently of engagement (corr ≈ 0), so this restores power on missing-at-random dropout. It adds power without de-biasing: the 2.08→1.27 attenuation reflects variance in the old 27-event cell, where 27 events gave an unstable estimate. That independence is generator-guaranteed; on real registration data it becomes an assumption a study must test directly, via a balance table (compare baseline covariate distributions between registered-and-followed-up users and registered-only users) or a placebo test (confirm the attenuation doesn't reappear in a subgroup or period where no nesting fix was needed).
- **Misspecification (13):** the Cox-Snell check tests baseline-hazard family adequacy alone; covariate, link, and omitted-confounder errors fall outside it, so a correct family with a wrong mean model can still pass. Median-matching the alternatives holds central tendency fixed while the event fraction still varies (weibull ~62%, lognormal ~77%, gengamma ~91% under 180-day censoring), so the rejections reflect both the family change and the event-rate/information change it induces. The detection power is demonstrated at a single CI-gated seed. The multi-seed rate version is `slow`-marked and off the CI hot path. The alternatives are detectable by design. A gengamma shape of 3 was tried and dropped because detection was inconsistent across seeds at this cohort's size, so the check demonstrates detection on clearly-misspecified families and stops short of any power claim against subtle ones.
- **ML (10):** the 0.839 AUC reflects clean synthetic signal. Real DHT churn AUCs typically fall around 0.65-0.78 (an illustrative band for context, drawn from several reported studies). The label (zero late-window events) and the features (early engagement) are both downstream of the same latent engagement propensity, so the classifier partly re-identifies that propensity by construction. The AUC marks a clean-signal ceiling set by the construction itself. The subgroup-AUC fairness cut covers only age and education, and the age gap (0.870 vs 0.816) is read from point estimates with no variance on the difference.
- **Benchmarks** provide context only. mHealth 6-month abstinence runs ~15-30% vs ~3-5% control (Whittaker et al. 2019, Cochrane CD006611, which reports pooled risk ratios, not raw percentages).
- **Scope:** exploratory on a single synthetic dataset with no multiple-comparison correction. Real-data use would require prospective design, endogeneity and missingness handling, pre-registration, and FDR control before any clinical reading.

## Citing

If you use this casebook, cite it via the [`CITATION.cff`](CITATION.cff) file. GitHub's "Cite this repository" button generates APA and BibTeX from it. For a specific reported number, cite the latest tagged release rather than a commit hash: tags are cut automatically on merge to `main` (see [CONTRIBUTING](CONTRIBUTING.md#branching-and-releases)), so the latest tag always matches the numbers on this page. The methods and calibration sources are listed with DOIs in [docs/REFERENCES.md](docs/REFERENCES.md).

## License

MIT. See [LICENSE](LICENSE).

## AI assistance

Built with AI coding tools under the review, testing, and parameter-recovery validation gates in [CONTRIBUTING](CONTRIBUTING.md#ai-assistance). The maintainer makes the design decisions and validates every result against those references.
