"""Analysis 01 -- Psychometric analysis pipeline for self-report scales.

Demonstrates the analysis pipeline for SDBS (Smoking Decisional Balance Scale,
20 items, 5-pt) and SSEQ-12 (Smoking Self-Efficacy Questionnaire, 12 items,
5-pt) on calibrated synthetic data.  Because the synthetic data are generated
from the same CFA-compatible correlation structure, results reflect parameter
recovery fidelity rather than real-world construct validity.  A pipeline
validation step (Section 7) fits the 2-factor model on 1-factor synthetic
data to confirm the analysis correctly detects misfit when the generating
model is wrong.

Analyses:
  1. Reliability (Cronbach alpha, item-total correlations)
  2. Dimensionality (parallel analysis + scree, EFA oblimin)
  3. Confirmatory factor analysis (SDBS 2-factor; SSEQ-12 2-factor)
  4. Graded-response IRT (SDBS Pros factor)
  5. Cross-group factor congruence (SDBS Pros loadings across TTM stage groups)
  6. MARS subscale reliability (Mobile App Rating Scale; standalone app-quality
     summary, outside the cessation-construct chain)

Run:  uv run python analysis/01_psychometrics.py
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import cast

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from cessation import config as C
from cessation import data
from cessation.viz import add_synthetic_footer

OUT = C.RESULTS
OUT.mkdir(parents=True, exist_ok=True)


# ── helpers ───────────────────────────────────────────────────────────────────

def _alpha(X: pd.DataFrame) -> float:
    """Cronbach's alpha."""
    k = X.shape[1]
    item_vars = cast(pd.Series, X.var(axis=0, ddof=1))
    total_var = X.sum(axis=1).var(ddof=1)
    return (k / (k - 1)) * (1 - item_vars.sum() / total_var)


def _item_total(X: pd.DataFrame) -> pd.DataFrame:
    total = X.sum(axis=1)
    mean_s = cast(pd.Series, X.mean(axis=0))
    sd_s = cast(pd.Series, X.std(axis=0, ddof=1))
    return pd.DataFrame({
        "item": X.columns,
        "mean": mean_s.round(3),
        "sd":   sd_s.round(3),
        "r_item_total": [
            X[c].corr(total - X[c]) for c in X.columns
        ],
    })


def _sdbs_items(s: pd.DataFrame, wave: str = "t1") -> tuple[pd.DataFrame, pd.DataFrame]:
    pros = [f"{wave}_sdbs_{i:02d}" for i in range(1, 11)]
    cons = [f"{wave}_sdbs_{i:02d}" for i in range(11, 21)]
    pros_cols = [c for c in pros if c in s.columns]
    cons_cols = [c for c in cons if c in s.columns]
    return (
        cast(pd.DataFrame, s[pros_cols].apply(pd.to_numeric, errors="coerce")),
        cast(pd.DataFrame, s[cons_cols].apply(pd.to_numeric, errors="coerce")),
    )


def _sseq_items(s: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    int_cols = [f"sseq_{i:02d}" for i in range(1, 7) if f"sseq_{i:02d}" in s.columns]
    ext_cols = [f"sseq_{i:02d}" for i in range(7, 13) if f"sseq_{i:02d}" in s.columns]
    return (
        cast(pd.DataFrame, s[int_cols].apply(pd.to_numeric, errors="coerce")),
        cast(pd.DataFrame, s[ext_cols].apply(pd.to_numeric, errors="coerce")),
    )


# ── 1. Reliability ────────────────────────────────────────────────────────────

def reliability_table(s: pd.DataFrame) -> None:
    print("\n=== 1. Reliability ===")
    pros, cons = _sdbs_items(s)
    sseq_int, sseq_ext = _sseq_items(s)
    rows = []
    for name, X in [("SDBS Pros", pros), ("SDBS Cons", cons),
                    ("SSEQ Internal", sseq_int), ("SSEQ External", sseq_ext)]:
        X = X.dropna()
        a = _alpha(X)
        print(f"  {name}: n_items={X.shape[1]}  alpha={a:.3f}")
        rows.append({"scale": name, "n_items": X.shape[1], "alpha": round(a, 3)})
    pd.DataFrame(rows).to_csv(OUT / "01_reliability.csv", index=False)


# ── 2. Item analysis ──────────────────────────────────────────────────────────

def item_analysis(s: pd.DataFrame) -> None:
    print("\n=== 2. Item analysis ===")
    pros, cons = _sdbs_items(s)
    sseq_int, sseq_ext = _sseq_items(s)
    for name, X in [("SDBS_Pros", pros), ("SDBS_Cons", cons),
                    ("SSEQ_Internal", sseq_int), ("SSEQ_External", sseq_ext)]:
        st = _item_total(X.dropna())
        weak = st[st["r_item_total"] < 0.30]
        print(f"  {name}: {len(X.columns)} items, {len(weak)} weak (r<.30)")
        st.round(3).to_csv(OUT / f"01_items_{name}.csv", index=False)


# ── 3. Dimensionality ─────────────────────────────────────────────────────────

def dimensionality(s: pd.DataFrame) -> None:
    from factor_analyzer import FactorAnalyzer
    from factor_analyzer.factor_analyzer import (
        calculate_bartlett_sphericity,
        calculate_kmo,
    )

    for name, X in [("SDBS", pd.concat(_sdbs_items(s), axis=1)),
                    ("SSEQ", pd.concat(_sseq_items(s), axis=1))]:
        X = X.dropna()
        print(f"\n=== 3. Dimensionality ({name}, n={len(X)}) ===")
        chi2, p = calculate_bartlett_sphericity(X)
        _, kmo  = calculate_kmo(X)
        print(f"  Bartlett chi2={chi2:.0f} p={p:.2e} | KMO={kmo:.3f}")

        # rotation=None is a valid factor_analyzer value (disables rotation);
        # the library's type hint omits Optional.
        no_rotation = cast(str, None)
        fa = FactorAnalyzer(n_factors=X.shape[1], rotation=no_rotation)
        fa.fit(X)
        ev, _ = fa.get_eigenvalues()

        rng = np.random.default_rng(C.SEED)
        rand_ev = np.zeros((1000, X.shape[1]))
        for i in range(1000):
            fa2 = FactorAnalyzer(n_factors=X.shape[1], rotation=no_rotation)
            fa2.fit(pd.DataFrame(rng.standard_normal(X.shape)))
            rand_ev[i], _ = fa2.get_eigenvalues()
        pa = rand_ev.mean(axis=0)

        n_kaiser = (ev > 1).sum()
        n_pa     = (ev > pa).sum()
        print(f"  Kaiser ev>1: {n_kaiser}  |  parallel analysis: {n_pa}")
        print(f"  First 5 ev: {np.round(ev[:5], 2)}")

        # EFA oblimin with n_factors from parallel analysis (min 1)
        n_efa = max(1, int(n_pa))
        fa_obl = FactorAnalyzer(n_factors=n_efa, rotation="oblimin")
        fa_obl.fit(X)
        loads_obl = pd.DataFrame(
            fa_obl.loadings_,
            index=X.columns,
            columns=pd.Index([f"F{j+1}" for j in range(n_efa)]),
        )
        loads_obl.round(3).to_csv(OUT / f"01_efa_{name}_oblimin.csv")
        print(f"  EFA oblimin (n_factors={n_efa}):")
        print(loads_obl.round(3).to_string())
        if name == "SSEQ" and n_efa < 2:
            print("  WARNING: EFA supports only 1 factor for SSEQ. "
                  "The 2-factor CFA (Internal/External) imposes a structure not "
                  "confirmed by EFA: subscale scores should be interpreted "
                  "with caution.")

        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(range(1, len(ev) + 1), ev, "o-", label="observed")
        ax.plot(range(1, len(ev) + 1), pa, "s--", color="grey", label="parallel")
        ax.axhline(1, color="red", lw=0.8, ls=":")
        ax.legend()
        ax.set(xlabel="factor", ylabel="eigenvalue",
               title=f"{name} scree + parallel analysis")
        fig.tight_layout()
        add_synthetic_footer(fig)
        fig.savefig(OUT / f"01_fig_{name}_scree.png", dpi=120)
        plt.close(fig)
        print(f"  saved 01_fig_{name}_scree.png")


# ── 4. CFA ────────────────────────────────────────────────────────────────────

def cfa_sdbs(s: pd.DataFrame) -> None:
    """CFA of the SDBS 2-factor (Pros + Cons) model.

    Items are 5-category ordinal but fit with semopy's default ML (continuous),
    while the generator draws them through a probit-threshold model. Treating
    ordinal indicators as continuous mildly biases RMSEA and chi-square relative
    to a polychoric/WLSMV fit, so the reported fit statistics are indicative,
    not exact. semopy does not expose WLSMV cleanly in this toolchain.
    """
    import semopy
    print("\n=== 4. CFA -- SDBS 2-factor (Pros + Cons) ===")
    pros, cons = _sdbs_items(s)
    X = pd.concat([pros, cons], axis=1).dropna()
    X.columns = ([f"p{i:02d}" for i in range(1, 11)]
                 + [f"c{i:02d}" for i in range(1, 11)])
    pros_names = [f"p{i:02d}" for i in range(1, 11)]
    cons_names = [f"c{i:02d}" for i in range(1, 11)]
    spec = (
        "Pros =~ " + " + ".join(pros_names) + "\n"
        + "Cons =~ " + " + ".join(cons_names) + "\n"
        + "Pros ~~ Cons"
    )
    m = semopy.Model(spec)
    m.fit(X)
    st = semopy.calc_stats(m).T
    fit = {k: round(float(st.loc[k].iloc[0]), 3)
           for k in ["chi2", "DoF", "CFI", "TLI", "RMSEA", "SRMR"] if k in st.index}
    print("  fit:", fit)
    pd.DataFrame([fit]).to_csv(OUT / "01_sdbs_cfa_fit.csv", index=False)
    ins = m.inspect()
    assert isinstance(ins, pd.DataFrame), \
        "semopy Model.inspect() returned non-DataFrame"
    loads = ins.loc[ins["op"] == "~", ["lval", "rval", "Estimate"]]
    loads.round(3).to_csv(OUT / "01_sdbs_cfa_loadings.csv", index=False)
    print(loads.round(3).to_string(index=False))


def cfa_sseq(s: pd.DataFrame) -> None:
    import semopy
    print("\n=== 4. CFA -- SSEQ-12 2-factor (Internal + External) ===")
    sint, sext = _sseq_items(s)
    X = pd.concat([sint, sext], axis=1).dropna()
    int_names = list(sint.columns)
    ext_names = list(sext.columns)
    spec = (
        "Internal =~ " + " + ".join(int_names) + "\n"
        + "External =~ " + " + ".join(ext_names) + "\n"
        + "Internal ~~ External"
    )
    m = semopy.Model(spec)
    m.fit(X)
    st = semopy.calc_stats(m).T
    fit = {k: round(float(st.loc[k].iloc[0]), 3)
           for k in ["chi2", "DoF", "CFI", "TLI", "RMSEA", "SRMR"] if k in st.index}
    print("  fit:", fit)
    pd.DataFrame([fit]).to_csv(OUT / "01_sseq_cfa_fit.csv", index=False)
    ins = m.inspect()
    assert isinstance(ins, pd.DataFrame), \
        "semopy Model.inspect() returned non-DataFrame"
    loads = ins.loc[ins["op"] == "~", ["lval", "rval", "Estimate"]]
    loads.round(3).to_csv(OUT / "01_sseq_cfa_loadings.csv", index=False)
    print(loads.round(3).to_string(index=False))


# ── 5. IRT (Graded Response Model) ────────────────────────────────────────────

def irt_sdbs_pros(s: pd.DataFrame) -> None:
    """Graded Response Model IRT for the SDBS Pros subscale.

    Scope: Pros only.  Decisional balance theory (Prochaska et al., 1985)
    assigns the Pros subscale primary discriminative value at the
    Precontemplation-to-Contemplation transition (the stage boundary most
    relevant for early DHT engagement.  The Cons subscale follows the same
    GRM structure and SSEQ subscales would extend identically); demonstrating
    the pipeline on one subscale avoids redundant output.
    """
    print("\n=== 5. IRT GRM -- SDBS Pros ===")
    try:
        from girth import grm_mml
    except ImportError as e:
        print(f"  girth not available: {e}")
        return
    pros, _ = _sdbs_items(s)
    X = pros.dropna().astype(int) - 1  # GRM expects 0-indexed categories (0–4)
    try:
        res = grm_mml(X.T.to_numpy())
        disc = res["Discrimination"]
        thresholds = res["Difficulty"]  # shape (n_items, n_thresholds)
        out = pd.DataFrame({"item": X.columns, "discrimination": np.round(disc, 3)})
        n_thresh = thresholds.shape[1] if thresholds.ndim > 1 else 1
        for k in range(n_thresh):
            col = thresholds[:, k] if thresholds.ndim > 1 else thresholds
            out[f"threshold_{k+1}"] = np.round(col, 3)
        print(out.to_string(index=False))
        print(f"  discrimination range {disc.min():.2f}–{disc.max():.2f}")
        out.to_csv(OUT / "01_sdbs_pros_irt.csv", index=False)
    except Exception as exc:
        print(f"  IRT fit failed: {exc}")


# ── 6. Metric invariance (SDBS T1 vs T2) ─────────────────────────────────────

def invariance_sdbs(s: pd.DataFrame) -> None:
    """Cross-group factor congruence for the SDBS Pros factor across TTM stages.

    Between-groups structural comparison (not repeated-measures): split the pilot
    cohort into early-stage (precontemplation + contemplation) vs prepared-or-quit
    (preparation + action + maintenance), fit the 1-factor Pros model separately
    in each group, and compare the two loading vectors with Tucker's congruence
    coefficient phi (Lorenzo-Seva & ten Berge, 2006). phi >= 0.95 indicates the
    factor has effectively the same pattern of loadings in both groups, the
    structural core of metric invariance.

    This is a congruence-based check, deliberately lighter than a full nested
    multigroup invariance LRT (configural vs equal-loading metric model), which
    needs multigroup SEM estimation that this toolchain (semopy) does not support
    cleanly. Each per-group 1-factor fit converges on its own, and congruence of
    their loadings is a robust, well-established factor-replicability index. It
    does not test scalar (intercept) invariance and so cannot license comparing
    factor mean scores across stages.

    Construction note: the generator injects the same per-item Pros loadings for
    every stage group (only the factor mean shifts by stage), so metric
    congruence holds by construction and phi is pinned near 1 up to sampling
    noise. This check verifies that the pipeline recovers that injected
    invariance; it does not discover invariance in the data and could not have
    failed given the DGP.
    """
    import semopy
    print("\n=== 6. SDBS Pros factor congruence across TTM stage groups ===")

    pros_cols = [f"t1_sdbs_{i:02d}" for i in range(1, 11)
                 if f"t1_sdbs_{i:02d}" in s.columns]
    if "mod_stage" not in s.columns or len(pros_cols) < 3:
        print("  mod_stage or SDBS Pros items missing; skipping")
        return

    spec = "Pros =~ " + " + ".join(pros_cols)

    def fit_loadings(data: pd.DataFrame) -> tuple[np.ndarray, float]:
        m = semopy.Model(spec)
        m.fit(data)
        insp = cast(pd.DataFrame, m.inspect())
        load = insp[(insp["op"] == "~") & (insp["rval"] == "Pros")]
        load = (load.set_index("lval")["Estimate"]
                .apply(pd.to_numeric, errors="coerce").reindex(pros_cols))
        cfi = float(semopy.calc_stats(m).T.loc["CFI"].iloc[0])
        return load.to_numpy(dtype=float), cfi

    def congruence(a: np.ndarray, b: np.ndarray) -> float:
        return float(a @ b / np.sqrt((a @ a) * (b @ b)))

    is_early = s["mod_stage"].isin(["precontemplation", "contemplation"])
    groups = {"early": s.loc[is_early], "prepared": s.loc[~is_early]}

    try:
        loadings: dict[str, np.ndarray] = {}
        cfis: dict[str, float] = {}
        ns: dict[str, int] = {}
        for name, gdf in groups.items():
            g = cast(pd.DataFrame,
                     gdf[pros_cols].apply(pd.to_numeric, errors="coerce").dropna())
            if len(g) < 50:
                print(f"  group '{name}' n={len(g)} < 50; too small for congruence")
                return
            load, cfi = fit_loadings(g)
            loadings[name] = load
            cfis[name] = cfi
            ns[name] = len(g)
            print(f"  group '{name}': n={len(g)} CFI={cfi:.3f}")

        phi = congruence(loadings["early"], loadings["prepared"])
        both_fit = min(cfis.values()) >= 0.90
        print(f"  Tucker's congruence phi={phi:.3f} "
              f"(>= 0.95 = congruent loading pattern)")
        if not both_fit:
            verdict = (f"INDETERMINATE (a group CFI < 0.90; "
                       f"min={min(cfis.values()):.3f})")
        else:
            verdict = "congruent" if phi >= 0.95 else "NOT congruent"
        print(f"  → cross-group factor congruence: {verdict}")
        print("  Scope: loading-pattern congruence only, a structural check, not a "
              "full invariance sequence. Scalar (intercept) invariance is untested "
              "and is the additional prerequisite before comparing factor mean "
              "scores across TTM stages.")
        pd.DataFrame([{
            "group_a": "early", "group_b": "prepared",
            "n_a": ns["early"], "n_b": ns["prepared"],
            "cfi_a": round(cfis["early"], 3), "cfi_b": round(cfis["prepared"], 3),
            "congruence_phi": round(phi, 4), "verdict": verdict,
        }]).to_csv(OUT / "01_sdbs_invariance.csv", index=False)
    except Exception as exc:
        print(f"  Congruence test failed: {exc}")


# ── 7. CFA pipeline validation (misspecification check) ──────────────────────

def cfa_misspec_check() -> None:
    """Pipeline validation: fit the 2-factor SDBS model on three DGPs.

    Case 1: 1-factor orthogonal data (trivially wrong) → should fail badly.
    Case 2: 2-factor orthogonal data (correct) → should fit well.
    Case 3: "Bifactor" data (general g + two specific factors) with uniform
            loadings on the two item blocks. Note this DGP is observationally
            equivalent to a correlated-2-factor model: each block gets one g
            loading and one specific loading identical across its items, so the
            block is rank-1 and g plus the specific factor are not separately
            identified within it. A good 2-factor fit here is therefore the
            correct answer, not a missed misspecification. This case documents
            that limit; a genuine bifactor stress-test needs specific factors
            that cross-cut the item blocks (or item-level cross-loadings).

    Case 1 should give CFI < 0.93 (real misfit detected); Case 2 and Case 3
    both should give CFI > 0.95 (Case 3 because it is not actually misspecified
    relative to the 2-factor model).
    """
    import semopy
    print("\n=== 7. CFA pipeline validation (misspecification check) ===")
    rng = np.random.default_rng(C.SEED)
    n, lam = 800, 0.6
    cols_p = [f"p{i:02d}" for i in range(1, 11)]
    cols_c = [f"c{i:02d}" for i in range(1, 11)]
    spec_2f = ("Pros =~ " + " + ".join(cols_p) + "\nCons =~ " + " + ".join(cols_c)
               + "\nPros ~~ Cons")
    spec_1f = "G =~ " + " + ".join(cols_p + cols_c)

    def _discretise(arr: np.ndarray) -> np.ndarray:
        return np.clip(np.round(arr * 1.5 + 3), 1, 5)

    rows = []

    # Case 1: 1-factor orthogonal (trivially wrong model)
    fa = rng.standard_normal(n)
    fb = rng.standard_normal(n)
    pros = lam * fa[:, None] + np.sqrt(1 - lam**2) * rng.standard_normal((n, 10))
    cons = lam * fb[:, None] + np.sqrt(1 - lam**2) * rng.standard_normal((n, 10))
    df_mis = pd.DataFrame(
        _discretise(np.hstack([pros, cons])), columns=pd.Index(cols_p + cols_c)
    )
    for label, spec in [("1-factor/wrong (Case 1)", spec_1f),
                         ("2-factor/correct (Case 1 data)", spec_2f)]:
        try:
            m = semopy.Model(spec)
            m.fit(df_mis)
            st = semopy.calc_stats(m).T
            fit = {k: round(float(st.loc[k].iloc[0]), 3)
                   for k in ["CFI", "RMSEA"] if k in st.index}
        except Exception as exc:
            fit = {"error": str(exc)}
        print(f"  {label}: {fit}")
        rows.append({"case": label, **fit})

    # Case 3: Bifactor (general factor + two specific factors)
    # All 20 items load on g (lam_g=0.4) plus their specific factor (lam_s=0.5)
    g  = rng.standard_normal(n)
    f1 = rng.standard_normal(n)
    f2 = rng.standard_normal(n)
    lam_g, lam_s = 0.40, 0.50
    unique_var = np.sqrt(max(0, 1 - lam_g**2 - lam_s**2))
    pros_bf = (lam_g * g[:, None] + lam_s * f1[:, None]
               + unique_var * rng.standard_normal((n, 10)))
    cons_bf = (lam_g * g[:, None] + lam_s * f2[:, None]
               + unique_var * rng.standard_normal((n, 10)))
    df_bf = pd.DataFrame(_discretise(np.hstack([pros_bf, cons_bf])),
                         columns=pd.Index(cols_p + cols_c))
    for label, spec in [("2-factor on bifactor data (Case 3 (hard))", spec_2f),
                         ("1-factor on bifactor data (Case 3)", spec_1f)]:
        try:
            m = semopy.Model(spec)
            m.fit(df_bf)
            st = semopy.calc_stats(m).T
            fit = {k: round(float(st.loc[k].iloc[0]), 3)
                   for k in ["CFI", "RMSEA"] if k in st.index}
        except Exception as exc:
            fit = {"error": str(exc)}
        print(f"  {label}: {fit}")
        rows.append({"case": label, **fit})

    pd.DataFrame(rows).to_csv(OUT / "01_cfa_misspec_check.csv", index=False)
    for r in rows:
        cfi_val, rmsea_val = r.get("CFI"), r.get("RMSEA")
        if (isinstance(cfi_val, float) and cfi_val > 1.0) or \
           (isinstance(rmsea_val, float) and rmsea_val == 0.0):
            print(f"  [warn] {r['case']}: CFI={cfi_val} RMSEA={rmsea_val} outside the "
                  "[0,1] fit range; CFI>1 or RMSEA=0 signals a near-saturated / "
                  "non-positive-df solution. Read as degenerate, not as ideal fit.")
    case1_wrong = next((r for r in rows if "1-factor/wrong" in r["case"]), {})
    case3_hard  = next((r for r in rows if "Case 3 (hard)" in r["case"]), {})
    detects_easy = case1_wrong.get("CFI", 1.0) < 0.93
    detects_hard = case3_hard.get("CFI", 1.0) < 0.95
    print(f"  detects trivial misfit (Case 1): {detects_easy}")
    print(f"  detects bifactor misfit (Case 3): {detects_hard}")
    if not detects_hard:
        print("  [note] 2-factor model fits the 'bifactor' data well because, with "
              "uniform loadings on the two item blocks, that DGP is observationally "
              "equivalent to a correlated-2-factor model (rank-1 blocks). A good fit "
              "is the correct result here, not a missed misspecification; a genuine "
              "bifactor test needs specific factors that cross-cut the item blocks.")


# ── 8. MARS subscale reliability ─────────────────────────────────────────────

def mars_reliability(s: pd.DataFrame) -> None:
    """MARS subscale reliability (Cronbach's alpha) + global-rating descriptive.

    MARS (Mobile App Rating Scale, Stoyanov 2015) is an app-quality instrument,
    outside the cessation-construct chain that drives the survival model, so it is
    reported here as a standalone measurement summary. The four subscales are
    generated as independent blocks, so this recovers per-subscale reliability
    (consistent with the injected 0.70 loadings), not a correlated 4-factor
    structure; inter-subscale correlations are near zero by construction. Item 17
    is a single global app-quality rating, summarized descriptively, not scaled.
    """
    print("\n=== 8. MARS subscale reliability ===")
    layout = {"engagement": (1, 5), "functionality": (6, 9),
              "aesthetics": (10, 12), "information": (13, 16)}
    rows = []
    sub_scores: dict[str, pd.Series] = {}
    for name, (lo, hi) in layout.items():
        cols = [f"mars_{i:02d}" for i in range(lo, hi + 1)
                if f"mars_{i:02d}" in s.columns]
        if len(cols) < 2:
            continue
        X = cast(pd.DataFrame,
                 s[cols].apply(pd.to_numeric, errors="coerce").dropna())
        a = _alpha(X)
        sub_scores[name] = cast(pd.Series, s[cols].apply(
            pd.to_numeric, errors="coerce").sum(axis=1))
        rows.append({"subscale": name, "n_items": len(cols), "n": len(X),
                     "cronbach_alpha": round(a, 3),
                     "mean_item": round(float(X.values.mean()), 3)})
        print(f"  {name}: {len(cols)} items, alpha={a:.3f}, n={len(X)}")

    if len(sub_scores) > 1:
        corr = pd.DataFrame(sub_scores).corr()
        offdiag = corr.where(~np.eye(len(corr), dtype=bool)).abs().stack()
        print(f"  mean |inter-subscale r| = {offdiag.mean():.3f} "
              "(≈0 expected: subscales generated independently)")

    if "mars_17" in s.columns:
        q = cast(pd.Series, pd.to_numeric(s["mars_17"], errors="coerce")).dropna()
        print(f"  global rating (item 17): mean={q.mean():.2f} sd={q.std():.2f} "
              "(single-item descriptive, not scaled)")
        rows.append({"subscale": "global_rating", "n_items": 1, "n": len(q),
                     "cronbach_alpha": None, "mean_item": round(float(q.mean()), 3)})

    pd.DataFrame(rows).to_csv(OUT / "01_mars_reliability.csv", index=False)


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    s = data.load_survey()
    print(f"Survey n={len(s)}")
    reliability_table(s)
    item_analysis(s)
    dimensionality(s)
    cfa_sdbs(s)
    cfa_sseq(s)
    irt_sdbs_pros(s)
    invariance_sdbs(s)
    cfa_misspec_check()
    mars_reliability(s)
    print(f"\nDone. Results in {OUT}")


if __name__ == "__main__":
    main()
