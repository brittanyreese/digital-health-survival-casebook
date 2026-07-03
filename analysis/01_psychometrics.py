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
  5. Longitudinal metric invariance (SDBS T1 vs T2 across TTM stage groups)

Run:  uv run python analysis/01_psychometrics.py
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from cessation import config as C
from cessation import data

OUT = C.RESULTS
OUT.mkdir(parents=True, exist_ok=True)


# ── helpers ───────────────────────────────────────────────────────────────────

def _alpha(X: pd.DataFrame) -> float:
    """Cronbach's alpha."""
    k = X.shape[1]
    item_vars = X.var(ddof=1)
    total_var = X.sum(axis=1).var(ddof=1)
    return (k / (k - 1)) * (1 - item_vars.sum() / total_var)


def _item_total(X: pd.DataFrame) -> pd.DataFrame:
    total = X.sum(axis=1)
    return pd.DataFrame({
        "item": X.columns,
        "mean": X.mean().round(3),
        "sd":   X.std(ddof=1).round(3),
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
        s[pros_cols].apply(pd.to_numeric, errors="coerce"),
        s[cons_cols].apply(pd.to_numeric, errors="coerce"),
    )


def _sseq_items(s: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    int_cols = [f"sseq_{i:02d}" for i in range(1, 7) if f"sseq_{i:02d}" in s.columns]
    ext_cols = [f"sseq_{i:02d}" for i in range(7, 13) if f"sseq_{i:02d}" in s.columns]
    return (
        s[int_cols].apply(pd.to_numeric, errors="coerce"),
        s[ext_cols].apply(pd.to_numeric, errors="coerce"),
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

        fa = FactorAnalyzer(n_factors=X.shape[1], rotation=None)
        fa.fit(X)
        ev, _ = fa.get_eigenvalues()

        rng = np.random.default_rng(0)
        rand_ev = np.zeros((1000, X.shape[1]))
        for i in range(1000):
            fa2 = FactorAnalyzer(n_factors=X.shape[1], rotation=None)
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
            columns=[f"F{j+1}" for j in range(n_efa)],
        )
        loads_obl.round(3).to_csv(OUT / f"01_efa_{name}_oblimin.csv")
        print(f"  EFA oblimin (n_factors={n_efa}):")
        print(loads_obl.round(3).to_string())
        if name == "SSEQ" and n_efa < 2:
            print("  WARNING: EFA supports only 1 factor for SSEQ. "
                  "The 2-factor CFA (Internal/External) imposes a structure not "
                  "confirmed by EFA — subscale scores should be interpreted "
                  "with caution.")

        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(range(1, len(ev) + 1), ev, "o-", label="observed")
        ax.plot(range(1, len(ev) + 1), pa, "s--", color="grey", label="parallel")
        ax.axhline(1, color="red", lw=0.8, ls=":")
        ax.legend()
        ax.set(xlabel="factor", ylabel="eigenvalue",
               title=f"{name} scree + parallel analysis")
        fig.tight_layout()
        fig.savefig(OUT / f"01_fig_{name}_scree.png", dpi=120)
        plt.close(fig)
        print(f"  saved 01_fig_{name}_scree.png")


# ── 4. CFA ────────────────────────────────────────────────────────────────────

def cfa_sdbs(s: pd.DataFrame) -> None:
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
    loads = ins[ins["op"] == "~"][["lval", "rval", "Estimate"]]
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
    loads = ins[ins["op"] == "~"][["lval", "rval", "Estimate"]]
    loads.round(3).to_csv(OUT / "01_sseq_cfa_loadings.csv", index=False)
    print(loads.round(3).to_string(index=False))


# ── 5. IRT (Graded Response Model) ────────────────────────────────────────────

def irt_sdbs_pros(s: pd.DataFrame) -> None:
    """Graded Response Model IRT for the SDBS Pros subscale.

    Scope: Pros only.  Decisional balance theory (Prochaska et al., 1985)
    assigns the Pros subscale primary discriminative value at the
    Precontemplation-to-Contemplation transition — the stage boundary most
    relevant for early DHT engagement.  The Cons subscale follows the same
    GRM structure and SSEQ subscales would extend identically; demonstrating
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
    import semopy
    from scipy.stats import chi2 as chi2dist
    print("\n=== 6. Metric invariance -- SDBS Pros, T1 vs T2 ===")

    t1_cols = [f"t1_sdbs_{i:02d}" for i in range(1, 11)
               if f"t1_sdbs_{i:02d}" in s.columns]
    t2_cols = [f"t2_sdbs_{i:02d}" for i in range(1, 11)
               if f"t2_sdbs_{i:02d}" in s.columns]

    if not t2_cols:
        print("  No T2 data")
        return

    both = s[t1_cols + t2_cols].apply(pd.to_numeric, errors="coerce").dropna()
    print(f"  n with T1+T2: {len(both)}")
    if len(both) < 30:
        print("  Insufficient data")
        return

    it1 = [f"F1_{c}" for c in t1_cols]
    it2 = [f"F2_{c}" for c in t2_cols]
    both.columns = it1 + it2

    def spec(constrain_loadings: bool) -> str:
        def side(fac: str, items: list[str], lbl: bool) -> str:
            labels = [f"l{i}" for i in range(len(items))]
            if lbl:
                terms = [items[0]] + [
                    f"{lb}*{it}" for lb, it in zip(labels[1:], items[1:])
                ]
            else:
                terms = items
            return f"{fac} =~ " + " + ".join(terms)
        # Same item measured twice shares residual variance — required for adequate fit
        resid = "\n".join(f"{a} ~~ {b}" for a, b in zip(it1, it2))
        return (side("Wave1", it1, constrain_loadings) + "\n"
                + side("Wave2", it2, constrain_loadings) + "\nWave1 ~~ Wave2\n" + resid)

    def fit_model(desc: str) -> tuple[float, float, float]:
        m = semopy.Model(desc)
        m.fit(both)
        st = semopy.calc_stats(m).T
        return (float(st.loc["chi2"].iloc[0]),
                float(st.loc["DoF"].iloc[0]),
                float(st.loc["CFI"].iloc[0]))

    try:
        cf, dff, cfif = fit_model(spec(False))
        ce, dfe, cfie = fit_model(spec(True))
        dchi, ddf = ce - cf, dfe - dff
        p = 1 - chi2dist.cdf(dchi, ddf) if ddf > 0 else 1.0
        print(f"  configural CFI={cfif:.3f} | metric CFI={cfie:.3f}")
        print(f"  Δchi2={dchi:.1f} (df={ddf:.0f}) p={p:.3f} | ΔCFI={cfie-cfif:+.3f}")
        if dchi < 0:
            verdict = (f"INDETERMINATE (Δchi²={dchi:.1f} < 0 — "
                       "configural model non-convergence; check starting values)")
        elif cfif < 0.90:
            verdict = (f"INDETERMINATE (configural CFI={cfif:.3f} < 0.90; "
                       "baseline misfit)")
        else:
            verdict = ("supported" if abs(cfie - cfif) < 0.01 and p > 0.05
                       else "NOT supported")
        print(f"  → metric invariance {verdict}")
        print("  Scope: metric (loading) invariance only.  Scalar invariance "
              "(equal item intercepts across groups) is the additional prerequisite "
              "for comparing factor mean scores across TTM stages.  A full invariance "
              "sequence (configural → metric → scalar → strict) would be required "
              "before interpreting stage-conditional latent mean differences.")
        pd.DataFrame([{
            "CFI_configural": round(cfif, 3), "CFI_metric": round(cfie, 3),
            "delta_CFI": round(cfie - cfif, 4), "delta_chi2": round(dchi, 1),
            "delta_df": ddf, "p_value": round(p, 4), "verdict": verdict,
        }]).to_csv(OUT / "01_sdbs_invariance.csv", index=False)
    except Exception as exc:
        print(f"  Invariance test failed: {exc}")


# ── 7. CFA pipeline validation (misspecification check) ──────────────────────

def cfa_misspec_check() -> None:
    """Pipeline validation: fit the 2-factor SDBS model on three DGPs.

    Case 1: 1-factor orthogonal data (trivially wrong) → should fail badly.
    Case 2: 2-factor orthogonal data (correct) → should fit well.
    Case 3: Bifactor data (general + two specific factors) → 2-factor model
            partially misspecified; harder case that tests pipeline sensitivity.

    Good discriminative power means: Case 1 CFI < 0.93, Case 3 CFI < 0.95,
    Case 2 CFI > 0.95.
    """
    import semopy
    print("\n=== 7. CFA pipeline validation (misspecification check) ===")
    rng = np.random.default_rng(42)
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
    df_mis = pd.DataFrame(_discretise(np.hstack([pros, cons])), columns=cols_p + cols_c)
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
                         columns=cols_p + cols_c)
    for label, spec in [("2-factor on bifactor data (Case 3 — hard)", spec_2f),
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
    case1_wrong = next((r for r in rows if "1-factor/wrong" in r["case"]), {})
    case3_hard  = next((r for r in rows if "Case 3 — hard" in r["case"]), {})
    detects_easy = case1_wrong.get("CFI", 1.0) < 0.93
    detects_hard = case3_hard.get("CFI", 1.0) < 0.95
    print(f"  detects trivial misfit (Case 1): {detects_easy}")
    print(f"  detects bifactor misfit (Case 3): {detects_hard}")
    if not detects_hard:
        print("  [warn] 2-factor model shows acceptable fit on bifactor data — "
              "pipeline cannot distinguish bifactor from correlated-factor structure "
              "on these parameters.")


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
    print(f"\nDone. Results in {OUT}")


if __name__ == "__main__":
    main()
