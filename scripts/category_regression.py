#!/usr/bin/env python3
"""
category_regression.py - Taxonomy construct validity and covariate control.

1. Renders the dimension table docs/taxonomy_dimensions.csv (intent to deceive,
   epistemic status, rhetorical style, role) to out/tables/taxonomy_dimensions.tex.
2. Fits logistic regressions of agent error on passage category, before and after
   adding surface covariates: passage length (tokens, standardised), explicit
   answer cue (intended wrong option named verbatim; judge flag when available,
   lexical flag otherwise) and hedging-marker count (standardised). Standard errors
   are clustered by question (questions are repeated across the nine categories),
   with model fixed effects when several models are pooled. A random-intercept
   (question) logistic model is fitted in addition when statsmodels' BinomialBayesMixedGLM
   is available.

Inputs: out/results/rescored/single_agent.csv (exp1 rows) and MINT_dataset_v1.2/<dataset>_passages.jsonl.
Outputs: out/results/category_regression_exp1.csv, out/tables/category_regression_exp1.tex
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from scripts.common import CATEGORY_ORDER, RESULTS_DIR, TABLES_DIR, ensure_dirs, read_csv, write_csv, write_latex_table  # noqa: E402

HEDGES = ["allegedly", "reportedly", "some say", "some people", "it is said", "rumor", "rumour", "may", "might", "could",
          "possibly", "perhaps", "supposedly", "claim", "claims", "unconfirmed", "sources say", "insiders", "believed to",
          "appears", "seems", "likely", "probably", "according to"]
_HEDGE_RE = re.compile(r"\b(" + "|".join(re.escape(h) for h in HEDGES) + r")\b", re.IGNORECASE)


def load_passages(data_dir: Path) -> Dict[tuple, dict]:
    out: Dict[tuple, dict] = {}
    for p in data_dir.glob("*_passages.jsonl"):
        with open(p, encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                r = json.loads(line)
                v = r.get("validation") or {}
                # judge explicit cue: any judge says yes (fallback lexical)
                judge_cue = None
                for k, fl in v.items():
                    if isinstance(fl, dict) and fl.get("explicit_cue") is not None:
                        judge_cue = bool(fl["explicit_cue"]) if judge_cue is None else (judge_cue or bool(fl["explicit_cue"]))
                out[(r["mint_id"], r["category"])] = {
                    "n_tokens": r.get("n_tokens") or r.get("n_tokens_ws") or 0,
                    "explicit_cue": judge_cue if judge_cue is not None else r.get("explicit_cue_lexical"),
                    "hedging": len(_HEDGE_RE.findall(r.get("passage") or "")),
                }
    return out


def build_design(rows: List[dict], passages: Dict[tuple, dict], reference: str) -> tuple:
    import pandas as pd

    recs = []
    for r in rows:
        cat = r["category"]
        if cat not in CATEGORY_ORDER:
            continue
        pv = passages.get((r["mint_id"], cat))
        if pv is None:
            continue
        recs.append({"error": 0 if str(r["is_correct"]).lower() in ("true", "1") else 1, "category": cat,
                     "n_tokens": float(pv["n_tokens"]), "explicit_cue": 1.0 if pv["explicit_cue"] else 0.0,
                     "hedging": float(pv["hedging"]), "question": r["mint_id"], "model": r["model"], "dataset": r["dataset"]})
    df = pd.DataFrame(recs)
    if df.empty:
        return df, None
    df["z_tokens"] = (df["n_tokens"] - df["n_tokens"].mean()) / (df["n_tokens"].std(ddof=0) or 1.0)
    df["z_hedging"] = (df["hedging"] - df["hedging"].mean()) / (df["hedging"].std(ddof=0) or 1.0)
    cats = [reference] + [c for c in CATEGORY_ORDER if c != reference]
    df["category"] = pd.Categorical(df["category"], categories=cats)
    return df, cats


def fit(df, formula: str):
    import statsmodels.formula.api as smf

    model = smf.glm(formula, data=df, family=__import__("statsmodels.api", fromlist=["families"]).families.Binomial())
    res = model.fit(cov_type="cluster", cov_kwds={"groups": df["question"].astype("category").cat.codes.values})
    return res


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--reference", default="false_fact", help="reference category")
    ap.add_argument("--data_dir", default=str(REPO_ROOT / "MINT_dataset_v1.2"))
    ap.add_argument("--rescored_dir", default=str(RESULTS_DIR / "rescored"))
    args = ap.parse_args()
    ensure_dirs()

    # 1) dimension table
    dims = read_csv(REPO_ROOT / "docs" / "taxonomy_dimensions.csv")
    write_latex_table(TABLES_DIR / "taxonomy_dimensions.tex",
                      ["Category", "Intent to deceive", "Epistemic status", "Rhetorical style", "Role"],
                      [[d["label_in_paper"], d["intent_to_deceive"], d["epistemic_status"], d["rhetorical_style"], d["role_in_analysis"]]
                       for d in dims],
                      caption="Dimensions of the nine MINT passage categories. Satire and the unconstrained category do not encode "
                              "an intent to deceive and are analysed as rhetorical-style controls.", label="tab:taxonomy_dimensions",
                      col_spec="lllll")

    # 2) regression
    rd = Path(args.rescored_dir)
    # seed 0 only: the run that covers all nine categories with identical settings (seeds 1-2 only re-ran
    # baseline/false_fact and would over-weight the reference category)
    rows = [r for r in read_csv(rd / "single_agent.csv") if r["experiment"] == "exp1" and r["relevance"] == "relevant"
            and str(r.get("seed") or "0") in ("0", "0.0")]
    passages = load_passages(Path(args.data_dir))
    df, cats = build_design(rows, passages, args.reference)
    if df is None or df.empty:
        print("no rows; aborting")
        return
    try:
        import statsmodels.api as sm  # noqa: F401
    except ImportError:
        print("statsmodels is required: pip install statsmodels")
        return

    out_rows = []
    fe = " + C(model)" if df["model"].nunique() > 1 else ""
    fe += " + C(dataset)" if df["dataset"].nunique() > 1 else ""
    f0 = f"error ~ C(category){fe}"
    f1 = f"error ~ C(category) + z_tokens + explicit_cue + z_hedging{fe}"
    res0, res1 = fit(df, f0), fit(df, f1)
    for name, res in (("before_covariates", res0), ("with_covariates", res1)):
        for term, coef in res.params.items():
            se = res.bse[term]
            out_rows.append({"experiment": "exp1", "model_spec": name, "term": term, "coef_logit": coef,
                             "odds_ratio": float(np.exp(coef)), "se_cluster": se, "ci_lo": coef - 1.96 * se,
                             "ci_hi": coef + 1.96 * se, "p": res.pvalues[term], "n_obs": int(res.nobs),
                             "n_questions": int(df["question"].nunique())})
    # mixed model (random intercept per question) as a robustness check
    try:
        from statsmodels.genmod.bayes_mixed_glm import BinomialBayesMixedGLM

        md = BinomialBayesMixedGLM.from_formula(f1, {"question": "0 + C(question)"}, df)
        mres = md.fit_vb()
        names = list(md.exog_names)
        for i, term in enumerate(names):
            out_rows.append({"experiment": "exp1", "model_spec": "mixed_random_intercept_question", "term": term,
                             "coef_logit": float(mres.fe_mean[i]), "odds_ratio": float(np.exp(mres.fe_mean[i])),
                             "se_cluster": float(mres.fe_sd[i]), "ci_lo": float(mres.fe_mean[i] - 1.96 * mres.fe_sd[i]),
                             "ci_hi": float(mres.fe_mean[i] + 1.96 * mres.fe_sd[i]), "p": None, "n_obs": int(len(df)),
                             "n_questions": int(df["question"].nunique())})
    except Exception as e:  # noqa: BLE001
        print(f"[WARN] mixed model skipped: {e}")

    write_csv(RESULTS_DIR / "category_regression_exp1.csv", out_rows)
    # LaTeX: category coefficients before/after (odds ratios vs reference)
    tex = []
    for c in cats[1:]:
        term = f"C(category)[T.{c}]"
        b = next((r for r in out_rows if r["model_spec"] == "before_covariates" and r["term"] == term), None)
        a = next((r for r in out_rows if r["model_spec"] == "with_covariates" and r["term"] == term), None)
        if b and a:
            tex.append([c, f"{b['odds_ratio']:.2f} [{np.exp(b['ci_lo']):.2f}, {np.exp(b['ci_hi']):.2f}]", f"{b['p']:.3f}",
                        f"{a['odds_ratio']:.2f} [{np.exp(a['ci_lo']):.2f}, {np.exp(a['ci_hi']):.2f}]", f"{a['p']:.3f}"])
    for cov, lab in (("z_tokens", "length (per SD)"), ("explicit_cue", "explicit answer cue"), ("z_hedging", "hedging (per SD)")):
        a = next((r for r in out_rows if r["model_spec"] == "with_covariates" and r["term"] == cov), None)
        if a:
            tex.append([lab, "--", "--", f"{a['odds_ratio']:.2f} [{np.exp(a['ci_lo']):.2f}, {np.exp(a['ci_hi']):.2f}]", f"{a['p']:.3f}"])
    write_latex_table(TABLES_DIR / "category_regression_exp1.tex",
                      ["Term (vs. plain false fact)", "OR before [95\\% CI]", "p", "OR with covariates [95\\% CI]", "p"], tex,
                      caption="Logistic regression of agent error on passage category (single agent), "
                              "with and without surface covariates (standardised token length, explicit answer cue, standardised hedging count). "
                              "Odds ratios relative to the plain false fact; standard errors clustered by question; model and dataset fixed effects.",
                      label="tab:category_regression_exp1", col_spec="lllll")
    print(f"n_obs={int(res1.nobs)} questions={df['question'].nunique()}; wrote out/results/category_regression_exp1.csv")
    for r in tex:
        print("  ", r)


if __name__ == "__main__":
    main()
