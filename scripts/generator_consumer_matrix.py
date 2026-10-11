#!/usr/bin/env python3
"""
generator_consumer_matrix.py - Same-model hypothesis: generator x consumer matrix.

Inputs are single-agent (exp1) result files produced with different passage sets:
  out/<consumer>/exp1_results_<dataset>_<consumer>.json                  generator = Llama (MINT v1.2)
  out/<consumer>/generators/<gen>/exp1_results_<dataset>_<consumer>.json  generator = glm | gptoss
  out/<consumer>/controls/paraphrase_gptoss/exp1_results_...              paraphrase style control
(produce them with exp1.py --data_dir MINT_dataset_v1.2/generators/<gen> --output_dir out/<consumer>/generators/<gen>).

For every consumer x generator x dataset the accuracy drop
    drop = acc(baseline) - acc(false_fact passages of that generator)
is computed with paired bootstrap CIs, and a logistic regression of per-question
correctness on generator, consumer, dataset and a same_model indicator (generator
family == consumer family) with question-clustered standard errors gives the
same-model effect. Outputs: out/results/generator_consumer_matrix.csv,
out/results/same_model_effect.csv, out/tables/generator_consumer_matrix.tex.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from answer_extraction import STATUS_PARSED, extract_free_form_answer, extract_mc_answer, free_form_matches  # noqa: E402
from mint_data import load_mint_items  # noqa: E402
from scripts.common import (  # noqa: E402
    DATASET_DISPLAY,
    DATASET_ORDER,
    MODEL_DIRS,
    MODEL_FAMILY,
    MODEL_ORDER,
    RESULTS_DIR,
    TABLES_DIR,
    ensure_dirs,
    write_csv,
    write_latex_table,
)
from scripts.stats import paired_bootstrap_diff  # noqa: E402

FILE_DS = {"cwq": "complex_web_questions", "ethics": "ethics", "winogrande": "winogrande"}
GEN_LABEL = {"llama": "Llama-3.3-70B", "glm": "GLM-4.7-Flash", "gptoss": "GPT-OSS-120B"}


def per_question_correct(path: Path, dataset: str, condition: str, items: Dict[str, dict]) -> Dict[str, float]:
    """Re-score a condition of an exp1 results file with the unified extractor."""
    data = json.load(open(path, encoding="utf-8"))
    payload = data.get(condition)
    if not payload:
        return {}
    out: Dict[str, float] = {}
    st = {"n": 0, "unparsed": 0, "ambiguous": 0}
    for r in payload.get("results", []):
        it = items.get(" ".join(str(r.get("sentence", "")).split()))
        if it is None:
            continue
        # --reprompt_unparsed runs: the constrained follow-up answer replaces the first response (as in exp1.py)
        text = r.get("reprompt_response") if (r.get("reprompted") and r.get("reprompt_response")) else r.get("response")
        if it["is_multiple_choice"]:
            ex = extract_mc_answer(text, it["options"])
            ok = ex.status == STATUS_PARSED and ex.index == it["gold_index"]
        else:
            ex = extract_free_form_answer(text)
            golds = [it["gold_text"]] + list(it["gold_aliases"])
            ok = free_form_matches(text, golds) if ex.status == STATUS_PARSED else False
        out[it["mint_id"]] = float(ok)
        st["n"] += 1
        st["unparsed"] += int(ex.status == "unparsed")
        st["ambiguous"] += int(ex.status == "ambiguous")
    STATUS_BY_FILE[(str(path), condition)] = st
    return out


# extraction status per (file, condition), filled by per_question_correct (unparsed answers count as incorrect)
STATUS_BY_FILE: Dict[tuple, dict] = {}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out_dir", default=str(REPO_ROOT / "out"))
    ap.add_argument("--condition", default="false_fact")
    ap.add_argument("--n_boot", type=int, default=10_000)
    args = ap.parse_args()
    ensure_dirs()
    out_dir = Path(args.out_dir)

    cells: List[dict] = []
    long: List[dict] = []
    for cdir, consumer in MODEL_DIRS.items():
        cfam = MODEL_FAMILY[consumer]
        for ds in DATASET_ORDER:
            items = {" ".join(it["sentence"].split()): it for it in load_mint_items(ds)}
            base_path = out_dir / cdir / f"exp1_results_{FILE_DS[ds]}_{cdir}.json"
            if not base_path.is_file():
                continue
            base = per_question_correct(base_path, ds, "baseline", items)
            sources = {"llama": base_path}
            for gen in ("glm", "gptoss"):
                p = out_dir / cdir / "generators" / gen / f"exp1_results_{FILE_DS[ds]}_{cdir}.json"
                if p.is_file():
                    sources[gen] = p
            para = out_dir / cdir / "controls" / "paraphrase_gptoss" / f"exp1_results_{FILE_DS[ds]}_{cdir}.json"
            if para.is_file():
                sources["llama_paraphrased_by_gptoss"] = para
            for gen, p in sources.items():
                mis = per_question_correct(p, ds, args.condition, items)
                qs = sorted(set(base) & set(mis))
                if len(qs) < 10:
                    continue
                res = paired_bootstrap_diff([base[q] for q in qs], [mis[q] for q in qs], n_boot=args.n_boot)
                gfam = gen if gen in ("llama", "glm", "gptoss") else "llama"
                st_b = STATUS_BY_FILE.get((str(base_path), "baseline"), {})
                st_m = STATUS_BY_FILE.get((str(p), args.condition), {})
                cells.append({"consumer": consumer, "generator": GEN_LABEL.get(gen, gen), "generator_key": gen, "dataset": ds,
                              "acc_baseline": res["mean_a"], "acc_misinformed": res["mean_b"], "drop": res["diff"],
                              "ci_lo": res["ci_lo"], "ci_hi": res["ci_hi"], "n": res["n"], "same_model": int(gfam == cfam),
                              "style_control": gen == "llama_paraphrased_by_gptoss",
                              "unparsed_rate_baseline": (st_b.get("unparsed", 0) / st_b["n"]) if st_b.get("n") else None,
                              "unparsed_rate_misinformed": (st_m.get("unparsed", 0) / st_m["n"]) if st_m.get("n") else None})
                for q in qs:
                    long.append({"consumer": consumer, "generator": gen, "dataset": ds, "question": q, "correct": mis[q],
                                 "baseline_correct": base[q], "same_model": int(gfam == cfam),
                                 "style_control": int(gen == "llama_paraphrased_by_gptoss")})
    write_csv(RESULTS_DIR / "generator_consumer_matrix.csv", cells)
    print(f"{len(cells)} generator x consumer x dataset cells")

    # ---- same-model effect (logistic regression, question-clustered SEs)
    eff_rows = []
    try:
        import pandas as pd
        import statsmodels.api as sm
        import statsmodels.formula.api as smf

        df = pd.DataFrame([r for r in long if not r["style_control"]])
        if not df.empty and df["generator"].nunique() >= 2:
            f = "correct ~ C(generator) + C(consumer) + C(dataset) + same_model"
            res = smf.glm(f, data=df, family=sm.families.Binomial()).fit(
                cov_type="cluster", cov_kwds={"groups": df["question"].astype("category").cat.codes.values})
            for term in res.params.index:
                eff_rows.append({"term": term, "coef_logit": float(res.params[term]), "odds_ratio": float(np.exp(res.params[term])),
                                 "se": float(res.bse[term]), "ci_lo": float(res.params[term] - 1.96 * res.bse[term]),
                                 "ci_hi": float(res.params[term] + 1.96 * res.bse[term]), "p": float(res.pvalues[term]),
                                 "n_obs": int(res.nobs)})
            # also with baseline correctness as covariate (question difficulty)
            f2 = f + " + baseline_correct"
            res2 = smf.glm(f2, data=df, family=sm.families.Binomial()).fit(
                cov_type="cluster", cov_kwds={"groups": df["question"].astype("category").cat.codes.values})
            eff_rows.append({"term": "same_model (adj. baseline_correct)", "coef_logit": float(res2.params["same_model"]),
                             "odds_ratio": float(np.exp(res2.params["same_model"])), "se": float(res2.bse["same_model"]),
                             "ci_lo": float(res2.params["same_model"] - 1.96 * res2.bse["same_model"]),
                             "ci_hi": float(res2.params["same_model"] + 1.96 * res2.bse["same_model"]),
                             "p": float(res2.pvalues["same_model"]), "n_obs": int(res2.nobs)})
            # one indicator per diagonal cell: which consumer carries the same-model effect
            for fam in ("llama", "glm", "gptoss"):
                df[f"same_{fam}"] = ((df["same_model"] == 1) & (df["generator"] == fam)).astype(int)
            f3 = "correct ~ C(generator) + C(consumer) + C(dataset) + same_llama + same_glm + same_gptoss"
            res3 = smf.glm(f3, data=df, family=sm.families.Binomial()).fit(
                cov_type="cluster", cov_kwds={"groups": df["question"].astype("category").cat.codes.values})
            for fam in ("llama", "glm", "gptoss"):
                t = f"same_{fam}"
                if t in res3.params.index:
                    eff_rows.append({"term": f"{t} (one indicator per diagonal cell)", "coef_logit": float(res3.params[t]),
                                     "odds_ratio": float(np.exp(res3.params[t])), "se": float(res3.bse[t]),
                                     "ci_lo": float(res3.params[t] - 1.96 * res3.bse[t]),
                                     "ci_hi": float(res3.params[t] + 1.96 * res3.bse[t]),
                                     "p": float(res3.pvalues[t]), "n_obs": int(res3.nobs)})
        else:
            print("same-model regression needs >= 2 generators (run exp1 on MINT_dataset_v1.2/generators/<gen>)")
    except ImportError:
        print("statsmodels not installed; skipping regression")
    write_csv(RESULTS_DIR / "same_model_effect.csv", eff_rows)

    # ---- LaTeX matrix
    gens = [g for g in ("Llama-3.3-70B", "GLM-4.7-Flash", "GPT-OSS-120B") if any(c["generator"] == g for c in cells)]
    para = "llama_paraphrased_by_gptoss"
    table_gens = gens + ([para] if any(c["generator"] == para for c in cells) else [])
    gen_head = {"Llama-3.3-70B": "Gen.: Llama-3.3", "GLM-4.7-Flash": "Gen.: GLM-4.7", "GPT-OSS-120B": "Gen.: GPT-OSS",
                para: "Llama-3.3, paraphrased by GPT-OSS"}
    tex = []
    for ds in DATASET_ORDER:
        for consumer in MODEL_ORDER:
            row = [DATASET_DISPLAY.get(ds, ds), consumer]
            any_cell = False
            for g in table_gens:
                c = next((x for x in cells if x["dataset"] == ds and x["consumer"] == consumer and x["generator"] == g), None)
                if c:
                    any_cell = True
                    mark = "*" if c["same_model"] and not c["style_control"] else ""
                    row.append(f"{c['drop']:+.3f} [{c['ci_lo']:+.3f}, {c['ci_hi']:+.3f}]{mark}")
                else:
                    row.append("--")
            if any_cell:
                tex.append(row)
    write_latex_table(TABLES_DIR / "generator_consumer_matrix.tex", ["Dataset", "Consumer"] + [gen_head.get(g, g) for g in table_gens],
                      tex,
                      caption="Accuracy drop (baseline $-$ misinformed, single agent, plain false-fact passages, all questions) for "
                              "every generator $\\times$ consumer cell with paired bootstrap 95\\% CIs; * marks the same-model "
                              "(diagonal) cells. Last column: the Llama-3.3 passages paraphrased by GPT-OSS-120B, consumed by "
                              "Llama-3.3 (style control; not run for the other consumers).",
                      label="tab:generator_consumer",
                      col_spec="ll" + "l" * len(table_gens))
    if eff_rows:
        sm_row = next((r for r in eff_rows if r["term"] == "same_model"), None)
        if sm_row:
            print(f"same_model effect: OR={sm_row['odds_ratio']:.3f} [{np.exp(sm_row['ci_lo']):.3f}, {np.exp(sm_row['ci_hi']):.3f}] p={sm_row['p']:.4f}")


if __name__ == "__main__":
    main()
