#!/usr/bin/env python3
"""
verified_subset_robustness.py - Results on the MINT-verified subset vs. the full set.

For every model x dataset x passage category the single-agent accuracy drop
    drop = acc(baseline) - acc(misinformed)
is computed (paired over questions, bootstrap CIs) on several question sets:
  all           every question of the sample
  verified      the dataset's verified subset (scripts.common.verification_criterion: both judges Q1 and Q2
                on CWQ/ETHICS; both judges Q2 on WinoGrande, where Q1 is ill-posed) - the questions of the debates
  unverified    the remaining questions
  strict        both judges Q1 and Q2 (identical to `verified` on CWQ/ETHICS; the n = 27 subset on WinoGrande)
  either_judge  at least one judge says yes to Q1 and Q2
  loo           leave-one-out: verified by the judge(s) that are NOT the consumer model (Llama: both judges;
                GLM: GPT-OSS only; GPT-OSS: GLM only), the consumer-disjoint check for the judge-consumer overlap
This shows whether the reported effects depend on passages that failed the automatic
validation and whether a consumer's drop depends on its own judgement of the passages.

Inputs: out/results/rescored/single_agent.csv (seed 0, exp1), MINT_dataset_v1.2/<dataset>_passages.jsonl
Outputs: out/results/verified_subset_robustness.csv, out/tables/verified_subset_robustness.tex,
         out/tables/verified_subset_loo.tex
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from scripts.common import (CATEGORY_ORDER, DATASET_DISPLAY, DATASET_ORDER, MODEL_ORDER, RESULTS_DIR, TABLES_DIR,  # noqa: E402
                            ensure_dirs, read_csv, verification_criterion, verified_counts_text, verified_questions, write_csv,
                            write_latex_table)
from scripts.stats import paired_bootstrap_diff  # noqa: E402

JUDGE_OF = {"Llama-3.3-70B": None, "GLM-4.7-Flash": "glm_4_7_flash", "GPT-OSS-120B": "gpt_oss_120b"}
ALL_JUDGES = ("gpt_oss_120b", "glm_4_7_flash")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rescored_dir", default=str(RESULTS_DIR / "rescored"))
    ap.add_argument("--data_dir", default=str(REPO_ROOT / "MINT_dataset_v1.2"))
    ap.add_argument("--n_boot", type=int, default=10_000)
    args = ap.parse_args()
    ensure_dirs()
    data_dir = Path(args.data_dir)
    sa = [r for r in read_csv(Path(args.rescored_dir) / "single_agent.csv")
          if r["experiment"] == "exp1" and str(r.get("seed") or "0") in ("0", "0.0")]
    base: Dict[tuple, Dict[str, float]] = defaultdict(dict)
    mis: Dict[tuple, Dict[str, float]] = defaultdict(dict)
    for r in sa:
        ok = float(str(r["is_correct"]).lower() in ("true", "1"))
        if r["condition"] == "baseline":
            base[(r["model"], r["dataset"])][r["mint_id"]] = ok
        elif r["relevance"] == "relevant" and r["category"] in CATEGORY_ORDER:
            mis[(r["model"], r["dataset"], r["category"])][r["mint_id"]] = ok
    rows = []
    for model in MODEL_ORDER:
        for ds in DATASET_ORDER:
            b = base.get((model, ds))
            if not b:
                continue
            for cat in CATEGORY_ORDER:
                m = mis.get((model, ds, cat))
                if not m:
                    continue
                qs_all = sorted(set(b) & set(m))
                V = verified_questions(ds, cat, data_dir=data_dir)
                strict = verified_questions(ds, cat, criterion="both_q1q2", data_dir=data_dir)
                either = verified_questions(ds, cat, criterion="either_q1q2", data_dir=data_dir)
                own = JUDGE_OF[model]
                loo_judges = [j for j in ALL_JUDGES if j != own]
                loo = verified_questions(ds, cat, judges=loo_judges, data_dir=data_dir)
                subsets = {"all": qs_all,
                           "verified": [q for q in qs_all if q in V],
                           "unverified": [q for q in qs_all if q not in V],
                           "strict": [q for q in qs_all if q in strict],
                           "either_judge": [q for q in qs_all if q in either],
                           "loo": [q for q in qs_all if q in loo]}
                for name, qs in subsets.items():
                    if len(qs) < 5:
                        rows.append({"model": model, "dataset": ds, "category": cat, "question_set": name, "n": len(qs),
                                     "acc_baseline": None, "acc_misinformed": None, "drop": None, "ci_lo": None, "ci_hi": None,
                                     "p_perm": None, "criterion": verification_criterion(ds),
                                     "loo_judges": ";".join(loo_judges) if name == "loo" else ""})
                        continue
                    res = paired_bootstrap_diff([b[q] for q in qs], [m[q] for q in qs], n_boot=args.n_boot)
                    rows.append({"model": model, "dataset": ds, "category": cat, "question_set": name, "n": res["n"],
                                 "acc_baseline": res["mean_a"], "acc_misinformed": res["mean_b"], "drop": res["diff"],
                                 "ci_lo": res["ci_lo"], "ci_hi": res["ci_hi"], "p_perm": res["p_perm"],
                                 "criterion": verification_criterion(ds), "loo_judges": ";".join(loo_judges) if name == "loo" else ""})
    write_csv(RESULTS_DIR / "verified_subset_robustness.csv", rows)

    def _fmt(r):
        return f"{r['drop']:+.3f} [{r['ci_lo']:+.3f}, {r['ci_hi']:+.3f}]" if r["drop"] is not None else "--"

    tex = []
    for r in rows:
        if r["category"] != "false_fact" or r["question_set"] not in ("all", "verified", "unverified", "strict", "either_judge"):
            continue
        if r["question_set"] == "strict" and verification_criterion(r["dataset"]) == "both_q1q2":
            continue  # identical to `verified` on CWQ/ETHICS
        tex.append([r["model"], DATASET_DISPLAY.get(r["dataset"], r["dataset"]), r["question_set"].replace("_", " "), r["n"], _fmt(r)])
    write_latex_table(TABLES_DIR / "verified_subset_robustness.tex", ["Model", "Dataset", "Questions", "n", "Drop [95\\% CI]"], tex,
                      caption="Single-agent accuracy drop (baseline $-$ misinformed, false\\_fact passages) on all questions, on the "
                              "MINT-verified subset used in the debates (" + verified_counts_text("false_fact") + "; on WinoGrande "
                              "both judges confirm Q2, on CWQ and ETHICS Q1 and Q2), on the unverified remainder, on the WinoGrande "
                              "subset verified on Q1 and Q2 (strict) and on the subset verified by at least one judge. "
                              "Paired bootstrap CIs over questions.", label="tab:verified_subset", col_spec="lllrl")
    # leave-one-out table: consumer x dataset, verified by the other judge(s) only
    tex = []
    for r in rows:
        if r["category"] != "false_fact" or r["question_set"] not in ("verified", "loo"):
            continue
        consumer_is_judge = JUDGE_OF.get(r["model"]) is not None
        if r["question_set"] == "loo" and not consumer_is_judge:
            continue  # identical to the "verified" row (Llama-3.3 is not a judge)
        label = ("both judges" + ("" if consumer_is_judge else " (consumer is not a judge)")) if r["question_set"] == "verified" else             {"gpt_oss_120b": "GPT-OSS-120B only", "glm_4_7_flash": "GLM-4.7-Flash only"}[r["loo_judges"]]
        tex.append([r["model"], DATASET_DISPLAY.get(r["dataset"], r["dataset"]), label, r["n"], _fmt(r)])
    write_latex_table(TABLES_DIR / "verified_subset_loo.tex", ["Consumer", "Dataset", "Verified by", "n", "Drop [95\\% CI]"], tex,
                      caption="Leave-one-out verification: the single-agent drop on the subset verified by both judges (the paper's "
                              "subset) and on the subset verified only by the judge that is not the consumer model (Llama-3.3 is not a "
                              "judge, so its subset is unchanged). Same verification rule per dataset as in "
                              "Table~\\ref{tab:verified_subset}; paired bootstrap CIs over questions.",
                      label="tab:verified_subset_loo", col_spec="lllrl")
    print(f"wrote {len(rows)} rows")


if __name__ == "__main__":
    main()
