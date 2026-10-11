#!/usr/bin/env python3
"""
exposure_curve.py - Exposure-matched comparison of single agent, debate and
misinformed self-consistency.

Builds, per model x dataset x protocol, the accuracy at exposure k/n for
  MAD(k)   multi-agent debate (exp_exposure results; exp2 supplies k=1 of 3 and
           exp3 supplies k/5 for WinoGrande when exposure runs are missing)
  mSC(k)   misinformed self-consistency (exp_msc results; exp2_ablation supplies a
           15-sample fully-exposed SC reference)
  SA       single agent: baseline (exposure 0) and false_fact (exposure 1/1) from exp1
and the debate effect  debate_effect(k) = acc_MAD(k) - acc_mSC(k)  with paired
bootstrap CIs over questions (scripts/stats.py).

Outputs
  out/results/exposure_curve.csv   (model, dataset, protocol, system, k, n, seed, acc, ci_lo, ci_hi, n_questions)
  out/results/debate_effect.csv    (paired MAD - mSC per k with CI and permutation p)
  out/results/exposure_long.csv    (per-question long table used by scripts/stats.py)
  out/tables/exposure_curve.tex, out/tables/debate_effect.tex
  out/figures/fig_exposure_curve.pdf
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from scripts.common import (  # noqa: E402
    DATASET_DISPLAY,
    DATASET_ORDER,
    FIGURES_DIR,
    MODEL_ORDER,
    RESULTS_DIR,
    TABLES_DIR,
    ensure_dirs,
    read_csv,
    verification_criterion,
    verified_counts_text,
    verified_questions,
    write_csv,
    write_latex_table,
)
from scripts.stats import bootstrap_mean_ci, holm_bonferroni, paired_bootstrap_diff  # noqa: E402


def _b(v) -> bool:
    return str(v).strip().lower() in ("true", "1", "1.0", "yes")


def _int(v, default=None):
    try:
        return int(float(v))
    except Exception:
        return default


def collect(category: str, sa_rows, mad_rows, sc_rows, seeds: Optional[List[int]] = None,
            msc_sample_rows: Optional[List[dict]] = None) -> List[dict]:
    """Return long rows: model, dataset, system, protocol, k, n, seed, mint_id, is_correct.

    Systems: SA (exp1 single agent, own prompt), MAD (debate), mSC (misinformed self-consistency),
    SC15 (15-sample self-consistency, exp2_ablation) and, when msc_sample_rows is given, SAdp = the
    individual samples of exp_msc, i.e. single agents under the debate prompt format (informed
    samples = exposure 1/1, uninformed samples = exposure 0/1)."""
    long: List[dict] = []
    # single agent (exp1): baseline -> k=0/1, category -> k=1/1. Only seed 0 is used as the single-agent
    # reference: it is the run that covers every condition. Seeds 1-2 repeat baseline and false_fact (with the
    # constrained re-prompt) and are not part of the exposure curve.
    for r in sa_rows:
        if r["experiment"] != "exp1":
            continue
        if r["condition"] == "baseline":
            k = 0
        elif r["condition"] == category or r["condition"] == f"strategy_{category}":
            k = 1
        else:
            continue
        seed = _int(r.get("seed"), 0)
        if seed != 0 or (seeds is not None and seed not in seeds):
            continue
        long.append({"model": r["model"], "dataset": r["dataset"], "system": "SA", "protocol": "single", "k": k, "n": 1,
                     "seed": seed, "mint_id": r["mint_id"], "is_correct": _b(r["is_correct"]), "status": r.get("status"),
                     "source": "exp1"})
    # debate. The exp2 runs (1 of 3, misinformed agent always Participant 3, full-sample
    # run set) are only used as a fallback for model x dataset pairs WITHOUT exp_exposure runs;
    # otherwise they would be pooled into the same k=1/3 cell as the exposure runs.
    has_exposure = {(r["model"], r["dataset"]) for r in mad_rows if r["experiment"] == "exp_exposure"}
    for r in mad_rows:
        exp = r["experiment"]
        k, n = _int(r["k_misinformed"]), _int(r["num_agents"])
        if k is None or n is None:
            continue
        if exp == "exp_exposure":
            if r.get("information_source") not in ("misinformation", "none") or r.get("mitigation") not in ("none", "", None):
                continue
            if k > 0 and r["category"] != category:
                continue
        elif exp == "exp2":
            if r["condition"] != category or (r["model"], r["dataset"]) in has_exposure:
                continue
        elif exp == "exp3":
            pass  # 5-agent WinoGrande composition sweep (false_fact)
        else:
            continue
        seed = _int(r.get("seed"), 0)
        if seeds is not None and seed not in seeds:
            continue
        long.append({"model": r["model"], "dataset": r["dataset"], "system": "MAD", "protocol": r["protocol"], "k": k,
                     "n": n, "seed": seed, "mint_id": r["mint_id"], "is_correct": _b(r["is_correct"]), "status": r.get("status"),
                     "source": exp})
    # self-consistency with the STRICT majority rule (a question without a majority counts as wrong):
    # mSC-strict (3 samples, exp_msc) and SC15 (15 samples, exp2_ablation)
    for r in sc_rows:
        exp = r["experiment"]
        if exp == "exp_msc":
            k, n = _int(r["k_misinformed"]), _int(r["num_samples_cfg"])
            if k is None or n is None:
                continue
            if r.get("information_source") not in ("misinformation", "none") or r.get("mitigation") not in ("none", "", None):
                continue
            if k > 0 and r["category"] != category:
                continue
            system = "mSC-strict"
        elif exp == "exp2_ablation":
            if r["condition"] != category:
                continue
            k, n, system = 15, 15, "SC15"  # fully exposed, 15 samples (exp2_ablation)
        else:
            continue
        seed = _int(r.get("seed"), 0)
        if seeds is not None and seed not in seeds:
            continue
        long.append({"model": r["model"], "dataset": r["dataset"], "system": system, "protocol": "vote-strict", "k": k, "n": n,
                     "seed": seed, "mint_id": r["mint_id"], "is_correct": _b(r["is_correct"]), "status": r.get("status"),
                     "source": exp})
    # mSC (the paper's baseline): majority over the three samples, falling back to the FIRST sample when
    # there is no majority (the rule of the runner). Free-form answers rarely match string-exactly
    # across samples, so 23-58% of CWQ questions have no majority and count as wrong under the strict rule.
    sample0: Dict[tuple, tuple] = {}
    for r in (msc_sample_rows or []):
        if r.get("experiment") == "exp_msc" and str(r.get("sample_idx")) in ("0", "0.0"):
            sample0[(r["model"], r["dataset"], r["condition"], r["mint_id"])] = (_b(r["is_correct"]), r.get("status"))
    if sample0:
        for r in sc_rows:
            if r["experiment"] != "exp_msc":
                continue
            k, n = _int(r["k_misinformed"]), _int(r["num_samples_cfg"])
            if k is None or n is None:
                continue
            if r.get("information_source") not in ("misinformation", "none") or r.get("mitigation") not in ("none", "", None):
                continue
            if k > 0 and r["category"] != category:
                continue
            seed = _int(r.get("seed"), 0)
            if seeds is not None and seed not in seeds:
                continue
            no_majority = _b(r.get("tie")) or r.get("status") != "parsed"
            s0 = sample0.get((r["model"], r["dataset"], r["condition"], r["mint_id"]), (False, "unparsed"))
            ok, status = s0 if no_majority else (_b(r["is_correct"]), r.get("status"))
            long.append({"model": r["model"], "dataset": r["dataset"], "system": "mSC", "protocol": "vote", "k": k, "n": n,
                         "seed": seed, "mint_id": r["mint_id"], "is_correct": ok, "status": status, "source": "exp_msc"})
    # single agent under the debate prompt: the individual samples of exp_msc
    for r in (msc_sample_rows or []):
        if r.get("experiment") != "exp_msc":
            continue
        if r.get("information_source") not in ("misinformation", "none") or r.get("mitigation") not in ("none", "", None):
            continue
        informed = str(r.get("informed")).strip().lower() in ("true", "1")
        if informed and r.get("category") != category:
            continue
        seed = _int(r.get("seed"), 0)
        if seeds is not None and seed not in seeds:
            continue
        long.append({"model": r["model"], "dataset": r["dataset"], "system": "SAdp", "protocol": "single",
                     "k": 1 if informed else 0, "n": 1, "seed": seed, "mint_id": r["mint_id"],
                     "is_correct": _b(r["is_correct"]), "status": r.get("status"), "source": "exp_msc_samples"})
    return long


def verified_question_sets(long: List[dict]) -> Dict[tuple, Dict[str, set]]:
    """Per (model, dataset): the question sets on which a curve is reported.

    'matched'  questions present in ALL k >= 1 exposure cells of the pair AND verified under the
               dataset's criterion (scripts.common.verification_criterion); for CWQ/ETHICS the k >= 1
               runs already contain only verified questions, for WinoGrande (run on all questions)
               this applies the Q2 rule of both judges
    'strict'   (only where the dataset criterion is not Q1-and-Q2) the Q1-and-Q2 subset, a sensitivity check
    """
    by_k: Dict[tuple, Dict[int, set]] = defaultdict(lambda: defaultdict(set))
    for r in long:
        if r["source"] == "exp_exposure" and r["k"] > 0 and r["n"] == 3:
            by_k[(r["model"], r["dataset"])][r["k"]].add(r["mint_id"])
    sets: Dict[tuple, Dict[str, set]] = {}
    for key, kmap in by_k.items():
        qs = None
        for _k, s in kmap.items():
            qs = set(s) if qs is None else (qs & s)
        if not qs:
            continue
        ds = key[1]
        matched = qs & verified_questions(ds)
        if matched:
            sets[key] = {"matched": matched}
            if verification_criterion(ds) != "both_q1q2":
                strict = qs & verified_questions(ds, criterion="both_q1q2")
                if strict:
                    sets[key]["strict"] = strict
    return sets


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--category", default="false_fact")
    ap.add_argument("--rescored_dir", default=str(RESULTS_DIR / "rescored"))
    ap.add_argument("--n_boot", type=int, default=10_000)
    ap.add_argument("--no_figure", action="store_true")
    args = ap.parse_args()
    ensure_dirs()
    rd = Path(args.rescored_dir)
    sa = read_csv(rd / "single_agent.csv") if (rd / "single_agent.csv").is_file() else []
    mad = read_csv(rd / "mad_final.csv") if (rd / "mad_final.csv").is_file() else []
    sc = read_csv(rd / "self_consistency.csv") if (rd / "self_consistency.csv").is_file() else []
    mss = read_csv(rd / "msc_samples.csv") if (rd / "msc_samples.csv").is_file() else []
    long = collect(args.category, sa, mad, sc, msc_sample_rows=mss)
    write_csv(RESULTS_DIR / "exposure_long.csv", long)
    print(f"{len(long)} per-question rows")

    # ---- question sets (verified_question_sets): "matched" = the verified questions present in every k >= 1
    # exposure cell of a model x dataset (CWQ/ETHICS: the --mint_filter verified runs; WinoGrande: the Q2 rule
    # applied to the all-question runs); "strict" = Q1-and-Q2 subset where that differs (WinoGrande, n = 27);
    # "all" keeps every question of a cell (full sample for k = 0 and the single agent). Paired differences
    # use the intersection of questions of the matched set.
    qsets = verified_question_sets(long)
    matched_q: Dict[tuple, set] = {key: s["matched"] for key, s in qsets.items()}

    # ---- accuracy per cell (pool seeds: per-question mean over seeds, then bootstrap over questions)
    cells: Dict[tuple, Dict[str, List[tuple]]] = defaultdict(lambda: defaultdict(list))
    status_cells: Dict[tuple, Dict[str, List[str]]] = defaultdict(lambda: defaultdict(list))
    for r in long:
        ck = (r["model"], r["dataset"], r["system"], r["protocol"], r["k"], r["n"])
        cells[ck][r["mint_id"]].append((r["seed"], float(r["is_correct"])))
        status_cells[ck][r["mint_id"]].append(r.get("status") or "")
    curve_rows = []
    per_q: Dict[tuple, Dict[str, float]] = {}       # matched questions (paired comparisons of the paper)
    per_q_all: Dict[tuple, Dict[str, float]] = {}   # every question of the cell
    for key, qmap in cells.items():
        sets_here = {"all": None}
        sets_here.update(qsets.get((key[0], key[1]), {}))
        for question_set, Q in sets_here.items():
            sub = qmap if Q is None else {q: v for q, v in qmap.items() if q in Q}
            if not sub:
                continue
            vals = {q: float(np.mean([x for _, x in v])) for q, v in sub.items()}
            if question_set == "all":
                per_q_all[key] = vals
            if question_set == "matched" or (question_set == "all" and (key[0], key[1]) not in qsets):
                per_q[key] = vals
            ci = bootstrap_mean_ci(list(vals.values()), n_boot=args.n_boot)
            by_seed: Dict[int, List[float]] = defaultdict(list)
            for v in sub.values():
                for s, x in v:
                    by_seed[s].append(x)
            seed_accs = [float(np.mean(v)) for v in by_seed.values()]
            # parsed-only (conditional) accuracy and the share of unparsed/ambiguous outcomes (extraction bound)
            st = [(x, s) for q in sub for (_, x), s in zip(cells[key][q], status_cells[key][q])]
            parsed = [x for x, s in st if s == "parsed" or s == "stored_vote"]
            curve_rows.append({"model": key[0], "dataset": key[1], "system": key[2], "protocol": key[3], "k": key[4],
                               "n": key[5], "exposure": key[4] / key[5] if key[5] else None, "question_set": question_set,
                               "acc": ci["mean"], "ci_lo": ci["ci_lo"], "ci_hi": ci["ci_hi"], "n_questions": ci["n"],
                               "n_seeds": len(by_seed), "seeds": ";".join(str(s) for s in sorted(by_seed)),
                               "acc_seed_sd": float(np.std(seed_accs, ddof=1)) if len(seed_accs) > 1 else None,
                               "acc_parsed_only": float(np.mean(parsed)) if parsed else None,
                               "unparsed_share": 1.0 - len(parsed) / len(st) if st else None})
    curve_rows.sort(key=lambda r: (MODEL_ORDER.index(r["model"]) if r["model"] in MODEL_ORDER else 9, r["dataset"],
                                   r["question_set"], r["system"], r["protocol"], r["n"], r["k"]))
    write_csv(RESULTS_DIR / "exposure_curve.csv", curve_rows)
    # primary rows for the table and the figure: matched questions where exposure runs exist, else all
    primary_rows = [r for r in curve_rows
                    if r["question_set"] == ("matched" if (r["model"], r["dataset"]) in matched_q else "all")]
    n_matched = {ds: sorted({len(s["matched"]) for key, s in qsets.items() if key[1] == ds}) for ds in DATASET_ORDER}
    n_text = ", ".join(f"{DATASET_DISPLAY[ds]} {'/'.join(str(v) for v in n_matched[ds])}" for ds in DATASET_ORDER if n_matched[ds])

    # ---- debate effect: MAD(k, protocol) - mSC(k) paired over questions (same n). Baselines: mSC (the
    # paper's baseline, tie -> first sample) and mSC-strict (tie -> wrong). k = 0 is paired on the full sample
    # when no exposure run restricts the pair (per_q holds the matched questions where they exist).
    eff_rows = []
    for key, mad_q in per_q.items():
        model, ds, system, proto, k, n = key
        if system != "MAD":
            continue
        for base_system, base_proto in (("mSC", "vote"), ("mSC-strict", "vote-strict")):
            msc_key = (model, ds, base_system, base_proto, k, n)
            if msc_key not in per_q:
                continue
            msc_q = per_q[msc_key]
            qs = sorted(set(mad_q) & set(msc_q))
            if len(qs) < 10:
                continue
            res = paired_bootstrap_diff([mad_q[q] for q in qs], [msc_q[q] for q in qs], n_boot=args.n_boot)
            eff_rows.append({"model": model, "dataset": ds, "protocol": proto, "k": k, "n": n, "baseline": base_system,
                             "acc_mad": res["mean_a"], "acc_msc": res["mean_b"], "debate_effect": res["diff"],
                             "ci_lo": res["ci_lo"], "ci_hi": res["ci_hi"], "p_perm": res["p_perm"], "n_questions": res["n"]})
    # Holm adjustment within the family of the paper's principal debate effects (mSC baseline, n = 3)
    fam = [r for r in eff_rows if r["baseline"] == "mSC" and r["n"] == 3]
    for r, p_adj in zip(fam, holm_bonferroni([r["p_perm"] for r in fam])):
        r["p_holm"] = p_adj
    eff_rows.sort(key=lambda r: (MODEL_ORDER.index(r["model"]) if r["model"] in MODEL_ORDER else 9, DATASET_ORDER.index(r["dataset"])
                                 if r["dataset"] in DATASET_ORDER else 9, r["protocol"], r["n"], r["k"], r["baseline"]))
    write_csv(RESULTS_DIR / "debate_effect.csv", eff_rows)

    # ---- LaTeX: full exposure table (appendix)
    strict_note = ""
    strict_ds = [ds for ds in DATASET_ORDER if verification_criterion(ds) != "both_q1q2"]
    if strict_ds:
        strict_note = (" On " + " and ".join(DATASET_DISPLAY[d] for d in strict_ds) + " verified means that both judges confirm "
                       "that the passage supports the intended wrong answer (Q2); the stricter Q1-and-Q2 subset is in "
                       "Table~\\ref{tab:exposure_curve_strict}.")
    tex_rows = [[r["model"], DATASET_DISPLAY.get(r["dataset"], r["dataset"]), r["system"], r["protocol"], f"{r['k']}/{r['n']}",
                 f"{r['acc']:.3f} [{r['ci_lo']:.3f}, {r['ci_hi']:.3f}]", r["n_questions"], r["n_seeds"]] for r in primary_rows]
    write_latex_table(TABLES_DIR / "exposure_curve.tex", ["Model", "Dataset", "System", "Protocol", "Exposure", "Accuracy [95\\% CI]",
                                                          "n", "seeds"], tex_rows,
                      caption=f"Accuracy by exposure level ({args.category} passages). SA = single agent (own prompt), SAdp = single "
                              "sample under the debate prompt, MAD = 3-agent debate (voting / consensus), mSC = misinformed "
                              "self-consistency (3 samples, majority vote, first sample when there is no majority), mSC-strict = the "
                              "same with ties counted as wrong, SC15 = 15-sample self-consistency (strict). Bootstrap 95\\% CIs over "
                              "questions; all cells of a model x dataset are restricted to the MINT-verified questions "
                              f"({n_text}; n = number of questions, seeds = runs pooled per question).{strict_note}",
                      label="tab:exposure_curve", col_spec="llllrlrr")
    strict_rows = [r for r in curve_rows if r["question_set"] == "strict"]
    if strict_rows:
        tex_rows = [[r["model"], DATASET_DISPLAY.get(r["dataset"], r["dataset"]), r["system"], r["protocol"], f"{r['k']}/{r['n']}",
                     f"{r['acc']:.3f} [{r['ci_lo']:.3f}, {r['ci_hi']:.3f}]", r["n_questions"], r["n_seeds"]] for r in strict_rows]
        write_latex_table(TABLES_DIR / "exposure_curve_strict.tex", ["Model", "Dataset", "System", "Protocol", "Exposure",
                                                                     "Accuracy [95\\% CI]", "n", "seeds"], tex_rows,
                          caption="Sensitivity check: the cells of Table~\\ref{tab:exposure_curve} on the questions whose passage both "
                                  "judges confirm on Q1 and Q2 (" + ", ".join(f"{DATASET_DISPLAY[d]} n = "
                                  f"{len(next(iter(s['strict'] for key, s in qsets.items() if key[1] == d and 'strict' in s)))}"
                                  for d in strict_ds) + ").", label="tab:exposure_curve_strict", col_spec="llllrlrr")
    tex_eff = [[r["model"], DATASET_DISPLAY.get(r["dataset"], r["dataset"]), r["protocol"], f"{r['k']}/{r['n']}", r["baseline"],
                f"{r['acc_mad']:.3f}", f"{r['acc_msc']:.3f}", f"{r['debate_effect']:+.3f} [{r['ci_lo']:+.3f}, {r['ci_hi']:+.3f}]",
                f"{r['p_perm']:.4f}", r["n_questions"]] for r in eff_rows]
    write_latex_table(TABLES_DIR / "debate_effect.tex", ["Model", "Dataset", "Protocol", "Exposure", "Baseline", "MAD", "Baseline acc.",
                                                         "Debate effect [95\\% CI]", "p", "n"], tex_eff,
                      caption="Debate effect at equal exposure: acc$_{\\mathrm{MAD}}(k)$ $-$ acc$_{\\mathrm{baseline}}(k)$ with paired "
                              "bootstrap 95\\% CIs and paired permutation p-values over questions. mSC falls back to the first sample "
                              "when the three samples have no majority (the baseline of the main text); mSC-strict counts such "
                              "questions as wrong (relevant for free-form CWQ answers, where 23--58\\% of the questions have no "
                              "string-identical majority). All exposure levels, including 0/3, are paired on the verified questions (" + verified_counts_text(args.category) + ").",
                      label="tab:debate_effect", col_spec="lllllrrlrr")

    # ---- LaTeX: compact main-text table of the principal debate effects (mSC baseline, 3 agents)
    def _cell(r):
        if r is None:
            return "--"
        mark = "$^{*}$" if r.get("p_holm", 1.0) < 0.05 else ("$^{\\dagger}$" if r["p_perm"] < 0.05 else "")
        return f"{r['debate_effect']:+.2f} [{r['ci_lo']:+.2f}, {r['ci_hi']:+.2f}]{mark}"

    main_rows = []
    ks = sorted({r["k"] for r in fam})
    for model in MODEL_ORDER:
        for ds in DATASET_ORDER:
            for proto in ("voting", "consensus"):
                sel = [r for r in fam if r["model"] == model and r["dataset"] == ds and r["protocol"] == proto]
                if not sel:
                    continue
                n_q = sorted({r["n_questions"] for r in sel if r["k"] > 0}) or sorted({r["n_questions"] for r in sel})
                main_rows.append([model, DATASET_DISPLAY[ds], proto] +
                                 [_cell(next((r for r in sel if r["k"] == k), None)) for k in ks] + ["/".join(str(v) for v in n_q)])
    write_latex_table(TABLES_DIR / "debate_effect_main.tex", ["Model", "Dataset", "Protocol"] + [f"{k}/3" for k in ks] + ["n"],
                      main_rows,
                      caption="Debate effect at equal exposure, acc$_{\\mathrm{MAD}}(k/3)$ $-$ acc$_{\\mathrm{mSC}}(k/3)$, paired over "
                              f"questions with 95\\% bootstrap CIs; all exposure levels on the same verified questions "
                              f"({verified_counts_text(args.category)}). " "$^{*}$: Holm-adjusted $p < 0.05$ across the 72 cells "
                              "of the table; $^{\\dagger}$: unadjusted "
                              "$p < 0.05$ only. Three seeds at 1/3 and 3/3, one at 0/3 and 2/3.",
                      label="tab:debate_effect_main", col_spec="lll" + "l" * len(ks) + "r", font="footnotesize")

    # ---- LaTeX: parsed-only accuracy for GLM-4.7 (extraction bound, appendix)
    glm_rows = [r for r in primary_rows if r["model"] == "GLM-4.7-Flash" and r["system"] in ("MAD", "mSC", "mSC-strict", "SAdp", "SA")
                and r["n"] in (1, 3) and r["acc_parsed_only"] is not None]
    if glm_rows:
        tex_rows = [[DATASET_DISPLAY[r["dataset"]], r["system"], r["protocol"], f"{r['k']}/{r['n']}", f"{r['acc']:.3f}",
                     f"{r['acc_parsed_only']:.3f}", f"{100 * r['unparsed_share']:.1f}", r["n_questions"]] for r in glm_rows]
        write_latex_table(TABLES_DIR / "glm_parsed_only.tex", ["Dataset", "System", "Protocol", "Exposure", "Acc. (unparsed = wrong)",
                                                               "Acc. (parsed only)", "Unparsed \\%", "n"], tex_rows,
                          caption="GLM-4.7-Flash: accuracy with unparsed or ambiguous answers counted as wrong (the paper's rule) next "
                                  "to the accuracy among parsed answers only and the share of unparsed outcomes, for every cell of "
                                  "Table~\\ref{tab:exposure_curve} (verified questions; mSC: status of the sample that decides the vote). "
                                  "The parsed-only column bounds the extraction component of GLM-4.7's debate-versus-mSC gains.",
                          label="tab:glm_parsed_only", col_spec="lllrrrrr")
    print(f"wrote {len(curve_rows)} curve cells, {len(eff_rows)} debate-effect rows; matched n: {n_text}")

    if args.no_figure or not curve_rows:
        return
    try:
        import matplotlib.pyplot as plt

        from scripts.figure_common import (SYSTEM_COLOR, annotate_value, apply_paper_style, dataset_label, finish_with_legend,
                                           model_label, panel_title, save_figure, series_handle, style_axes)

        fs = apply_paper_style()
        models = [m for m in MODEL_ORDER if any(r["model"] == m for r in curve_rows)]
        datasets = [d for d in DATASET_ORDER if any(r["dataset"] == d for r in curve_rows)]
        # system -> (colour, linestyle, marker, legend label, value labels above/below/None)
        style = {("MAD", "voting"): (SYSTEM_COLOR["MAD voting"], "-", "o", "MAD voting", True),
                 ("MAD", "consensus"): (SYSTEM_COLOR["MAD consensus"], "--", "s", "MAD consensus", None),
                 ("mSC", "vote"): (SYSTEM_COLOR["mSC"], "-.", "^", "mSC (no interaction)", False),
                 ("SA", "single"): (SYSTEM_COLOR["single agent"], ":", "D", "single agent", None),
                 ("SAdp", "single"): (SYSTEM_COLOR["single agent (debate prompt)"], ":", "v", "single agent (debate prompt)", None)}
        # compact grid: 2.9 in per model row (was 4.2), x tick labels only on the bottom row, ticks 0 / 0.5 / 1
        fig, axes = plt.subplots(len(models), len(datasets), figsize=(15.5, 2.9 * len(models) + 0.6), sharey=True, sharex=True,
                                 squeeze=False)
        for i, m in enumerate(models):
            for j, ds in enumerate(datasets):
                ax = axes[i][j]
                for (system, proto), (col, ls, mk, lab, label_side) in style.items():
                    rows = sorted([r for r in primary_rows if r["model"] == m and r["dataset"] == ds and r["system"] == system
                                   and r["protocol"] == proto and (r["n"] in (1, 3))], key=lambda r: r["exposure"])
                    if not rows:
                        continue
                    x = [r["exposure"] for r in rows]
                    y = [r["acc"] for r in rows]
                    yerr = [[r["acc"] - r["ci_lo"] for r in rows], [r["ci_hi"] - r["acc"] for r in rows]]
                    ax.errorbar(x, y, yerr=yerr, color=col, linestyle=ls if len(rows) > 1 else "none", marker=mk,
                                markersize=8, markeredgecolor="#2e2e2e", markeredgewidth=0.6, capsize=3, linewidth=2.8,
                                elinewidth=1.0, alpha=0.95, zorder=3)
                    if label_side is not None:
                        for xi, yi in zip(x, y):
                            annotate_value(ax, xi, yi, yi, col, fs, above=label_side, offset=16)
                if i == 0:
                    panel_title(ax, dataset_label(ds), fs)
                if j == 0:
                    ax.set_ylabel(f"{model_label(m)}\nAccuracy", fontsize=fs["label"])
                if i == len(models) - 1 and j == len(datasets) // 2:  # one x label under the middle column
                    ax.set_xlabel("Share of exposed agents (k/n)", fontsize=fs["label"])
                ax.set_xticks([0, 1 / 3, 2 / 3, 1])
                ax.set_xticklabels(["0", "1/3", "2/3", "1"])
                ax.set_xlim(-0.08, 1.08)
                ax.set_ylim(-0.16, 1.05)  # room for value labels below points near 0
                ax.set_yticks([0, 0.5, 1.0])
                style_axes(ax, "y", fs)
                if i < len(models) - 1:
                    ax.tick_params(axis="x", labelbottom=False)
        handles = [series_handle(lab, col, mk, ls) for (col, ls, mk, lab, _s) in style.values()]
        finish_with_legend(fig, handles, [h.get_label() for h in handles], fs, ncol=5, top=0.95)
        save_figure(fig, FIGURES_DIR / "fig_exposure_curve")
        print(f"figure -> {FIGURES_DIR / 'fig_exposure_curve.pdf'}")
    except Exception as e:  # noqa: BLE001
        print(f"[WARN] figure skipped: {e}")


if __name__ == "__main__":
    main()
