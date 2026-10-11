#!/usr/bin/env python3
"""
mitigation_effects.py - Effect of the two mitigations with paired bootstrap CIs.

  (a) source-skepticism prompt under false context: exp_exposure debates run with --mitigation
      source_skepticism (k = 1 and 3 of 3, seed 0, false_fact passages, MINT-verified questions) vs. the
      same debates without the prompt (same questions, same seed -> same informed slots), voting and consensus.
  (b) verifier agent: exp_mitigation_verifier re-runs the final stage of the stored seed-0 k = 1 debates
      with a verifier note shown to every agent, vs. a regenerated final stage without the note (primary
      control, same transcripts) and vs. the stored vote (secondary).
  (c) the cost side of the prompt: the same instruction in debates whose k = 1 or 3 informed agents
      read the RELEVANT-TRUE passage (--information_source relevant_true) and in uninformed debates (k = 0,
      no context), each vs. the same debates without the prompt; and the net effect per question,
      gain under false context + change under true context (both at the same k).

Costs: the verifier adds one generation per debate (its note) - the regenerated final answers and votes are
produced in both arms; token counts come from the run metadata. The source-skepticism prompt adds no
generation (about 40 prompt tokens per call).

Inputs: out/results/rescored/mad_final.csv (scripts/rescore_results.py), out/<model>/mitigation/run_metadata_verifier_*.json
Outputs: out/results/mitigation_effects.csv (every paired comparison; column `context` = false / true / none / net),
         out/tables/mitigation_effects.tex (false context and verifier), out/tables/mitigation_context.tex
         (prompt effect by context), out/figures/fig_mitigation.pdf (the prompt with no, relevant-true and
         false context next to the verifier, voting)
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from scripts.common import (  # noqa: E402
    DATASET_DISPLAY,
    DATASET_ORDER,
    FIGURES_DIR,
    MODEL_DIRS,
    MODEL_ORDER,
    RESULTS_DIR,
    TABLES_DIR,
    ensure_dirs,
    read_csv,
    verified_counts_text,
    verified_questions,
    write_csv,
    write_latex_table,
)
from scripts.stats import paired_bootstrap_diff  # noqa: E402

CONTROL_PROMPT = "same debates without the prompt (seed 0)"
CONTEXT_LABEL = {"false": "false passage", "true": "relevant-true passage", "none": "no context", "net": "net (false + true)"}


def _b(v) -> float:
    return float(str(v).strip().lower() in ("true", "1", "1.0", "yes"))


def _pq(rows: List[dict]) -> Dict[str, float]:
    """Per-question outcome (mean over duplicate rows, e.g. several seeds)."""
    acc: Dict[str, List[float]] = defaultdict(list)
    for r in rows:
        acc[r["mint_id"]].append(_b(r["is_correct"]))
    return {q: sum(v) / len(v) for q, v in acc.items()}


def verifier_costs(out_dir: Path) -> Dict[tuple, dict]:
    """Extra cost of the verifier per debate from the run metadata token ledgers."""
    costs: Dict[tuple, dict] = {}
    for mdir, model in MODEL_DIRS.items():
        for p in glob.glob(str(out_dir / mdir / "mitigation" / "run_metadata_verifier_*.json")):
            try:
                meta = json.load(open(p, encoding="utf-8"))
            except Exception:  # noqa: BLE001
                continue
            for key, led in (meta.get("tokens") or {}).items():
                if key.startswith("verifier_"):
                    ds = key[len("verifier_"):]
                    calls = led.get("calls") or 0
                    if calls:
                        costs[(model, ds)] = {"verifier_calls_per_debate": 1.0,
                                             "verifier_prompt_tokens_per_debate": led.get("prompt_tokens", 0) / calls,
                                             "verifier_completion_tokens_per_debate": led.get("completion_tokens", 0) / calls}
    return costs


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rescored_dir", default=str(RESULTS_DIR / "rescored"))
    ap.add_argument("--out_dir", default=str(REPO_ROOT / "out"))
    ap.add_argument("--category", default="false_fact")
    ap.add_argument("--n_boot", type=int, default=10_000)
    ap.add_argument("--no_figure", action="store_true")
    args = ap.parse_args()
    ensure_dirs()
    mad = read_csv(Path(args.rescored_dir) / "mad_final.csv")
    costs = verifier_costs(Path(args.out_dir))

    # every debate cell is restricted to the verified questions of its dataset (the WinoGrande runs cover all
    # questions; CWQ/ETHICS runs were made with --mint_filter verified and are unaffected; the k = 0 debates without
    # the prompt cover the full samples and are paired with the verified k = 0 skepticism runs)
    mad = [r for r in mad if r["mint_id"] in verified_questions(r["dataset"], args.category)
           or r["experiment"] not in ("exp_exposure", "exp_mitigation_verifier")]

    def sel(model: str, ds: str, mitigation: str, k: str, proto: str, src: str = "misinformation", seed: str = "0") -> List[dict]:
        """Seed-0 exposure debates of one cell. k = 0 debates carry category 'none' and source 'none'."""
        return [r for r in mad if r["model"] == model and r["dataset"] == ds and r["experiment"] == "exp_exposure"
                and (r.get("mitigation") or "none") == mitigation and str(r.get("k_misinformed")) == k
                and r["protocol"] == proto and str(r.get("seed")) == seed and str(r.get("num_agents")) == "3"
                and (r.get("information_source") or "misinformation") == src
                and (k == "0" or r.get("category") == args.category)]

    def _answers(rows: List[dict]) -> Dict[str, str]:
        return {r["mint_id"]: (r.get("predicted_text") or r.get("predicted_index") or "").strip().lower() for r in rows}

    def _paired(a: Dict[str, float], b: Dict[str, float]) -> Optional[dict]:
        qs = sorted(set(a) & set(b))
        if len(qs) < 5:
            return None
        return paired_bootstrap_diff([a[q] for q in qs], [b[q] for q in qs], n_boot=args.n_boot)

    rows = []
    contexts = (("false", "misinformation", ("1", "3")), ("true", "relevant_true", ("1", "3")), ("none", "none", ("0",)))
    for model in MODEL_ORDER:
        for ds in DATASET_ORDER:
            if not any(r["model"] == model and r["dataset"] == ds for r in mad):
                continue
            # (a) + (c) source skepticism by context
            diffs: Dict[tuple, Dict[str, float]] = {}  # (context, k, proto) -> per-question difference (prompt - no prompt)
            for ctx, src, ks in contexts:
                for k in ks:
                    for proto in ("voting", "consensus"):
                        a = _pq(sel(model, ds, "source_skepticism", k, proto, src))
                        b = _pq(sel(model, ds, "none", k, proto, src))
                        res = _paired(a, b)
                        if res is None:
                            continue
                        diffs[(ctx, k, proto)] = {q: a[q] - b[q] for q in set(a) & set(b)}
                        rows.append({"model": model, "dataset": ds, "mitigation": "source_skepticism", "context": ctx, "k": int(k),
                                     "n": 3, "protocol": proto, "control": CONTROL_PROMPT,
                                     "acc_mitigated": res["mean_a"], "acc_control": res["mean_b"], "effect": res["diff"],
                                     "ci_lo": res["ci_lo"], "ci_hi": res["ci_hi"], "p_perm": res["p_perm"], "n_questions": res["n"],
                                     "extra_generations_per_debate": 0.0, "extra_completion_tokens_per_debate": 0.0})
            # net per question: gain under false context + change under true context (same k, same questions)
            for k in ("1", "3"):
                for proto in ("voting", "consensus"):
                    df, dt = diffs.get(("false", k, proto)), diffs.get(("true", k, proto))
                    if not df or not dt:
                        continue
                    qs = sorted(set(df) & set(dt))
                    if len(qs) < 5:
                        continue
                    res = paired_bootstrap_diff([df[q] + dt[q] for q in qs], [0.0] * len(qs), n_boot=args.n_boot)
                    rows.append({"model": model, "dataset": ds, "mitigation": "source_skepticism", "context": "net", "k": int(k),
                                 "n": 3, "protocol": proto, "control": CONTROL_PROMPT,
                                 "acc_mitigated": None, "acc_control": None, "effect": res["diff"],
                                 "ci_lo": res["ci_lo"], "ci_hi": res["ci_hi"], "p_perm": res["p_perm"], "n_questions": res["n"],
                                 "extra_generations_per_debate": 0.0, "extra_completion_tokens_per_debate": 0.0})
            # (b) verifier agent (k = 1, voting, false context)
            mad_md = [r for r in mad if r["model"] == model and r["dataset"] == ds]
            ver_rows = [r for r in mad_md if r["experiment"] == "exp_mitigation_verifier" and r.get("mitigation") == "verifier"
                        and r["protocol"] == "voting"]
            reg_rows = [r for r in mad_md if r["experiment"] == "exp_mitigation_verifier" and r.get("mitigation") == "none_regen"
                        and r["protocol"] == "voting"]
            ver, reg = _pq(ver_rows), _pq(reg_rows)
            stored_rows = sel(model, ds, "none", "1", "voting")
            stored = _pq(stored_rows)
            c = costs.get((model, ds), {})
            for ctrl_name, ctrl, ctrl_rows in (("regenerated final stage without the note", reg, reg_rows),
                                               ("stored vote of the same debates", stored, stored_rows)):
                res = _paired(ver, ctrl)
                if res is None:
                    continue
                qs = sorted(set(ver) & set(ctrl))
                # how often the verifier changed the final answer at all, and how often that changed its correctness
                a_v, a_c = _answers(ver_rows), _answers(ctrl_rows)
                changed = sum(1 for q in qs if a_v.get(q) != a_c.get(q))
                corr_changed = sum(1 for q in qs if ver[q] != ctrl[q])
                rows.append({"model": model, "dataset": ds, "mitigation": "verifier", "context": "false", "k": 1, "n": 3,
                             "protocol": "voting", "control": ctrl_name, "acc_mitigated": res["mean_a"], "acc_control": res["mean_b"],
                             "effect": res["diff"], "ci_lo": res["ci_lo"], "ci_hi": res["ci_hi"], "p_perm": res["p_perm"],
                             "n_questions": res["n"], "extra_generations_per_debate": c.get("verifier_calls_per_debate"),
                             "extra_completion_tokens_per_debate": c.get("verifier_completion_tokens_per_debate"),
                             "n_answer_changed": changed, "share_answer_changed": changed / len(qs),
                             "n_correctness_changed": corr_changed, "share_correctness_changed": corr_changed / len(qs)})
    write_csv(RESULTS_DIR / "mitigation_effects.csv", rows)

    # ---- LaTeX 1: the two mitigations under false context (the table of the draft, now with ETHICS)
    control_short = {CONTROL_PROMPT: "no prompt", "regenerated final stage without the note": "regenerated",
                     "stored vote of the same debates": "stored vote"}
    main_rows = [r for r in rows if r["context"] == "false"]
    tex = [[r["model"], DATASET_DISPLAY.get(r["dataset"], r["dataset"]),
            "skepticism" if r["mitigation"] == "source_skepticism" else "verifier",
            f"{r['k']}/{r['n']}", r["protocol"], control_short.get(r["control"], r["control"]),
            f"{r['acc_mitigated']:.3f}", f"{r['acc_control']:.3f}",
            f"{r['effect']:+.3f} [{r['ci_lo']:+.3f}, {r['ci_hi']:+.3f}]", f"{r['p_perm']:.4f}", r["n_questions"]] for r in main_rows]
    write_latex_table(TABLES_DIR / "mitigation_effects.tex",
                      ["Model", "Dataset", "Mitigation", "Exposure", "Protocol", "Control", "Acc. mitig.", "Acc. control",
                       "Effect [95\\% CI]", "p", "n"], tex,
                      caption="Mitigations under false context (false\\_fact passages, MINT-verified questions: "
                              + verified_counts_text(args.category) + "). Source skepticism: every agent's task "
                              "instruction states that extra information may be unreliable (no extra generation); control = the same "
                              "debates without the prompt (seed 0). Verifier agent: "
                              "before the final answer and vote, a verifier lists claims of the discussion that the task input does "
                              "not support (one extra generation per debate); control = the final stage regenerated without the "
                              "verifier note, or the stored vote of the same debates. Paired bootstrap CIs and permutation p-values "
                              "over questions. The effect of the skepticism prompt under relevant-true and no context is in "
                              "Table~\\ref{tab:mitigation_context}.",
                      label="tab:mitigation_effects", col_spec="llllllrrlrr")

    # ---- LaTeX 2: the prompt's effect by context (false / true / none) and the net
    def _find(model, ds, ctx, k, proto):
        return next((r for r in rows if r["model"] == model and r["dataset"] == ds and r["mitigation"] == "source_skepticism"
                     and r["context"] == ctx and r["k"] == k and r["protocol"] == proto), None)

    def _cell(r):
        if r is None:
            return "--"
        mark = "$^{\\dagger}$" if r["p_perm"] < 0.05 else ""
        return f"{r['effect']:+.2f} [{r['ci_lo']:+.2f}, {r['ci_hi']:+.2f}]{mark}"

    ctx_rows = []
    for model in MODEL_ORDER:
        for ds in DATASET_ORDER:
            for proto in ("voting", "consensus"):
                none0 = _find(model, ds, "none", 0, proto)
                for k in (1, 3):
                    f, t, net = _find(model, ds, "false", k, proto), _find(model, ds, "true", k, proto), _find(model, ds, "net", k, proto)
                    if f is None and t is None:
                        continue
                    n_q = (net or f or t)["n_questions"]
                    ctx_rows.append([model, DATASET_DISPLAY.get(ds, ds), proto, f"{k}/3", _cell(f), _cell(t),
                                     _cell(none0) if k == 1 else "", _cell(net), n_q])
    if ctx_rows:
        write_latex_table(TABLES_DIR / "mitigation_context.tex",
                          ["Model", "Dataset", "Protocol", "Exposure", "$\\Delta$ false context", "$\\Delta$ true context",
                           "$\\Delta$ no context (0/3)", "Net (false + true)", "n"], ctx_rows,
                          caption="Effect of the source-skepticism instruction by context: change in debate accuracy against the same "
                                  "debates without the instruction (seed 0, MINT-verified questions: " + verified_counts_text(args.category)
                                  + "), when the $k$ informed agents read the false passage, when they read the relevant-true passage "
                                  "of the same question, and in uninformed debates (0/3; one value per model and dataset). Net = "
                                  "per-question gain under false context plus change under true context at the same exposure, "
                                  "i.e. the benefit of the instruction when the group cannot tell whether the context is true. Paired "
                                  "bootstrap 95\\% CIs; $^{\\dagger}$: permutation $p < 0.05$. n = questions of the net column; the "
                                  "WinoGrande relevant-true and no-context runs used a 182-question verified list of which 148 lie in "
                                  "the 183-question set, so those cells and the net rest on 148 questions.",
                          label="tab:mitigation_context", col_spec="llllllllr", font="footnotesize")
    print(f"wrote {len(rows)} mitigation rows ({len(main_rows)} false-context rows, {len(ctx_rows)} context-table rows)")

    if args.no_figure or not rows:
        return
    try:
        from scripts.figure_common import (MODEL_COLOR, dataset_label, finish_with_legend, layout_style, model_label,
                                           panel_axes, panel_title, save_figure, shared_labels, style_axes)

        datasets = [d for d in DATASET_ORDER if any(r["dataset"] == d for r in rows)]
        models = [m for m in MODEL_ORDER if any(r["model"] == m for r in rows)]

        def draw_point(ax, x, r, m):
            """Dot and whisker: filled when the 95% CI excludes zero, open otherwise; no caps."""
            col = MODEL_COLOR.get(m, "#999999")
            ms, elw = 3.6, 0.9
            sig = r["ci_lo"] > 0 or r["ci_hi"] < 0
            ax.errorbar(x, r["effect"], yerr=[[r["effect"] - r["ci_lo"]], [r["ci_hi"] - r["effect"]]], fmt="none",
                        ecolor=col, elinewidth=elw, capsize=0, zorder=2)
            ax.plot(x, r["effect"], marker="o", linestyle="none", markersize=ms, markeredgewidth=elw * 0.9,
                    markeredgecolor=col, markerfacecolor=col if sig else "white", zorder=3)

        def grouped_figure(stem: str, groups: List[tuple], find, ylabel: str, height: float = 2.1, shade=()):
            """One panel per dataset, the groups side by side (three 1-inch panels at column width); dot-and-whisker
            per model, filled when the 95% CI excludes zero. groups: (group key, x tick label); shade: (x0, x1) spans
            drawn as a light grey band."""
            L = layout_style()
            fs = L["fs"]
            vals = [find(m, ds, g[0]) for m in models for ds in datasets for g in groups]
            vals = [r for r in vals if r]
            if not vals:
                return
            y_lo, y_hi = min(r["ci_lo"] for r in vals), max(r["ci_hi"] for r in vals)
            fig, axs = panel_axes(len(datasets), height=height)
            for ax, ds in zip(axs, datasets):
                for gi, g in enumerate(groups):
                    for mi, m in enumerate(models):
                        r = find(m, ds, g[0])
                        if r:
                            draw_point(ax, gi + (mi - (len(models) - 1) / 2) * 0.24, r, m)
                ax.axhline(0, color="#2e2e2e", linewidth=0.7, linestyle="--", zorder=1)
                for x0, x1 in shade:
                    ax.axvspan(x0, x1, color="#ececec", linewidth=0, zorder=0)
                ax.set_xlim(-0.5, len(groups) - 0.5)
                ax.set_xticks(range(len(groups)))
                ax.set_xticklabels([g[1] for g in groups], fontsize=fs["tick"])
                panel_title(ax, dataset_label(ds), fs)
                style_axes(ax, "y", fs)
            for ax in axs:
                ax.set_ylim(min(y_lo, 0.0) - 0.05, y_hi + 0.05)
            shared_labels(axs, "", ylabel, fs)
            from matplotlib.lines import Line2D
            handles = [Line2D([], [], marker="o", linestyle="none", markersize=3.6, color=MODEL_COLOR.get(m, "#999999"),
                              label=model_label(m)) for m in models]
            finish_with_legend(fig, handles, [h.get_label() for h in handles], fs,
                               ncol=min(len(models), L["legend_ncol_max"]), top=0.90)
            out = FIGURES_DIR / stem
            save_figure(fig, out)
            print(f"figure -> {out}.pdf")

        # paper Figure 6: the skepticism prompt by context (no / relevant-true / false passage) next to the
        # verifier under false context, all under voting, so the "costs nothing when the context is true" claim is on
        # the same figure as the gain under false context
        def find_main(m, ds, g):
            mit, ctx, k = g
            return next((x for x in rows if x["model"] == m and x["dataset"] == ds and x["mitigation"] == mit and x["k"] == k
                         and x["protocol"] == "voting" and x["context"] == ctx
                         and (mit != "verifier" or x["control"].startswith("regenerated"))), None)

        grouped_figure("fig_mitigation",
                       [(("source_skepticism", "none", 0), "0\n"),
                        (("source_skepticism", "true", 1), "T\n1"),
                        (("source_skepticism", "true", 3), "T\n3"),
                        (("source_skepticism", "false", 1), "F\n1"),
                        (("source_skepticism", "false", 3), "F\n3"),
                        (("verifier", "false", 1), "V\n1")],
                       find_main, r"$\Delta$ Accuracy vs. control", height=2.2, shade=((4.5, 5.5),))

    except Exception as e:  # noqa: BLE001
        print(f"[WARN] figure skipped: {e}")


if __name__ == "__main__":
    main()
