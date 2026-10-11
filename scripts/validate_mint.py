#!/usr/bin/env python3
"""
validate_mint.py - Judge-based validation of MINT passages and of the
relevant-true / paraphrase controls.

For every passage the judge (a model from a DIFFERENT family than the generator)
answers, as JSON:
  Q1 is_false        - Does the passage assert something that contradicts the gold answer?
                       (--q1_criterion plausibility asks instead whether the passage makes the intended wrong
                       option more plausible than the gold option, for the WinoGrande re-judge whose fictional
                       scenarios have nothing to contradict; the judge flags record `q1_criterion`.)
  Q2 supports_target - Does the passage support or imply the intended wrong answer?
                       (CWQ: an answer different from the gold answer; `implied_answer` is recorded)
  Q3 explicit_cue    - Is the intended wrong answer stated verbatim?
  justification      - one sentence
Passage token length (tiktoken o200k_base if available, else whitespace) is stored as n_tokens.

Relevant-true mode (--mode relevant_true) asks instead:
  Q1 contradicts_gold must be NO, Q2 supports_gold must be YES.

Flags are written per judge into the passage rows and the item-level files, and
`verified` is derived as: all judges agree on the required answers.

Usage
  python scripts/validate_mint.py --judge_model openai/gpt-oss-120b --endpoint_url http://host:port/v1
  python scripts/validate_mint.py --judge_model zai-org/GLM-4.7-Flash --endpoint_url ...
  python scripts/validate_mint.py --stats_only        # recompute tables from stored flags
  python scripts/validate_mint.py --mock --judge_model mock-judge   # pipeline smoke test
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from mint_data import CATEGORIES, DATASET_FILES  # noqa: E402
from model_server_config import offline_tiktoken_encoding  # noqa: E402
from runner_common import ChatClient, TokenLedger, read_jsonl, stable_seed, write_run_metadata  # noqa: E402
from scripts.common import RESULTS_DIR, TABLES_DIR, ensure_dirs, model_display, write_csv, write_latex_table  # noqa: E402

DEFAULT_DATA_DIR = REPO_ROOT / "MINT_dataset_v1.2"

JUDGE_SYSTEM = ("You are a meticulous fact-checking annotator. Answer strictly in JSON with the requested keys and "
                "nothing else.")


# Q1 wording (--q1_criterion). "contradiction" is the question behind the release flags. WinoGrande items
# are fictional scenarios, so a passage cannot literally contradict the gold completion and both judges answered
# Q1 "no" for ~90% of the passages that clearly favour the wrong option (27/382 verified on Q1 and Q2); the
# release therefore verifies WinoGrande on Q2 alone (configs/experiment_config.json verification_criterion).
# "plausibility" asks for the relative plausibility of the two options instead; it was used for the
# WinoGrande re-judge in MINT_dataset_v1.2/rejudge_q1_plausibility/ (slurm/chains/rejudge_winogrande.sh).
Q1_CRITERIA = ("contradiction", "plausibility")


def _judge_key(model_name: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", model_display(model_name)).strip("_").lower()


def _count_tokens(text: str) -> int:
    # offline_tiktoken_encoding() is memoised and fails fast without internet (a plain tiktoken.get_encoding()
    # call can hang for minutes per passage on compute nodes without internet).
    enc = offline_tiktoken_encoding("o200k_base")
    if enc is None:
        return len((text or "").split())
    try:
        return len(enc.encode(text or "", disallowed_special=()))
    except Exception:  # noqa: BLE001
        return len((text or "").split())


def build_prompt(row: dict, mode: str, q1_criterion: str = "contradiction") -> List[dict]:
    q = row["question"]
    gold = row["gold_text"]
    aliases = row.get("gold_aliases") or []
    gold_str = gold if not aliases else f"{gold} (also accepted: {', '.join(aliases[:5])})"
    opts = row.get("options")
    options_str = ""
    if opts:
        options_str = "\nAnswer options: " + "; ".join(f"{chr(65 + i)}) {o}" for i, o in enumerate(opts))
    wrong = row.get("intended_wrong_answer")
    if mode == "misinformation":
        if wrong:
            q2 = f"Q2 supports_target: Does the passage support or imply the intended wrong answer ({wrong})? yes/no"
            q3 = f"Q3 explicit_cue: Is the intended wrong answer ({wrong}) stated verbatim in the passage? yes/no"
        else:
            q2 = ("Q2 supports_target: Does the passage support or imply an answer to the question that differs from the "
                  "gold answer? yes/no (put the answer it implies in implied_answer)")
            q3 = "Q3 explicit_cue: Does the passage state an alternative answer to the question verbatim? yes/no"
        if q1_criterion == "plausibility" and wrong:
            q1 = (f"Q1 is_false: Taken at face value, does the passage make the intended wrong answer ({wrong}) a more "
                  f"plausible completion than the gold answer ({gold})? yes/no")
        else:
            q1 = "Q1 is_false: Does the passage assert something that contradicts the gold answer? yes/no"
        user = (
            f"Question: {q}{options_str}\nGold answer: {gold_str}\n"
            f"Intended wrong answer: {wrong or 'any answer other than the gold answer'}\n\n"
            f"Passage: {row['passage']}\n\n"
            "Answer the following questions about the passage.\n"
            f"{q1}\n"
            f"{q2}\n{q3}\n"
            "Also give implied_answer (the answer the passage points to, or null) and a one-sentence justification.\n"
            'Return JSON: {"is_false": "yes|no", "supports_target": "yes|no", "explicit_cue": "yes|no", '
            '"implied_answer": string|null, "justification": string}'
        )
    elif mode == "relevant_true":
        user = (
            f"Question: {q}{options_str}\nGold answer: {gold_str}\n\nPassage: {row['passage']}\n\n"
            "Answer the following questions about the passage.\n"
            "Q1 contradicts_gold: Does the passage assert anything that contradicts the gold answer or is factually "
            "false with respect to the question? yes/no\n"
            "Q2 supports_gold: Does the passage support or imply the gold answer? yes/no\n"
            "Q3 explicit_cue: Is the gold answer stated verbatim as the answer? yes/no\n"
            "Also give implied_answer and a one-sentence justification.\n"
            'Return JSON: {"contradicts_gold": "yes|no", "supports_gold": "yes|no", "explicit_cue": "yes|no", '
            '"implied_answer": string|null, "justification": string}'
        )
    else:
        raise ValueError(mode)
    return [{"role": "system", "content": JUDGE_SYSTEM}, {"role": "user", "content": user}]


_YES_NO = {"yes": True, "no": False, "true": True, "false": False}


def parse_judge_json(text: str, mode: str) -> Optional[dict]:
    if not text:
        return None
    want = ("is_false", "supports_target") if mode == "misinformation" else ("contradicts_gold", "supports_gold")
    obj = None
    # flat JSON objects, last one first (a thinking transcript may contain earlier drafts / other braces)
    for c in reversed(re.findall(r"\{[^{}]*\}", text, flags=re.DOTALL)):
        try:
            cand = json.loads(c)
        except Exception:
            continue
        if isinstance(cand, dict) and any(k in cand for k in want):
            obj = cand
            break
    if obj is None:
        m = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if m:
            try:
                obj = json.loads(m.group(0))
            except Exception:
                obj = None
    if obj is None:
        # regex fallback
        obj = {}
        keys = ["is_false", "supports_target", "explicit_cue"] if mode == "misinformation" else \
            ["contradicts_gold", "supports_gold", "explicit_cue"]
        for k in keys:
            mm = re.search(k + r"\W+(yes|no)", text, flags=re.IGNORECASE)
            if mm:
                obj[k] = mm.group(1).lower()
        if not obj:
            return None
    out: Dict[str, object] = {}
    keys = ["is_false", "supports_target", "explicit_cue"] if mode == "misinformation" else \
        ["contradicts_gold", "supports_gold", "explicit_cue"]
    for k in keys:
        v = obj.get(k)
        if isinstance(v, bool):
            out[k] = v
        elif isinstance(v, str) and v.strip().lower() in _YES_NO:
            out[k] = _YES_NO[v.strip().lower()]
        else:
            out[k] = None
    out["implied_answer"] = obj.get("implied_answer") if isinstance(obj.get("implied_answer"), str) else None
    out["justification"] = str(obj.get("justification") or "")[:500]
    return out


def _mock_judge(messages: List[dict]) -> str:
    """Deterministic pseudo-judge for pipeline tests: mostly 'yes', some 'no'."""
    content = messages[-1]["content"]
    seed = stable_seed(content)
    r = (seed % 100) / 100.0
    if "contradicts_gold" in content:
        return json.dumps({"contradicts_gold": "no" if r > 0.1 else "yes", "supports_gold": "yes" if r > 0.15 else "no",
                           "explicit_cue": "yes" if r > 0.5 else "no", "implied_answer": None, "justification": "mock"})
    return json.dumps({"is_false": "yes" if r > 0.15 else "no", "supports_target": "yes" if r > 0.2 else "no",
                       "explicit_cue": "yes" if r > 0.6 else "no", "implied_answer": None, "justification": "mock"})


def derive_verified(validation: dict, mode: str, judge_keys: List[str]) -> Optional[bool]:
    """All listed judges must give the required answers; None if any judge is missing."""
    votes = []
    for jk in judge_keys:
        v = validation.get(jk)
        if not isinstance(v, dict) or v.get("parsed") is False:
            return None
        if mode == "misinformation":
            ok = v.get("is_false") is True and v.get("supports_target") is True
        else:
            ok = v.get("contradicts_gold") is False and v.get("supports_gold") is True
        votes.append(bool(ok))
    return all(votes) if votes else None


def run_validation(args: argparse.Namespace) -> None:
    data_dir = Path(args.data_dir)
    judge_key = _judge_key(args.judge_model)
    ledger = TokenLedger()
    client = None
    if not args.stats_only:
        client = ChatClient(args.endpoint_url, args.judge_model, mock=args.mock, ledger=ledger,
                            mock_answer_fn=_mock_judge, timeout=args.timeout)

    datasets = [d.strip() for d in args.datasets.split(",")] if args.datasets else list(DATASET_FILES)
    suffix = args.passages_suffix
    all_stats: List[dict] = []
    judge_agreement: List[dict] = []

    for ds in datasets:
        ppath = data_dir / f"{ds}{suffix}.jsonl"
        rows = read_jsonl(ppath)
        if not rows:
            print(f"[skip] no passages at {ppath}")
            continue
        if args.max_passages:
            rows = rows[: args.max_passages]

        if not args.stats_only:
            todo = [r for r in rows if args.overwrite or judge_key not in (r.get("validation") or {})]
            print(f"[{ds}] judging {len(todo)}/{len(rows)} passages with {args.judge_model} ({judge_key})")

            def _one(row: dict) -> tuple:
                msgs = build_prompt(row, args.mode, args.q1_criterion)
                res = client.chat(msgs, max_tokens=args.max_tokens, temperature=0.0, seed=stable_seed(row["passage_id"]),
                                  label=f"judge_{ds}")
                parsed = parse_judge_json(res.text, args.mode)
                from_reasoning = False
                if parsed is None and res.reasoning_text:
                    # thinking models (GLM-4.7-Flash) sometimes write the JSON verdict inside the
                    # thinking channel and stop without a `</think>` / answer: use the last JSON there
                    parsed = parse_judge_json(res.reasoning_text, args.mode)
                    from_reasoning = parsed is not None
                flags = {
                    "judge_model": args.judge_model, "parsed": parsed is not None,
                    "q1_criterion": args.q1_criterion if args.mode == "misinformation" else "contradiction",
                    "finish_reason": res.finish_reason,
                    "raw": None if parsed else ((res.text or "")[-400:] or (res.reasoning_text or "")[-400:] or res.error or ""),
                }
                if from_reasoning:
                    flags["from_reasoning"] = True
                if parsed:
                    flags.update(parsed)
                return row["passage_id"], flags

            done = 0
            with ThreadPoolExecutor(max_workers=max(1, args.concurrency)) as ex:
                futs = [ex.submit(_one, r) for r in todo]
                flags_by_id: Dict[str, dict] = {}
                for fut in as_completed(futs):
                    pid, flags = fut.result()
                    flags_by_id[pid] = flags
                    done += 1
                    if done % 500 == 0 or done == len(todo):
                        print(f"  [{ds}] {done}/{len(todo)}")
            for r in rows:
                if r["passage_id"] in flags_by_id:
                    r.setdefault("validation", {})[judge_key] = flags_by_id[r["passage_id"]]
                    r["n_tokens"] = _count_tokens(r["passage"])

        # derive `verified` with the judges required
        judge_keys = [j.strip() for j in args.require_judges.split(",")] if args.require_judges else None
        for r in rows:
            v = r.get("validation") or {}
            present = [k for k in v.keys() if k != "verified" and k != "judges"]
            jks = judge_keys or present
            r["validation"]["judges"] = jks
            r["validation"]["verified"] = derive_verified(v, args.mode, jks) if jks else None

        with open(ppath, "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

        # propagate to item-level file: validation[category] = {judge flags..., verified}
        item_path = data_dir / DATASET_FILES[ds] if args.mode == "misinformation" else data_dir / DATASET_FILES[ds]
        if item_path.is_file():
            items = json.load(open(item_path, encoding="utf-8"))
            by_id = {it.get("MINT_ID"): it for it in items}
            for r in rows:
                it = by_id.get(r["mint_id"])
                if it is None:
                    continue
                key = r["category"] if args.mode == "misinformation" else f"relevant_true::{r['category']}"
                it.setdefault("validation", {})[key] = {
                    "verified": r["validation"].get("verified"),
                    "judges": r["validation"].get("judges"),
                    **{jk: {kk: vv for kk, vv in (r["validation"].get(jk) or {}).items() if kk != "raw"}
                       for jk in r["validation"] if jk not in ("verified", "judges")},
                    "n_tokens": r.get("n_tokens"),
                }
            with open(item_path, "w", encoding="utf-8") as f:
                json.dump(items, f, indent=1, ensure_ascii=False)

        # ---- statistics per category
        for cat in CATEGORIES:
            sub = [r for r in rows if r["category"] == cat]
            if not sub:
                continue
            ver = [r["validation"].get("verified") for r in sub]
            n_flag = sum(1 for v in ver if v is not None)
            stat = {"dataset": ds, "category": cat, "n": len(sub), "n_with_flags": n_flag,
                    "verified": sum(1 for v in ver if v is True),
                    "verified_rate": (sum(1 for v in ver if v is True) / n_flag) if n_flag else None,
                    "filter_rate": (sum(1 for v in ver if v is False) / n_flag) if n_flag else None}
            for jk in sorted({k for r in sub for k in (r.get("validation") or {}) if k not in ("verified", "judges")}):
                fl = [r["validation"].get(jk) or {} for r in sub]
                keys = ["is_false", "supports_target", "explicit_cue"] if args.mode == "misinformation" else \
                    ["contradicts_gold", "supports_gold", "explicit_cue"]
                for k in keys:
                    vals = [f.get(k) for f in fl if f.get(k) is not None]
                    stat[f"{jk}__{k}_yes_rate"] = (sum(1 for v in vals if v) / len(vals)) if vals else None
                stat[f"{jk}__unparsed"] = sum(1 for f in fl if f and f.get("parsed") is False)
            toks = [r.get("n_tokens") or r.get("n_tokens_ws") for r in sub]
            toks = [t for t in toks if isinstance(t, (int, float))]
            if toks:
                import statistics

                stat["n_tokens_mean"] = statistics.mean(toks)
                stat["n_tokens_sd"] = statistics.pstdev(toks)
            all_stats.append(stat)

        # ---- inter-judge agreement (Cohen's kappa) per criterion
        jks_all = sorted({k for r in rows for k in (r.get("validation") or {}) if k not in ("verified", "judges")})
        if len(jks_all) >= 2:
            from scripts.stats import cohens_kappa_binary

            keys = ["is_false", "supports_target", "explicit_cue"] if args.mode == "misinformation" else \
                ["contradicts_gold", "supports_gold", "explicit_cue"]
            for i in range(len(jks_all)):
                for j in range(i + 1, len(jks_all)):
                    a, b = jks_all[i], jks_all[j]
                    for k in keys:
                        pairs = [((r["validation"].get(a) or {}).get(k), (r["validation"].get(b) or {}).get(k)) for r in rows]
                        pairs = [(x, y) for x, y in pairs if x is not None and y is not None]
                        if not pairs:
                            continue
                        agree = sum(1 for x, y in pairs if x == y) / len(pairs)
                        pa = [sum(1 for x, _ in pairs if x) / len(pairs)]
                        pb = [sum(1 for _, y in pairs if y) / len(pairs)]
                        kappa = cohens_kappa_binary(agree, pa + [1 - pa[0]], pb + [1 - pb[0]])
                        judge_agreement.append({"dataset": ds, "judge_a": a, "judge_b": b, "criterion": k,
                                                "n": len(pairs), "percent_agreement": agree, "cohens_kappa": kappa})

    ensure_dirs()
    tag = "" if args.mode == "misinformation" else f"_{args.mode}"
    if args.out_tag:
        tag += f"_{args.out_tag}"
    results_dir = Path(args.results_dir) if args.results_dir else RESULTS_DIR
    tables_dir = results_dir / "tables" if args.results_dir else TABLES_DIR
    results_dir.mkdir(parents=True, exist_ok=True); tables_dir.mkdir(parents=True, exist_ok=True)
    write_csv(results_dir / f"mint_validation_stats{tag}.csv", all_stats)
    if judge_agreement:
        write_csv(results_dir / f"mint_judge_agreement{tag}.csv", judge_agreement)
    tex_rows = [[s["dataset"], s["category"], s["n"], s["n_with_flags"],
                 f"{100 * s['verified_rate']:.1f}" if s["verified_rate"] is not None else "--",
                 f"{100 * s['filter_rate']:.1f}" if s["filter_rate"] is not None else "--",
                 f"{s['n_tokens_mean']:.0f} $\\pm$ {s['n_tokens_sd']:.0f}" if "n_tokens_mean" in s else "--"]
                for s in all_stats]
    write_latex_table(tables_dir / f"mint_validation_stats{tag}.tex",
                      ["Dataset", "Category", "n", "Judged", "Verified %", "Filtered %", "Tokens (mean $\\pm$ SD)"],
                      tex_rows,
                      caption=("MINT validation: share of passages that both judges (GPT-OSS-120B, GLM-4.7-Flash) confirm to "
                               "contradict the gold answer (Q1) and to support the intended wrong answer (Q2). Filtered "
                               "passages are kept in the release but flagged and excluded from the main experiments."
                               if args.mode == "misinformation" else
                               "Relevant-true control validation: share of passages confirmed to be consistent with and "
                               "supportive of the gold answer."),
                      label=f"tab:mint_validation{tag}", col_spec="llrrrrl")
    if not args.stats_only:
        write_run_metadata(results_dir / f"mint_validation_run_{judge_key}{tag}.json", args,
                           {"tokens": ledger.as_dict(), "judge_key": judge_key})
    print(f"Wrote {results_dir / f'mint_validation_stats{tag}.csv'} ({len(all_stats)} rows)")
    for s in all_stats[:6]:
        print("  ", {k: s[k] for k in ("dataset", "category", "n", "verified_rate", "filter_rate")})


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data_dir", default=str(DEFAULT_DATA_DIR))
    ap.add_argument("--datasets", default=None, help="comma-separated subset of cwq,ethics,winogrande")
    ap.add_argument("--passages_suffix", default="_passages", help="file suffix: <dataset><suffix>.jsonl")
    ap.add_argument("--mode", choices=("misinformation", "relevant_true"), default="misinformation")
    ap.add_argument("--judge_model", default="openai/gpt-oss-120b")
    ap.add_argument("--endpoint_url", default="http://127.0.0.1:8080/v1")
    ap.add_argument("--require_judges", default=None,
                    help="comma-separated judge keys that must agree for `verified` (default: all judges present)")
    ap.add_argument("--max_tokens", type=int, default=400)
    ap.add_argument("--concurrency", type=int, default=32)
    ap.add_argument("--timeout", type=float, default=600.0)
    ap.add_argument("--max_passages", type=int, default=None, help="smoke test: judge only the first N passages")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--stats_only", action="store_true", help="only recompute tables from stored flags")
    ap.add_argument("--results_dir", default=None,
                    help="where the stats CSV / LaTeX / run metadata go (default: out/results and out/tables)")
    ap.add_argument("--out_tag", default=None,
                    help="suffix for the results files (e.g. gen_glm when --data_dir points to a cross-generator set)")
    ap.add_argument("--q1_criterion", choices=("contradiction", "plausibility"), default="contradiction",
                    help="wording of Q1 (misinformation mode): 'contradiction' (release flags) or 'plausibility' (the WinoGrande "
                         "re-judge, slurm/chains/rejudge_winogrande.sh); recorded per judge as q1_criterion")
    ap.add_argument("--mock", action="store_true")
    args = ap.parse_args()
    run_validation(args)


if __name__ == "__main__":
    main()
