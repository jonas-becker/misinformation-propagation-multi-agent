#!/usr/bin/env python3
"""
exp1.py - Single-agent experiment with (mis)information context.

Overview:
  * gold answers and passages come from mint_data;
  * answers are extracted with answer_extraction (status parsed/ambiguous/unparsed is
    stored for every record; only unambiguous answers can be correct);
  * conditions cover the 2x2 (+filler) design:
        baseline                 no extra information
        false_fact / strategy_<s> relevant FALSE passage (MINT category)
        irrelevant_misinformed   FALSE passage of another question (irrelevant-false)
        irrelevant_true          length-matched true passage about another topic (MINT v1.1)
        relevant_true            true passage supporting the gold answer (controls/relevant_true) - also relevant_true_strategy_<s>
        filler                   content-free length-matched text (scripts/generate_filler.py)
  * --seed (per-item seeds), --temperature/--top_p/--max_tokens are explicit and stored;
    --reprompt_unparsed issues a constrained follow-up ('Answer with exactly one of: A / B')
    when the first answer cannot be parsed (the original answer is kept in the record);
  * --data_dir selects cross-generator or control passage sets, --mint_filter
    verified restricts to MINT-verified passages;
  * prompt/completion tokens and finish_reason are recorded;
  * the HF path applies the chat template.

Outputs (consumed by scripts/rescore_results.py): <out_dir>/exp1_results_<dataset>_<model>.json with
{condition: {accuracy, correct_count, total_count, results: [...], config: {...}}}.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

from answer_extraction import (
    STATUS_PARSED,
    extract_free_form_answer,
    extract_mc_answer,
    free_form_matches,
    letter_for_index,
)
from mint_data import STRATEGIES, canonical_dataset, filter_items_for_condition, list_categories, load_mint_items
from model_server_config import mallm_concurrency_default, normalize_model_name
from runner_common import ChatClient, TokenLedger, load_experiment_config, stable_seed, write_run_metadata
from shared_utils import (
    PROMPT_BASELINE,
    PROMPT_BASELINE_FREE_FORM,
    PROMPT_MISINFORMED,
    PROMPT_MISINFORMED_FREE_FORM,
    build_common_arg_parser,
    format_options,
    model_name_to_safe_path,
)

REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_CONDITIONS = ["baseline", "false_fact"] + [f"strategy_{s}" for s in STRATEGIES] + ["irrelevant_misinformed"]
CONTROL_CONDITIONS = ["irrelevant_true", "relevant_true", "filler"]
REPROMPT_MC = "Your previous answer could not be parsed. Answer with exactly one of: {letters}. Output only the letter."
REPROMPT_FF = "Your previous answer could not be parsed. State your final answer after the text 'Final Answer:'."


# ---------------------------------------------------------------------------
# Condition -> information text
# ---------------------------------------------------------------------------

def information_for(item: dict, condition: str, items: List[dict], seed: int) -> tuple:
    """Return (information_text or None, category_label, source_mint_id)."""
    if condition == "baseline":
        return None, "none", None
    if condition == "false_fact":
        return item["passages"].get("false_fact"), "false_fact", item["mint_id"]
    if condition.startswith("strategy_"):
        s = condition[len("strategy_"):]
        return item["passages"].get(s), s, item["mint_id"]
    if condition == "irrelevant_misinformed":
        # false fact of another (deterministically chosen) question: irrelevant-false control
        rng = random.Random(stable_seed(seed, item["mint_id"], "irrelevant"))
        others = [it for it in items if it["mint_id"] != item["mint_id"] and it["passages"].get("false_fact")]
        if not others:
            return None, "irrelevant_false_fact", None
        o = rng.choice(others)
        return o["passages"]["false_fact"], "irrelevant_false_fact", o["mint_id"]
    if condition == "irrelevant_true":
        return item.get("irrelevant_true_information"), "irrelevant_true_information", item["mint_id"]
    if condition == "relevant_true":
        by = (item.get("raw") or {}).get("relevant_true_by_strategy") or {}
        return by.get("false_fact") or item.get("relevant_true_information"), "relevant_true", item["mint_id"]
    if condition.startswith("relevant_true_strategy_"):
        s = condition[len("relevant_true_strategy_"):]
        by = (item.get("raw") or {}).get("relevant_true_by_strategy") or {}
        return by.get(s), f"relevant_true_{s}", item["mint_id"]
    if condition == "filler":
        return item.get("filler_information"), "filler", item["mint_id"]
    raise ValueError(f"Unknown condition {condition}")


def build_prompt(item: dict, information: Optional[str]) -> str:
    if item["is_multiple_choice"]:
        options_text = format_options(item["options"])
        if information:
            return PROMPT_MISINFORMED.format(sentence=item["sentence"], options_text=options_text, misinformation=information)
        return PROMPT_BASELINE.format(sentence=item["sentence"], options_text=options_text)
    if information:
        return PROMPT_MISINFORMED_FREE_FORM.format(sentence=item["sentence"], misinformation=information)
    return PROMPT_BASELINE_FREE_FORM.format(sentence=item["sentence"])


def score(item: dict, response: str) -> dict:
    if item["is_multiple_choice"]:
        ex = extract_mc_answer(response, item["options"])
        return {"predicted_answer": ex.text, "predicted_index": ex.index, "extraction_status": ex.status,
                "extraction_method": ex.method, "is_correct": bool(ex.status == STATUS_PARSED and ex.index == item["gold_index"]),
                "is_correct_best_guess": bool(ex.index is not None and ex.index == item["gold_index"])}
    ex = extract_free_form_answer(response)
    golds = [item["gold_text"]] + list(item.get("gold_aliases") or [])
    lenient = free_form_matches(response, golds) if ex.status == STATUS_PARSED else False
    strict = free_form_matches(ex.text, golds) if ex.method == "marker" else None
    return {"predicted_answer": (ex.text or "")[:500], "predicted_index": None, "extraction_status": ex.status,
            "extraction_method": ex.method, "is_correct": bool(strict) if strict is not None else bool(lenient),
            "is_correct_lenient": bool(lenient), "is_correct_strict": strict, "is_correct_best_guess": bool(lenient)}


# ---------------------------------------------------------------------------
# Inference back-ends
# ---------------------------------------------------------------------------

def _dummy_answer(messages: List[dict]) -> str:
    content = messages[-1]["content"]
    if "Answer Choices" in content:
        return random.choice(["The correct answer is A).", "The answer is B) option.", "I think A"])
    return "Final Answer: Sweden"


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def run_condition(items: List[dict], all_items: List[dict], condition: str, dataset: str, args: argparse.Namespace,
                  client: Optional[ChatClient], ledger: TokenLedger) -> dict:
    jobs = []
    skipped = 0
    for it in items:
        info, category, src = information_for(it, condition, all_items, args.seed)
        if condition != "baseline" and not (isinstance(info, str) and info.strip()):
            skipped += 1
            continue
        jobs.append((it, info, category, src, build_prompt(it, info)))
    if skipped:
        print(f"  [{dataset}/{condition}] skipped {skipped} items without the required passage")

    def _finish(it, info, category, src, prompt, gen: dict, reprompt: Optional[dict]) -> dict:
        sc = score(it, gen["text"])
        rec = {
            "sentence": it["sentence"], "mint_id": it["mint_id"], "prompt": prompt, "response": gen["text"],
            "finish_reason": gen.get("finish_reason"), "prompt_tokens": gen.get("prompt_tokens"),
            "completion_tokens": gen.get("completion_tokens"), "information": info, "information_source_mint_id": src,
            "correct_answer": it["gold_text"], "gold_index": it["gold_index"], "is_multiple_choice": it["is_multiple_choice"],
            "options": it["options"] if it["is_multiple_choice"] else [it["gold_text"]] + list(it.get("gold_aliases") or []),
            "misinformation_strategy": category, "condition": condition, "seed": args.seed, **sc,
            "reprompted": False, "reprompt_response": None,
        }
        if reprompt is not None:
            sc2 = score(it, reprompt["text"])
            rec["reprompted"] = True
            rec["reprompt_response"] = reprompt["text"]
            rec["extraction_status_first"] = rec["extraction_status"]
            rec.update({k: v for k, v in sc2.items()})
        return rec

    results: List[dict] = []
    if client is None and args.dummy_test:
        # dummy answers (pipeline test without a model)
        bs = max(1, args.batch_size)
        for i in range(0, len(jobs), bs):
            chunk = jobs[i:i + bs]
            gens = [{"text": _dummy_answer([{"role": "user", "content": j[4]}]), "finish_reason": "stop",
                     "prompt_tokens": 0, "completion_tokens": 0} for j in chunk]
            for j, g in zip(chunk, gens):
                results.append(_finish(*j, g, None))
    else:
        def _one(job):
            it, info, category, src, prompt = job
            res = client.chat([{"role": "user", "content": prompt}], max_tokens=args.max_tokens, temperature=args.temperature,
                              top_p=args.top_p if args.top_p is not None else 1.0,
                              seed=stable_seed(args.seed, it["mint_id"], condition), label=f"exp1_{dataset}_{condition}")
            gen = {"text": res.text, "finish_reason": res.finish_reason, "prompt_tokens": res.prompt_tokens,
                   "completion_tokens": res.completion_tokens, "error": res.error}
            rep = None
            if args.reprompt_unparsed and score(it, res.text)["extraction_status"] != STATUS_PARSED:
                if it["is_multiple_choice"]:
                    letters = " / ".join(letter_for_index(i) for i in range(len(it["options"])))
                    follow = REPROMPT_MC.format(letters=letters)
                else:
                    follow = REPROMPT_FF
                r2 = client.chat([{"role": "user", "content": prompt}, {"role": "assistant", "content": res.text or "(no answer)"},
                                  {"role": "user", "content": follow}], max_tokens=args.max_tokens, temperature=0.0,
                                 seed=stable_seed(args.seed, it["mint_id"], condition, "reprompt"), label=f"exp1_reprompt_{dataset}")
                rep = {"text": r2.text, "finish_reason": r2.finish_reason}
            return _finish(it, info, category, src, prompt, gen, rep)

        with ThreadPoolExecutor(max_workers=max(1, args.concurrent_requests)) as ex:
            futs = [ex.submit(_one, j) for j in jobs]
            done = 0
            for fut in as_completed(futs):
                results.append(fut.result())
                done += 1
                if done % 100 == 0 or done == len(futs):
                    print(f"  [{dataset}/{condition}] {done}/{len(futs)}")
    results.sort(key=lambda r: r["mint_id"])
    correct = sum(1 for r in results if r["is_correct"])
    statuses = {}
    for r in results:
        statuses[r["extraction_status"]] = statuses.get(r["extraction_status"], 0) + 1
    return {
        "accuracy": (correct / len(results)) if results else 0.0, "correct_count": correct, "total_count": len(results),
        "results": results,
        "config": {"dataset": dataset, "condition": condition, "model_name": args.model_name,
                   "temperature": args.temperature, "top_p": args.top_p, "max_tokens": args.max_tokens, "seed": args.seed,
                   "data_dir": args.data_dir, "mint_filter": args.mint_filter, "reprompt_unparsed": args.reprompt_unparsed,
                   "n_skipped_no_passage": skipped, "extraction_status_counts": statuses,
                   "tokens": ledger.as_dict().get(f"exp1_{dataset}_{condition}"),
                   "timestamp_utc": datetime.now(timezone.utc).isoformat()},
    }


def resolve_conditions(spec: str, items: List[dict]) -> List[str]:
    if spec == "default":
        conds = ["baseline", "false_fact"] + [f"strategy_{s}" for s in list_categories(items) if s != "false_fact"] + ["irrelevant_misinformed"]
    elif spec == "controls":
        conds = ["baseline"] + CONTROL_CONDITIONS
    elif spec == "all":
        conds = ["baseline", "false_fact"] + [f"strategy_{s}" for s in list_categories(items) if s != "false_fact"] + \
                ["irrelevant_misinformed"] + CONTROL_CONDITIONS
    else:
        conds = [c.strip() for c in spec.split(",") if c.strip()]
    return conds


def main() -> None:
    cfg = load_experiment_config()
    parser = build_common_arg_parser("Run single agent experiment with (mis)information context")
    parser.add_argument("--endpoint_url", default="http://127.0.0.1:8080/v1")
    parser.add_argument("--no_autostart", action="store_true")
    parser.add_argument("--concurrent_requests", type=int, default=None)
    parser.add_argument("--output_dir", default=None, help="default: out/<model>/ (seed 0) or out/<model>/seeds/seed<k>/")
    parser.add_argument("--conditions", default="default", help="'default' | 'controls' | 'all' | comma list")
    parser.add_argument("--data_dir", default=None, help="passage set (MINT_dataset_v1.2 default; controls/generators via this flag)")
    parser.add_argument("--mint_filter", choices=("all", "verified"), default="all")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=float(cfg["single_agent"]["temperature"]))
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--reprompt_unparsed", action="store_true")
    args = parser.parse_args()
    args.model_name = normalize_model_name(args.model_name)
    if args.max_tokens == 512:  # parser default -> model-aware default
        low = args.model_name.lower()
        args.max_tokens = int(cfg["single_agent"]["max_tokens_reasoning_models"]) if ("glm" in low or "gpt-oss" in low) else 512
    args.concurrent_requests = args.concurrent_requests or mallm_concurrency_default(args.model_name)

    print(f"=== exp1 (single agent) model={args.model_name} temperature={args.temperature} "
          f"max_tokens={args.max_tokens} seed={args.seed} conditions={args.conditions} data_dir={args.data_dir}")

    ledger = TokenLedger()
    client: Optional[ChatClient] = None
    if not args.dummy_test:
        from runner_common import resolve_endpoint

        endpoint = resolve_endpoint(args.endpoint_url, args.model_name, args.no_autostart)
        client = ChatClient(endpoint, args.model_name, ledger=ledger, timeout=float(os.environ.get("MALLM_OPENAI_TIMEOUT_SECONDS", "600")))

    model_suffix = "dummy_model" if args.dummy_test else model_name_to_safe_path(args.model_name)
    if args.output_dir:
        out_dir = Path(args.output_dir)
    elif args.seed == 0:
        out_dir = REPO_ROOT / "out" / model_suffix
    else:
        out_dir = REPO_ROOT / "out" / model_suffix / "seeds" / f"seed{args.seed}"
    out_dir.mkdir(parents=True, exist_ok=True)

    datasets = [canonical_dataset(d) for d in (args.datasets.split(",") if args.datasets else ["winogrande", "ethics", "cwq"])]
    file_names = {"cwq": "complex_web_questions", "ethics": "ethics", "winogrande": "winogrande"}
    for ds in datasets:
        items = load_mint_items(ds, args.data_dir, max_items=args.max_samples)
        conds = resolve_conditions(args.conditions, items)
        out_path = out_dir / f"exp1_results_{file_names[ds]}_{model_suffix}.json"
        existing = {}
        if out_path.is_file():
            try:
                existing = json.load(open(out_path, encoding="utf-8"))
            except Exception:
                existing = {}
        for cond in conds:
            if getattr(args, "continue") and cond in existing and existing[cond].get("total_count", 0) >= len(items) - 5:
                print(f"[CONTINUE] {ds}/{cond} present, skipping")
                continue
            cat = cond[len("strategy_"):] if cond.startswith("strategy_") else ("false_fact" if cond == "false_fact" else None)
            cond_items = filter_items_for_condition(items, cat, args.mint_filter) if cat else items
            t0 = time.time()
            res = run_condition(cond_items, items, cond, ds, args, client, ledger)
            existing[cond] = res
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(existing, f, indent=1, ensure_ascii=False)
            print(f"[{ds}] {cond}: acc={res['accuracy']:.3f} ({res['correct_count']}/{res['total_count']}) "
                  f"status={res['config']['extraction_status_counts']} in {time.time() - t0:.0f}s -> {out_path.name}")
    write_run_metadata(out_dir / f"run_metadata_exp1_seed{args.seed}_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')}.json",
                       args, {"tokens": ledger.as_dict()})


if __name__ == "__main__":
    main()
