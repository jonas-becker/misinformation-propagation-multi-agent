#!/usr/bin/env python3
"""
exp_exposure.py - Exposure-matched multi-agent debate (also used for the debates with
control passages and for the source-skepticism mitigation).

Runs MALLM debates with n agents of which exactly k receive a passage, for every
k in --k, every dataset and every category requested. Compared to exp2.py:
  * k in {0,...,n} instead of a fixed 1-of-3;
  * informed slots are shuffled per question with a fixed --seed (exp2 always
    made Participant 3 the misinformed agent);
  * --passage_sharing shared|independent (same passage for all k agents or
    different category passages of the same question);
  * --information_source misinformation|relevant_true|irrelevant_true|filler;
  * --mitigation none|source_skepticism (prompt-level mitigation);
  * --mint_filter verified restricts to MINT-verified passages;
  * outputs use the MINT id as example id, store the full transcript, and are
    re-scored with the unified extractor (status per answer); run metadata is
    written next to the results.

Outputs: <out_dir>/exposure_results_<dataset>_seed<seed>.json  (exp2-compatible
schema keyed by exposure_common.condition_key) and <out_dir>/exposure_io/*.

Example (smoke test):  python exp_exposure.py --mock --datasets winogrande --k 0,3 --max_items 5
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import List

import exp2 as exp2_base
from answer_extraction import extract_free_form_answer, extract_mc_answer, free_form_matches
from exposure_common import (
    INFO_SOURCES,
    MITIGATIONS,
    assign_slots,
    condition_key,
    expand_categories,
    independent_passages,
    select_passage,
    task_instruction,
)
from mint_data import (canonical_dataset, filter_items_by_ids, filter_items_for_condition, format_mc_input, load_mint_id_subset,
                       load_mint_items)
from model_server_config import normalize_model_name
from runner_common import load_experiment_config, resolve_endpoint, write_run_metadata

REPO_ROOT = Path(__file__).resolve().parent


def default_max_tokens(model_name: str, cfg: dict) -> int:
    low = model_name.lower()
    if "glm" in low or "gpt-oss" in low:
        return int(cfg["debate"].get("max_tokens_reasoning_models", 4096))
    return int(cfg["debate"].get("max_tokens", 1024))


def build_mallm_input(items: List[dict], dataset: str, category: str, k: int, n: int, source: str, sharing: str,
                      seed: int) -> List[dict]:
    out = []
    for it in items:
        if sharing == "shared":
            passages = [select_passage(it, category, source)] * k
        else:
            passages = independent_passages(it, category, source, k, seed)
        if k > 0 and any(p is None or not str(p).strip() for p in passages):
            continue  # item lacks the requested passage/source -> skip (reported in config)
        slots = assign_slots(it, k, n, seed, passages)
        refs = [it["gold_text"]] + list(it.get("gold_aliases") or []) if not it["is_multiple_choice"] else [it["gold_text"]]
        out.append({
            "example_id": it["mint_id"],
            "dataset_id": dataset,
            "inputs": [format_mc_input(it) if it["is_multiple_choice"] else it["sentence"]],
            "context": None,
            "references": [r for r in refs if r],
            "metadata": {"mint_id": it["mint_id"], "category": category, "k": k, "n": n, "information_source": source,
                         "passage_sharing": sharing, "seed": seed,
                         "informed_slots": [i for i, s in enumerate(slots) if s is not None]},
            "informations": slots,
        })
    return out


def evaluate_output(output_path: str, items_by_id: dict, condition: str) -> dict:
    try:
        with open(output_path, "r", encoding="utf-8") as f:
            records = json.load(f)
    except Exception:
        records = []
    results = []
    correct = 0
    for rec in records:
        it = items_by_id.get(rec.get("exampleId"))
        final = rec.get("finalAnswer")
        if it is None:
            continue
        if it["is_multiple_choice"]:
            ex = extract_mc_answer(final, it["options"])
            is_correct = ex.status == "parsed" and ex.index == it["gold_index"]
            status = ex.status
            pred = ex.text
        else:
            ex = extract_free_form_answer(final)
            golds = [it["gold_text"]] + list(it.get("gold_aliases") or [])
            is_correct = bool(ex.status == "parsed" and free_form_matches(ex.text, golds))
            status = ex.status
            pred = ex.text
        correct += int(is_correct)
        results.append({
            "sentence": (rec.get("input") or [""])[0],
            "mint_id": it["mint_id"],
            "response": str(final) if final is not None else "",
            "predicted_answer": pred,
            "extraction_status": status,
            "correct_answer": it["gold_text"],
            "is_correct": bool(is_correct),
            "is_multiple_choice": it["is_multiple_choice"],
            "options": it["options"],
            "misinformation_strategy": condition,
            "mallm_log": rec,
        })
    return {"accuracy": (correct / len(results)) if results else 0.0, "correct_count": correct,
            "total_count": len(results), "results": results}


def main() -> None:
    cfg = load_experiment_config()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model_name", "--model", dest="model_name", default="openai/gpt-oss-120b")
    ap.add_argument("--endpoint_url", default="http://127.0.0.1:8080/v1")
    ap.add_argument("--no_autostart", action="store_true")
    ap.add_argument("--mock", action="store_true", help="MALLM mock endpoint (pipeline test)")
    ap.add_argument("--datasets", default="cwq,ethics,winogrande")
    ap.add_argument("--k", default="0,1,2,3", help="comma-separated exposure levels")
    ap.add_argument("--num_agents", type=int, default=3)
    ap.add_argument("--categories", default="false_fact", help="'false_fact', comma list, or 'all'")
    ap.add_argument("--data_dir", default=None)
    ap.add_argument("--mint_filter", choices=("all", "verified"), default="all")
    ap.add_argument("--mint_ids_file", default=None,
                    help="restrict every condition (including k=0) to the MINT ids listed in this file; "
                         "{dataset} is replaced by the dataset key (e.g. configs/subsets/verified_{dataset}.txt)")
    ap.add_argument("--information_source", choices=INFO_SOURCES, default="misinformation")
    ap.add_argument("--passage_sharing", choices=("shared", "independent"), default="shared")
    ap.add_argument("--mitigation", choices=MITIGATIONS, default="none")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max_turns", type=int, default=5)
    ap.add_argument("--max_tokens", type=int, default=None)
    ap.add_argument("--concurrent_api_requests", type=int, default=None)
    ap.add_argument("--max_items", type=int, default=None, help="first N questions per dataset (smoke tests)")
    ap.add_argument("--output_dir", default=None)
    ap.add_argument("--continue", dest="continue_mode", action="store_true")
    args = ap.parse_args()

    model_name = normalize_model_name(args.model_name)
    exp2_base._ensure_mallm_on_path()
    if args.mock:
        import exp3 as exp3_base  # its stub set also covers torch (MALLM type imports)

        exp3_base._install_runtime_stubs()
        endpoint = "mock://local"
    else:
        endpoint = resolve_endpoint(args.endpoint_url, model_name, args.no_autostart)
        timeout_s = int(os.environ.get("MALLM_OPENAI_TIMEOUT_SECONDS", "1800"))
        exp2_base._patch_mallm_openai_timeout(timeout_seconds=timeout_s, max_retries=5)
    from mallm import scheduler
    from mallm.utils.config import Config

    model_suffix = "mock_model" if args.mock else exp2_base._safe_model_name(model_name)
    out_dir = Path(args.output_dir) if args.output_dir else REPO_ROOT / "out" / model_suffix / "exposure"
    io_dir = out_dir / "exposure_io"
    io_dir.mkdir(parents=True, exist_ok=True)
    max_tokens = args.max_tokens or default_max_tokens(model_name, cfg)
    conc = args.concurrent_api_requests or int(os.environ.get("MALLM_CONCURRENT_API_REQUESTS", "250"))
    ks = [int(x) for x in args.k.split(",") if x.strip() != ""]
    n = args.num_agents
    print(f"[INFO] exp_exposure model={model_name} endpoint={endpoint} out={out_dir} max_tokens={max_tokens} k={ks} n={n}")

    for ds_raw in [d.strip() for d in args.datasets.split(",") if d.strip()]:
        ds = canonical_dataset(ds_raw)
        items = load_mint_items(ds, args.data_dir, max_items=args.max_items)
        id_subset = load_mint_id_subset(args.mint_ids_file, ds)
        if id_subset is not None:
            items = filter_items_by_ids(items, id_subset)
            print(f"[INFO] {ds}: restricted to {len(items)} of {len(id_subset)} listed questions ({args.mint_ids_file})")
        items_by_id = {it["mint_id"]: it for it in items}
        categories = expand_categories(args.categories, items)
        agg_path = out_dir / f"exposure_results_{ds}_seed{args.seed}.json"
        existing = {}
        if agg_path.is_file():
            try:
                existing = json.load(open(agg_path, encoding="utf-8"))
            except Exception:
                existing = {}
        for category in categories:
            for k in ks:
                if k > n:
                    continue
                cond = condition_key(category, k, n, args.information_source, args.passage_sharing, args.mitigation, args.seed)
                if k == 0:
                    # k=0 does not depend on category/source; run it once under the false_fact key
                    if category != categories[0]:
                        continue
                    cond = condition_key("none", 0, n, "none", args.passage_sharing, args.mitigation, args.seed)
                if args.continue_mode and cond in existing and existing[cond].get("total_count", 0) > 0:
                    print(f"[CONTINUE] skipping {ds} {cond}")
                    continue
                cond_items = filter_items_for_condition(items, category, args.mint_filter) if k > 0 else items
                mallm_in = build_mallm_input(cond_items, ds, category, k, n, args.information_source,
                                             args.passage_sharing, args.seed)
                if not mallm_in:
                    print(f"[WARN] no items for {ds} {cond}; skipping")
                    continue
                safe = cond.replace("|", "__").replace("=", "-")
                in_path = io_dir / f"exposure_input_{ds}_{safe}.json"
                out_path = io_dir / f"exposure_output_{ds}_{safe}.json"
                with open(in_path, "w", encoding="utf-8") as f:
                    json.dump(mallm_in, f)
                mcfg = Config(
                    input_json_file_path=str(in_path),
                    output_json_file_path=str(out_path),
                    task_instruction_prompt=task_instruction(ds, args.mitigation),
                    endpoint_url=endpoint,
                    model_name=model_name,
                    discussion_paradigm="memory",
                    decision_protocol="simple_voting",
                    max_turns=args.max_turns,
                    num_agents=n,
                    max_tokens=max_tokens,
                    agent_generator="informed",
                    agent_generators_list=["informed"] * n,
                    use_chain_of_thought=False,
                    concurrent_api_requests=conc,
                    shuffle_input_samples=False,
                )
                print(f"[RUN] {ds} {cond} ({len(mallm_in)} debates)")
                scheduler.Scheduler(mcfg).run()
                res = evaluate_output(str(out_path), items_by_id, category)
                res["config"] = {
                    "dataset": ds, "category": category, "k": k, "n": n, "information_source": args.information_source,
                    "passage_sharing": args.passage_sharing, "mitigation": args.mitigation, "seed": args.seed,
                    "max_turns": args.max_turns, "max_tokens": max_tokens, "model_name": model_name,
                    "endpoint_url": endpoint, "mint_filter": args.mint_filter, "mint_ids_file": args.mint_ids_file,
                    "n_items_requested": len(items), "n_items_run": len(mallm_in),
                    "task_instruction": task_instruction(ds, args.mitigation),
                    "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                }
                existing[cond] = res
                with open(agg_path, "w", encoding="utf-8") as f:
                    json.dump(existing, f)
                print(f"[DONE] {ds} {cond}: acc={res['accuracy']:.3f} ({res['correct_count']}/{res['total_count']}) -> {agg_path}")
    write_run_metadata(out_dir / f"run_metadata_seed{args.seed}_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')}.json",
                       args, {"model_name": model_name, "endpoint": endpoint, "max_tokens": max_tokens})


if __name__ == "__main__":
    main()
