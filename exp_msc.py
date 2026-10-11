#!/usr/bin/env python3
"""
exp_msc.py - Misinformed self-consistency (mSC) baseline.

For every question, draw n independent single-agent samples of which exactly k
receive the passage (same slot assignment as exp_exposure.py for the same seed),
and aggregate by majority vote. This has the same exposure and the same
aggregation as voting MAD but no interaction, so
    debate_effect(k) = acc_MAD(k) - acc_mSC(k).
k = 0 with n samples is plain self-consistency SC(uninformed, n) used by the
decomposition.

Prompting mirrors exp2_ablation.py (MALLM's baseline system prompt + the same task
instruction as the debate agents, the passage in a 'You have the following
information:' line). Sampling defaults to the server's default temperature (like
MALLM); set --temperature/--top_p explicitly to change that.

Outputs: <out_dir>/msc_results_<dataset>_seed<seed>.json (keyed by
exposure_common.condition_key) and <out_dir>/msc_samples_<dataset>_seed<seed>.jsonl
(one row per question x sample).
"""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

from answer_extraction import extract_free_form_answer, extract_mc_answer, free_form_matches, normalize_text
from exp2_ablation import SC_INFORMATION_TEMPLATE, SC_SYSTEM_PROMPT
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
from runner_common import ChatClient, JsonlWriter, TokenLedger, load_experiment_config, resolve_endpoint, stable_seed, write_run_metadata

REPO_ROOT = Path(__file__).resolve().parent


def build_prompt(instr: str, input_str: str, information: Optional[str]) -> List[dict]:
    user = f"Task: {instr}\nInput: {input_str}"
    if information and str(information).strip():
        user += "\n" + SC_INFORMATION_TEMPLATE.format(info=information)
    return [{"role": "system", "content": SC_SYSTEM_PROMPT}, {"role": "user", "content": user}]


def _mock_answer(messages: List[dict]) -> str:
    content = messages[-1]["content"]
    r = stable_seed(content) % 100
    if "Options:" in content:
        return f"Reasoning...\nFinal Solution: {'A' if r < 60 else 'B'}"
    return "Final Solution: Sweden"


def main() -> None:
    cfg = load_experiment_config()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model_name", "--model", dest="model_name", default="openai/gpt-oss-120b")
    ap.add_argument("--endpoint_url", default="http://127.0.0.1:8080/v1")
    ap.add_argument("--no_autostart", action="store_true")
    ap.add_argument("--mock", action="store_true")
    ap.add_argument("--datasets", default="cwq,ethics,winogrande")
    ap.add_argument("--k", default="0,1,2,3")
    ap.add_argument("--num_samples", type=int, default=3, help="n independent samples (= number of debate agents)")
    ap.add_argument("--categories", default="false_fact")
    ap.add_argument("--data_dir", default=None)
    ap.add_argument("--mint_filter", choices=("all", "verified"), default="all")
    ap.add_argument("--mint_ids_file", default=None,
                    help="restrict every condition (including k=0) to the MINT ids listed in this file ({dataset} placeholder)")
    ap.add_argument("--information_source", choices=INFO_SOURCES, default="misinformation")
    ap.add_argument("--passage_sharing", choices=("shared", "independent"), default="shared")
    ap.add_argument("--mitigation", choices=MITIGATIONS, default="none")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--temperature", type=float, default=None,
                    help="default 1.0 (vLLM default; equals the Llama-3.3 server default only if its generation_config is "
                         "ignored). Debates send no sampling parameters, so pass the model's server default here to be "
                         "sampling-matched (Llama-3.3-70B-Instruct: --temperature 0.6 --top_p 0.9).")
    ap.add_argument("--top_p", type=float, default=None)
    ap.add_argument("--max_tokens", type=int, default=None)
    ap.add_argument("--concurrent_requests", type=int, default=64)
    ap.add_argument("--max_items", type=int, default=None)
    ap.add_argument("--output_dir", default=None)
    ap.add_argument("--continue", dest="continue_mode", action="store_true")
    args = ap.parse_args()

    model_name = normalize_model_name(args.model_name)
    endpoint = None if args.mock else resolve_endpoint(args.endpoint_url, model_name, args.no_autostart)
    ledger = TokenLedger()
    client = ChatClient(endpoint, model_name, mock=args.mock, ledger=ledger, mock_answer_fn=_mock_answer,
                        timeout=float(os.environ.get("MALLM_OPENAI_TIMEOUT_SECONDS", "600")))
    low = model_name.lower()
    max_tokens = args.max_tokens or (int(cfg["debate"].get("max_tokens_reasoning_models", 4096))
                                     if ("glm" in low or "gpt-oss" in low) else int(cfg["debate"].get("max_tokens", 1024)))
    model_suffix = "mock_model" if args.mock else model_name.replace("/", "_")
    out_dir = Path(args.output_dir) if args.output_dir else REPO_ROOT / "out" / model_suffix / "msc"
    out_dir.mkdir(parents=True, exist_ok=True)
    ks = [int(x) for x in args.k.split(",") if x.strip() != ""]
    n = args.num_samples
    print(f"[INFO] exp_msc model={model_name} out={out_dir} n={n} k={ks} temperature={args.temperature} max_tokens={max_tokens}")

    for ds_raw in [d.strip() for d in args.datasets.split(",") if d.strip()]:
        ds = canonical_dataset(ds_raw)
        items = load_mint_items(ds, args.data_dir, max_items=args.max_items)
        id_subset = load_mint_id_subset(args.mint_ids_file, ds)
        if id_subset is not None:
            items = filter_items_by_ids(items, id_subset)
            print(f"[INFO] {ds}: restricted to {len(items)} of {len(id_subset)} listed questions ({args.mint_ids_file})")
        categories = expand_categories(args.categories, items)
        agg_path = out_dir / f"msc_results_{ds}_seed{args.seed}.json"
        samples_writer = JsonlWriter(out_dir / f"msc_samples_{ds}_seed{args.seed}.jsonl")
        existing = json.load(open(agg_path, encoding="utf-8")) if agg_path.is_file() else {}
        instr = task_instruction(ds, args.mitigation)
        for category in categories:
            for k in ks:
                if k > n:
                    continue
                cond = condition_key(category, k, n, args.information_source, args.passage_sharing, args.mitigation, args.seed)
                if k == 0:
                    if category != categories[0]:
                        continue
                    cond = condition_key("none", 0, n, "none", args.passage_sharing, args.mitigation, args.seed)
                if args.continue_mode and cond in existing and existing[cond].get("total_count", 0) > 0:
                    print(f"[CONTINUE] skipping {ds} {cond}")
                    continue
                cond_items = filter_items_for_condition(items, category, args.mint_filter) if k > 0 else items
                jobs = []
                for it in cond_items:
                    if args.passage_sharing == "shared":
                        passages = [select_passage(it, category, args.information_source)] * k
                    else:
                        passages = independent_passages(it, category, args.information_source, k, args.seed)
                    if k > 0 and any(p is None or not str(p).strip() for p in passages):
                        continue
                    slots = assign_slots(it, k, n, args.seed, passages)
                    input_str = format_mc_input(it) if it["is_multiple_choice"] else it["sentence"]
                    for j in range(n):
                        jobs.append((it, j, slots[j], build_prompt(instr, input_str, slots[j])))
                print(f"[RUN] {ds} {cond}: {len(jobs)} completions")

                def _one(job):
                    it, j, info, msgs = job
                    res = client.chat(msgs, max_tokens=max_tokens,
                                      temperature=args.temperature if args.temperature is not None else 1.0,
                                      top_p=args.top_p if args.top_p is not None else 1.0,
                                      seed=stable_seed(args.seed, it["mint_id"], j), label=f"msc_{ds}")
                    return it["mint_id"], j, info, res

                by_item: dict = {}
                with ThreadPoolExecutor(max_workers=max(1, args.concurrent_requests)) as ex:
                    futs = [ex.submit(_one, jb) for jb in jobs]
                    done = 0
                    for fut in as_completed(futs):
                        mid, j, info, res = fut.result()
                        by_item.setdefault(mid, {})[j] = (info, res)
                        done += 1
                        if done % 500 == 0 or done == len(futs):
                            print(f"  {done}/{len(futs)}")

                results = []
                correct = 0
                items_by_id = {it["mint_id"]: it for it in cond_items}
                for mid, samples in by_item.items():
                    it = items_by_id[mid]
                    votes = []
                    sample_rows = []
                    for j in sorted(samples):
                        info, res = samples[j]
                        if it["is_multiple_choice"]:
                            ex_ = extract_mc_answer(res.text, it["options"])
                            vote = ex_.index if ex_.status == "parsed" else None
                            is_corr = bool(ex_.status == "parsed" and ex_.index == it["gold_index"])
                            pred_text = ex_.text
                        else:
                            ex_ = extract_free_form_answer(res.text)
                            vote = normalize_text(ex_.text) if ex_.status == "parsed" and ex_.text else None
                            golds = [it["gold_text"]] + list(it.get("gold_aliases") or [])
                            is_corr = bool(ex_.status == "parsed" and free_form_matches(ex_.text, golds))
                            pred_text = (ex_.text or "")[:300]
                        if vote is not None:
                            votes.append(vote)
                        row = {
                            "dataset": ds, "condition": cond, "category": category, "k": k, "n": n, "mint_id": mid,
                            "sample_idx": j, "informed": info is not None, "response": res.text,
                            "finish_reason": res.finish_reason, "prompt_tokens": res.prompt_tokens,
                            "completion_tokens": res.completion_tokens, "error": res.error,
                            "predicted_index": ex_.index, "predicted_text": pred_text, "status": ex_.status,
                            "method": ex_.method, "is_correct": is_corr, "gold_index": it["gold_index"],
                        }
                        sample_rows.append(row)
                    samples_writer.write_many(sample_rows)
                    winner, tie, share = None, False, 0.0
                    if votes:
                        cnt = Counter(votes).most_common()
                        winner = cnt[0][0]
                        tie = len(cnt) > 1 and cnt[1][1] == cnt[0][1]
                        share = cnt[0][1] / len(votes)
                    if it["is_multiple_choice"]:
                        is_correct = bool(winner is not None and not tie and winner == it["gold_index"])
                        pred = it["options"][winner] if winner is not None else None
                    else:
                        golds = [it["gold_text"]] + list(it.get("gold_aliases") or [])
                        is_correct = bool(winner is not None and not tie and free_form_matches(winner, golds))
                        pred = winner
                    correct += int(is_correct)
                    results.append({
                        "sentence": it["sentence"], "mint_id": mid, "response": pred, "predicted_answer": pred,
                        "correct_answer": it["gold_text"], "is_correct": is_correct,
                        "is_multiple_choice": it["is_multiple_choice"], "options": it["options"],
                        "misinformation_strategy": category,
                        "msc_log": {"num_samples": n, "num_parsed": len(votes), "tie": tie, "winner_share": share,
                                    "samples": [{k_: v for k_, v in r.items() if k_ not in ("response",)} for r in sample_rows]},
                    })
                res_agg = {"accuracy": (correct / len(results)) if results else 0.0, "correct_count": correct,
                           "total_count": len(results), "results": results,
                           "config": {"dataset": ds, "category": category, "k": k, "n": n,
                                      "information_source": args.information_source, "passage_sharing": args.passage_sharing,
                                      "mitigation": args.mitigation, "seed": args.seed, "temperature": args.temperature,
                                      "top_p": args.top_p, "max_tokens": max_tokens, "model_name": model_name,
                                      "mint_filter": args.mint_filter, "mint_ids_file": args.mint_ids_file,
                                      "task_instruction": instr,
                                      "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                                      "tokens": ledger.as_dict().get(f"msc_{ds}")}}
                existing[cond] = res_agg
                with open(agg_path, "w", encoding="utf-8") as f:
                    json.dump(existing, f)
                print(f"[DONE] {ds} {cond}: acc={res_agg['accuracy']:.3f} ({correct}/{len(results)})")
    write_run_metadata(out_dir / f"run_metadata_seed{args.seed}_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')}.json",
                       args, {"model_name": model_name, "tokens": ledger.as_dict()})


if __name__ == "__main__":
    main()
