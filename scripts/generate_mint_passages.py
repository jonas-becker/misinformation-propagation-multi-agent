#!/usr/bin/env python3
"""
generate_mint_passages.py - Generate passages with the MINT pipeline via an
OpenAI-compatible endpoint (cross-generator sets, relevant-true control, paraphrase
style control).

Modes
  misinformation : false fact + one passage per rhetorical category (prompts
                   identical to download_datasets.py, generator model swapped)
                   -> MINT_dataset_v1.2/generators/<gen>/<dataset>_misinformed.json (+ _passages.jsonl)
  relevant_true  : true fact + one TRUE passage per style, supporting the gold answer,
                   length target = the item's Llama misinformation length per category
                   -> controls/relevant_true/<dataset>_misinformed.json
                      (fields: relevant_true_by_strategy, relevant_true_information)
  paraphrase     : content-preserving paraphrase of the existing (Llama) passages
                   -> controls/paraphrase_<gen>/<dataset>_misinformed.json

All outputs keep the item-level schema so exp1.py / exp_exposure.py can consume
them via --data_dir. Every run writes run metadata (model, params, token counts).
Use --mock for a pipeline test without a model server.
"""

from __future__ import annotations

import argparse
import threading
import json
import re
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from mint_data import DATASET_FILES, load_mint_items, resolve_dataset_path  # noqa: E402
from mint_prompts import (  # noqa: E402
    STRATEGIES,
    false_fact_prompt,
    misinformation_from_fact_prompt,
    paraphrase_prompt,
    remove_meta_sentences,
    strip_wrapping,
    true_fact_prompt,
    true_passage_from_fact_prompt,
)
from runner_common import ChatClient, TokenLedger, resolve_endpoint, stable_seed, write_run_metadata  # noqa: E402
from scripts.build_mint_v1_2 import explicit_cue, n_tokens  # noqa: E402
from scripts.common import model_display  # noqa: E402

GEN_KEYS = {"llama": "llama", "glm": "glm", "gptoss": "gptoss"}


def gen_key(model_name: str) -> str:
    fam = model_display(model_name)
    return {"Llama-3.3-70B": "llama", "GLM-4.7-Flash": "glm", "GPT-OSS-120B": "gptoss"}.get(fam,
                                                                                            re.sub(r"[^a-z0-9]+", "_", fam.lower()))


def _mock_generate(messages: List[dict]) -> str:
    content = messages[-1]["content"]
    if "Paraphrase" in content:
        return "Paraphrased: " + content.split("Text:", 1)[-1].strip()[:200]
    if "true fact" in content.lower():
        return "A verifiable and relevant true statement about the situation that points toward the right option."
    return "A believable but incorrect statement about the situation that points toward the wrong option."


empty_calls: Dict[str, int] = {}
_empty_lock = threading.Lock()


def generate_for_dataset(ds: str, items: List[dict], client: ChatClient, args: argparse.Namespace) -> List[dict]:
    mode = args.mode
    out_items: List[dict] = []

    def _call(prompt: str, seed_parts: tuple, label: str, max_tokens: int) -> str:
        seed = stable_seed(*seed_parts) if args.seed is not None else None
        res = client.chat([{"role": "user", "content": prompt}], max_tokens=max_tokens, temperature=args.temperature,
                          top_p=args.top_p, seed=seed, label=label)
        if not res.text.strip() and res.finish_reason == "length":
            # thinking models (GLM-4.7-Flash, gpt-oss) can spend the whole budget on reasoning: one retry, doubled
            res = client.chat([{"role": "user", "content": prompt}], max_tokens=max_tokens * 2, temperature=args.temperature,
                              top_p=args.top_p, seed=seed, label=label)
        if not res.text.strip():
            with _empty_lock:
                empty_calls[label] = empty_calls.get(label, 0) + 1
        return strip_wrapping(res.text)

    # Stage 1: base fact (misinformation: false fact; relevant_true: true fact). Paraphrase has no stage 1.
    base_facts: Dict[int, str] = {}
    if mode in ("misinformation", "relevant_true"):
        def _fact(i: int) -> tuple:
            it = items[i]
            prompt = false_fact_prompt(it) if mode == "misinformation" else true_fact_prompt(it)
            return i, _call(prompt, (args.seed, it["mint_id"], mode, "fact"), f"{mode}_fact_{ds}", args.max_tokens_fact)

        with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
            for fut in as_completed([ex.submit(_fact, i) for i in range(len(items))]):
                i, txt = fut.result()
                base_facts[i] = txt
        print(f"  [{ds}] stage 1 done ({len(base_facts)} base facts)")

    # Stage 2: per-category passages
    def _passage(i: int, strat: str) -> tuple:
        it = items[i]
        if mode == "misinformation":
            prompt = misinformation_from_fact_prompt(it, base_facts[i], strat)
            txt = remove_meta_sentences(_call(prompt, (args.seed, it["mint_id"], mode, strat), f"{mode}_{ds}", args.max_tokens))
        elif mode == "relevant_true":
            target = None
            if args.length_match:
                ref = it["passages"].get(strat)
                target = n_tokens(ref) if ref else None
            prompt = true_passage_from_fact_prompt(it, base_facts[i], strat, target_tokens=target)
            txt = _call(prompt, (args.seed, it["mint_id"], mode, strat), f"{mode}_{ds}", args.max_tokens)
        else:  # paraphrase
            src = it["passages"].get(strat)
            if not src:
                return i, strat, None
            prompt = paraphrase_prompt(src)
            txt = _call(prompt, (args.seed, it["mint_id"], mode, strat), f"{mode}_{ds}", args.max_tokens)
        return i, strat, txt

    cats = ["false_fact"] + STRATEGIES if mode != "paraphrase" else ["false_fact"] + STRATEGIES
    passages: Dict[int, Dict[str, str]] = {i: {} for i in range(len(items))}
    with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
        futs = []
        for i in range(len(items)):
            for strat in cats:
                if mode == "misinformation" and strat == "false_fact":
                    continue  # the false fact itself is the 'false_fact' passage
                if mode == "relevant_true" and strat == "false_fact":
                    continue
                futs.append(ex.submit(_passage, i, strat))
        done = 0
        for fut in as_completed(futs):
            i, strat, txt = fut.result()
            if txt is not None:
                passages[i][strat] = txt
            done += 1
            if done % 500 == 0 or done == len(futs):
                print(f"  [{ds}] stage 2: {done}/{len(futs)}")

    for i, it in enumerate(items):
        raw = json.loads(json.dumps(it["raw"]))
        raw["MINT_ID"] = it["mint_id"]
        raw["gold_index"] = it["gold_index"]
        raw["gold_text"] = it["gold_text"]
        if mode == "misinformation":
            raw["false_fact"] = base_facts[i]
            raw["misinformation_by_strategy"] = {s: passages[i].get(s, "") for s in STRATEGIES}
            raw["generator_model"] = args.model_name
            raw["validation"] = {}
        elif mode == "relevant_true":
            by_strat = {"false_fact": base_facts[i]}  # 'false_fact' slot holds the neutral TRUE fact
            by_strat.update({s: passages[i].get(s, "") for s in STRATEGIES})
            raw["relevant_true_information"] = base_facts[i]
            raw["relevant_true_by_strategy"] = by_strat
            raw["relevant_true_generator_model"] = args.model_name
        else:
            raw["false_fact"] = passages[i].get("false_fact", raw.get("false_fact"))
            raw["misinformation_by_strategy"] = {s: passages[i].get(s, "") for s in STRATEGIES}
            raw["paraphrase_model"] = args.model_name
            raw["paraphrase_of_generator"] = "meta-llama/Llama-3.3-70B-Instruct"
            raw["validation"] = {}
        out_items.append(raw)
    return out_items


def write_passages_jsonl(ds: str, out_items: List[dict], path: Path, mode: str, generator: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for it in out_items:
            gi = it.get("gold_index")
            opts = it.get("options")
            gold_text = it.get("gold_text")
            wrong = None
            if isinstance(opts, list) and len(opts) == 2 and gi is not None:
                wrong = str(opts[1 - int(gi)])
            if mode == "relevant_true":
                by = it.get("relevant_true_by_strategy") or {}
            else:
                by = {"false_fact": it.get("false_fact"), **(it.get("misinformation_by_strategy") or {})}
            for cat, p in by.items():
                if not p:
                    continue
                f.write(json.dumps({
                    "passage_id": f"{it['MINT_ID']}::{cat}", "mint_id": it["MINT_ID"], "dataset": ds, "category": cat,
                    "question": it.get("sentence"), "options": opts, "gold_index": gi, "gold_text": gold_text,
                    "gold_aliases": (it.get("gold_answers") or [None])[1:] if it.get("gold_answers") else [],
                    "intended_wrong_answer": wrong, "passage": p, "n_tokens_ws": n_tokens(p),
                    "explicit_cue_lexical": explicit_cue(p, wrong) if mode != "relevant_true" else explicit_cue(p, gold_text),
                    "generator_model": generator, "mode": mode, "validation": {},
                }, ensure_ascii=False) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", choices=("misinformation", "relevant_true", "paraphrase"), required=True)
    ap.add_argument("--model_name", required=True, help="generator model served at --endpoint_url")
    ap.add_argument("--endpoint_url", default="http://127.0.0.1:8080/v1")
    ap.add_argument("--no_autostart", action="store_true")
    ap.add_argument("--data_dir", default=str(REPO_ROOT / "MINT_dataset_v1.2"), help="source items (questions, gold, Llama passages)")
    ap.add_argument("--out_dir", default=None)
    ap.add_argument("--datasets", default="cwq,ethics,winogrande")
    ap.add_argument("--max_items", type=int, default=None, help="use the first N questions per dataset (all by default; MINT has ~380)")
    ap.add_argument("--temperature", type=float, default=0.7, help="MINT v1 used 0.7")
    ap.add_argument("--top_p", type=float, default=0.9, help="MINT v1 used 0.9")
    ap.add_argument("--max_tokens", type=int, default=512)
    ap.add_argument("--max_tokens_fact", type=int, default=160)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--length_match", action="store_true", default=True,
                    help="relevant_true: ask for the Llama passage's word count per category (default on)")
    ap.add_argument("--concurrency", type=int, default=32)
    ap.add_argument("--mock", action="store_true")
    args = ap.parse_args()
    low = args.model_name.lower()
    if "glm" in low or "gpt-oss" in low:
        # thinking models: reasoning tokens count against max_tokens (same rule as exp1.py / exp_exposure.py);
        # with the parser defaults GLM-4.7-Flash returned empty content (all budget spent thinking).
        if args.max_tokens == 512:
            args.max_tokens = 4096
        if args.max_tokens_fact == 160:
            args.max_tokens_fact = 2048
    print(f"generation budgets: max_tokens={args.max_tokens} max_tokens_fact={args.max_tokens_fact} model={args.model_name}")

    gk = gen_key(args.model_name)
    if args.out_dir:
        out_dir = Path(args.out_dir)
    elif args.mode == "misinformation":
        out_dir = REPO_ROOT / "MINT_dataset_v1.2" / "generators" / gk
    elif args.mode == "relevant_true":
        out_dir = REPO_ROOT / "controls" / "relevant_true"
    else:
        out_dir = REPO_ROOT / "controls" / f"paraphrase_{gk}"
    out_dir.mkdir(parents=True, exist_ok=True)

    endpoint = None
    if not args.mock:
        endpoint = resolve_endpoint(args.endpoint_url, args.model_name, args.no_autostart)
    ledger = TokenLedger()
    client = ChatClient(endpoint, args.model_name, mock=args.mock, ledger=ledger, mock_answer_fn=_mock_generate)

    for ds in [d.strip() for d in args.datasets.split(",") if d.strip()]:
        items = load_mint_items(ds, args.data_dir, max_items=args.max_items)
        print(f"[{ds}] generating mode={args.mode} with {args.model_name} for {len(items)} questions "
              f"(source: {resolve_dataset_path(ds, args.data_dir)})")
        out_items = generate_for_dataset(ds, items, client, args)
        item_path = out_dir / DATASET_FILES[ds]
        with open(item_path, "w", encoding="utf-8") as f:
            json.dump(out_items, f, indent=1, ensure_ascii=False)
        suffix = "_passages" if args.mode != "relevant_true" else "_relevant_true_passages"
        write_passages_jsonl(ds, out_items, out_dir / f"{ds}{suffix}.jsonl", args.mode, args.model_name)
        print(f"  wrote {item_path}")
    if empty_calls:
        print(f"[WARN] empty generations (passage skipped) per label: {empty_calls}")
    write_run_metadata(out_dir / f"generation_run_{gk}.json", args,
                       {"tokens": ledger.as_dict(), "generator_key": gk, "empty_generations": dict(empty_calls)})
    print("done")


if __name__ == "__main__":
    main()
