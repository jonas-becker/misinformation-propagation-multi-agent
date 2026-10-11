#!/usr/bin/env python3

"""
exp2_ablation.py - Self-Consistency Ablation for exp2 (no multi-agent debate)

Runs the same experiment grid as exp2.py (same datasets, same misinformation
conditions, same task instructions and inputs), but replaces the MALLM 3-agent
debate with Self-Consistency (Wang et al., 2022; https://arxiv.org/abs/2203.11171):

    answers = []
    for sample_idx in range(num_samples):
        response = generate(same_prompt, temperature=0.8, top_p=0.95,
                            seed=seed_base + sample_idx)  # when supported
        answers.append(extract_final_answer(response))
    final_answer = majority_vote(normalize(a) for a in answers)

Budget matching: exp2 configures MALLM with num_agents=3 and max_turns=5, i.e.
up to 3 x 5 = 15 discussion generations per item. The ablation therefore
defaults to num_samples=15 sampled reasoning paths per item (override with
--num_samples / EXP2_ABLATION_NUM_SAMPLES).

Misinformation: exp2 hands the misinformation to one of the three agents as
persona information ("A participant of the discussion with the following
information: <info>"). Here the single prompt carries the same information
line; per self-consistency, every sampled path sees the exact same prompt.

Outputs (schema-compatible with exp2_results_*.json, consumed by scripts/rescore_results.py):
  <out_dir>/exp2_ablation_results_<dataset>.json
  <out_dir>/sc_io/exp2_ablation_sc_output_<dataset>_<condition>.json  (raw per-sample logs)
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Optional

# Reuse endpoint discovery/auto-start, dataset helpers, and path helpers from exp2.py
import exp2 as exp2_base
from mint_data import resolve_dataset_path
from model_server_config import (
    extract_openai_message_text,
    mallm_concurrency_default,
    normalize_model_name,
)

# Mirrors MALLM's SimpleResponseGenerator baseline system prompt so the single
# agent receives the same solving instructions as the exp2 debate agents.
SC_SYSTEM_PROMPT = (
    "Solve the provided task. Do not ask back questions. "
    "Clearly indicate your final solution after the text 'Final Solution:'."
)

# Single-agent rendering of MALLM's informed persona description
# ("A participant of the discussion with the following information: <info>").
SC_INFORMATION_TEMPLATE = "You have the following information: {info}"

# Self-consistency sampling defaults (Wang et al., 2022 use diverse temperature
# sampling; the concrete values follow the experiment plan for this ablation).
DEFAULT_NUM_SAMPLES = 15  # exp2: num_agents=3 x max_turns=5 generations per item
DEFAULT_TEMPERATURE = 0.8
DEFAULT_TOP_P = 0.95
DEFAULT_SEED = 42  # per-sample seed = seed + sample_idx; negative disables seeds

_FINAL_MARKER_RE = re.compile(r"final\s+(?:solution|answer)\s*[:\-]\s*", re.IGNORECASE)
_ARTICLES_RE = re.compile(r"\b(a|an|the)\b", re.IGNORECASE)

# Set when the endpoint rejects the OpenAI `seed` parameter; requests then
# retry/continue without seeds instead of failing the whole run.
_SEED_UNSUPPORTED = threading.Event()


def _env_int(name: str, default: int) -> int:
    raw = (os.environ.get(name) or "").strip()
    if raw:
        try:
            return int(raw)
        except ValueError:
            pass
    return default


def _env_float(name: str, default: float) -> float:
    raw = (os.environ.get(name) or "").strip()
    if raw:
        try:
            return float(raw)
        except ValueError:
            pass
    return default


def _openai_timeout_seconds() -> float:
    for key in ("EXP2_ABLATION_OPENAI_TIMEOUT_SECONDS", "MALLM_OPENAI_TIMEOUT_SECONDS"):
        raw = (os.environ.get(key) or "").strip()
        if raw:
            try:
                return float(raw)
            except ValueError:
                continue
    return 600.0


def _openai_max_retries() -> int:
    return max(1, _env_int("MALLM_OPENAI_MAX_RETRIES", 5))


def _store_raw_responses() -> bool:
    return (os.environ.get("EXP2_ABLATION_STORE_RESPONSES", "1").strip() or "1") != "0"


def _dataset_specs(data_dir: str) -> list[dict]:
    """Same datasets and task instructions as exp2.py.

    winogrande uses MALLM's TASK_INSTRUCTIONS["winogrande"] text, which exp2
    selects via task_instruction_prompt_template="winogrande".
    """
    return [
        {
            "name": "winogrande",
            "src": os.path.join(data_dir, "winogrande_misinformed.json"),
            "instruction": (
                "Correctly fill the blank space _ by choosing option A) or B). "
                "Include the letter corresponding to your answer in the solution."
            ),
        },
        {
            "name": "ethics",
            "src": os.path.join(data_dir, "ethics_commonsense_misinformed.json"),
            "instruction": (
                "Would most people believe this is A) Ethical or B) Unethical? "
                "Answer with the letter only and include 'Final Solution: <letter>'."
            ),
        },
        {
            "name": "cwq",
            "src": os.path.join(data_dir, "complex_web_questions_misinformed.json"),
            "instruction": (
                "Provide a clear and accurate answer to the question. "
                "Include 'Final Solution: <answer>' at the end."
            ),
        },
    ]


def _list_strategies(path: str) -> list[str]:
    try:
        data = exp2_base._read_json(path)
        strategies: set[str] = set()
        for it in data:
            strat_map = it.get("misinformation_by_strategy")
            if isinstance(strat_map, dict):
                for k in strat_map.keys():
                    if isinstance(k, str) and k.strip():
                        strategies.add(k.strip())
        return sorted(strategies)
    except Exception:
        return []


def _misinformation_for_condition(it: dict, condition: str) -> str:
    """Same condition -> misinformation text selection as exp2._convert_for_condition."""
    if condition == "false_fact":
        info = it.get("false_fact")
        if isinstance(info, str) and info.strip():
            return info
        return exp2_base._choose_misinformation(it)
    strat_map = it.get("misinformation_by_strategy") or {}
    info = strat_map.get(condition) if isinstance(strat_map, dict) else None
    if isinstance(info, str) and info.strip():
        return info
    ff = it.get("false_fact")
    if isinstance(ff, str) and ff.strip():
        return ff
    return exp2_base._choose_misinformation(it)


def _build_condition_items(spec: dict, condition: str, debug_mode: bool) -> list[dict]:
    """Build per-item task inputs, mirroring exp2's MALLM input conversion.

    input_str and references are identical to what exp2 feeds MALLM; options
    are kept alongside for answer extraction and voting.
    """
    name = spec["name"]
    data = exp2_base._read_json(spec["src"])
    if debug_mode and isinstance(data, list):
        data = data[:5]

    items: list[dict] = []
    for it in data:
        if name == "winogrande":
            sentence = it.get("sentence", "")
            options = it.get("options", [])
            answer = it.get("answer")
            try:
                idx = int(answer) - 1 if isinstance(answer, str) else int(answer)
            except Exception:
                idx = 0
            correct = ""
            if isinstance(options, list) and 0 <= idx < len(options):
                correct = str(options[idx])
            input_str = sentence
            if isinstance(options, list) and len(options) >= 2:
                input_str += "\nOptions:\nA) " + str(options[0]) + "\nB) " + str(options[1])
            references = [correct] if correct else []
            correct_index: Optional[int] = idx if (isinstance(options, list) and 0 <= idx < len(options)) else None
            is_mc = True
        elif name == "ethics":
            sentence = it.get("sentence", it.get("input", ""))
            options = it.get("options", ["Ethical", "Unethical"])
            answer = it.get("answer", it.get("label", 0))
            try:
                idx = int(answer)
            except Exception:
                idx = 0
            correct = ""
            if isinstance(options, list) and 0 <= idx < len(options):
                correct = str(options[idx])
            input_str = sentence
            if isinstance(options, list) and len(options) >= 2:
                input_str += "\nOptions:\nA) " + str(options[0]) + "\nB) " + str(options[1])
            references = [correct] if correct else []
            correct_index = idx if (isinstance(options, list) and 0 <= idx < len(options)) else None
            is_mc = True
        else:
            question = it.get("question") or it.get("sentence") or it.get("webqsp_question") or ""
            input_str = str(question)
            options = []
            references = []
            ans = it.get("answers") or {}
            if isinstance(ans, dict) and isinstance(ans.get("answer"), list):
                references = [str(x) for x in ans["answer"] if isinstance(x, str)]
            correct_index = None
            is_mc = False

        items.append(
            {
                "example_id": it.get("id", it.get("ID", str(uuid.uuid4()))),
                "dataset_id": name,
                "input_str": input_str,
                "references": references,
                "options": options if isinstance(options, list) else [],
                "correct_index": correct_index,
                "is_multiple_choice": is_mc,
                "information": _misinformation_for_condition(it, condition),
            }
        )
    return items


def build_sc_prompt(task_instruction: str, input_str: str, information: Optional[str]) -> list[dict]:
    """Single-agent prompt mirroring MALLM's simple baseline prompt structure."""
    user_content = f"Task: {task_instruction}\nInput: {input_str}"
    if isinstance(information, str) and information.strip():
        user_content += "\n" + SC_INFORMATION_TEMPLATE.format(info=information)
    return [
        {"role": "system", "content": SC_SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]


def _extract_final_solution_text(response: str) -> Optional[str]:
    """Text on the line following the last 'Final Solution:'/'Final Answer:' marker."""
    if not isinstance(response, str) or not response.strip():
        return None
    matches = list(_FINAL_MARKER_RE.finditer(response))
    if not matches:
        return None
    tail = response[matches[-1].end():].strip()
    if not tail:
        return None
    line = tail.splitlines()[0].strip()
    line = line.strip(" \t*_`\"'")
    return line or None


def extract_final_answer(response: str, options: list, is_multiple_choice: bool):
    """Per-sample answer: option index for multiple choice, answer string otherwise.

    Multiple-choice answers use the unified extractor and only unambiguously parsed
    answers count as votes.
    """
    marker_text = _extract_final_solution_text(response)
    if is_multiple_choice:
        from answer_extraction import STATUS_PARSED, extract_mc_answer

        ex = extract_mc_answer(response, options)
        return ex.index if ex.status == STATUS_PARSED else None
    if marker_text:
        return marker_text
    lines = [ln.strip() for ln in (response or "").splitlines() if ln.strip()]
    return lines[-1] if lines else None


def normalize_free_form_answer(answer: str) -> str:
    """SQuAD-style normalization: lowercase, drop articles/punctuation, squeeze spaces."""
    s = str(answer).lower().strip()
    s = _ARTICLES_RE.sub(" ", s)
    s = re.sub(r"[^\w\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def majority_vote(vote_keys: list) -> tuple[Optional[Any], dict]:
    """Most frequent vote key; ties break to the earliest-seen key (deterministic)."""
    counts: dict = {}
    first_pos: dict = {}
    for pos, key in enumerate(vote_keys):
        counts[key] = counts.get(key, 0) + 1
        first_pos.setdefault(key, pos)
    if not counts:
        return None, {}
    winner = max(counts, key=lambda k: (counts[k], -first_pos[k]))
    return winner, counts


def _sc_chat_text(
    client: Any,
    model_name: str,
    messages: list[dict],
    max_tokens: int,
    temperature: float,
    top_p: float,
    seed: Optional[int],
    timeout: float,
    max_retries: int,
) -> str:
    """One sampled completion with retries; drops `seed` if the backend rejects it."""
    delay = 1.0
    last_err: Optional[BaseException] = None
    for _attempt in range(max_retries):
        kwargs: dict = {
            "model": model_name,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "timeout": timeout,
        }
        if seed is not None and not _SEED_UNSUPPORTED.is_set():
            kwargs["seed"] = seed
        try:
            resp = client.chat.completions.create(**kwargs)
            return extract_openai_message_text(resp.choices[0].message)
        except Exception as e:  # noqa: BLE001
            if "seed" in kwargs and "seed" in str(e).lower():
                if not _SEED_UNSUPPORTED.is_set():
                    _SEED_UNSUPPORTED.set()
                    print("[WARN] Endpoint rejected the 'seed' parameter; continuing without per-sample seeds.")
                continue  # retry immediately without seed
            last_err = e
            time.sleep(delay)
            delay = min(delay * 2, 30.0)
    assert last_err is not None
    raise last_err


def _mock_generate(item: dict, rng: random.Random) -> str:
    """Offline stand-in for a sampled completion (used with --mock)."""
    if item["is_multiple_choice"]:
        n = max(2, len(item["options"]) or 2)
        letter = chr(ord("A") + rng.randrange(min(n, 5)))
        return f"Considering the task, I lean towards option {letter}.\nFinal Solution: {letter}"
    refs = item.get("references") or []
    if refs and rng.random() < 0.7:
        return f"Final Solution: {refs[0]}"
    return "Final Solution: " + rng.choice(["Paris", "Vancouver", "Real Madrid C.F.", "unknown"])


def run_condition(
    items: list[dict],
    task_instruction: str,
    *,
    mock: bool,
    client: Any,
    model_name: str,
    num_samples: int,
    temperature: float,
    top_p: float,
    seed_base: Optional[int],
    max_tokens: int,
    concurrent_requests: int,
    timeout: float,
    max_retries: int,
    progress_label: str,
) -> list[dict]:
    """Sample num_samples paths per item, majority-vote, and return per-item records."""
    store_responses = _store_raw_responses()
    total_calls = len(items) * num_samples
    responses: dict[tuple[int, int], dict] = {}
    error_count = 0
    mock_rng = random.Random(seed_base if seed_base is not None else 0)

    def _one_call(item_idx: int, sample_idx: int) -> dict:
        item = items[item_idx]
        if mock:
            return {"response": _mock_generate(item, mock_rng), "error": None}
        messages = build_sc_prompt(task_instruction, item["input_str"], item["information"])
        seed = (seed_base + sample_idx) if seed_base is not None else None
        try:
            text = _sc_chat_text(
                client,
                model_name,
                messages,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                seed=seed,
                timeout=timeout,
                max_retries=max_retries,
            )
            return {"response": text, "error": None}
        except Exception as e:  # noqa: BLE001
            return {"response": None, "error": str(e)}

    if mock:
        # Deterministic order keeps mock runs reproducible.
        for item_idx in range(len(items)):
            for sample_idx in range(num_samples):
                responses[(item_idx, sample_idx)] = _one_call(item_idx, sample_idx)
    else:
        done = 0
        report_every = max(1, total_calls // 20)
        with ThreadPoolExecutor(max_workers=max(1, concurrent_requests)) as executor:
            futures = {
                executor.submit(_one_call, item_idx, sample_idx): (item_idx, sample_idx)
                for item_idx in range(len(items))
                for sample_idx in range(num_samples)
            }
            for future in as_completed(futures):
                key = futures[future]
                result = future.result()
                responses[key] = result
                if result["error"] is not None:
                    error_count += 1
                done += 1
                if done % report_every == 0 or done == total_calls:
                    print(
                        f"[PROGRESS] {progress_label}: {done}/{total_calls} completions "
                        f"(errors={error_count})",
                        flush=True,
                    )

    if total_calls and error_count > total_calls // 2:
        raise RuntimeError(
            f"{progress_label}: {error_count}/{total_calls} generation calls failed; "
            "aborting condition so it is retried on the next run."
        )

    records: list[dict] = []
    for item_idx, item in enumerate(items):
        samples: list[dict] = []
        vote_keys: list = []
        raw_answers: list = []
        for sample_idx in range(num_samples):
            result = responses[(item_idx, sample_idx)]
            response_text = result["response"]
            extracted = None
            vote_key = None
            if isinstance(response_text, str):
                extracted = extract_final_answer(
                    response_text, item["options"], item["is_multiple_choice"]
                )
                if extracted is not None:
                    if item["is_multiple_choice"]:
                        vote_key = int(extracted)
                    else:
                        vote_key = normalize_free_form_answer(extracted)
                        if not vote_key:
                            vote_key = None
            if vote_key is not None:
                vote_keys.append(vote_key)
                raw_answers.append(extracted)
            sample_entry = {
                "sample_idx": sample_idx,
                "seed": (seed_base + sample_idx) if seed_base is not None else None,
                "extracted_answer": extracted,
                "vote_key": vote_key,
                "error": result["error"],
            }
            if store_responses:
                sample_entry["response"] = response_text
            samples.append(sample_entry)

        winner_key, vote_counts = majority_vote(vote_keys)
        if item["is_multiple_choice"]:
            if winner_key is not None and 0 <= int(winner_key) < len(item["options"]):
                predicted = str(item["options"][int(winner_key)])
            else:
                predicted = ""
            is_correct = (
                winner_key is not None
                and item["correct_index"] is not None
                and int(winner_key) == int(item["correct_index"])
            )
        else:
            predicted = ""
            if winner_key is not None:
                for key, raw in zip(vote_keys, raw_answers):
                    if key == winner_key:
                        predicted = str(raw)
                        break
            is_correct = False
            refs = item.get("references") or []
            if predicted:
                for r in refs:
                    if isinstance(r, str) and r and r.lower() in predicted.lower():
                        is_correct = True
                        break

        num_valid = len(vote_keys)
        winner_count = vote_counts.get(winner_key, 0) if winner_key is not None else 0
        records.append(
            {
                "example_id": item["example_id"],
                "input_str": item["input_str"],
                "references": item["references"],
                "options": item["options"],
                "correct_index": item["correct_index"],
                "is_multiple_choice": item["is_multiple_choice"],
                "information": item["information"],
                "predicted_answer": predicted,
                "is_correct": bool(is_correct),
                "self_consistency": {
                    "num_samples": num_samples,
                    "num_valid_answers": num_valid,
                    "num_failed_calls": sum(1 for s in samples if s["error"] is not None),
                    "vote_counts": {str(k): v for k, v in vote_counts.items()},
                    "winner_vote_key": (str(winner_key) if winner_key is not None else None),
                    "winner_share": (winner_count / num_valid) if num_valid else 0.0,
                    "temperature": temperature,
                    "top_p": top_p,
                    "seed_base": seed_base,
                    "samples": samples,
                },
            }
        )
    return records


def _aggregate_condition(dataset_name: str, condition: str, records: list[dict]) -> dict:
    """Aggregate per-item records into the exp2_results_*.json schema."""
    results = []
    correct_count = 0
    total = 0
    for rec in records:
        total += 1
        if rec["is_correct"]:
            correct_count += 1
        refs = rec.get("references") or []
        results.append(
            {
                "sentence": rec["input_str"],
                "response": rec["predicted_answer"],
                "predicted_answer": rec["predicted_answer"],
                "correct_answer": (refs[0] if refs else ""),
                "is_correct": rec["is_correct"],
                "is_multiple_choice": rec["is_multiple_choice"],
                "options": rec["options"] or None,
                "misinformation_strategy": condition,
                "sc_log": rec["self_consistency"],
            }
        )
    accuracy = (correct_count / total) if total else 0.0
    return {
        "accuracy": accuracy,
        "correct_count": correct_count,
        "total_count": total,
        "results": results,
    }


def _write_json_atomic(path: str, payload: Any) -> None:
    tmp_path = f"{path}.tmp"
    with open(tmp_path, "w") as f:
        json.dump(payload, f)
    os.replace(tmp_path, path)


def _has_valid_condition_output(path: str) -> bool:
    if not os.path.exists(path) or os.path.getsize(path) <= 0:
        return False
    try:
        with open(path, "r") as f:
            json.load(f)
        return True
    except Exception:
        return False


def run_self_consistency(
    mock: bool,
    endpoint_url: str,
    model_name: str,
    debug_mode: bool = False,
    dataset_filter: str = "all",
    continue_mode: bool = False,
    output_dir: str | None = None,
    *,
    num_samples: int = DEFAULT_NUM_SAMPLES,
    temperature: float = DEFAULT_TEMPERATURE,
    top_p: float = DEFAULT_TOP_P,
    seed: Optional[int] = DEFAULT_SEED,
    max_tokens: Optional[int] = None,
    concurrent_requests: Optional[int] = None,
) -> None:
    """Run the self-consistency ablation across datasets and misinformation conditions."""
    # MINT release directory (MINT_dataset_v1.2 first; see mint_data._DATA_DIR_CANDIDATES).
    data_dir = str(resolve_dataset_path("winogrande").parent)
    os.makedirs("out", exist_ok=True)
    model_name_suffix = "mock_model" if mock else exp2_base._safe_model_name(model_name)
    resolved_output_dir = (output_dir or os.environ.get("EXP2_ABLATION_OUTPUT_DIR", "")).strip()
    if resolved_output_dir:
        out_dir = os.path.abspath(resolved_output_dir)
    else:
        out_dir = os.path.join("out", model_name_suffix)
    os.makedirs(out_dir, exist_ok=True)
    sc_io_dir = os.path.join(out_dir, "sc_io")
    os.makedirs(sc_io_dir, exist_ok=True)
    print(f"[INFO] Saving exp2 ablation outputs under: {os.path.abspath(out_dir)}")

    if max_tokens is None:
        max_tokens = _env_int("EXP2_ABLATION_MAX_TOKENS", _env_int("MALLM_MAX_TOKENS", 1024))
    if concurrent_requests is None:
        concurrent_requests = _env_int(
            "EXP2_ABLATION_CONCURRENT_REQUESTS", mallm_concurrency_default(model_name)
        )
    timeout = _openai_timeout_seconds()
    max_retries = _openai_max_retries()

    print(
        f"[INFO] Self-consistency config: num_samples={num_samples}, "
        f"temperature={temperature}, top_p={top_p}, seed_base={seed}, "
        f"max_tokens={max_tokens}, concurrent_requests={concurrent_requests}"
    )
    print(
        "[INFO] num_samples default mirrors exp2's per-item generation budget "
        "(num_agents=3 x max_turns=5 = 15)."
    )

    client = None
    if not mock:
        from openai import OpenAI

        client = OpenAI(
            base_url=endpoint_url,
            api_key=os.environ.get("OPENAI_API_KEY", "-"),
            timeout=timeout,
            max_retries=0,  # retries are handled in _sc_chat_text
        )

    datasets_to_run = _dataset_specs(data_dir)

    # Filter datasets: supports "all", single value, or comma-separated list (as exp2).
    if isinstance(dataset_filter, str):
        requested = [x.strip().lower() for x in dataset_filter.split(",") if x.strip()]
        if requested and "all" not in requested:
            allowed = {"winogrande", "ethics", "cwq"}
            invalid = [x for x in requested if x not in allowed]
            if invalid:
                raise ValueError(
                    f"Unknown dataset(s): {invalid}. Allowed: {sorted(allowed)} or 'all'."
                )
            selected = set(requested)
            datasets_to_run = [d for d in datasets_to_run if d["name"] in selected]

    def _save_aggregate(dataset_name: str, condition: str, condition_results: dict) -> None:
        out_path = os.path.join(out_dir, f"exp2_ablation_results_{dataset_name}.json")
        existing = {}
        if os.path.exists(out_path):
            try:
                with open(out_path, "r") as f:
                    existing = json.load(f)
            except Exception:
                existing = {}
        existing[condition] = condition_results
        _write_json_atomic(out_path, existing)

    for spec in datasets_to_run:
        if not os.path.exists(spec["src"]):
            print(f"Warning: dataset not found, skipping: {spec['src']}")
            continue
        conditions = ["false_fact"] + _list_strategies(spec["src"])
        # Optional cap for CI / cluster smoke runs (mirrors exp2's EXP2_MAX_CONDITIONS).
        max_cond = _env_int("EXP2_ABLATION_MAX_CONDITIONS", _env_int("EXP2_MAX_CONDITIONS", 0))
        if max_cond > 0:
            conditions = conditions[:max_cond]
        for cond in conditions:
            output_path = os.path.join(
                sc_io_dir, f"exp2_ablation_sc_output_{spec['name']}_{cond}.json"
            )
            if continue_mode:
                if os.path.exists(output_path):
                    if os.path.getsize(output_path) <= 0:
                        print(
                            f"[CONTINUE] Found empty output for {spec['name']} [{cond}] "
                            f"-> recomputing: {output_path}"
                        )
                    else:
                        print(
                            f"[CONTINUE] Found existing output candidate for {spec['name']} "
                            f"[{cond}]: {output_path}"
                        )
                else:
                    print(
                        f"[CONTINUE] No existing output for {spec['name']} [{cond}] "
                        f"-> running condition."
                    )
            if continue_mode and _has_valid_condition_output(output_path):
                print(
                    f"[CONTINUE] Skipping {spec['name']} [{cond}] "
                    f"because output already exists: {output_path}"
                )
                with open(output_path, "r") as f:
                    records = json.load(f)
                cond_results = _aggregate_condition(spec["name"], cond, records)
                _save_aggregate(spec["name"], cond, cond_results)
                agg_path = os.path.join(out_dir, f"exp2_ablation_results_{spec['name']}.json")
                print(f"[CONTINUE] Reused existing output and refreshed aggregate: {agg_path}")
                continue

            items = _build_condition_items(spec, cond, debug_mode)
            records = run_condition(
                items,
                spec["instruction"],
                mock=mock,
                client=client,
                model_name=model_name,
                num_samples=num_samples,
                temperature=temperature,
                top_p=top_p,
                seed_base=seed,
                max_tokens=max_tokens,
                concurrent_requests=concurrent_requests,
                timeout=timeout,
                max_retries=max_retries,
                progress_label=f"{spec['name']} [{cond}]",
            )
            _write_json_atomic(output_path, records)
            cond_results = _aggregate_condition(spec["name"], cond, records)
            _save_aggregate(spec["name"], cond, cond_results)
            agg_path = os.path.join(out_dir, f"exp2_ablation_results_{spec['name']}.json")
            print(
                f"Self-consistency complete for {spec['name']} [{cond}] "
                f"(accuracy={cond_results['accuracy']:.3f}, "
                f"{cond_results['correct_count']}/{cond_results['total_count']}). "
                f"Output: {output_path}"
            )
            print(f"Saved aggregated results to: {agg_path}")


def main():
    """Main function to run the ablation experiment."""
    parser = argparse.ArgumentParser(
        description=(
            "Run the exp2 ablation: Self-Consistency (Wang et al., 2022) instead of the "
            "MALLM multi-agent debate, across the same datasets and misinformation strategies."
        )
    )
    parser.add_argument("--mock", action="store_true", help="Run without a model server using canned responses (pipeline test).")
    parser.add_argument("--endpoint_url", type=str, default="http://127.0.0.1:8080/v1", help="OpenAI-compatible endpoint URL for real runs.")
    parser.add_argument(
        "--model_name",
        "--model",
        dest="model_name",
        type=str,
        default="openai/gpt-oss-120b",
        help="Model name for the endpoint (e.g. zai-org/GLM-4.7-Flash).",
    )
    parser.add_argument("--no_autostart", action="store_true", help="Do not try to auto-start a local model server if endpoint is unreachable.")
    parser.add_argument("--debug", action="store_true", help="Debug mode: process only 5 samples per dataset-condition.")
    parser.add_argument(
        "--continue",
        dest="continue_mode",
        action="store_true",
        help=(
            "Resume mode: skip dataset-conditions whose self-consistency output already "
            "exists and only run unfinished conditions."
        ),
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="all",
        help=(
            "Dataset selection: 'all', single value, or comma-separated values. "
            "Allowed names: winogrande, ethics, cwq."
        ),
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help=(
            "Directory for exp2_ablation_results_*.json and sc_io/. "
            "Default: out/<model>/ (or EXP2_ABLATION_OUTPUT_DIR if set)."
        ),
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=_env_int("EXP2_ABLATION_NUM_SAMPLES", DEFAULT_NUM_SAMPLES),
        help=(
            "Sampled reasoning paths per item. Default 15 matches exp2's MALLM budget "
            "(num_agents=3 x max_turns=5)."
        ),
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=_env_float("EXP2_ABLATION_TEMPERATURE", DEFAULT_TEMPERATURE),
        help="Sampling temperature for self-consistency (default 0.8).",
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=_env_float("EXP2_ABLATION_TOP_P", DEFAULT_TOP_P),
        help="Nucleus sampling top_p for self-consistency (default 0.95).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=_env_int("EXP2_ABLATION_SEED", DEFAULT_SEED),
        help=(
            "Base seed; sample k uses seed+k when the endpoint supports it. "
            "Negative disables per-sample seeds."
        ),
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=None,
        help="Max tokens per sampled completion (default: MALLM_MAX_TOKENS or 1024, as exp2).",
    )
    parser.add_argument(
        "--concurrent_requests",
        type=int,
        default=None,
        help="Max parallel chat completions (default: MALLM_CONCURRENT_API_REQUESTS or 64).",
    )

    args = parser.parse_args()
    resolved_model_name = normalize_model_name(args.model_name)
    if args.num_samples < 1:
        raise SystemExit("--num_samples must be >= 1")
    seed = None if args.seed is not None and args.seed < 0 else args.seed

    sc_kwargs = dict(
        num_samples=args.num_samples,
        temperature=args.temperature,
        top_p=args.top_p,
        seed=seed,
        max_tokens=args.max_tokens,
        concurrent_requests=args.concurrent_requests,
    )

    if args.mock:
        run_self_consistency(
            True,
            "mock://local",
            resolved_model_name,
            debug_mode=args.debug,
            dataset_filter=args.dataset,
            continue_mode=args.continue_mode,
            output_dir=args.output_dir,
            **sc_kwargs,
        )
        return

    # Resolve endpoint from env overrides and verify availability; auto-discover if needed.
    resolved_endpoint = exp2_base._env_or_default_endpoint(args.endpoint_url)
    if resolved_endpoint.startswith("http") and not exp2_base._endpoint_alive(resolved_endpoint, timeout=1.5):
        auto = exp2_base._discover_endpoint_from_mallm(resolved_model_name)
        if auto:
            print(f"[INFO] Provided endpoint '{resolved_endpoint}' is unreachable. Using discovered endpoint '{auto}' instead.")
            resolved_endpoint = auto
        else:
            print(f"[WARN] Endpoint '{resolved_endpoint}' is unreachable.")
            if not args.no_autostart:
                try:
                    host_port = re.search(r"https?://([^:/]+):(\d+)/v1", resolved_endpoint)
                    desired_port = 8080
                    if host_port and host_port.group(1) in ("127.0.0.1", "localhost"):
                        try:
                            desired_port = int(host_port.group(2))
                        except Exception:
                            desired_port = 8080
                    started = exp2_base._start_model_server(resolved_model_name, desired_port=desired_port)
                    if started:
                        resolved_endpoint = started["endpoint_url"]  # type: ignore[index]
                    else:
                        print("[ERROR] Could not auto-start a model server. Please start one manually or provide a reachable --endpoint_url.")
                except Exception as e:
                    print(f"[ERROR] Auto-start failed: {e}. Please start a model server or set --no_autostart.")
            else:
                print("[INFO] Auto-start disabled (--no_autostart).")
    # Fail fast: do not run when the endpoint is still unreachable.
    if resolved_endpoint.startswith("http") and not exp2_base._endpoint_alive(resolved_endpoint, timeout=1.5):
        raise SystemExit(
            "[FATAL] No reachable model endpoint. "
            "Start a server manually, pass a reachable --endpoint_url, "
            "or remove --no_autostart to let exp2_ablation try starting one."
        )
    run_self_consistency(
        False,
        resolved_endpoint,
        resolved_model_name,
        debug_mode=args.debug,
        dataset_filter=args.dataset,
        continue_mode=args.continue_mode,
        output_dir=args.output_dir,
        **sc_kwargs,
    )


if __name__ == "__main__":
    main()
