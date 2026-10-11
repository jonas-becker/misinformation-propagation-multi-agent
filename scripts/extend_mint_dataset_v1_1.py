#!/usr/bin/env python3
"""Extend MINT_dataset_v1 with irrelevant true information from Wikimedia Wikipedia.

For each sample:
- Compute average token length across all entries in misinformation_by_strategy.
- Sample one unique text from facebook/wiki_dpr with similar token length.
- Optionally trim sampled text to sentence boundary near target length.

Outputs updated files to MINT-dataset_v1.1 (or custom output dir).
"""

from __future__ import annotations

import argparse
import bisect
import json
import random
import re
from pathlib import Path
from typing import Dict, List, Tuple

from datasets import load_dataset


TOKEN_RE = re.compile(r"\S+")
GOOD_SENTENCE_START_RE = re.compile(r'^[A-Z0-9"\']')
COMMON_ABBREVIATIONS = {
    "mr.",
    "mrs.",
    "ms.",
    "dr.",
    "prof.",
    "sr.",
    "jr.",
    "st.",
    "mt.",
    "etc.",
    "vs.",
    "e.g.",
    "i.e.",
    "u.s.",
    "u.k.",
}


def tokenize_len(text: str) -> int:
    return len(TOKEN_RE.findall(text or ""))


def split_sentences_robust(text: str) -> List[str]:
    """Split text into sentences while avoiding common false boundaries.

    Handles frequent abbreviation and initial patterns so strings like
    "St. Petersburg" do not trigger incorrect sentence breaks.
    """
    text = (text or "").strip()
    if not text:
        return []

    boundaries: List[int] = []
    n = len(text)
    for i, ch in enumerate(text):
        if ch not in ".!?":
            continue
        if i == n - 1:
            boundaries.append(i + 1)
            continue

        nxt = text[i + 1]
        if not nxt.isspace():
            continue

        left = text[: i + 1]
        word_match = re.search(r"([A-Za-z](?:[A-Za-z.]*[A-Za-z])?\.)$", left)
        last_word = word_match.group(1).lower() if word_match else ""

        # Avoid splitting on common abbreviations and initials.
        if last_word in COMMON_ABBREVIATIONS:
            continue
        if re.fullmatch(r"[A-Za-z]\.", last_word):
            continue
        if re.fullmatch(r"(?:[A-Za-z]\.){2,}", last_word):
            continue

        # Avoid decimal-number boundaries.
        prev_char = text[i - 1] if i > 0 else ""
        next_non_space_idx = i + 1
        while next_non_space_idx < n and text[next_non_space_idx].isspace():
            next_non_space_idx += 1
        next_char = text[next_non_space_idx] if next_non_space_idx < n else ""
        if ch == "." and prev_char.isdigit() and next_char.isdigit():
            continue

        boundaries.append(i + 1)

    if not boundaries:
        return [text]

    out: List[str] = []
    start = 0
    for end in boundaries:
        seg = text[start:end].strip()
        if seg:
            out.append(seg)
        start = end
    tail = text[start:].strip()
    if tail:
        out.append(tail)
    return out


def trim_to_sentence_target(text: str, target_tokens: int) -> str:
    """Trim text to a sentence boundary near target token count."""
    text = (text or "").strip()
    if not text:
        return text
    if tokenize_len(text) <= target_tokens:
        return text

    sentences = split_sentences_robust(text)
    if not sentences:
        toks = TOKEN_RE.findall(text)[: max(1, target_tokens)]
        out = " ".join(toks).strip()
        return out if out.endswith((".", "!", "?")) else f"{out}."

    built: List[str] = []
    cur_tokens = 0
    for sent in sentences:
        st = tokenize_len(sent)
        if built and cur_tokens + st > target_tokens:
            break
        built.append(sent)
        cur_tokens += st
        if cur_tokens >= target_tokens:
            break

    if not built:
        toks = TOKEN_RE.findall(sentences[0])[: max(1, target_tokens)]
        out = " ".join(toks).strip()
        return out if out.endswith((".", "!", "?")) else f"{out}."

    out = " ".join(built).strip()
    return out if out.endswith((".", "!", "?")) else f"{out}."


def normalize_fragment_start(text: str, min_tokens: int = 8) -> str:
    """Try to avoid obvious mid-sentence starts from passage chunks.

    DPR passages are fixed-size blocks and can begin with trailing connective fragments
    (e.g., 'and that ...'). We prefer the first sentence-like segment that starts with
    an uppercase letter/quote/number and has enough tokens.
    """
    text = (text or "").strip()
    if not text:
        return text

    sentences = split_sentences_robust(text)
    if not sentences:
        return text

    # If already sentence-like, keep as-is.
    if GOOD_SENTENCE_START_RE.match(sentences[0]):
        return text

    # Otherwise, move to first sentence-like segment with minimal content.
    for i, sent in enumerate(sentences):
        if GOOD_SENTENCE_START_RE.match(sent) and tokenize_len(sent) >= min_tokens:
            out = " ".join(sentences[i:]).strip()
            return out if out.endswith((".", "!", "?")) else f"{out}."

    return text


def avg_misinfo_tokens(sample: Dict) -> int:
    strat = sample.get("misinformation_by_strategy") or {}
    if not isinstance(strat, dict) or not strat:
        return max(1, tokenize_len(str(sample.get("false_fact", ""))))
    lens = [tokenize_len(str(v)) for v in strat.values()]
    return max(1, int(round(sum(lens) / max(1, len(lens)))))


def collect_targets(
    files: List[Path],
    limit_per_dataset: int | None,
) -> Tuple[List[Tuple[str, int, int]], Dict[str, List[Dict]]]:
    """Return flat targets and loaded data by filename.

    target tuple: (filename, sample_index, target_tokens)
    """
    loaded: Dict[str, List[Dict]] = {}
    targets: List[Tuple[str, int, int]] = []
    for p in files:
        data = json.loads(p.read_text(encoding="utf-8"))
        if not isinstance(data, list):
            raise ValueError(f"Expected list in {p}")
        if limit_per_dataset is not None:
            data = data[:limit_per_dataset]
        loaded[p.name] = data
        for i, sample in enumerate(data):
            targets.append((p.name, i, avg_misinfo_tokens(sample)))
    return targets, loaded


def _iter_sentence_chunks(
    text: str,
    *,
    min_tokens: int,
    max_tokens: int,
) -> List[str]:
    """Create sentence-bounded chunks from full article text."""
    sentences = split_sentences_robust(text)
    if not sentences:
        return []

    chunks: List[str] = []
    i = 0
    n = len(sentences)
    while i < n:
        if not GOOD_SENTENCE_START_RE.match(sentences[i]):
            i += 1
            continue
        buf: List[str] = []
        cur = 0
        j = i
        emitted = 0
        while j < n:
            s = sentences[j]
            st = tokenize_len(s)
            if st == 0:
                j += 1
                continue
            if cur + st > max_tokens and buf:
                break
            if cur + st > max_tokens and not buf:
                # Single very long sentence; hard-trim.
                toks = TOKEN_RE.findall(s)[:max_tokens]
                s = " ".join(toks).strip()
                st = tokenize_len(s)
            buf.append(s)
            cur += st
            j += 1
            # Emit multiple sizes per start position to better cover target lengths.
            if cur >= min_tokens:
                out = " ".join(buf).strip()
                if out and not out.endswith((".", "!", "?")):
                    out += "."
                chunks.append(out)
                emitted += 1
        # Advance with overlap to retain diversity while avoiding explosion.
        if emitted > 0:
            i += 2
        else:
            i += 1
    return chunks


def load_wiki_pool(
    needed: int,
    seed: int,
    max_factor: float,
    min_tokens: int,
    max_tokens: int,
) -> List[str]:
    """Stream Wikimedia Wikipedia and collect sentence-bounded candidate texts."""
    rng = random.Random(seed)
    target_pool = int(max(needed * max_factor, needed + 1000))
    dataset = load_dataset(
        "wikimedia/wikipedia",
        name="20231101.en",
        split="train",
        streaming=True,
    )

    pool: List[str] = []
    seen = set()
    for row in dataset:
        article = str(row.get("text", "")).strip()
        if not article:
            continue
        for chunk in _iter_sentence_chunks(article, min_tokens=min_tokens, max_tokens=max_tokens):
            tlen = tokenize_len(chunk)
            if tlen < min_tokens or tlen > max_tokens:
                continue
            if chunk in seen:
                continue
            seen.add(chunk)
            pool.append(chunk)
            if len(pool) >= target_pool:
                break
        if len(pool) >= target_pool:
            break

    rng.shuffle(pool)
    if len(pool) < needed:
        raise RuntimeError(f"Not enough candidate passages collected: {len(pool)} < {needed}")
    return pool


def assign_unique_by_nearest(
    targets: List[Tuple[str, int, int]],
    pool: List[str],
) -> Dict[Tuple[str, int], str]:
    """Assign each target a unique pool text nearest in token length."""
    pool_lens = [tokenize_len(t) for t in pool]
    order = sorted(range(len(pool)), key=lambda i: pool_lens[i])
    sorted_lens = [pool_lens[i] for i in order]
    used = [False] * len(order)
    used_texts = set()

    out: Dict[Tuple[str, int], str] = {}
    for fname, idx, tgt in targets:
        pos = bisect.bisect_left(sorted_lens, tgt)
        candidates: List[int] = []
        max_span = len(order)
        for step in range(max_span):
            li = pos - 1 - step
            ri = pos + step
            if li >= 0:
                candidates.append(li)
            if ri < len(order):
                candidates.append(ri)
            if li < 0 and ri >= len(order):
                break

        chosen = None
        chosen_text = None
        for cand in candidates:
            if used[cand]:
                continue
            src_text = pool[order[cand]]
            normalized = normalize_fragment_start(trim_to_sentence_target(src_text, tgt))
            if normalized in used_texts:
                continue
            chosen = cand
            chosen_text = normalized
            break

        if chosen is None or chosen_text is None:
            raise RuntimeError("Ran out of unique pool samples during assignment")
        used[chosen] = True
        used_texts.add(chosen_text)
        out[(fname, idx)] = chosen_text
    return out


def mint_id_for_sample(file_name: str, sample_index: int) -> str:
    """Build a deterministic globally unique ID for each MINT v1.1 sample."""
    dataset_name = Path(file_name).stem.replace("_misinformed", "").upper()
    return f"MINT_ID_{dataset_name}_{sample_index + 1:06d}"


def infer_answer_value(sample: Dict, file_name: str) -> object:
    """Return a harmonized top-level answer value from subset-specific fields."""
    if "answer" in sample:
        value = sample.get("answer")
    elif "answers" in sample:
        value = sample.get("answers")
    else:
        value = None

    # Keep richer CWQ answer structures as-is.
    if file_name == "complex_web_questions_misinformed.json":
        return value

    # Other subsets should always expose answer as string when present.
    if value is None:
        return None
    if isinstance(value, str):
        return value
    return str(value)


def infer_options_value(sample: Dict) -> object:
    """Return a harmonized top-level options value (None if unavailable)."""
    if "options" in sample:
        return sample.get("options")
    return None


def normalize_multiple_choice_answer(sample: Dict, file_name: str) -> None:
    """Normalize multiple-choice answer indexing to 0-based (0/1 for binary)."""
    if file_name == "complex_web_questions_misinformed.json":
        return

    options = sample.get("options")
    if not isinstance(options, list) or not options:
        return

    value = sample.get("answer")
    if isinstance(value, int):
        # Keep already 0-based values.
        if 0 <= value < len(options):
            sample["answer"] = str(value)
            return
        # Convert likely 1-based values.
        if 1 <= value <= len(options):
            sample["answer"] = str(value - 1)
            return
        return

    if isinstance(value, str):
        raw = value.strip()
        if not raw.isdigit():
            return
        idx = int(raw)
        if 0 <= idx < len(options):
            sample["answer"] = str(idx)
            return
        if 1 <= idx <= len(options):
            sample["answer"] = str(idx - 1)
            return


def harmonize_subset_fields(loaded: Dict[str, List[Dict]]) -> List[str]:
    """Ensure all subset samples share top-level fields, moving extras to metadata."""
    for file_name, data in loaded.items():
        for sample in data:
            sample["answer"] = infer_answer_value(sample, file_name)
            sample["options"] = infer_options_value(sample)

    per_subset_keys: List[set[str]] = []
    for data in loaded.values():
        subset_keys: set[str] = set()
        for sample in data:
            subset_keys.update(sample.keys())
        per_subset_keys.append(subset_keys)

    if not per_subset_keys:
        return []

    common_keys = set.intersection(*per_subset_keys)
    common_keys.add("metadata")

    # Keep a stable and readable key order in output JSON.
    preferred_order = [
        "MINT_ID",
        "id",
        "sentence",
        "answer",
        "options",
        "false_fact",
        "misinformation_by_strategy",
        "irrelevant_true_information",
        "avg_misinformation_tokens",
        "irrelevant_true_information_tokens",
        "metadata",
    ]
    ordered_common = [k for k in preferred_order if k in common_keys]
    ordered_common.extend(sorted(k for k in common_keys if k not in ordered_common))
    common_key_set = set(ordered_common)

    for fname, data in loaded.items():
        for i, sample in enumerate(data):
            existing_meta = sample.get("metadata")
            if isinstance(existing_meta, dict):
                metadata = dict(existing_meta)
            else:
                metadata = {}

            extras = {
                k: v
                for k, v in sample.items()
                if k not in common_key_set and k not in {"metadata", "answers", "options"}
            }
            metadata.update(extras)

            normalized = {k: sample.get(k) for k in ordered_common if k != "metadata"}
            normalized["metadata"] = metadata
            sample.clear()
            sample.update(normalized)

    return ordered_common


def main() -> None:
    parser = argparse.ArgumentParser(description="Create MINT-dataset_v1.1 with irrelevant true information")
    parser.add_argument("--input_dir", type=str, default="MINT_dataset_v1")
    parser.add_argument("--output_dir", type=str, default="MINT-dataset_v1.1")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit_per_dataset", type=int, default=None)
    parser.add_argument("--pool_factor", type=float, default=4.0, help="candidate pool size = needed * factor")
    parser.add_argument("--min_tokens", type=int, default=18)
    parser.add_argument("--max_tokens", type=int, default=520)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    _ = rng  # deterministic seed retained for future extension.

    in_dir = Path(args.input_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    files = [
        in_dir / "complex_web_questions_misinformed.json",
        in_dir / "ethics_commonsense_misinformed.json",
        in_dir / "winogrande_misinformed.json",
    ]
    for f in files:
        if not f.exists():
            raise FileNotFoundError(f"Missing expected dataset file: {f}")

    targets, loaded = collect_targets(files, args.limit_per_dataset)
    if not targets:
        raise RuntimeError("No targets found to process")

    print(f"Need unique irrelevant texts: {len(targets)}")
    pool = load_wiki_pool(
        needed=len(targets),
        seed=args.seed,
        max_factor=args.pool_factor,
        min_tokens=args.min_tokens,
        max_tokens=args.max_tokens,
    )
    print(f"Collected pool size: {len(pool)}")
    assignment = assign_unique_by_nearest(targets, pool)

    for fname, data in loaded.items():
        for i, sample in enumerate(data):
            irr = assignment[(fname, i)]
            avg_toks = avg_misinfo_tokens(sample)
            normalize_multiple_choice_answer(sample, fname)
            sample["MINT_ID"] = mint_id_for_sample(fname, i)
            sample["irrelevant_true_information"] = irr
            sample["avg_misinformation_tokens"] = avg_toks
            sample["irrelevant_true_information_tokens"] = tokenize_len(irr)

    common_fields = harmonize_subset_fields(loaded)
    print(f"Harmonized subset fields: {common_fields}")

    for fname, data in loaded.items():
        out_path = out_dir / fname
        out_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Wrote {out_path} ({len(data)} rows)")

    print("Done.")


if __name__ == "__main__":
    main()

