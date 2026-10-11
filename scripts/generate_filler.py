#!/usr/bin/env python3
"""
generate_filler.py - Content-free, length-matched filler control.

The existing irrelevant-true control (MINT v1.1 `irrelevant_true_information`) is
length-matched encyclopedia text about an unrelated topic; it therefore carries
*content* (facts, entities) even though it is irrelevant to the question. To test
whether the irrelevant-true gain is a pure length / deliberation effect, this
script adds `filler_information`: a passage assembled deterministically from a
fixed bank of topic-neutral, entity-free sentences (about formatting conventions,
units, generic procedures) and trimmed to the item's mean misinformation length
(`avg_misinformation_tokens`, whitespace tokens, same convention as v1.1).

Writes the field into MINT_dataset_v1.2/<dataset>_misinformed.json in place (or --out_dir).
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from mint_data import DATASET_FILES, resolve_dataset_path  # noqa: E402

FILLER_BANK = [
    "A standard sheet of paper in the A4 format measures 210 by 297 millimetres.",
    "Most word processors place a default margin of about one inch on every side of a page.",
    "A minute contains sixty seconds, and an hour contains sixty minutes.",
    "The metric system expresses length in metres and mass in kilograms.",
    "A typical paragraph begins with an indented first line or is separated by a blank line.",
    "Page numbers are commonly printed in the footer, centred or aligned to the outer edge.",
    "A leap year adds one day to February, giving the month twenty-nine days.",
    "Bullet points are often used to list items whose order does not matter.",
    "A kilometre is equal to one thousand metres.",
    "Line spacing of one and a half lines is a common setting for drafts.",
    "The week is divided into seven days, five of which are usually working days in many offices.",
    "A standard deck of playing cards is organised into four suits.",
    "Headings are typically set in a larger or bolder typeface than body text.",
    "A litre of water at room temperature weighs approximately one kilogram.",
    "Footnotes appear at the bottom of the page on which they are referenced.",
    "Temperatures are reported in degrees Celsius in most countries and in degrees Fahrenheit in a few others.",
    "Tables usually have a header row that names the content of each column.",
    "The alphabet used for English has twenty-six letters.",
    "A byte consists of eight bits.",
    "Abbreviations are conventionally explained the first time they are used in a document.",
    "A dozen refers to a group of twelve items.",
    "Documents are often saved in portable formats so that they render identically on different devices.",
    "A right angle measures ninety degrees.",
    "Serif typefaces have small strokes at the ends of letters, whereas sans-serif typefaces do not.",
    "Quotation marks enclose text that is reproduced verbatim from another source.",
    "The number of days in a year is 365, or 366 in a leap year.",
    "A colon may introduce a list, an explanation or a quotation.",
    "An index at the end of a book lists topics alphabetically with their page numbers.",
    "A gram is one thousandth of a kilogram.",
    "Captions describe figures and tables and are usually placed directly below or above them.",
    "Wide tables can be rotated to landscape orientation so that all columns fit on one page.",
    "Ordinal numbers such as first, second and third indicate position in a sequence.",
    "A hyphen joins words, while a longer dash separates parts of a sentence.",
    "Appendices contain supplementary material that would interrupt the main text.",
    "Percentages express a quantity as a fraction of one hundred.",
    "A glossary defines specialised terms used in a document.",
    "Roman numerals are sometimes used to number the front matter of a book.",
    "A centimetre contains ten millimetres.",
    "Line breaks separate lines within a paragraph without starting a new paragraph.",
    "The twenty-four-hour clock counts hours from zero to twenty-three.",
]
_TOKEN_RE = re.compile(r"\S+")


def _stable_seed(*parts) -> int:
    import hashlib

    return int(hashlib.md5("::".join(str(p) for p in parts).encode("utf-8")).hexdigest()[:8], 16)


def make_filler(seed: int, target_tokens: int) -> str:
    rng = random.Random(seed)
    bank = FILLER_BANK[:]
    rng.shuffle(bank)
    words: list = []
    i = 0
    while len(words) < target_tokens and i < 5 * len(bank):
        words += _TOKEN_RE.findall(bank[i % len(bank)])
        i += 1
    return " ".join(words[: max(1, target_tokens)])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data_dir", default=str(REPO_ROOT / "MINT_dataset_v1.2"))
    ap.add_argument("--out_dir", default=None, help="default: overwrite files in --data_dir")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    out_dir = Path(args.out_dir) if args.out_dir else Path(args.data_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for ds, fname in DATASET_FILES.items():
        path = resolve_dataset_path(ds, args.data_dir)
        items = json.load(open(path, encoding="utf-8"))
        lens = []
        for it in items:
            target = it.get("avg_misinformation_tokens")
            if not isinstance(target, int) or target <= 0:
                texts = [it.get("false_fact") or ""] + list((it.get("misinformation_by_strategy") or {}).values())
                target = max(1, round(sum(len(_TOKEN_RE.findall(t)) for t in texts) / max(1, len(texts))))
            fill = make_filler(_stable_seed(args.seed, it.get("MINT_ID")), target)
            it["filler_information"] = fill
            it["filler_information_tokens"] = len(_TOKEN_RE.findall(fill))
            lens.append((target, it["filler_information_tokens"]))
        with open(out_dir / fname, "w", encoding="utf-8") as f:
            json.dump(items, f, indent=1, ensure_ascii=False)
        mean_t = sum(t for t, _ in lens) / len(lens)
        mean_f = sum(f for _, f in lens) / len(lens)
        print(f"{ds}: n={len(items)} target tokens mean={mean_t:.1f} filler tokens mean={mean_f:.1f} -> {out_dir / fname}")


if __name__ == "__main__":
    main()
