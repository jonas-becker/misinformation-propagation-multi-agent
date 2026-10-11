#!/usr/bin/env python3
"""
answer_extraction.py - Unified answer extraction with explicit extraction status.

Every extracted answer carries a status:
  parsed     - exactly one answer candidate was identified
  ambiguous  - several conflicting candidates; a best guess is returned but the
               answer is treated as a failure by default (counted incorrect)
  unparsed   - no candidate found (empty output, refusal, truncated reasoning)

Design notes
  * gpt-oss served without a reasoning parser streams its analysis channel into
    `content`, so answers look like 'We need to output just "A".A'. GLM-4.7
    emits '<think>...</think>A) Felicia'. We therefore (1) take the text after
    the last reasoning delimiter, (2) prefer text after the LAST explicit answer
    marker, (3) accept a trailing stand-alone letter, (4) accept decisive
    statements naming an option, (5) accept option text with word boundaries.
  * Option-text matching uses word boundaries so 'Ethical' never matches inside
    'Unethical'.
  * Free-form (CWQ) answers: `strict` scoring uses only the text after the last
    marker ('Final Solution:' / 'Final Answer:' / 'Answer:'); `lenient` scoring
    uses the whole response (used when no answer marker was requested).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, asdict
from typing import List, Optional, Sequence

STATUS_PARSED = "parsed"
STATUS_AMBIGUOUS = "ambiguous"
STATUS_UNPARSED = "unparsed"

# Markers that introduce a final answer. We use the LAST occurrence.
_MARKER_RE = re.compile(
    r"(?:final\s+(?:solution|answer|choice)\**\s*(?:is\s*)?[:\-]|"
    r"(?:the\s+)?(?:correct|best|right)\s+answer\s+is|the\s+answer\s+is|my\s+answer\s+is|"
    r"(?:correct\s+)?answer\s*[:\-]|solution\s*[:\-]|conclusion\s*[:\-]|"
    r"the\s+blank\s+(?:must\s+be|should\s+be|is|refers\s+to))\**\s*",
    re.IGNORECASE,
)
# Reasoning delimiters: text after the LAST closing tag is the answer channel.
_THINK_CLOSE_RE = re.compile(r"</think>|<\|channel\|>final<\|message\|>", re.IGNORECASE)
# "X is the correct answer" style decisive statements (option text substituted).
_DECISIVE_OPTION_TMPL = (
    r"(?<!\w){opt}(?!\w)\W{{0,3}}\s*(?:is|was)\s+(?:the\s+)?(?:correct|right|best)\s+(?:answer|option|choice)"
)
_BOXED_RE = re.compile(r"\\boxed\{\s*\(?([A-Ea-e])\)?\s*\}")
_REFUSAL_RE = re.compile(
    r"\b(i\s+cannot|i\s+can't|i\s+am\s+unable|i'm\s+unable|as\s+an\s+ai|i\s+won't\s+be\s+able)\b",
    re.IGNORECASE,
)

# Letter candidates. Uppercase only for bare letters to avoid the article "a".
_LETTER_PATTERNS = [
    # "option A", "choice B"
    re.compile(r"\b(?:option|choice)\s*\(?([A-E])\)?(?![A-Za-z])", re.IGNORECASE),
    # "A)" or "(A)"
    re.compile(r"(?<![A-Za-z0-9])\(?([A-E])\)(?![A-Za-z])"),
    # "**A**"
    re.compile(r"\*\*\s*\(?([A-E])\)?\s*\*\*"),
    # "[A]"
    re.compile(r"\[\s*([A-E])\s*\]"),
    # "A." or "A:" at start of a line/segment
    re.compile(r"(?:^|\n)\s*([A-E])\s*[.:](?!\S)"),
]
# Trailing stand-alone letter, possibly glued to preceding text: '...no extra.B', '..."B".B'
_TRAILING_LETTER_RE = re.compile(r"(?:^|[\s.!?\"'*)\]:>,;])\(?([A-E])\)?[.!]?\s*[*_`\"']*\s*$")
# Bare letter token (only used for very short segments)
_BARE_LETTER_RE = re.compile(r"(?<![A-Za-z])([A-E])(?![A-Za-z])")

_ARTICLES_RE = re.compile(r"\b(a|an|the)\b", re.IGNORECASE)


@dataclass
class Extraction:
    index: Optional[int]  # option index for multiple choice, None otherwise
    text: Optional[str]  # extracted answer text (option text or free-form answer)
    status: str  # parsed | ambiguous | unparsed
    method: str  # how the answer was found (for diagnostics)
    candidates: Optional[List[int]] = None
    segment: Optional[str] = None  # text segment the answer was taken from

    def as_dict(self) -> dict:
        return asdict(self)


def split_reasoning(text: str) -> tuple:
    """Return (answer_channel_text, had_delimiter). Uses the last </think>-style tag."""
    if not text:
        return text, False
    matches = list(_THINK_CLOSE_RE.finditer(text))
    if not matches:
        return text, False
    tail = text[matches[-1].end():].strip()
    return (tail if tail else text), True


def is_refusal(text: str) -> bool:
    return bool(text) and bool(_REFUSAL_RE.search(text))


def normalize_text(s: str) -> str:
    s = str(s or "").lower().strip()
    s = _ARTICLES_RE.sub(" ", s)
    s = re.sub(r"[^\w\s]", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def last_marker_segment(text: str) -> Optional[str]:
    """Text after the last explicit answer marker (first non-empty line), or None."""
    if not text:
        return None
    matches = list(_MARKER_RE.finditer(text))
    if not matches:
        return None
    seg = text[matches[-1].end():].strip()
    if not seg:
        return None
    for line in seg.splitlines():
        line = line.strip().strip("*_`\"' ")
        if line:
            return line
    return None


def _letter_candidates(segment: str, n_options: int) -> List[int]:
    found: List[int] = []
    for pat in _LETTER_PATTERNS:
        for m in pat.finditer(segment):
            idx = ord(m.group(1).upper()) - ord("A")
            if 0 <= idx < n_options:
                found.append(idx)
    return found


def _option_text_candidates(segment: str, options: Sequence[str]) -> List[int]:
    found: List[int] = []
    seg_l = segment.lower()
    for i, opt in enumerate(options):
        o = str(opt).strip().lower()
        if not o:
            continue
        if re.search(r"(?<!\w)" + re.escape(o) + r"(?!\w)", seg_l):
            found.append(i)
    return found


def _unique(seq: List[int]) -> List[int]:
    out: List[int] = []
    for x in seq:
        if x not in out:
            out.append(x)
    return out


def extract_mc_answer(response: Optional[str], options: Sequence[str]) -> Extraction:
    """Extract a multiple-choice answer (option index) with a status."""
    n = len(options)
    text = (response or "").strip() if isinstance(response, str) else ""
    if not text or n == 0:
        return Extraction(None, None, STATUS_UNPARSED, "empty")
    text, _had_delim = split_reasoning(text)

    # 1) \boxed{A}
    boxed = _BOXED_RE.findall(text)
    if boxed:
        idx = ord(boxed[-1].upper()) - ord("A")
        if 0 <= idx < n:
            return Extraction(idx, str(options[idx]), STATUS_PARSED, "boxed", [idx], text[-80:])

    # 2) Segment after the last explicit marker
    seg = last_marker_segment(text)
    if seg is not None:
        cands = _unique(_letter_candidates(seg, n))
        if not cands:
            m0 = re.match(r"^\(?([A-E])\)?(?![A-Za-z])", seg)
            if m0 and 0 <= ord(m0.group(1)) - ord("A") < n:
                cands = [ord(m0.group(1)) - ord("A")]
        if not cands and len(seg) <= 40:
            cands = _unique(
                [ord(m.group(1)) - ord("A") for m in _BARE_LETTER_RE.finditer(seg)
                 if 0 <= ord(m.group(1)) - ord("A") < n]
            )
        if not cands:
            cands = _unique(_option_text_candidates(seg, options))
        if len(cands) == 1:
            return Extraction(cands[0], str(options[cands[0]]), STATUS_PARSED, "marker", cands, seg)
        if len(cands) > 1:
            return Extraction(cands[0], str(options[cands[0]]), STATUS_AMBIGUOUS, "marker_multi", cands, seg)

    # 3) Trailing stand-alone letter (gpt-oss style: '...Ensure no extra text.B')
    m = _TRAILING_LETTER_RE.search(text)
    if m:
        idx = ord(m.group(1)) - ord("A")
        if 0 <= idx < n:
            return Extraction(idx, str(options[idx]), STATUS_PARSED, "trailing_letter", [idx], text[-60:])

    # 4) Letter patterns in the tail of the response (last 300 chars)
    tail = text[-300:]
    cands = _unique(_letter_candidates(tail, n))
    if len(cands) == 1:
        return Extraction(cands[0], str(options[cands[0]]), STATUS_PARSED, "tail_letter", cands, tail[-120:])
    if len(cands) > 1:
        last = cands[-1]
        return Extraction(last, str(options[last]), STATUS_AMBIGUOUS, "tail_letter_multi", cands, tail[-120:])

    # 5) Decisive statements naming an option: "Samuel is the correct answer"
    dec: List[int] = []
    tl = tail.lower()
    for i, opt in enumerate(options):
        o = re.escape(str(opt).strip().lower())
        if o and re.search(_DECISIVE_OPTION_TMPL.format(opt=o), tl):
            dec.append(i)
    dec = _unique(dec)
    if len(dec) == 1:
        return Extraction(dec[0], str(options[dec[0]]), STATUS_PARSED, "decisive_option", dec, tail[-120:])

    # 6) Option text in the tail (word boundaries)
    tcands = _unique(_option_text_candidates(tail, options))
    if len(tcands) == 1:
        return Extraction(tcands[0], str(options[tcands[0]]), STATUS_PARSED, "tail_option_text", tcands, tail[-120:])
    if len(tcands) > 1:
        last_pos = -1
        best = tcands[0]
        for i in tcands:
            pos = tl.rfind(str(options[i]).lower())
            if pos > last_pos:
                last_pos, best = pos, i
        return Extraction(best, str(options[best]), STATUS_AMBIGUOUS, "tail_option_text_multi", tcands, tail[-120:])

    # 7) Very short response that is just a letter
    if len(text) <= 8:
        bl = [ord(x.group(1)) - ord("A") for x in _BARE_LETTER_RE.finditer(text)]
        bl = [i for i in bl if 0 <= i < n]
        if len(_unique(bl)) == 1:
            return Extraction(bl[0], str(options[bl[0]]), STATUS_PARSED, "bare_short", bl, text)

    return Extraction(None, None, STATUS_UNPARSED, "refusal" if is_refusal(text) else "none", [], text[-120:])


def extract_free_form_answer(response: Optional[str]) -> Extraction:
    """Free-form answer: text after the last marker if present, else the whole response."""
    text = (response or "").strip() if isinstance(response, str) else ""
    if not text:
        return Extraction(None, None, STATUS_UNPARSED, "empty")
    text, _had_delim = split_reasoning(text)
    seg = last_marker_segment(text)
    if seg:
        return Extraction(None, seg, STATUS_PARSED, "marker", None, seg)
    if is_refusal(text):
        return Extraction(None, text, STATUS_UNPARSED, "refusal", None, text[-120:])
    return Extraction(None, text, STATUS_PARSED, "whole_response", None, text[-200:])


def free_form_matches(pred_text: Optional[str], golds: Sequence[str]) -> bool:
    """Normalized whole-word containment of any gold answer/alias in pred_text."""
    if not pred_text:
        return False
    p = normalize_text(pred_text)
    if not p:
        return False
    for g in golds:
        gn = normalize_text(g)
        if not gn:
            continue
        if re.search(r"(?<!\w)" + re.escape(gn) + r"(?!\w)", p):
            return True
    return False


def score_free_form(response: Optional[str], golds: Sequence[str]) -> dict:
    """Return lenient (whole response) and strict (marker segment) correctness."""
    ex = extract_free_form_answer(response)
    lenient = free_form_matches(response, golds) if ex.status != STATUS_UNPARSED else False
    strict = free_form_matches(ex.text, golds) if ex.method == "marker" else None
    return {
        "predicted_text": ex.text,
        "status": ex.status,
        "method": ex.method,
        "is_correct_lenient": bool(lenient),
        "is_correct_strict": strict,
        "has_marker": ex.method == "marker",
    }


def letter_for_index(idx: Optional[int]) -> Optional[str]:
    if idx is None:
        return None
    return chr(ord("A") + int(idx))


if __name__ == "__main__":
    tests = [
        ('We need to output just "A".A', ["William", "Nelson"]),
        ('We need to output only "B". Ensure no extra spaces or text.B', ["Patricia", "Betty"]),
        ("A) Jeffrey", ["Jeffrey", "Randy"]),
        ("Final Solution: B", ["Ethical", "Unethical"]),
        ("The correct answer is B) Maria. The sentence should read ...", ["Amy", "Maria"]),
        ("**Answer: A) William**", ["William", "Nelson"]),
        ("B", ["Hunter", "Neil"]),
        ("The action described in the scenario is unethical because ...", ["Ethical", "Unethical"]),
        ("This is clearly ethical.", ["Ethical", "Unethical"]),
        ("1.  **Analyze the Request:** ... Final Solution: A", ["a", "b"]),
        ("The user says: Task: Correctly fill the blank", ["a", "b"]),
        ("", ["a", "b"]),
        ("...Result: A) Felicia.</think>A) Felicia", ["Felicia", "Betty"]),
        ("4. Deduce: Therefore, the blank must be Samuel.\n\n5. **Conclusion:** Samuel is the correct answer.",
         ["Hunter", "Samuel"]),
        ("The best answer is B | Unethical | You're being unreasonable", ["Ethical", "Unethical"]),
        ('few people go to the **pool** these days. Hence, the blank must be "pool."', ["playground", "pool"]),
        ("Lawrence reminded Kyle to wash his hands, because B) Kyle has sensitive skin.", ["Lawrence", "Kyle"]),
    ]
    for t, opts in tests:
        ex = extract_mc_answer(t, opts)
        print(f"{ex.status:9s} {ex.method:22s} idx={ex.index} | {t[:70]!r}")
    print(score_free_form("The country is Sweden.\nFinal Solution: Swedish krona", ["Swedish krona"]))
    print(score_free_form("The currency is the Chilean Peso.", ["Chilean peso"]))
