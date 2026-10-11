#!/usr/bin/env python3
"""Shared helpers for the analysis scripts (paths, names, table writers)."""

from __future__ import annotations

import csv
import json
import re
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

OUT_DIR = REPO_ROOT / "out"
RESULTS_DIR = OUT_DIR / "results"     # CSVs with the numbers behind the tables and figures
FIGURES_DIR = OUT_DIR / "figures"    # fig_*.pdf / .png
TABLES_DIR = OUT_DIR / "tables"      # LaTeX tables of the main text and appendix
APPENDIX_DIR = OUT_DIR / "appendix"  # model settings, experimental details, run inventory, prompts

MODEL_DIRS = {
    "meta-llama_Llama-3.3-70B-Instruct": "Llama-3.3-70B",
    "zai-org_GLM-4.7-Flash": "GLM-4.7-Flash",
    "openai_gpt-oss-120b": "GPT-OSS-120B",
}
MODEL_ORDER = ["Llama-3.3-70B", "GLM-4.7-Flash", "GPT-OSS-120B"]
MODEL_FAMILY = {"Llama-3.3-70B": "llama", "GLM-4.7-Flash": "glm", "GPT-OSS-120B": "gptoss"}
HF_IDS = {
    "Llama-3.3-70B": "meta-llama/Llama-3.3-70B-Instruct",
    "GLM-4.7-Flash": "zai-org/GLM-4.7-Flash",
    "GPT-OSS-120B": "openai/gpt-oss-120b",
}
DATASET_DISPLAY = {"winogrande": "WinoGrande", "ethics": "ETHICS", "cwq": "CWQ"}  # ETHICS = commonsense-morality subset
DATASET_ORDER = ["cwq", "ethics", "winogrande"]
CATEGORY_ORDER = ["false_fact", "clickbait", "conspiracy", "framing", "hoax", "other", "propaganda", "rumor", "satire"]


def model_display(model_dir_or_id: str) -> str:
    s = str(model_dir_or_id)
    if s in MODEL_DIRS:
        return MODEL_DIRS[s]
    for k, v in HF_IDS.items():
        if s == v:
            return k
    low = s.lower()
    if "llama" in low:
        return "Llama-3.3-70B"
    if "glm" in low:
        return "GLM-4.7-Flash"
    if "gpt-oss" in low or "gpt_oss" in low or "gptoss" in low:
        return "GPT-OSS-120B"
    return s


def ensure_dirs() -> None:
    for d in (RESULTS_DIR, TABLES_DIR, FIGURES_DIR, APPENDIX_DIR):
        d.mkdir(parents=True, exist_ok=True)


def write_csv(path: Path, rows: Iterable[dict], fieldnames: Optional[Sequence[str]] = None) -> int:
    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        keys: List[str] = []
        for r in rows:
            for k in r.keys():
                if k not in keys:
                    keys.append(k)
        fieldnames = keys
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(fieldnames), extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)
    return len(rows)


def read_csv(path: Path) -> List[dict]:
    with open(path, "r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def load_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# LaTeX output. Every generated table must compile unchanged inside the ACL template
# (two-column, T1 fontenc, booktabs loaded in the main file).
# ---------------------------------------------------------------------------
_UNICODE_TO_LATEX = {
    "\u00d7": "$\\times$", "\u2212": "$-$", "\u2265": "$\\geq$", "\u2264": "$\\leq$", "\u00b1": "$\\pm$",
    "\u2192": "$\\rightarrow$", "\u2190": "$\\leftarrow$", "\u2248": "$\\approx$", "\u2260": "$\\neq$",
    "\u221e": "$\\infty$", "\u0394": "$\\Delta$", "\u03b4": "$\\delta$", "\u03ba": "$\\kappa$", "\u03b1": "$\\alpha$",
    "\u03b2": "$\\beta$", "\u03bc": "$\\mu$", "\u03c3": "$\\sigma$", "\u00b0": "$^{\\circ}$",
    "\u2013": "--", "\u2014": "---", "\u00a0": "~", "\u2018": "`", "\u2019": "'", "\u201c": "``", "\u201d": "''",
    "\u2026": "\\ldots{}", "\u2713": "yes", "\u2717": "no", "\u00e9": "\\'e", "\u00fc": "\\\"u", "\u00f6": "\\\"o",
    "\u00e4": "\\\"a", "\u00df": "\\ss{}",
}
# ASCII digraphs that read as operators in prose; rendered as math outside $...$
_TEXT_DIGRAPHS = {"->": "$\\rightarrow$", ">=": "$\\geq$", "<=": "$\\leq$"}
_DIGRAPH_SPLIT = re.compile(r"(->|>=|<=)")
# an already escaped special character (\_ \% \& \# \$ \{ \}) is kept as it is
_PRE_ESCAPED = re.compile(r"\\([&%$#_{}])")
# arguments of referencing commands are never escaped (labels may contain underscores)
_REF_COMMAND = re.compile(r"\\(?:ref|eqref|autoref|cref|Cref|label|cite|citet|citep|citealp|url|input|include)\*?"
                          r"(?:\[[^\]]*\])?\{[^}]*\}")
_MATH_SPLIT = re.compile(r"(?<!\\)\$")
_ESCAPES = {"\\": "\\textbackslash{}", "&": "\\&", "%": "\\%", "$": "\\$", "#": "\\#", "_": "\\_", "{": "\\{",
            "}": "\\}", "~": "\\textasciitilde{}", "^": "\\textasciicircum{}", "<": "\\textless{}",
            ">": "\\textgreater{}", "|": "\\textbar{}"}
_SOFT_ESCAPES = {"&": "\\&", "%": "\\%", "#": "\\#", "_": "\\_", "<": "\\textless{}", ">": "\\textgreater{}"}


def _escape_text_segment(seg: str, soft: bool) -> str:
    kept: List[str] = []

    def _keep(m):
        kept.append(m.group(0))
        return f"\x00{len(kept) - 1}\x00"

    if soft:
        seg = _REF_COMMAND.sub(_keep, seg)
    seg = _PRE_ESCAPED.sub(_keep, seg)
    table = _SOFT_ESCAPES if soft else _ESCAPES
    out = []
    for piece in _DIGRAPH_SPLIT.split(seg):
        if piece in _TEXT_DIGRAPHS:
            out.append(_TEXT_DIGRAPHS[piece])
        else:
            piece = "".join(table.get(ch, ch) for ch in piece)
            # command-line flags such as --inference must not become an en dash; "10--20" and "--" stay ligatures
            out.append(re.sub(r"--(?=[A-Za-z])", "-{}-", piece))
    return re.sub(r"\x00(\d+)\x00", lambda m: kept[int(m.group(1))], "".join(out))


def latex_escape(s: object, soft: bool = False) -> str:
    """Escape text for LaTeX while leaving inline math ($...$) untouched.

    * unicode symbols (x, minus, >=, +-, arrows, Greek letters, dashes, quotes) become LaTeX;
    * '->', '>=', '<=' outside math become arrows / inequalities in math mode;
    * already escaped characters (\\_ \\% ...) are kept, so pre-escaped strings survive;
    * soft=True (captions written by the scripts) escapes only _ % & # < > and leaves commands,
      braces and backslashes alone.
    """
    s = str(s)
    for a, b in _UNICODE_TO_LATEX.items():
        s = s.replace(a, b)
    parts = _MATH_SPLIT.split(s)
    if len(parts) % 2 == 1:  # balanced $: odd segments are math
        return "$".join(p if i % 2 else _escape_text_segment(p, soft) for i, p in enumerate(parts))
    return _escape_text_segment(s.replace("$", "\\$"), soft)  # unbalanced: no math in this string


# ACL two-column geometry (pt): column width 3.03in, text width 6.3in; approximate glyph width per font size
_COLUMN_WIDTH_PT, _TEXT_WIDTH_PT = 218.0, 455.0
_FONT_PT = {"normalsize": 10.0, "small": 9.0, "footnotesize": 8.0, "scriptsize": 7.0}
_ROWS_PER_PAGE = {"normalsize": 34, "small": 40, "footnotesize": 46, "scriptsize": 54}
_P_COL = re.compile(r"p\{([0-9.]+)(cm|in|pt|mm)\}")


def _visible_len(cell: str) -> int:
    t = re.sub(r"\\[a-zA-Z]+\{?", "", cell)  # drop command names
    t = re.sub(r"[{}$^_]", "", t)
    return len(t)


_COL_TOKEN = re.compile(r"p\{[^}]*\}|[lcr]")


def _estimate_width_pt(header: Sequence[str], body: Sequence[Sequence[str]], col_spec: str, font: str,
                       colsep: float = 6.0) -> float:
    """Rough natural width of the tabular: fixed width for p{} columns, average glyph width (0.46 em in
    Times) times the longest cell otherwise, plus the column padding (2 x tabcolsep per column)."""
    ncol = len(header)
    glyph = 0.46 * _FONT_PT[font]
    tokens = _COL_TOKEN.findall(col_spec)
    to_pt = {"cm": 28.45, "in": 72.27, "pt": 1.0, "mm": 2.845}
    widths = []
    for j in range(ncol):
        tok = tokens[j] if j < len(tokens) else "l"
        m = _P_COL.fullmatch(tok)
        if m:
            widths.append(float(m.group(1)) * to_pt[m.group(2)])
        else:
            chars = max([_visible_len(header[j])] + [_visible_len(r[j]) for r in body if j < len(r)])
            widths.append(chars * glyph)
    return sum(widths) + 2 * colsep * ncol


def write_latex_table(
    path: Path,
    header: Sequence[str],
    rows: Sequence[Sequence[object]],
    caption: str,
    label: str,
    col_spec: Optional[str] = None,
    note: Optional[str] = None,
    small: bool = True,
    font: Optional[str] = None,
    wide: Optional[bool] = None,
    max_rows: Optional[int] = None,
    float_spec: str = "t",
) -> None:
    """Write a booktabs table that compiles as is in the ACL template.

    * text cells and headers are escaped with `latex_escape` (inline math is preserved), floats get
      three decimals, captions are escaped softly (they may contain LaTeX);
    * `wide=None` picks `table*` automatically when the estimated width exceeds one column; if the
      table does not fit the text width either, the font is reduced to scriptsize;
    * tables longer than one page are split into several floats; the first carries the label, the
      others a "continued" caption that refers back to it.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    if col_spec is None:
        col_spec = "l" + "r" * (len(header) - 1)
    font = font or ("small" if small else "normalsize")
    if font not in _FONT_PT:
        raise ValueError(f"unknown font size {font!r}")

    def _cell(c: object) -> str:
        if isinstance(c, bool):
            return "yes" if c else "no"
        if isinstance(c, float):
            if c != c:  # nan
                return "--"
            return f"{c:.3f}" if abs(c) < 10 else f"{c:.1f}"
        if c is None:
            return "--"
        return latex_escape(c)

    head = [latex_escape(h) for h in header]
    body = [[_cell(c) for c in r] for r in rows]
    for r in body:
        if len(r) != len(header):
            raise ValueError(f"{path.name}: row has {len(r)} cells, header has {len(header)}: {r}")

    width = _estimate_width_pt(head, body, col_spec, font)
    if wide is None:
        wide = width > _COLUMN_WIDTH_PT
    limit = _TEXT_WIDTH_PT if wide else _COLUMN_WIDTH_PT
    if wide and width > limit and font != "scriptsize":
        font = "scriptsize"
        width = _estimate_width_pt(head, body, col_spec, font)
    colsep = 6.0
    for cs in (6.0, 4.0, 3.0):  # tighten the column padding before giving up
        colsep = cs
        width = _estimate_width_pt(head, body, col_spec, font, colsep)
        if width <= limit:
            break
    if width > limit:
        print(f"[WARN] {path.name}: estimated width {width:.0f}pt exceeds the "
              f"{'text' if wide else 'column'} width; shorten cells or pass a p{{}} col_spec")

    env = "table*" if wide else "table"
    max_rows = max_rows or _ROWS_PER_PAGE[font]
    chunks = [body[i:i + max_rows] for i in range(0, len(body), max_rows)] or [[]]
    full_caption = latex_escape(caption, soft=True) + (" " + latex_escape(note, soft=True) if note else "")

    lines = [f"% generated by scripts/common.py (write_latex_table); requires \\usepackage{{booktabs}}; \\input this file as is",
             f"% {len(body)} rows in {len(chunks)} float(s), estimated width {width:.0f}pt, font {font}"
             + (f", tabcolsep {colsep:g}pt" if colsep != 6.0 else "")]
    for ci, chunk in enumerate(chunks):
        lines += [f"\\begin{{{env}}}[{float_spec}]"]
        if ci > 0:  # keep the table number when the caption package is loaded (it is in the ACL template)
            lines += ["\\ifdefined\\ContinuedFloat\\ContinuedFloat\\fi"]
        lines += ["\\centering", f"\\{font}"]
        if colsep != 6.0:  # local to this float
            lines += [f"\\setlength{{\\tabcolsep}}{{{colsep:g}pt}}"]
        lines += [f"\\begin{{tabular}}{{{col_spec}}}", "\\toprule", " & ".join(head) + " \\\\", "\\midrule"]
        lines += [" & ".join(r) + " \\\\" for r in chunk]
        lines += ["\\bottomrule", "\\end{tabular}"]
        if ci == 0:
            lines += [f"\\caption{{{full_caption}}}", f"\\label{{{label}}}"]
        else:
            lines += [f"\\caption{{Continuation of Table~\\ref{{{label}}} (part {ci + 1} of {len(chunks)}).}}"]
        lines += [f"\\end{{{env}}}", ""]
    path.write_text("\n".join(lines).rstrip("\n") + "\n", encoding="utf-8")


def condition_to_relevance_category(condition: str, misinformation_strategy: Optional[str]) -> tuple:
    """Map an exp1 condition to (relevance, category)."""
    strat = (misinformation_strategy or "").strip().lower()
    cond = (condition or "").strip()
    if cond == "baseline" or strat in {"", "none", "baseline"} and cond == "baseline":
        return "none", "none"
    if strat == "irrelevant_true_information" or cond == "irrelevant_true_information":
        return "irrelevant_true", "irrelevant_true"
    if cond == "relevant_true":
        return "relevant_true", "relevant_true"
    if cond == "filler":
        return "filler", "filler"
    if strat.startswith("irrelevant_"):
        return "irrelevant", strat[len("irrelevant_"):]
    if cond == "irrelevant_misinformed":
        return "irrelevant", strat or "unknown"
    if cond == "false_fact" or strat == "false_fact":
        return "relevant", "false_fact"
    if cond.startswith("strategy_"):
        return "relevant", cond[len("strategy_"):]
    if cond in CATEGORY_ORDER:
        return "relevant", cond
    return "relevant", strat or cond or "unknown"


# ---------------------------------------------------------------------------
# MINT verification. The judges answer Q1 (does the passage contradict the gold answer) and Q2
# (does it support the intended wrong answer). CWQ and ETHICS are verified when BOTH judges say
# yes to Q1 and Q2 (the runs with --mint_filter verified used exactly this rule). On WinoGrande
# Q1 is ill-posed (a false fact about a fictional scenario is rarely called "false"), so the
# WinoGrande debates were run on all questions and the analyses restrict them to the questions
# whose plain false fact BOTH judges confirm to support the intended wrong answer (Q2). The
# criterion per dataset lives in configs/experiment_config.json (mint_generation.verification_criterion).
# ---------------------------------------------------------------------------
_JUDGE_KEYS = ("gpt_oss_120b", "glm_4_7_flash")
_FLAG_CACHE: Dict[str, Dict[tuple, dict]] = {}


def _config() -> dict:
    p = REPO_ROOT / "configs" / "experiment_config.json"
    try:
        return load_json(p)
    except Exception:  # noqa: BLE001
        return {}


def verification_criterion(dataset: str) -> str:
    """Rule that defines the verified subset of a dataset ('both_q1q2' unless the config says otherwise)."""
    crit = ((_config().get("mint_generation") or {}).get("verification_criterion") or {})
    return crit.get(dataset, "both_q1q2")


def load_passage_flags(data_dir: Optional[Path] = None) -> Dict[tuple, dict]:
    """(dataset, mint_id, category) -> {"judges": {key: {"parsed", "q1", "q2"}}, "verified_flag": bool|None}."""
    data_dir = Path(data_dir) if data_dir else REPO_ROOT / "MINT_dataset_v1.2"
    ck = str(data_dir)
    if ck in _FLAG_CACHE:
        return _FLAG_CACHE[ck]
    flags: Dict[tuple, dict] = {}
    for p in sorted(data_dir.glob("*_passages.jsonl")):
        with open(p, encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                r = json.loads(line)
                v = r.get("validation") or {}
                judges = {}
                for jk, jv in v.items():
                    if jk in ("verified", "judges") or not isinstance(jv, dict):
                        continue
                    judges[jk] = {"parsed": bool(jv.get("parsed")), "q1": jv.get("is_false") is True,
                                  "q2": jv.get("supports_target") is True}
                flags[(r["dataset"], r["mint_id"], r["category"])] = {"judges": judges, "verified_flag": v.get("verified")}
    _FLAG_CACHE[ck] = flags
    return flags


def passage_meets(flag: Optional[dict], criterion: str, judges: Optional[Sequence[str]] = None) -> bool:
    """Apply a verification rule to one passage's judge flags; `judges` restricts the rule to these judges."""
    if not flag:
        return False
    js = flag["judges"]
    keys = [j for j in (judges or _JUDGE_KEYS) if j in js]
    if not keys:
        return False
    if criterion == "both_q1q2":
        return all(js[j]["q1"] and js[j]["q2"] for j in keys)
    if criterion == "both_q2":
        return all(js[j]["q2"] for j in keys)
    if criterion == "either_q1q2":
        return any(js[j]["q1"] and js[j]["q2"] for j in keys)
    raise ValueError(f"unknown verification criterion {criterion!r}")


def verified_questions(dataset: str, category: str = "false_fact", criterion: Optional[str] = None,
                       judges: Optional[Sequence[str]] = None, data_dir: Optional[Path] = None) -> set:
    """Question ids whose `category` passage passes the dataset's verification rule (or `criterion`)."""
    crit = criterion or verification_criterion(dataset)
    flags = load_passage_flags(data_dir)
    return {mid for (ds, mid, cat), fl in flags.items() if ds == dataset and cat == category and passage_meets(fl, crit, judges)}


def verified_counts_text(category: str = "false_fact") -> str:
    """'CWQ 98, ETHICS 114, WinoGrande 183' for captions."""
    return ", ".join(f"{DATASET_DISPLAY[d]} {len(verified_questions(d, category))}" for d in DATASET_ORDER)
