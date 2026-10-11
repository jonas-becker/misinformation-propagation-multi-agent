#!/usr/bin/env python3
"""
runner_common.py - Shared infrastructure for the experiment runners
(exp_exposure.py, exp_msc.py, exp_message_ablation.py, exp_mitigation_verifier.py,
scripts/validate_mint.py, scripts/generate_mint_passages.py).

Provides
  * ChatClient: OpenAI-compatible chat wrapper with retries, per-call seeds,
    token accounting, finish_reason capture and a --mock mode.
  * load_experiment_config(): reads configs/experiment_config.json.
  * JSONL writer for one-row-per-question x agent x turn outputs.
  * Endpoint resolution reusing exp2's discovery/autostart helpers.
"""

from __future__ import annotations

import json
import os
import random
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parent
CONFIG_PATH = REPO_ROOT / "configs" / "experiment_config.json"


def load_experiment_config() -> dict:
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


@dataclass
class TokenLedger:
    """Thread-safe prompt/completion token accounting per (stage) label."""

    counts: Dict[str, Dict[str, int]] = field(default_factory=dict)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def add(self, label: str, prompt_tokens: int, completion_tokens: int, n_calls: int = 1) -> None:
        with self._lock:
            c = self.counts.setdefault(label, {"prompt_tokens": 0, "completion_tokens": 0, "calls": 0})
            c["prompt_tokens"] += int(prompt_tokens or 0)
            c["completion_tokens"] += int(completion_tokens or 0)
            c["calls"] += n_calls

    def as_dict(self) -> dict:
        with self._lock:
            return json.loads(json.dumps(self.counts))


THINK_END = "</think>"


def split_think_channels(content: Optional[str], reasoning: Optional[str]) -> Tuple[str, Optional[str]]:
    """Normalise (content, reasoning) from an OpenAI-compatible server with a reasoning parser.

    vLLM's glm45 parser (GLM-4.7-Flash) has two failure modes:
      * the closing tag leaks into `content` ("...nal Solution: 4</think>The answer ...");
      * when the model never emits `</think>` the whole output lands in `reasoning` and
        `content` is empty, although the answer is at the end of that text.
    The answer is everything after the LAST `</think>`; anything before it is thinking.
    """
    content = content or ""
    reasoning = reasoning or None
    if THINK_END in content:
        pre, _, post = content.rpartition(THINK_END)
        reasoning = ((reasoning or "") + pre).strip() or None
        content = post
    elif not content.strip() and reasoning and THINK_END in reasoning:
        pre, _, post = reasoning.rpartition(THINK_END)
        reasoning, content = pre.strip() or None, post
    if content.lstrip().startswith("<think>"):
        content = content.lstrip()[len("<think>"):]
    if reasoning and reasoning.lstrip().startswith("<think>"):
        reasoning = reasoning.lstrip()[len("<think>"):].strip() or None
    return content, reasoning


@dataclass
class ChatResult:
    text: str
    finish_reason: Optional[str]
    prompt_tokens: int
    completion_tokens: int
    reasoning_text: Optional[str] = None
    error: Optional[str] = None


class ChatClient:
    """Minimal OpenAI-compatible chat client with retries, seeds and token accounting."""

    def __init__(
        self,
        endpoint_url: Optional[str],
        model_name: str,
        *,
        mock: bool = False,
        timeout: float = 600.0,
        max_retries: int = 5,
        ledger: Optional[TokenLedger] = None,
        mock_answer_fn=None,
    ) -> None:
        self.model_name = model_name
        self.mock = mock
        self.timeout = timeout
        self.max_retries = max_retries
        self.ledger = ledger or TokenLedger()
        self._seed_unsupported = threading.Event()
        self._mock_answer_fn = mock_answer_fn
        self._mock_rng = random.Random(0)
        self.client = None
        if not mock:
            from openai import OpenAI  # lazy import

            self.client = OpenAI(
                base_url=(endpoint_url or "http://127.0.0.1:8080/v1").rstrip("/"),
                api_key=os.environ.get("OPENAI_API_KEY") or "EMPTY",
                timeout=timeout,
                max_retries=0,
            )

    def chat(
        self,
        messages: List[dict],
        *,
        max_tokens: int,
        temperature: float,
        top_p: float = 1.0,
        seed: Optional[int] = None,
        label: str = "default",
        stop: Optional[List[str]] = None,
    ) -> ChatResult:
        if self.mock:
            text = self._mock_answer_fn(messages) if self._mock_answer_fn else "Final Solution: A"
            pt = sum(len(str(m.get("content", "")).split()) for m in messages)
            ct = len(text.split())
            self.ledger.add(label, pt, ct)
            return ChatResult(text=text, finish_reason="stop", prompt_tokens=pt, completion_tokens=ct)

        delay = 1.0
        last_err: Optional[BaseException] = None
        for _attempt in range(self.max_retries):
            kwargs: Dict[str, Any] = {
                "model": self.model_name,
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "top_p": top_p,
                "timeout": self.timeout,
            }
            if stop:
                kwargs["stop"] = stop
            if seed is not None and not self._seed_unsupported.is_set():
                kwargs["seed"] = seed
            try:
                resp = self.client.chat.completions.create(**kwargs)
                choice = resp.choices[0]
                msg = choice.message
                content = getattr(msg, "content", None) or ""
                reasoning = None
                for attr in ("reasoning_content", "reasoning"):
                    alt = getattr(msg, attr, None)
                    if isinstance(alt, str) and alt.strip():
                        reasoning = alt
                        break
                content, reasoning = split_think_channels(content, reasoning)
                usage = getattr(resp, "usage", None)
                pt = int(getattr(usage, "prompt_tokens", 0) or 0)
                ct = int(getattr(usage, "completion_tokens", 0) or 0)
                self.ledger.add(label, pt, ct)
                return ChatResult(text=content.strip(), finish_reason=getattr(choice, "finish_reason", None),
                                  prompt_tokens=pt, completion_tokens=ct, reasoning_text=reasoning)
            except Exception as e:  # noqa: BLE001
                if "seed" in kwargs and "seed" in str(e).lower():
                    self._seed_unsupported.set()
                    print("[WARN] endpoint rejected 'seed'; continuing without per-call seeds")
                    continue
                last_err = e
                time.sleep(delay)
                delay = min(delay * 2, 30.0)
        return ChatResult(text="", finish_reason="error", prompt_tokens=0, completion_tokens=0, error=str(last_err))


def resolve_endpoint(endpoint_url: str, model_name: str, no_autostart: bool) -> str:
    """Reuse exp2's endpoint discovery / autostart logic."""
    import exp2 as exp2_base  # local import: heavy module

    resolved = exp2_base._env_or_default_endpoint(endpoint_url)
    if resolved.startswith("http") and not exp2_base._endpoint_alive(resolved, timeout=2.0):
        auto = exp2_base._discover_endpoint_from_mallm(model_name)
        if auto:
            print(f"[INFO] endpoint {resolved!r} unreachable; using discovered endpoint {auto!r}")
            resolved = auto
        elif not no_autostart:
            started = exp2_base._start_model_server(model_name, desired_port=8080)
            if started:
                resolved = started["endpoint_url"]
    if resolved.startswith("http") and not exp2_base._endpoint_alive(resolved, timeout=2.5):
        raise SystemExit(f"[FATAL] no reachable model endpoint at {resolved!r}")
    return resolved


class JsonlWriter:
    """Append-only JSONL writer (one record per question x agent x turn)."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def write(self, record: dict) -> None:
        line = json.dumps(record, ensure_ascii=False)
        with self._lock:
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(line + "\n")

    def write_many(self, records: List[dict]) -> None:
        with self._lock:
            with open(self.path, "a", encoding="utf-8") as f:
                for r in records:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")


def read_jsonl(path: Path) -> List[dict]:
    out: List[dict] = []
    if not Path(path).is_file():
        return out
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def stable_seed(*parts: Any) -> int:
    """Deterministic 31-bit seed from arbitrary parts (used for per-question assignment)."""
    import hashlib

    h = hashlib.md5("::".join(str(p) for p in parts).encode("utf-8")).hexdigest()
    return int(h[:8], 16) & 0x7FFFFFFF


def write_run_metadata(path: Path, args: Any, extra: Optional[dict] = None) -> None:
    """Dump the effective configuration next to the results (reproducibility)."""
    import platform
    import sys
    from datetime import datetime, timezone

    meta = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "argv": sys.argv,
        "args": {k: (v if isinstance(v, (int, float, str, bool, list, dict, type(None))) else str(v))
                 for k, v in vars(args).items()},
        "python": sys.version,
        "platform": platform.platform(),
        "git_commit": _git_commit(),
    }
    if extra:
        meta.update(extra)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)


def _git_commit() -> Optional[str]:
    try:
        import subprocess

        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=str(REPO_ROOT), text=True,
                                       stderr=subprocess.DEVNULL).strip()
    except Exception:
        return None
