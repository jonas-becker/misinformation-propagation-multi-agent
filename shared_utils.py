#!/usr/bin/env python3

"""
Shared utilities of the single-agent runners (prompt templates, argument parser, dataset helpers).
"""

import argparse
import json
import re
from typing import Dict, List

# Model configuration
MODEL_NAME = "openai/gpt-oss-120b"
GLM47_FLASH_MODEL_NAME = "zai-org/GLM-4.7-Flash"

# Prompt templates
PROMPT_BASELINE = """
{sentence}

{options_text}

Solve the task.
"""

PROMPT_MISINFORMED = """
{sentence}

{options_text}

You have this extra information: {misinformation}

Solve the task.
"""

# Special prompt templates for complex_web_questions (free-form questions)
PROMPT_BASELINE_FREE_FORM = """
{sentence}

Provide a clear and accurate answer to the question above.
"""

PROMPT_MISINFORMED_FREE_FORM = """
{sentence}

You have this extra information: {misinformation}

Provide a clear and accurate answer to the question above.
"""

def build_common_arg_parser(description: str = "Run experiment") -> argparse.ArgumentParser:
    """Build ArgumentParser with flags shared by experiment scripts (exp1, exp1a, ...)."""
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--max_samples", type=int, default=None, 
                       help="Maximum number of samples to evaluate per dataset")
    parser.add_argument(
        "--model_name",
        "--model",
        dest="model_name",
        type=str,
        default=MODEL_NAME,
        help=(
            "Model name to use for evaluation "
            f"(e.g. {GLM47_FLASH_MODEL_NAME})"
        ),
    )
    parser.add_argument("--batch_size", type=int, default=8,
                       help="Batch size for processing (default: 8)")
    parser.add_argument("--max_tokens", type=int, default=512, help="Maximum tokens to generate per prompt")
    parser.add_argument("--continue", action="store_true", help="Continue from previous runs (don't clear existing files)")
    parser.add_argument("--dummy_test", action="store_true", help="Run in dummy test mode without loading models")
    parser.add_argument(
        "--datasets",
        type=str,
        default=None,
        help=(
            "Comma-separated datasets to run, e.g. "
            "'winogrande,ethics,complex_web_questions'. "
            "If omitted, run all available datasets."
        ),
    )
    return parser


def model_name_to_safe_path(model_name: str) -> str:
    """Create a filesystem-safe token from a model name."""
    safe = (model_name or "").strip().replace("/", "_")
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", safe)
    return safe or "unknown_model"

def format_options(options: List[str]) -> str:
    """Format options as A) B) C) etc."""
    if not options:
        return ""
    formatted = []
    for i, option in enumerate(options):
        formatted.append(f"{chr(65 + i)}) {option}")
    return "Answer Choices:\n" + "\n".join(formatted)

def save_results(results: Dict, output_file: str):
    """Save results to JSON file."""
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
