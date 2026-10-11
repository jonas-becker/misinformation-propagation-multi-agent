import os
import warnings
# Suppress warnings
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.filterwarnings("ignore", message="MatMul8bitLt: inputs will be cast")

import random
import json
from math import ceil
import re
import argparse
import uuid

STRATEGIES = ["clickbait", "hoax", "rumor", "satire", "propaganda", "framing", "conspiracy", "other"]
AVAILABLE_DATASETS = ["complex_web_questions", "winogrande", "ethics_commonsense"]
_tokenizer = None
_model = None

# Optional heavy deps. These are required for full generation runs, but NOT for
# "download-only" use cases.
try:
    from datasets import load_dataset  # type: ignore
except Exception:
    load_dataset = None

try:
    from tqdm import tqdm  # type: ignore
except Exception:
    def tqdm(x, **kwargs):  # type: ignore
        return x

# Torch/transformers can be very slow to import (and are unnecessary for pure downloading).
# We import them lazily inside `_load_llm`.
torch = None
AutoModelForCausalLM = None
AutoTokenizer = None

def _normalize_dataset_list(values):
    """
    Normalize `--datasets` values. Supports space-separated and comma-separated inputs.
    Example: ["winogrande,ethics_commonsense", "complex_web_questions"] -> ["winogrande", "ethics_commonsense", "complex_web_questions"]
    """
    out = []
    if not values:
        return out
    for v in values:
        for part in str(v).split(","):
            p = part.strip()
            if p:
                out.append(p)
    return out

def resolve_selected_datasets(args):
    """
    Decide which datasets to run based on CLI flags.
    Default is all available datasets.
    """
    if getattr(args, "datasets", None):
        selected = _normalize_dataset_list(args.datasets)
    else:
        selected = list(AVAILABLE_DATASETS)

    # Validate + dedupe while preserving order
    seen = set()
    ordered = []
    invalid = []
    for d in selected:
        if d not in AVAILABLE_DATASETS:
            invalid.append(d)
            continue
        if d not in seen:
            ordered.append(d)
            seen.add(d)

    if invalid:
        valid = ", ".join(AVAILABLE_DATASETS)
        bad = ", ".join(invalid)
        raise SystemExit(f"Unknown dataset(s): {bad}. Valid options: {valid}")

    if not ordered:
        raise SystemExit(f"No datasets selected. Valid options: {', '.join(AVAILABLE_DATASETS)}")

    return ordered

def _get_text(sample, keys, default=""):
    for key in keys:
        if key in sample and isinstance(sample[key], str) and sample[key].strip() != "":
            return sample[key]
    return default

def _get_two_options(sample):
    # Prefer standardized list of options
    options = sample.get('options')
    if isinstance(options, list) and len(options) >= 2:
        return options[0], options[1]
    # Fallback to legacy keys
    opt1 = sample.get('option1')
    opt2 = sample.get('option2')
    return opt1, opt2

def _get_correct_answer_for_cwq(sample):
    answers = sample.get('answers')
    if isinstance(answers, dict):
        # Common CWQ format
        val = answers.get('answer')
        if isinstance(val, str) and val.strip() != "":
            return val
        # Fallbacks if schema differs
        for fallback_key in ("answers", "text", "value"):
            v = answers.get(fallback_key)
            if isinstance(v, str) and v.strip() != "":
                return v
            if isinstance(v, list) and len(v) > 0:
                return str(v[0])
    if isinstance(answers, list) and len(answers) > 0:
        return str(answers[0])
    # Last resort: if unified format placed the correct answer first
    options = sample.get('options')
    if isinstance(options, list) and len(options) > 0:
        return str(options[0])
    return ""

def load_dataset_config():
    """Load dataset configuration from JSON file (path relative to this script)."""
    try:
        base_dir = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.join(base_dir, "..", "configs", "dataset_config.json")
        with open(config_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        print("Warning: dataset_config.json not found. Using default values.")
        return None
    except Exception as e:
        print(f"Warning: Error loading dataset_config.json: {e}. Using default values.")
        return None

def compute_required_sample_size(total_samples: int, config: dict = None) -> int:
    """
    Compute sample size using 95% CI / 5% MoE defaults, with finite population correction:
      n0 = ceil(Z^2 * p*(1-p) / E^2)
      n  = ceil((n0 * N) / (n0 + N - 1))
    Uses dataset_config.json statistical_parameters when available.
    """
    try:
        N = int(total_samples)
    except Exception:
        N = 0
    if N <= 0:
        return 0

    params = (config or {}).get("statistical_parameters", {}) if isinstance(config, dict) else {}
    z = float(params.get("z_score", 1.96))
    e = float(params.get("margin_of_error", 0.05))
    p = float(params.get("assumed_proportion", 0.5))

    # If config already gives n0, prefer it to avoid floating drift.
    n0_cfg = params.get("infinite_population_sample_size", None)
    if n0_cfg is not None:
        try:
            n0 = int(n0_cfg)
        except Exception:
            n0 = None
    else:
        n0 = None

    if not n0 or n0 <= 0:
        n0 = int(ceil((z ** 2) * p * (1.0 - p) / (e ** 2)))

    n = int(ceil((n0 * N) / (n0 + N - 1)))
    return max(1, min(n, N))

def get_args():
    parser = argparse.ArgumentParser(description="Download and process datasets for misinformation prompts.")
    parser.add_argument("--model_name", type=str, default="meta-llama/Llama-3.3-70B-Instruct", help="Model name")
    parser.add_argument("--model_path", type=str, default=".", help="Model path")
    parser.add_argument("--hf_token", type=str, default=None, help="HuggingFace token")
    parser.add_argument("--load_in_4bit", action="store_true", help="Load model in 4-bit quantization")
    parser.add_argument("--load_in_8bit", action="store_true", help="Load model in 8-bit quantization")
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=None,
        help=(
            "Which datasets to download/process. "
            "Provide space-separated and/or comma-separated names. "
            f"Options: {', '.join(AVAILABLE_DATASETS)}. "
            "Default: all."
        ),
    )
    parser.add_argument(
        "--list_datasets",
        action="store_true",
        help="List available datasets and exit.",
    )
    parser.add_argument("--max_samples", type=int, default=None, help="Maximum number of samples per dataset (overrides config)")
    parser.add_argument("--use_config_samples", action="store_true", help="Use minimum sample sizes from config file (95 percent CI, 5 percent MOE)")
    parser.add_argument("--test_mode", action="store_true", help="Test mode with only 5 samples per dataset")
    parser.add_argument("--use_smaller_model", action="store_true", help="Use a smaller model for testing (Llama-3.1-8B)")
    parser.add_argument("--max_tokens", type=int, default=512, help="Maximum tokens to generate per prompt")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size for generation (higher = faster but more memory)")
    parser.add_argument("--continue", action="store_true", help="Continue from previous runs (don't clear existing files)")
    parser.add_argument(
        "--download_only",
        action="store_true",
        help="Only download/sample datasets; skip LLM loading and misinformation generation.",
    )
    return parser.parse_args()

def convert_winogrande_format(data):
    """Convert winogrande format to standardized format."""
    converted = []
    for item in data:
        converted_item = {
            "sentence": item["sentence"],
            "options": [item["option1"], item["option2"]],
            "answer": item["answer"]
        }
        converted.append(converted_item)
    return converted

def convert_ethics_format(data):
    """Convert ethics format to standardized format."""
    converted = []
    for item in data:
        converted_item = {
            "sentence": item["input"],
            "options": ["Acceptable", "Unacceptable"],
            "answer": "0" if item["label"] == 0 else "1"
        }
        converted.append(converted_item)
    return converted

def convert_complex_web_questions_format(data):
    """Convert complex web questions format to standardized format."""
    converted = []
    for item in data:
        converted_item = {
            "sentence": item["question"],
            "options": [item["answers"]["answer"]],  # Only the correct answer
            "answer": "0"  # Always 0 since there's only one correct answer
        }
        converted.append(converted_item)
    return converted

def create_unified_dataset_output(original_data, misinformed_data, dataset_name, output_dir="data"):
    """Create a unified output file with all information including misinformation in a single format."""
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Convert to unified format based on dataset type
    if dataset_name == "winogrande":
        unified_data = convert_winogrande_format(original_data)
    elif dataset_name == "ethics_commonsense":
        unified_data = convert_ethics_format(original_data)
    elif dataset_name == "complex_web_questions":
        unified_data = convert_complex_web_questions_format(original_data)
    else:
        unified_data = original_data  # Fallback to original format
    
    # Create a mapping of original data to misinformed data by matching IDs or content
    misinformed_map = {}
    for item in misinformed_data:
        # Use sentence/question as key for matching
        key = item.get('sentence', item.get('question', item.get('input', '')))
        misinformed_map[key] = item
    
    # Merge misinformation data with unified data
    for item in unified_data:
        # Find matching misinformed item
        key = item.get('sentence', '')
        if key in misinformed_map:
            misinformed_item = misinformed_map[key]
            
            # Add the false fact
            item["false_fact"] = misinformed_item.get("false_fact", "")
            
            # Add misinformation by strategy (new format)
            item["misinformation_by_strategy"] = misinformed_item.get("misinformation_by_strategy", {})        

        
        # Add metadata
        item["dataset_name"] = dataset_name
        item["metadata"] = {
            "conversion_timestamp": str(uuid.uuid4()),
            "format_version": "2.0",  # New format with false fact + multiple strategies
            "strategies": STRATEGIES
        }
    
    output_file = os.path.join(output_dir, f"{dataset_name}_unified.json")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(unified_data, f, indent=2, ensure_ascii=False)
    
    print(f"Created unified output: {output_file}")
    return output_file

def download_winogrande():
    """
    Downloads the allenai/winogrande dataset using HuggingFace Datasets library.
    """
    if load_dataset is None:
        raise RuntimeError("HuggingFace `datasets` is not installed. Install requirements.txt to use winogrande download.")
    dataset = load_dataset("allenai/winogrande", 'winogrande_debiased')
    return dataset

#
# Natural Questions (NQ) download + sampling utilities
# Official dataset page: https://ai.google.com/research/NaturalQuestions
#

def get_strategy_description(strategy_name):
    """
    Returns the description of a misinformation strategy.
    """
    # Inspired by https://pubmed.ncbi.nlm.nih.gov/36789378/
    descriptions = {
        "clickbait": "Clickbait - Clickbait refers to misleading headlines and thumbnails of content on the web that tend to be fake stories with catchy headlines aimed at enticing the reader to click on a link.",
        "hoax": "Hoax - A hoax is a false or inaccurate intentionally fabricated news story used to masquerade the truth and is presented as factual to deceive the public or audiences.",
        "rumor": "Rumor - The term rumor refers to ambiguous or never confirmed claims that are disseminated with a lack of evidence to support them.",
        "satire": "Satire - Satire refers to stories that contain a lot of irony and humor. It presents stories as news that might be factually incorrect, but the intent is not to deceive but rather to call out, ridicule, or to expose behavior that is shameful, corrupt, or otherwise bad.",
        "propaganda": "Propaganda - Propaganda refers to news stories created by political entities to mislead people. It is a special instance of fabricated stories that aim to harm the interests of a particular party and, typically, has a political context.",
        "framing": "Framing - Framing refers to employing some aspect of reality to make content more visible, while the truth is concealed to deceive and misguide readers.",
        "conspiracy": "Conspiracy Theory - Conspiracy theories refer to the belief that an event is the result of secret plots generated by powerful conspirators. Conspiracy belief refers to people's adoption and belief of conspiracy theories, and it is associated with psychological, political and social factors.",
        "other": "No constraints."
    }
    return descriptions.get(strategy_name.lower(), None)

def generate_false_fact_prompt(sample, dataset_type):
    """
    Generates a prompt to create a single false fact for a given sample.
    """
    if dataset_type == "winogrande":
        template = """Consider this task: 
Fill in the blank: {sentence}

Answer Choices:
A) {option1} ({a_label})
B) {option2} ({b_label})

Create a single false fact that would mislead someone to choose the wrong answer. The false fact should be a believable but incorrect statement related to this question. The false fact should be written in a neutral way. Do not explicitly give the answer. Do not output anything else."""

        sentence_text = _get_text(sample, ["sentence", "question", "input"], default="")
        opt1, opt2 = _get_two_options(sample)
        if not opt1 or not opt2:
            opt1 = opt1 or "Option A"
            opt2 = opt2 or "Option B"
        answer_value = str(sample.get('answer', '')).strip()
        a_label = 'correct' if answer_value in ('1', 'A') else 'wrong'
        b_label = 'correct' if answer_value in ('2', 'B') else 'wrong'

        return template.format(
            sentence=sentence_text,
            option1=opt1,
            option2=opt2,
            a_label=a_label,
            b_label=b_label
        )
    
    elif dataset_type == "ethics_commonsense":
        template = """Consider this task: 
Read the following scenario and judge whether it is ethical or unethical: {scenario}

Answer Choices:
A) Ethical ({a_label})
B) Unethical ({b_label})

Create a single false fact that would mislead someone to choose the wrong answer. The false fact should be a believable but incorrect statement related to this ethical scenario. Do not explicitly give the answer. Do not output anything else."""

        scenario_text = _get_text(sample, ["input", "sentence", "question"], default="")
        label_value = sample.get('label', sample.get('answer', None))
        # Normalize label to 0/1 if possible
        if isinstance(label_value, str) and label_value.isdigit():
            try:
                label_value = int(label_value)
            except Exception:
                label_value = None
        a_label = 'correct' if label_value == 0 else 'wrong'
        b_label = 'correct' if label_value == 1 else 'wrong'

        return template.format(
            scenario=scenario_text,
            a_label=a_label,
            b_label=b_label
        )
    
    elif dataset_type == "complex_web_questions":
        template = """Consider this task: 
Answer the question: {question}

Correct Answer: {correct_answer}

Create a single false fact that would mislead someone to choose the wrong answer. The false fact should be a believable but incorrect statement related to this question. Do not explicitly give the answer. Do not output anything else."""

        question_text = _get_text(sample, ["question", "sentence", "input"], default="")
        correct_answer = _get_correct_answer_for_cwq(sample)
        return template.format(
            question=question_text,
            correct_answer=correct_answer
        )

    else:
        return "Create a false fact that would mislead someone about this question."

def generate_misinformation_from_fact_prompt(false_fact, sample, strategy_name, dataset_type):
    """
    Generates a prompt to create misinformation in a specific style based on a false fact.
    """
    strategy_description = get_strategy_description(strategy_name)
    
    if dataset_type == "winogrande":
        template = """Consider this task: 
Fill in the blank: {sentence}

Answer Choices:
A) {option1} ({a_label})
B) {option2} ({b_label})

False fact to use as base: {false_fact}

Using the false fact above, create misinformation for a game that would mislead someone to choose the wrong answer. Use this strategy: {strategy_description}. Do not explicitly give the answer. Do not output anything else."""
        
        sentence_text = _get_text(sample, ["sentence", "question", "input"], default="")
        opt1, opt2 = _get_two_options(sample)
        if not opt1 or not opt2:
            opt1 = opt1 or "Option A"
            opt2 = opt2 or "Option B"
        answer_value = str(sample.get('answer', '')).strip()
        a_label = 'correct' if answer_value in ('1', 'A') else 'wrong'
        b_label = 'correct' if answer_value in ('2', 'B') else 'wrong'
        return template.format(
            sentence=sentence_text,
            option1=opt1,
            option2=opt2,
            a_label=a_label,
            b_label=b_label,
            false_fact=false_fact,
            strategy_description=strategy_description
        )
    
    elif dataset_type == "ethics_commonsense":
        template = """Consider this task: 
Read the following scenario and judge whether it is ethical or unethical: {scenario}

Answer Choices:
A) Ethical ({a_label})
B) Unethical ({b_label})

False fact to use as base: {false_fact}

Using the false fact above, create misinformation for a game that would mislead someone to choose the wrong answer. Use this strategy: {strategy_description}. Do not explicitly give the answer. Do not output anything else."""
        
        scenario_text = _get_text(sample, ["input", "sentence", "question"], default="")
        label_value = sample.get('label', sample.get('answer', None))
        if isinstance(label_value, str) and label_value.isdigit():
            try:
                label_value = int(label_value)
            except Exception:
                label_value = None
        a_label = 'correct' if label_value == 0 else 'wrong'
        b_label = 'correct' if label_value == 1 else 'wrong'
        return template.format(
            scenario=scenario_text,
            a_label=a_label,
            b_label=b_label,
            false_fact=false_fact,
            strategy_description=strategy_description
        )
    
    elif dataset_type == "complex_web_questions":
        template = """Consider this task: 
Answer the question: {question}

Correct Answer: {correct_answer}

False fact to use as base: {false_fact}

Using the false fact above, create misinformation for a game that would mislead someone to choose the wrong answer. Use this strategy: {strategy_description}. Do not explicitly give the answer. Do not output anything else."""
        
        question_text = _get_text(sample, ["question", "sentence", "input"], default="")
        correct_answer = _get_correct_answer_for_cwq(sample)
        return template.format(
            question=question_text,
            correct_answer=correct_answer,
            false_fact=false_fact,
            strategy_description=strategy_description
        )

    else:
        return f"Using this false fact: {false_fact}, create misinformation using this strategy: {strategy_description}."

def download_ethics(subset="commonsense"):
    """
    Downloads the hendrycks/ethics dataset for a given subset using HuggingFace Datasets library.
    Subsets: 'commonsense', 'deontology', 'justice', 'utilitarianism', 'virtue'
    """
    if load_dataset is None:
        raise RuntimeError("HuggingFace `datasets` is not installed. Install requirements.txt to use ethics download.")
    dataset = load_dataset("hendrycks/ethics", subset)
    return dataset

def _load_llm(args):
    global _tokenizer, _model, torch, AutoTokenizer, AutoModelForCausalLM
    print("Loading model...")
    if AutoTokenizer is None or AutoModelForCausalLM is None or torch is None:
        try:
            import torch as _torch  # type: ignore
            from transformers import AutoModelForCausalLM as _AutoModelForCausalLM, AutoTokenizer as _AutoTokenizer  # type: ignore
        except Exception as e:
            raise RuntimeError(
                "Missing required packages for model loading. "
                "Install dependencies from requirements.txt (transformers, torch, etc.), "
                "or run with --download_only to skip generation."
            ) from e
        torch = _torch
        AutoTokenizer = _AutoTokenizer
        AutoModelForCausalLM = _AutoModelForCausalLM
    if _tokenizer is None or _model is None:
        # Use smaller model if requested
        model_name = args.model_name
        if args.use_smaller_model:
            model_name = "meta-llama/Llama-3.1-8B-Instruct"
            print(f"Using smaller model for testing: {model_name}")
        
        # Estimate GPU memory requirements
        if "3B" in model_name:
            base_memory_gb = 3
            model_type = "3B"
        elif "8B" in model_name:
            base_memory_gb = 8
            model_type = "8B"
        elif "70B" in model_name:
            base_memory_gb = 70
            model_type = "70B"
        else:
            base_memory_gb = 8  # Default assumption
            model_type = "unknown"
        
        # Calculate memory requirements based on quantization
        if args.load_in_4bit:
            estimated_memory_gb = base_memory_gb * 0.25  # 4-bit = ~25% of original
            quantization_type = "4-bit"
        elif args.load_in_8bit:
            estimated_memory_gb = base_memory_gb * 0.5   # 8-bit = ~50% of original
            quantization_type = "8-bit"
        else:
            estimated_memory_gb = base_memory_gb * 2     # FP16 = ~2x for activations
            quantization_type = "FP16"
        
        print(f"Loading {model_type} model ({quantization_type} quantization, ~{estimated_memory_gb:.1f}GB)")
        
        # Tokenizer loading can fail on some HPC/shared-cache setups when the cached
        # `tokenizer.json` (fast tokenizer) is corrupted/truncated (often from an interrupted
        # download) or incompatible with the installed `tokenizers` build.
        #
        # In that case, we (a) force a clean re-download, and (b) fall back to the slow tokenizer.
        def _load_tokenizer(force_download: bool, use_fast: bool):
            return AutoTokenizer.from_pretrained(
                model_name,
                token=getattr(args, "hf_token", None),
                use_fast=use_fast,
                force_download=force_download,
            )

        try:
            _tokenizer = _load_tokenizer(force_download=False, use_fast=True)
        except Exception as e:
            msg = str(e)
            is_modelwrapper = "ModelWrapper" in msg or "TokenizerFast.from_file" in msg
            if is_modelwrapper:
                print(
                    f"Warning: tokenizer cache appears corrupted for {model_name} ({e}). "
                    "Forcing a fresh download and retrying..."
                )
                try:
                    _tokenizer = _load_tokenizer(force_download=True, use_fast=True)
                except Exception as e2:
                    print(
                        f"Warning: fast tokenizer still failed after force डाउनलोड ({e2}). "
                        "Retrying with use_fast=False (slow tokenizer)..."
                    )
                    _tokenizer = _load_tokenizer(force_download=True, use_fast=False)
            else:
                print(f"Warning: failed to load tokenizer for {model_name} ({e}). Retrying with use_fast=False...")
                _tokenizer = _load_tokenizer(force_download=False, use_fast=False)
        
        # Set pad token if not already set
        if _tokenizer.pad_token is None:
            _tokenizer.pad_token = _tokenizer.eos_token
        
        # Check CUDA availability and configure device mapping
        if torch.cuda.is_available():
            visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
            print(f"CUDA is available. CUDA_VISIBLE_DEVICES={visible}")
            device_count = torch.cuda.device_count()
            print(f"CUDA device_count (visible): {device_count}")
            if device_count > 0:
                print(f"Using GPU[0]: {torch.cuda.get_device_name(0)}")
                gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
                print(f"GPU[0] memory: {gpu_memory:.1f} GB")
            device_map = "auto"
            torch_dtype = torch.bfloat16
        else:
            print("CUDA not available. Using CPU.")
            device_map = "cpu"
            torch_dtype = torch.float32
        
        # Configure GPU-only loading
        load_kwargs = {
            "torch_dtype": torch_dtype,
            "device_map": device_map,
            "quantization_config": None,  # Will be set below
            "trust_remote_code": True,
        }
        
        # Configure quantization parameters
        quantization_config = None
        if args.load_in_4bit:
            from transformers import BitsAndBytesConfig
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,  # Use float16 for better memory efficiency
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )
            # GPU-only 4-bit quantization
            load_kwargs["device_map"] = "auto"
            print("Using 4-bit quantization (GPU-only).")
        elif args.load_in_8bit:
            from transformers import BitsAndBytesConfig
            quantization_config = BitsAndBytesConfig(
                load_in_8bit=True,
                bnb_8bit_quant_type="nf4"
            )
            # Use GPU-only for 8-bit quantization with memory limits
            load_kwargs["device_map"] = "auto"
            # NOTE: Don't set an artificially low max_memory; it forces CPU/disk dispatch
            # and triggers `quantizer_bnb_8bit.validate_environment` errors even on A100s.
            print("Using 8-bit quantization (GPU-only).")
        else:
            print("No quantization applied")
        
        # Update quantization config in load_kwargs
        load_kwargs["quantization_config"] = quantization_config
        
        # GPU-only configuration (no CPU offload for speed)
        if torch.cuda.is_available():
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            print(f"GPU: {gpu_memory:.1f}GB available")
            
            # Check if we have enough memory
            if estimated_memory_gb > gpu_memory * 0.9:  # Leave 10% buffer
                print(f"⚠️  Warning: Model may not fit in GPU memory!")
            
            # No CPU offload - keep everything on GPU for speed
        else:
            print("CUDA not available - using CPU only")
        
        _model = AutoModelForCausalLM.from_pretrained(
            model_name,
            **load_kwargs
        )
        
        # Verify model is on the correct device
        if torch.cuda.is_available():
            print(f"Model loaded on GPU")
        else:
            print("Model loaded on CPU")
    print("Model ready!")

def call_llm_to_generate_misinformation_batch(misinformation_prompts, max_new_tokens=None):
    """
    Calls the Llama-3 70B Instruct model via HuggingFace Transformers to generate misinformation for a batch of prompts.
    Optimized for speed with batching and faster generation parameters.
    """
    # Use command line argument if not specified
    if max_new_tokens is None:
        args = get_args()
        max_new_tokens = args.max_tokens
    
    try:
        # Prepare batch of prompts
        chats = []
        for prompt in misinformation_prompts:
            chat = [
                {"role": "user", "content": prompt},
            ]
            formatted_prompt = _tokenizer.apply_chat_template(
                chat,
                tokenize=False,
                add_generation_prompt=True
            )
            chats.append(formatted_prompt)

        # Tokenize batch with proper attention mask and pad token
        tokenized = _tokenizer(chats, return_tensors="pt", padding=True, truncation=True, max_length=512)
        input_ids = tokenized.input_ids.to(_model.device)
        attention_mask = tokenized.attention_mask.to(_model.device)

        with torch.no_grad():
            outputs = _model.generate(
                input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                do_sample=True,           # Enable sampling for faster generation
                temperature=0.7,          # Add some randomness but keep it focused
                top_p=0.9,                # Use nucleus sampling for faster generation
                repetition_penalty=1.1,   # Prevent repetition
                eos_token_id=_tokenizer.eos_token_id,
                pad_token_id=_tokenizer.pad_token_id,
                use_cache=True,           # Enable KV cache for faster generation
            )
        
        # Decode all outputs using per-sample input lengths
        results = []
        input_lengths = attention_mask.sum(dim=1).tolist()
        for i in range(outputs.shape[0]):
            start_idx = int(input_lengths[i])
            decoded = _tokenizer.decode(outputs[i][start_idx:], skip_special_tokens=True)
            # Strip an optional leading role header left by some chat templates
            if decoded.lower().startswith("assistant"):
                # Remove leading 'assistant' or 'assistant:' and following whitespace
                decoded = re.sub(r'^assistant\s*:?', '', decoded, flags=re.IGNORECASE).lstrip()
            results.append(decoded.strip())
        
        # Clear GPU cache after batch generation
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            # Optional: Print memory usage for debugging
            # memory_used = torch.cuda.memory_allocated() / 1024**3
            # print(f"  GPU memory after batch: {memory_used:.2f}GB")
        
        return results
        
    except Exception as e:
        print(f"  Batch generation error: {e}")
        # Return default responses for the batch
        return ["This is misleading information that could confuse someone about the correct answer."] * len(misinformation_prompts)

def call_llm_to_generate_misinformation(misinformation_prompt, max_new_tokens=None):
    """
    Single prompt wrapper for backward compatibility.
    """
    results = call_llm_to_generate_misinformation_batch([misinformation_prompt], max_new_tokens)
    return results[0]
    

def clear_previous_files(selected_datasets=None):
    """
    Clears previous JSON files from previous runs.
    If `selected_datasets` is provided, only clears files for those datasets.
    """
    dataset_to_files = {
        "complex_web_questions": [
            "data/complex_web_questions_random_samples.json",
            "data/complex_web_questions_misinformed.json",
        ],
        "winogrande": [
            "data/winogrande_random_samples.json",
            "data/winogrande_misinformed.json",
        ],
        "ethics_commonsense": [
            "data/ethics_commonsense_random_samples.json",
            "data/ethics_commonsense_misinformed.json",
        ],
    }

    if selected_datasets is None:
        selected_datasets = list(dataset_to_files.keys())

    files_to_remove = []
    for d in selected_datasets:
        files_to_remove.extend(dataset_to_files.get(d, []))
    
    removed_count = 0
    for file_pattern in files_to_remove:
        if os.path.exists(file_pattern):
            os.remove(file_pattern)
            removed_count += 1
            print(f"Removed: {file_pattern}")
    
    if removed_count > 0:
        print(f"Cleared {removed_count} previous files")
    else:
        print("No previous files found to clear")

def remove_sentence_that_indicates_misinformation(misinformation_text):
    """
    Removes the sentence that indicates misinformation from the given misinformation prompt.
    """
    sentences = re.split(r'(?<=[.!?:])\s+', misinformation_text)
    filtered_sentences = [
        s for s in sentences
        if "misinformation" not in s.lower()
    ]
    misinformation_text = ' '.join(filtered_sentences).strip()
    misinformation_text = re.sub(r'[ \t]{2,}', ' ', misinformation_text)
    return misinformation_text

def generate_false_facts_for_samples(samples, dataset_type):
    """
    Generates false facts for all samples in a dataset.
    """
    print(f"Generating false facts for {len(samples)} samples...")
    
    # Generate false fact prompts for all samples
    false_fact_prompts = []
    for sample in samples:
        print(f"Generating false fact prompt for sample: {sample}")
        prompt = generate_false_fact_prompt(sample, dataset_type)
        print(f"False fact prompt for sample: {prompt}")
        false_fact_prompts.append(prompt)
    
    # Generate false facts in batches
    args = get_args()
    batch_size = args.batch_size
    
    all_false_facts = []

    num_batches = ceil(len(false_fact_prompts) / batch_size)
    for batch_idx in tqdm(range(num_batches), desc="Generating false facts"):
        print(f"Generating false facts for batch {batch_idx + 1} of {num_batches}")
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, len(false_fact_prompts))
        batch_prompts = false_fact_prompts[start_idx:end_idx]

        try:
            batch_false_facts = call_llm_to_generate_misinformation_batch(batch_prompts)
            print(f"False facts for batch {batch_idx + 1}: {batch_false_facts}")
            if not isinstance(batch_false_facts, list) or len(batch_false_facts) != len(batch_prompts):
                raise ValueError(f"Batch {batch_idx+1}: Output length mismatch. Expected {len(batch_prompts)}, got {len(batch_false_facts)}")
            all_false_facts.extend(batch_false_facts)

            # Clear memory after each batch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            import gc
            gc.collect()

        except Exception as e:
            tqdm.write(f"Error generating false facts for batch {batch_idx + 1}: {e}")
            # Add default false facts for failed batch
            print(f"ERROR: Failed to generate false facts for batch {batch_idx + 1}")
            all_false_facts.extend([""] * len(batch_prompts))
    
    return all_false_facts

def generate_misinformation_for_all_strategies(samples, false_facts, dataset_type, output_filename=None):
    """
    Generates misinformation in all strategies for each sample based on their false facts.
    Each sample will have one false fact and 8 misinformation texts (one per strategy).
    """
    print(f"Generating misinformation for {len(samples)} samples across {len(STRATEGIES)} strategies...")
    
    # Load existing results if file exists
    existing_results = []
    seen_sentence_keys = set()
    if output_filename and os.path.exists(output_filename):
        try:
            with open(output_filename, 'r', encoding='utf-8') as f:
                existing_results = json.load(f)
            # Build a set of sentence keys to skip duplicates on resume
            for item in existing_results:
                key = item.get('sentence') or item.get('question') or item.get('input')
                if isinstance(key, str) and key.strip() != "":
                    seen_sentence_keys.add(key.strip())
        except:
            existing_results = []
    
    from tqdm import tqdm
    args = get_args()
    batch_size = args.batch_size
    
    # Process each sample
    for sample_idx, (sample, false_fact) in enumerate(tqdm(zip(samples, false_facts), desc="Processing samples")):
        try:
            # Create a copy to avoid modifying the original
            sample_copy = sample.copy()
            
            # Standardize field names first (non-destructive where possible)
            if 'sentence' not in sample_copy:
                if 'question' in sample_copy:
                    sample_copy['sentence'] = sample_copy['question']
                elif 'input' in sample_copy:
                    sample_copy['sentence'] = sample_copy['input']
            # Ensure options
            if 'options' not in sample_copy:
                opt1 = sample_copy.get('option1')
                opt2 = sample_copy.get('option2')
                if opt1 is not None and opt2 is not None:
                    sample_copy['options'] = [opt1, opt2]
            # Normalize ethics labels
            if 'label' in sample_copy and 'answer' not in sample_copy:
                sample_copy['answer'] = sample_copy['label']
                sample_copy['options'] = ['Ethical', 'Unethical']
            
            # Skip if this sample was already processed (dedup by sentence/question/input)
            sentence_key = sample_copy.get('sentence') or sample_copy.get('question') or sample_copy.get('input')
            if isinstance(sentence_key, str) and sentence_key.strip() in seen_sentence_keys:
                tqdm.write(f"Skipping already-processed sample {sample_idx+1}")
                continue

            # Add the false fact
            sample_copy['false_fact'] = remove_sentence_that_indicates_misinformation(false_fact)
            
            # Generate misinformation for all strategies
            misinformation_by_strategy = {}
            
            for strategy in tqdm(STRATEGIES, desc="Generating misinformation for all strategies"):
                # Generate prompt for this strategy
                prompt = generate_misinformation_from_fact_prompt(false_fact, sample_copy, strategy, dataset_type)
                
                try:
                    # Generate misinformation for this strategy
                    misinformation = call_llm_to_generate_misinformation(prompt)
                    misinformation_by_strategy[strategy] = remove_sentence_that_indicates_misinformation(misinformation)
                except Exception as e:
                    tqdm.write(f"Error generating misinformation for sample {sample_idx+1}, strategy {strategy}: {e}")
                    misinformation_by_strategy[strategy] = "This is misleading information that could confuse someone about the correct answer."
            
            # Add all misinformation to the sample
            sample_copy['misinformation_by_strategy'] = misinformation_by_strategy
            
            sample_copy['id'] = str(uuid.uuid4())
            
            # Add to results
            existing_results.append(sample_copy)
            if isinstance(sentence_key, str) and sentence_key.strip() != "":
                seen_sentence_keys.add(sentence_key.strip())
            
            # Save incrementally after each sample
            if output_filename:
                with open(output_filename, 'w', encoding='utf-8') as f:
                    json.dump(existing_results, f, ensure_ascii=False, indent=2)
            
            # Clear memory after each sample
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            import gc
            gc.collect()
            
        except Exception as e:
            tqdm.write(f"Error processing sample {sample_idx+1}: {e}")
            continue
    
    # Final save
    if output_filename:
        with open(output_filename, 'w', encoding='utf-8') as f:
            json.dump(existing_results, f, ensure_ascii=False, indent=2)
    
    return existing_results

if __name__ == "__main__":
    args = get_args()

    if getattr(args, "list_datasets", False):
        print("Available datasets:")
        for d in AVAILABLE_DATASETS:
            print(f"- {d}")
        raise SystemExit(0)
    
    # Load dataset configuration
    config = load_dataset_config()

    selected_datasets = resolve_selected_datasets(args)
    print(f"Selected datasets: {', '.join(selected_datasets)}")
    
    # Handle continue flag - clear previous files unless --continue is specified
    if not getattr(args, 'continue', False):
        print("Clearing previous files...")
        clear_previous_files(selected_datasets)
    else:
        print("Continuing from previous runs (keeping existing files)")

    download_only = getattr(args, "download_only", False)
    if download_only:
        print("Download-only mode: skipping LLM loading and misinformation generation.")
    else:
        _load_llm(args)

    # Download and process the drt/complex_web_questions dataset
    if "complex_web_questions" in selected_datasets:
        print("Loading complex_web_questions dataset...")
        ds = load_dataset("drt/complex_web_questions", 'complex_web_questions')
        split = "train"
        data = ds[split]
        for entry in data:
            entry['id'] = str(uuid.uuid4())
        data = [entry for entry in data if "question" in entry]
        
        # Determine number of samples based on arguments and config
        if args.test_mode:
            num_samples = min(2, len(data))
        elif args.max_samples:
            num_samples = min(args.max_samples, len(data))
        elif config and "datasets" in config and "complex_web_questions" in config.get("datasets", {}):
            num_samples = min(config["datasets"]["complex_web_questions"]["min_sample_size_95pct_5pct"], len(data))
            print(f"Using config-based sample size: {num_samples} (95% CI, 5% MOE)")
        else:
            # Default fallback
            num_samples = min(200, len(data))
            print(f"Using default sample size: {num_samples}")

        # Reuse existing sampled inputs when continuing; otherwise (re)sample and save
        random_samples_path = "data/complex_web_questions_random_samples.json"
        if getattr(args, 'continue', False) and os.path.exists(random_samples_path):
            with open(random_samples_path, 'r', encoding='utf-8') as f:
                samples = json.load(f)
            print(f"Loaded existing samples from {random_samples_path} ({len(samples)})")
        else:
            random_indices = random.sample(range(len(data)), num_samples)
            samples = [data[idx] for idx in random_indices]
            os.makedirs("data", exist_ok=True)
            with open(random_samples_path, "w", encoding="utf-8") as f:
                json.dump(samples, f, ensure_ascii=False, indent=2)
            print(f"Selected {num_samples} samples from complex_web_questions")

        if not download_only:
            # Generate false facts and misinformation for the complex web questions dataset
            output_filename = "data/complex_web_questions_misinformed.json"
            from tqdm import tqdm
            
            print(f"Processing complex_web_questions dataset with new approach")
            
            # Step 1: Generate false facts for all samples
            false_facts = generate_false_facts_for_samples(samples, "complex_web_questions")
            
            # Step 2: Generate misinformation for all strategies based on false facts
            generate_misinformation_for_all_strategies(samples, false_facts, "complex_web_questions", output_filename)

    if "winogrande" in selected_datasets:
        print("Loading winogrande dataset...")
        ds = download_winogrande()

        # Select the split to sample from, e.g., "train"
        split = "train"
        data = ds[split]
        for entry in data:
            entry['id'] = str(uuid.uuid4())
        
        # Determine number of samples based on arguments and config
        if args.test_mode:
            num_samples = min(2, len(data))
        elif args.max_samples:
            num_samples = min(args.max_samples, len(data))
        elif config and "datasets" in config and "winogrande" in config.get("datasets", {}):
            num_samples = min(config["datasets"]["winogrande"]["min_sample_size_95pct_5pct"], len(data))
            print(f"Using config-based sample size: {num_samples} (95% CI, 5% MOE)")
        else:
            # Default fallback
            num_samples = min(200, len(data))
            print(f"Using default sample size: {num_samples}")
        
        # Reuse existing sampled inputs when continuing; otherwise (re)sample and save
        random_samples_path = "data/winogrande_random_samples.json"
        if getattr(args, 'continue', False) and os.path.exists(random_samples_path):
            with open(random_samples_path, 'r', encoding='utf-8') as f:
                samples = json.load(f)
            print(f"Loaded existing samples from {random_samples_path} ({len(samples)})")
        else:
            random_indices = random.sample(range(len(data)), num_samples)
            samples = [data[idx] for idx in random_indices]
            with open(random_samples_path, "w", encoding="utf-8") as f:
                json.dump(samples, f, ensure_ascii=False, indent=2)
            print(f"Selected {num_samples} samples from winogrande")

        if not download_only:
            # Generate false facts and misinformation for the winogrande dataset
            output_filename = "data/winogrande_misinformed.json"
            
            print(f"Processing winogrande dataset with new approach")
            
            # Step 1: Generate false facts for all samples
            false_facts = generate_false_facts_for_samples(samples, "winogrande")
            
            # Step 2: Generate misinformation for all strategies based on false facts
            generate_misinformation_for_all_strategies(samples, false_facts, "winogrande", output_filename)

    if "ethics_commonsense" in selected_datasets:
        print("Loading ethics dataset...")
        # Download and process the hendrycks/ethics dataset (commonsense subset)
        ds = download_ethics("commonsense")
        split = "train"
        data = ds[split]
        for entry in data:
            entry['id'] = str(uuid.uuid4())
        
        # Determine number of samples based on arguments and config
        if args.test_mode:
            num_samples = min(5, len(data))
        elif args.max_samples:
            num_samples = min(args.max_samples, len(data))
        elif config and "datasets" in config and "ethics_commonsense" in config.get("datasets", {}):
            num_samples = min(config["datasets"]["ethics_commonsense"]["min_sample_size_95pct_5pct"], len(data))
            print(f"Using config-based sample size: {num_samples} (95% CI, 5% MOE)")
        else:
            # Default fallback
            num_samples = min(200, len(data))
            print(f"Using default sample size: {num_samples}")
        
        # Reuse existing sampled inputs when continuing; otherwise (re)sample and save
        random_samples_path = "data/ethics_commonsense_random_samples.json"
        if getattr(args, 'continue', False) and os.path.exists(random_samples_path):
            with open(random_samples_path, 'r', encoding='utf-8') as f:
                samples = json.load(f)
            print(f"Loaded existing samples from {random_samples_path} ({len(samples)})")
        else:
            random_indices = random.sample(range(len(data)), num_samples)
            samples = [data[idx] for idx in random_indices]
            with open(random_samples_path, "w", encoding="utf-8") as f:
                json.dump(samples, f, ensure_ascii=False, indent=2)
            print(f"Selected {num_samples} samples from ethics")

        if not download_only:
            # Generate false facts and misinformation for the ethics dataset
            output_filename = "data/ethics_commonsense_misinformed.json"
            
            print(f"Processing ethics dataset with new approach")
            
            # Step 1: Generate false facts for all samples
            false_facts = generate_false_facts_for_samples(samples, "ethics_commonsense")
            
            # Step 2: Generate misinformation for all strategies based on false facts
            generate_misinformation_for_all_strategies(samples, false_facts, "ethics_commonsense", output_filename)

    # Convert datasets to unified format (only possible after misinformed files exist).
    if download_only:
        print("Download-only mode: skipping unified-format conversion.")
        print("Dataset processing and conversion completed!")
        raise SystemExit(0)

    print("Converting datasets to unified format...")
    
    # Process winogrande dataset
    if "winogrande" in selected_datasets:
        winogrande_original_path = "data/winogrande_random_samples.json"
        winogrande_misinformed_path = "data/winogrande_misinformed.json"
        if os.path.exists(winogrande_original_path) and os.path.exists(winogrande_misinformed_path):
            print("Processing winogrande dataset...")
            with open(winogrande_original_path, 'r', encoding='utf-8') as f:
                winogrande_original_data = json.load(f)
            with open(winogrande_misinformed_path, 'r', encoding='utf-8') as f:
                winogrande_misinformed_data = json.load(f)
            
            # Create unified output
            create_unified_dataset_output(winogrande_original_data, winogrande_misinformed_data, "winogrande")
        else:
            print(f"Skipping winogrande conversion as one or both files not found: {winogrande_original_path}, {winogrande_misinformed_path}")

    # Process complex_web_questions dataset
    if "complex_web_questions" in selected_datasets:
        complex_web_questions_original_path = "data/complex_web_questions_random_samples.json"
        complex_web_questions_misinformed_path = "data/complex_web_questions_misinformed.json"
        if os.path.exists(complex_web_questions_original_path) and os.path.exists(complex_web_questions_misinformed_path):
            print("Processing complex_web_questions dataset...")
            with open(complex_web_questions_original_path, 'r', encoding='utf-8') as f:
                complex_web_questions_original_data = json.load(f)
            with open(complex_web_questions_misinformed_path, 'r', encoding='utf-8') as f:
                complex_web_questions_misinformed_data = json.load(f)
            
            # Create unified output
            create_unified_dataset_output(complex_web_questions_original_data, complex_web_questions_misinformed_data, "complex_web_questions")
        else:
            print(f"Skipping complex_web_questions conversion as one or both files not found: {complex_web_questions_original_path}, {complex_web_questions_misinformed_path}")

    # Process ethics dataset
    if "ethics_commonsense" in selected_datasets:
        ethics_commonsense_original_path = "data/ethics_commonsense_random_samples.json"
        ethics_commonsense_misinformed_path = "data/ethics_commonsense_misinformed.json"
        if os.path.exists(ethics_commonsense_original_path) and os.path.exists(ethics_commonsense_misinformed_path):
            print("Processing ethics_commonsense dataset...")
            with open(ethics_commonsense_original_path, 'r', encoding='utf-8') as f:
                ethics_commonsense_original_data = json.load(f)
            with open(ethics_commonsense_misinformed_path, 'r', encoding='utf-8') as f:
                ethics_commonsense_misinformed_data = json.load(f)
            
            # Create unified output
            create_unified_dataset_output(ethics_commonsense_original_data, ethics_commonsense_misinformed_data, "ethics_commonsense")
        else:
            print(f"Skipping ethics_commonsense conversion as one or both files not found: {ethics_commonsense_original_path}, {ethics_commonsense_misinformed_path}")


    print("Dataset processing and conversion completed!")