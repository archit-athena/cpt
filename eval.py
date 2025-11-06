#!/usr/bin/env python3
"""
Evaluate filename prediction using vLLM directly with pass@n sampling.
Generates n predictions per example and calculates pass@n metrics.
"""
from __future__ import annotations

import argparse
import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from datasets import Dataset, load_dataset, load_from_disk
from tqdm import tqdm
from vllm import LLM, SamplingParams

# ---------- Constants ----------
SYSTEM_PROMPT = (
    "You are an expert at analyzing code changes and predicting which files will be modified "
    "in a Rust codebase. Think step by step about which files are likely to be affected "
    "based on the problem description and hints."
)


# ---------- Data classes ----------
@dataclass
class ExampleResult:
    index: int
    instance_id: str
    all_predictions: List[List[str]]  # n predictions
    all_reasoning: List[str] = None  # reasoning for each prediction
    actual_filenames: List[str] = None
    success: bool = True
    pass_at_n: Dict[int, bool] = None  # {n: passed}
    error: Optional[str] = None

    def __post_init__(self):
        if self.pass_at_n is None:
            self.pass_at_n = {}
        if self.all_reasoning is None:
            self.all_reasoning = []
        if self.actual_filenames is None:
            self.actual_filenames = []


@dataclass
class EvalSummary:
    total_examples: int
    successful_predictions: int
    error_rate: float
    pass_at_n: Dict[int, float] = None  # {n: pass_rate}

    def __post_init__(self):
        if self.pass_at_n is None:
            self.pass_at_n = {}


# ---------- Utils ----------
def create_prompt(problem_statement: str, hints_text: Optional[str], max_chars: Optional[int] = None) -> str:
    """Create prompt for filename prediction."""
    prompt = f"{SYSTEM_PROMPT}\n\n"
    prompt += "You are analyzing a code change request for the Hyperswitch payment processing system (a Rust project).\n\n"
    prompt += f"**Problem Statement:**\n{problem_statement.strip()}\n\n"
    if hints_text and hints_text.strip():
        prompt += f"**Hints:**\n{hints_text.strip()}\n\n"
    prompt += (
        "Based on this information, predict which files in the codebase will need to be modified.\n"
        "Respond with a JSON object containing:\n"
        '{"filenames": ["file1.rs", "file2.rs", ...], "reasoning": "explanation"}\n\n'
        "JSON:"
    )

    # Truncate if needed (rough estimate: 1 token ~= 4 chars)
    if max_chars and len(prompt) > max_chars:
        # Keep system prompt and instructions, truncate problem/hints
        header = f"{SYSTEM_PROMPT}\n\nYou are analyzing a code change request for the Hyperswitch payment processing system (a Rust project).\n\n**Problem Statement:**\n"
        footer = (
            "\n\nBased on this information, predict which files in the codebase will need to be modified.\n"
            "Respond with a JSON object containing:\n"
            '{"filenames": ["file1.rs", "file2.rs", ...], "reasoning": "explanation"}\n\n'
            "JSON:"
        )

        available_chars = max_chars - len(header) - len(footer) - 50  # buffer

        # Truncate problem statement and hints
        content = problem_statement.strip()
        if hints_text and hints_text.strip():
            content += f"\n\n**Hints:**\n{hints_text.strip()}"

        if len(content) > available_chars:
            content = content[:available_chars] + "...[truncated]"

        prompt = header + content + footer

    return prompt


def parse_json_response(text: str) -> tuple[List[str], str]:
    """Extract filenames and reasoning from JSON response."""
    try:
        # Try to find JSON in the response
        start = text.find("{")
        end = text.rfind("}") + 1
        if start == -1 or end == 0:
            return [], ""

        json_str = text[start:end]
        data = json.loads(json_str)
        return data.get("filenames", []), data.get("reasoning", "")
    except Exception:
        return [], ""


def calculate_pass_at_k(predictions: List[List[str]], actual: List[str], k: int) -> bool:
    """
    Check if any of the top-k predictions contain at least one actual file.
    predictions: list of n predicted file lists
    actual: ground truth files
    k: number of predictions to consider
    """
    if not actual:
        return True

    actual_set = set(actual)

    # Check if any of the first k predictions has overlap with actual
    for pred_list in predictions[:k]:
        pred_set = set(pred_list)
        if pred_set & actual_set:  # If there's any overlap
            return True

    return False


# ---------- vLLM Evaluation ----------
def evaluate_with_vllm(
    llm: LLM,
    dataset: Dataset,
    n: int = 10,
    max_examples: Optional[int] = None,
    temperature: float = 0.8,
    top_p: float = 0.95,
    max_tokens: int = 2048,
    max_model_len: int = 16192
) -> List[ExampleResult]:
    """
    Evaluate using vLLM with n samples per example.
    """
    limit = min(len(dataset), max_examples) if max_examples is not None else len(dataset)
    results: List[ExampleResult] = []

    logging.info(f"Generating {n} predictions per example for {limit} examples")

    # Calculate max prompt length (reserve space for response)
    max_prompt_tokens =  2048 # bigger buffer

    # Get tokenizer from llm
    tokenizer = llm.get_tokenizer()

    # Prepare all prompts with proper truncation
    prompts = []
    examples = []
    for idx in range(limit):
        example = dataset[idx]

        # Create initial prompt
        problem = example["problem_statement"]
        hints = example.get("hints_text", "")

        # Try full prompt first
        prompt = create_prompt(problem, hints, max_chars=None)

        # Tokenize and check length
        tokens = tokenizer.encode(prompt)

        # If too long, truncate
        if len(tokens) > max_prompt_tokens:
            logging.warning(f"Example {idx}: prompt too long ({len(tokens)} tokens), truncating...")

            # Calculate how much to keep
            header = f"{SYSTEM_PROMPT}\n\nYou are analyzing a code change request for the Hyperswitch payment processing system (a Rust project).\n\n**Problem Statement:**\n"
            footer = (
                "\n\nBased on this information, predict which files in the codebase will need to be modified.\n"
                "Respond with a JSON object containing:\n"
                '{"filenames": ["file1.rs", "file2.rs", ...], "reasoning": "explanation"}\n\n'
                "JSON:"
            )

            header_tokens = len(tokenizer.encode(header))
            footer_tokens = len(tokenizer.encode(footer))

            available_tokens = max_prompt_tokens - header_tokens - footer_tokens - 50

            # Truncate content
            content = problem.strip()
            if hints and hints.strip():
                content += f"\n\n**Hints:**\n{hints.strip()}"

            content_tokens = tokenizer.encode(content)
            if len(content_tokens) > available_tokens:
                content_tokens = content_tokens[:available_tokens]
                content = tokenizer.decode(content_tokens) + "...[truncated]"

            prompt = header + content + footer

        prompts.append(prompt)
        examples.append(example)

    # Generate n outputs per prompt
    sampling_params = SamplingParams(
        n=n,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
    )

    logging.info("Generating predictions with vLLM...")
    outputs = llm.generate(prompts, sampling_params, use_tqdm=True)

    # Process outputs
    for idx, (output, example) in enumerate(zip(outputs, examples)):
        try:
            # Extract filenames and reasoning from all n outputs
            all_predictions = []
            all_reasoning = []
            for completion in output.outputs:
                filenames, reasoning = parse_json_response(completion.text)
                all_predictions.append(filenames)
                all_reasoning.append(reasoning)

            actual = example.get("filenames", []) or []

            result = ExampleResult(
                index=idx,
                instance_id=str(example.get("instance_id", "")),
                all_predictions=all_predictions,
                all_reasoning=all_reasoning,
                actual_filenames=actual,
                success=True
            )
            results.append(result)

        except Exception as exc:
            result = ExampleResult(
                index=idx,
                instance_id=str(example.get("instance_id", "")),
                all_predictions=[],
                actual_filenames=example.get("filenames", []) or [],
                success=False,
                error=str(exc)
            )
            results.append(result)

    return results


def aggregate_metrics(results: List[ExampleResult], k_values: List[int]) -> EvalSummary:
    """
    Aggregate pass@k metrics.
    k_values: list of k values to compute pass@k for
    """
    total = len(results)
    successes = [r for r in results if r.success]
    successful_count = len(successes)
    error_rate = 1.0 - (successful_count / total) if total > 0 else 1.0

    if successful_count == 0:
        return EvalSummary(total_examples=total, successful_predictions=0, error_rate=error_rate, pass_at_n={})

    # Calculate pass@k for each k
    pass_at_k = {}
    for k in k_values:
        passed = sum(1 for r in successes if calculate_pass_at_k(r.all_predictions, r.actual_filenames, k))
        pass_at_k[k] = passed / successful_count

    return EvalSummary(
        total_examples=total,
        successful_predictions=successful_count,
        error_rate=error_rate,
        pass_at_n=pass_at_k
    )


# ---------- I/O ----------
def save_results(results: List[ExampleResult], summary: EvalSummary, output_path: Path) -> None:
    """Save results to JSON."""
    payload = {
        "metrics": asdict(summary),
        "results": [asdict(r) for r in results]
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, ensure_ascii=False)
    logging.info(f"Results written to {output_path}")


def print_summary(summary: EvalSummary) -> None:
    """Print evaluation summary."""
    print("\n" + "=" * 80)
    print("EVALUATION METRICS")
    print("=" * 80)
    print(f"Total Examples:           {summary.total_examples}")
    print(f"Successful Predictions:   {summary.successful_predictions}")
    print(f"Error Rate:               {summary.error_rate * 100:.2f}%")

    if summary.successful_predictions > 0:
        print(f"\n--- Pass@K Metrics ---")
        for k in sorted(summary.pass_at_n.keys()):
            print(f"Pass@{k:2d}:                 {summary.pass_at_n[k] * 100:.2f}%")
    else:
        print("\nNo successful predictions")

    print("=" * 80)


def load_dataset_safely(dataset_ref: str, split: str) -> Dataset:
    """Load dataset from HuggingFace or disk."""
    try:
        ds = load_dataset(dataset_ref, split=split)
        logging.info(f"Loaded dataset from HuggingFace: {dataset_ref} (len={len(ds)})")
        return ds
    except Exception:
        logging.info(f"Could not load from HuggingFace, trying disk path: {dataset_ref}")
        ds = load_from_disk(dataset_ref)
        logging.info(f"Loaded dataset from disk: {dataset_ref} (len={len(ds)})")
        return ds


# ---------- CLI & main ----------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate filename prediction with vLLM pass@n")
    p.add_argument("--dataset", type=str, default="archit11/evals", help="HuggingFace dataset ID or local path")
    p.add_argument("--split", type=str, default="train", help="Dataset split")
    p.add_argument("--model", type=str, default="Qwen/Qwen3-Coder-30B-A3B-Instruct", help="Model name/path")
    p.add_argument("--tokenizer", type=str, default=None, help="Tokenizer name/path (if different from model)")
    p.add_argument("--n", type=int, default=8, help="Number of predictions to generate per example")
    p.add_argument("--max_examples", type=int, default=None, help="Max examples to evaluate")
    p.add_argument("--temperature", type=float, default=0.8, help="Sampling temperature")
    p.add_argument("--top_p", type=float, default=0.95, help="Top-p sampling")
    p.add_argument("--max_tokens", type=int, default=16322, help="Max tokens per generation")
    p.add_argument("--tensor_parallel", type=int, default=4, help="Tensor parallel size (for multi-GPU)")
    p.add_argument("--output", type=Path, default=Path("eval_passn_results.json"), help="Output file")
    p.add_argument("--pass_k", type=str, default="1,3,8", help="Comma-separated k values for pass@k")
    return p.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = parse_args()

    # Parse pass@k values
    k_values = [int(k.strip()) for k in args.pass_k.split(",")]
    # Ensure k values don't exceed n
    k_values = [k for k in k_values if k <= args.n]
    logging.info(f"Computing pass@k for k={k_values}")

    # Load dataset
    dataset = load_dataset_safely(args.dataset, args.split)

    # Initialize vLLM
    logging.info(f"Initializing vLLM with model: {args.model}")
    logging.info(f"Tensor parallel size: {args.tensor_parallel}")

    llm = LLM(
        model=args.model,
        tokenizer=args.tokenizer if args.tokenizer else args.model,
        tensor_parallel_size=args.tensor_parallel,
        trust_remote_code=True,
        max_model_len=16192,
        gpu_memory_utilization=0.9,
        max_num_seqs=256,  # Process up to 256 sequences in parallel
    )

    # Run evaluation
    results = evaluate_with_vllm(
        llm=llm,
        dataset=dataset,
        n=args.n,
        max_examples=args.max_examples,
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
        max_model_len=8192
    )

    # Aggregate and display metrics
    summary = aggregate_metrics(results, k_values)
    print_summary(summary)
    save_results(results, summary, args.output)

    # Print sample predictions
    logging.info("\nSample predictions (first 3 examples):")
    for r in results[:3]:
        logging.info(f"\nExample {r.index} ({r.instance_id}):")
        logging.info(f"  Actual: {r.actual_filenames[:5]}")
        logging.info(f"  Pred 1: {r.all_predictions[0][:5] if r.all_predictions else 'N/A'}")
        logging.info(f"  Pred 2: {r.all_predictions[1][:5] if len(r.all_predictions) > 1 else 'N/A'}")


if __name__ == "__main__":
    main()