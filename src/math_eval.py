"""
MATH benchmark evaluation for perturbation experiments (Fig 4b).

Evaluates model accuracy on the MATH (competition_math) dataset.
Extracts \\boxed{} answers from model generations and compares
to ground truth after normalization.
"""

import logging
import re
from typing import Dict, List, Optional, Tuple

import torch
from tqdm import tqdm

logger = logging.getLogger(__name__)

# Few-shot prompt template for MATH problems
MATH_PROMPT_TEMPLATE = (
    "Solve the following math problem. Put your final answer in \\boxed{{}}.\n\n"
    "Problem: {problem}\n\n"
    "Solution:"
)


def extract_boxed_answer(text: str) -> Optional[str]:
    """
    Extract the last \\boxed{...} answer from text, handling nested braces.

    Returns None if no boxed answer is found.
    """
    # Find all \boxed{ occurrences and take the last one
    idx = text.rfind("\\boxed{")
    if idx == -1:
        # Try without backslash (some models output boxed{ directly)
        idx = text.rfind("boxed{")
        if idx == -1:
            return None
        idx += len("boxed{")
    else:
        idx += len("\\boxed{")

    # Track brace depth to handle nested braces
    depth = 1
    end = idx
    while end < len(text) and depth > 0:
        if text[end] == '{':
            depth += 1
        elif text[end] == '}':
            depth -= 1
        end += 1

    if depth != 0:
        return None

    # end-1 because we went one past the closing brace
    return text[idx:end - 1].strip()


def normalize_answer(answer: str) -> str:
    """
    Normalize a mathematical answer for comparison.

    Handles common variations:
    - Whitespace
    - Leading/trailing dollars
    - \\text{} wrappers
    - \\frac{a}{b} -> a/b
    - \\left, \\right
    - \\cdot -> *
    """
    if not answer:
        return ""

    s = answer.strip()

    # Remove surrounding dollar signs
    s = s.strip('$')

    # Remove \text{...} -> ...
    s = re.sub(r'\\text\s*\{([^}]*)\}', r'\1', s)

    # Remove \left \right
    s = s.replace('\\left', '').replace('\\right', '')

    # \frac{a}{b} -> a/b (simple cases)
    s = re.sub(r'\\frac\s*\{([^}]*)\}\s*\{([^}]*)\}', r'(\1)/(\2)', s)

    # \sqrt{a} -> sqrt(a)
    s = re.sub(r'\\sqrt\s*\{([^}]*)\}', r'sqrt(\1)', s)

    # \cdot -> *
    s = s.replace('\\cdot', '*')

    # \times -> *
    s = s.replace('\\times', '*')

    # Remove remaining backslash commands that don't affect the value
    s = re.sub(r'\\[a-zA-Z]+', '', s)

    # Normalize whitespace
    s = re.sub(r'\s+', ' ', s).strip()

    # Remove surrounding parens if they wrap the whole thing
    if s.startswith('(') and s.endswith(')'):
        inner = s[1:-1]
        # Only strip if balanced
        depth = 0
        balanced = True
        for c in inner:
            if c == '(':
                depth += 1
            elif c == ')':
                depth -= 1
            if depth < 0:
                balanced = False
                break
        if balanced and depth == 0:
            s = inner

    return s


def answers_match(predicted: str, ground_truth: str) -> bool:
    """Check if predicted answer matches ground truth after normalization."""
    if not predicted or not ground_truth:
        return False

    pred_norm = normalize_answer(predicted)
    gt_norm = normalize_answer(ground_truth)

    if pred_norm == gt_norm:
        return True

    # Try numeric comparison for simple numbers
    try:
        pred_val = float(pred_norm.replace(',', ''))
        gt_val = float(gt_norm.replace(',', ''))
        return abs(pred_val - gt_val) < 1e-6
    except (ValueError, ZeroDivisionError):
        pass

    return False


@torch.no_grad()
def evaluate_math_accuracy(
    model,
    tokenizer,
    device: torch.device,
    num_problems: int = 500,
    max_new_tokens: int = 512,
    dataset_split: str = "test",
) -> Tuple[float, List[Dict]]:
    """
    Evaluate model accuracy on the MATH benchmark.

    Args:
        model: HuggingFace causal LM (may have perturbed weights)
        tokenizer: corresponding tokenizer
        device: torch device
        num_problems: max number of problems to evaluate
        max_new_tokens: max tokens for generation per problem
        dataset_split: "test" or "train"

    Returns:
        accuracy: float (0 to 1)
        results: list of dicts with problem, prediction, ground_truth, correct
    """
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError(
            "datasets library required. Install with: pip install datasets"
        )

    # Load MATH dataset (cached if pre-downloaded)
    logger.info("Loading MATH dataset...")
    try:
        ds = load_dataset("hendrycks/competition_math", split=dataset_split,
                          trust_remote_code=True)
    except Exception:
        # Fallback: try loading from local cache
        ds = load_dataset("hendrycks/competition_math", split=dataset_split,
                          trust_remote_code=True,
                          download_mode="reuse_cache_if_exists")

    if num_problems < len(ds):
        ds = ds.select(range(num_problems))

    logger.info(f"Evaluating on {len(ds)} MATH problems...")

    results = []
    correct = 0

    for i, example in enumerate(tqdm(ds, desc="MATH evaluation")):
        problem = example["problem"]
        solution = example["solution"]

        # Extract ground truth answer
        gt_answer = extract_boxed_answer(solution)
        if gt_answer is None:
            # Skip problems without boxed answers
            continue

        # Format prompt
        prompt = MATH_PROMPT_TEMPLATE.format(problem=problem)
        input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)

        # Truncate if input is too long (leave room for generation)
        max_input_len = getattr(model.config, 'max_position_embeddings', 2048) - max_new_tokens
        if input_ids.shape[1] > max_input_len:
            input_ids = input_ids[:, -max_input_len:]

        # Generate
        outputs = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=False,  # greedy
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )

        # Decode only the generated portion
        gen_tokens = outputs[0, input_ids.shape[1]:]
        generation = tokenizer.decode(gen_tokens, skip_special_tokens=True)

        # Extract predicted answer
        pred_answer = extract_boxed_answer(generation)

        is_correct = answers_match(pred_answer or "", gt_answer)
        if is_correct:
            correct += 1

        results.append({
            "problem_idx": i,
            "problem": problem[:200],  # truncate for storage
            "ground_truth": gt_answer,
            "prediction": pred_answer,
            "correct": is_correct,
        })

        if (i + 1) % 50 == 0:
            running_acc = correct / len(results)
            logger.info(f"  [{i+1}/{len(ds)}] Running accuracy: {running_acc:.3f}")

    accuracy = correct / len(results) if results else 0.0
    logger.info(f"MATH accuracy: {accuracy:.4f} ({correct}/{len(results)})")

    return accuracy, results
