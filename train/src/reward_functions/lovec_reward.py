# Based on "Reinforcement Learning for Better Verbalized Confidence in Long-Form Generation"
# https://arxiv.org/abs/2505.23912
import math
import re
from typing import Optional

def extract_confidence(completion: str) -> Optional[float]:
    confidence_pattern = r"<confidence>(.*?)</confidence>"
    match = re.search(confidence_pattern, completion, re.DOTALL)
    if match:
        result = match.group(1).strip().replace("%", "")
        if is_valid_confidence(result):
            return float(result)
    return None

def extract_answer(completion: str) -> Optional[str]:
    answer_pattern = r"<answer>(.*?)</answer>"
    match = re.search(answer_pattern, completion, re.DOTALL)
    if match:
        return match.group(1).strip()
    return None

def is_valid_confidence(confidence_str: str) -> bool:
    try:
        # Remove % sign if present
        confidence_str = confidence_str.strip().replace("%", "")
        confidence_value = float(confidence_str)
        return 0 <= confidence_value and confidence_value <= 100
    except ValueError:
        return False


def lovec_reward_function(confidence: Optional[float], is_correct: bool,
                          lambda_scale: float = 1.0, epsilon: float = 1e-7) -> float:
    # Handle invalid confidence with penalty
    if confidence is None:
        return -lambda_scale * 2.0  # Format penalty

    # Normalize confidence to [0, 1]
    c = confidence / 100.0
    # Clip to avoid log(0)
    c = max(epsilon, min(1.0 - epsilon, c))

    # Factuality: 1 if correct, 0 if incorrect
    f = 1.0 if is_correct else 0.0

    # Calculate log-based reward components
    if is_correct:
        # When correct: reward = log(c), higher confidence is better
        log_component = math.log(c)
    else:
        # When incorrect: reward = log(1-c), lower confidence is better
        log_component = math.log(1.0 - c)

    # R_max normalization factor (using the maximum possible value)
    # For correct: max is at c=1, so log(1) = 0, but we use log(1-epsilon) for incorrect
    # For incorrect: max is at c=0, so log(1) = 0
    # We normalize by the range of log values: log(1-epsilon) to log(epsilon)
    R_max = abs(math.log(epsilon))  # Maximum absolute log value

    # Normalize and scale the reward
    # Formula: λ · (1 + log_component / R_max)
    # This maps the log reward to approximately [0, 2*λ] range
    reward = lambda_scale * (1.0 + log_component / R_max)

    return float(reward)


def lovec_reward_func(prompts, completions, **kwargs) -> list[float]:
    rewards = []
    completions = [c[0]["content"] for c in completions]
    answers = kwargs.get("output")

    # Get lambda scale from kwargs if provided
    lambda_scale = kwargs.get("lovec_lambda", 1.0)

    for completion, ground_truth in zip(completions, answers):
        model_answer = extract_answer(completion)
        confidence_raw = extract_confidence(completion)

        # 1. Determine Correctness
        is_correct = (model_answer and model_answer == ground_truth)

        # 2. Calculate LoVeC Reward
        reward = lovec_reward_function(confidence_raw, is_correct, lambda_scale)
        rewards.append(reward)

    return rewards


def lovec_grpo_reward_function(confidence: Optional[float], is_correct: bool,
                                lambda_scale: float = 1.0, epsilon: float = 1e-7) -> float:
    # Handle invalid confidence with strong penalty
    if confidence is None:
        return -lambda_scale * 3.0

    # Normalize confidence to [0, 1]
    c = confidence / 100.0
    c = max(epsilon, min(1.0 - epsilon, c))

    # Factuality
    f = 1.0 if is_correct else 0.0

    # GRPO formula: combines both correct and incorrect log terms
    # This creates a stronger penalty surface for miscalibration
    correct_term = f * math.log(c)
    incorrect_term = (1.0 - f) * math.log(1.0 - c)

    # Normalizing factor (element-wise product of f and c)
    R_max = abs(math.log(epsilon))

    # Combined reward
    reward = lambda_scale * (1.0 + (correct_term + incorrect_term) / R_max)

    return float(reward)


def lovec_grpo_reward_func(prompts, completions, **kwargs) -> list[float]:
    rewards = []
    completions = [c[0]["content"] for c in completions]
    answers = kwargs.get("output")

    lambda_scale = kwargs.get("lovec_lambda", 1.0)

    for completion, ground_truth in zip(completions, answers):
        model_answer = extract_answer(completion)
        confidence_raw = extract_confidence(completion)

        is_correct = (model_answer and model_answer == ground_truth)

        reward = lovec_grpo_reward_function(confidence_raw, is_correct, lambda_scale)
        rewards.append(reward)

    return rewards
