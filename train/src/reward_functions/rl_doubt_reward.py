# Based on RL-Doubt reward function from paper
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


# RL-Doubt reward function parameters
scale = 10.0
max_reward = -0.0010005003335835344
min_reward = -6.907755278982137 / 2
wrong_format_penalty = -scale * 3.0

def reward_function(confidence: Optional[float], is_answer_correct: bool) -> float:
    if confidence is None or confidence > 10 or confidence < 0:
        return wrong_format_penalty

    normalized_confidence = min(0.999, max(0.001, confidence / 10))

    if is_answer_correct:
        score = math.log(normalized_confidence)
    else:
        score = math.log(1 - normalized_confidence)

    norm_score = (score - min_reward) / (max_reward - min_reward)
    if is_answer_correct:
        norm_score += 0.25
    return float(scale * norm_score)


def rl_doubt_reward_func(prompts, completions, **kwargs) -> list[float]:
    rewards = []
    completions = [c[0]["content"] for c in completions]
    answers = kwargs.get("output")

    for completion, ground_truth in zip(completions, answers):
        model_answer = extract_answer(completion)
        confidence_raw = extract_confidence(completion)

        # 1. Determine Correctness
        is_correct = (model_answer and model_answer == ground_truth)

        # 2. Convert confidence from 0-100 scale to 0-10 scale
        # The original paper uses 0-10 scale
        if confidence_raw is not None:
            confidence = confidence_raw / 10.0  # Convert 0-100 to 0-10
        else:
            confidence = None

        # 3. Calculate RL-Doubt Reward
        reward = reward_function(confidence, is_correct)
        rewards.append(reward)

    return rewards
