# Based on https://openreview.net/pdf?id=ASQ649zdHm
import re
from typing import List, Optional

def extract_confidence(completion: str) -> Optional[str]:
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


def rlcr_brier_reward_func(prompts, completions, **kwargs) -> list[float]:
    rewards = []
    completions = [c[0]["content"] for c in completions]
    answers = kwargs.get("output")
    for completion, ground_truth in zip(completions, answers):
        model_answer = extract_answer(completion)
        confidence_q = extract_confidence(completion)
        
        # 1. Determine Correctness (I)
        is_correct = 1.0 if (model_answer and model_answer == ground_truth) else 0.0
        
        # 2. Determine Confidence (q)
        # If confidence is missing/malformed, we cannot calculate Brier score.
        # We assign a penalty (e.g., 0.5) to encourage valid formatting, 
        # or defaults to 0.5 (maximum uncertainty).
        if confidence_q is None:
            q = 0.5 
        else:
            q = int(confidence_q) / 100.0
            
        # 3. Calculate RLCR Reward
        # Formula: Correctness - (Confidence - Correctness)^2
        # If correct (1.0): 1.0 - (q - 1.0)^2  -> Max reward 1.0 (at q=1.0)
        # If wrong (0.0):   0.0 - (q - 0.0)^2  -> Max reward 0.0 (at q=0.0)
        
        brier_score_component = (q - is_correct) ** 2
        reward = is_correct - brier_score_component
        
        rewards.append(reward)
        
    return rewards