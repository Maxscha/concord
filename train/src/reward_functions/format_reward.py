import re
from typing import List



def is_valid_confidence(confidence_str: str) -> bool:
    try:
        # Remove % sign if present
        confidence_str = confidence_str.strip().replace("%", "")
        confidence_value = float(confidence_str)
        return 0 <= confidence_value and confidence_value <= 100
    except ValueError:
        return False


def format_reward(completions: List[str]) -> List[float]:
    rewards = []
    for completion in completions:
        score = 0.0
        penalty = 0.0

        # We get maximum o 1.0 for steps if we have all steps in the text

        # Check for think tag
        # think_pattern = r"<think>(.*?)</think>"
        # think_match = re.search(think_pattern, completion, re.DOTALL)
        # if think_match and think_match.group(1).strip():
        #     score += 0.25

        # Check for answer tag
        answer_pattern = r"<answer>(.*?)</answer>"
        answer_match = re.search(answer_pattern, completion, re.DOTALL)
        if answer_match and answer_match.group(1).strip():
            score += 0.25

        # Check for confidence tag
        confidence_pattern = r"<confidence>(.*?)</confidence>"
        confidence_match = re.search(confidence_pattern, completion, re.DOTALL)

        if confidence_match and confidence_match.group(1).strip():
            score += 0.25
            confidence_content = confidence_match.group(1).strip().replace("%", "")  # Optional %
            if is_valid_confidence(confidence_content):
                score += 0.25  # Extra points for valid confidence

        # Now check for text in wrong places (penalties)
        # Find positions of all tags
        # think_matches = list(re.finditer(think_pattern, completion, re.DOTALL))
        answer_matches = list(re.finditer(answer_pattern, completion, re.DOTALL))
        confidence_matches = list(re.finditer(confidence_pattern, completion, re.DOTALL))

        # Check for text before first step
        # if think_matches:
        #     first_step_start = think_matches[0].start()
        #     text_before_first_step = completion[:first_step_start].strip()
        #     # Ignore common preamble phrases
        #     preamble_phrases = [
        #         "let's solve this step by step",
        #         "let's think step by step",
        #         "let me solve this",
        #         "solution:",
        #         "answer:",
        #         "### Output Format",
        #     ]
        #     text_before_lower = text_before_first_step.lower().strip()
        #     is_preamble = any(phrase == text_before_lower for phrase in preamble_phrases)

        #     if text_before_first_step and not is_preamble:
        #         if len(text_before_first_step) > 70:
        #             penalty += 0.6
        #         elif len(text_before_first_step) > 10:
        #             penalty += 0.3
        #         else:
        #             penalty += 0.1

        # # Check for text between steps and answer
        # if think_matches and answer_matches:
        #     last_step_end = think_matches[-1].end()
        #     first_answer_start = answer_matches[0].start()
        #     text_between = completion[last_step_end:first_answer_start].strip()
        #     if text_between and len(text_between) > 3:
        #         penalty += 0.3

        # Check for text between answer and confidence
        if answer_matches and confidence_matches:
            answer_end = answer_matches[0].end()
            confidence_start = confidence_matches[0].start()
            text_between = completion[answer_end:confidence_start].strip()
            if text_between and len(text_between) > 3:
                penalty += 0.3

        # Check for text after confidence (heavily penalized)
        if confidence_matches:
            last_confidence_end = confidence_matches[-1].end()
            text_after = completion[last_confidence_end:].strip()
            if text_after and len(text_after) > 3:
                penalty += 0.3  # Heavy penalty

        # Apply penalty to score (ensure score doesn't go below 0)
        final_score = max(0.0, score - penalty)
        rewards.append(final_score)

    return rewards

def format_reward_func(prompts, completions, **kwargs):
    completions = [c[0]["content"] for c in completions]
    return format_reward(completions)

