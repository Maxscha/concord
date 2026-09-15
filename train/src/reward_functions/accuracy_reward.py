import re
from typing import List, Optional, Callable


def extract_answer(completion: str) -> Optional[str]:
    answer_pattern = r"<answer>(.*?)</answer>"
    match = re.search(answer_pattern, completion, re.DOTALL)
    if match:
        return match.group(1).strip()
    return None


def check_accuracy(
    completions: List[str],
    expected_answers: List[str],
    check_answer_fn: Callable[[str, str], bool]
) -> List[float]:
    rewards = []
    for completion, expected in zip(completions, expected_answers):
        extracted = extract_answer(completion)

        # No answer extracted
        if extracted is None:
            rewards.append(0.0)
            continue

        # Empty answer
        if extracted.strip() == "":
            rewards.append(-0.5)
            continue

        # Use dataset-specific checker
        is_correct = check_answer_fn(extracted, expected)
        if is_correct:
            rewards.append(1.5)
        else:
            rewards.append(0.0)

    return rewards



def accuracy_reward_func(prompts, completions, **kwargs):
    completions = [c[0]["content"] for c in completions]
    expected_answers = kwargs['output']
    check_answer_fn = kwargs.get('check_answer_fn')

    if check_answer_fn is None:
        raise ValueError(
            "check_answer_fn must be provided in kwargs. "
            "Get it from get_dataset_and_lambdas() in src/data.py"
        )

    return check_accuracy(completions, expected_answers, check_answer_fn)

