
import re
from typing import List, Union

from datasets import load_dataset


def normalize_numeric_answer(answer: str) -> str:
    answer = re.sub(r"[$,]", "", answer)
    return answer.strip()


def normalize_text_answer(answer: str) -> str:
    answer = answer.lower().strip()
    answer = re.sub(r"[^\w\s]", "", answer)
    return re.sub(r"\s+", " ", answer)


def _check_numeric_or_substring(predicted: str, expected: Union[str, List[str]]) -> bool:
    if not predicted:
        return False
    expected_answers = expected if isinstance(expected, list) else [expected]
    predicted_norm_num = normalize_numeric_answer(predicted)
    predicted_norm_txt = normalize_text_answer(predicted)
    for exp in expected_answers:
        exp_str = str(exp)
        exp_norm_num = normalize_numeric_answer(exp_str)
        exp_norm_txt = normalize_text_answer(exp_str)
        if predicted_norm_num == exp_norm_num or predicted_norm_txt == exp_norm_txt:
            return True
        try:
            if abs(float(predicted_norm_num) - float(exp_norm_num)) < 1e-6:
                return True
        except (ValueError, TypeError):
            pass
        if exp_norm_txt and (exp_norm_txt in predicted_norm_txt or predicted_norm_txt in exp_norm_txt):
            return True
    return False


def get_dataset_and_lambdas(dataset_name: str, max_samples: int | None = None) -> tuple:
    match dataset_name:
        case "Maxscha/concord-training-data":
            dataset = load_dataset(dataset_name, split="train")
            question_fn = lambda x: x["question"]
            answer_fn = lambda x: ""  # label-free: no reference answers in the uploaded subset
            check_fn = _check_numeric_or_substring

        case "openai/gsm8k":
            dataset = load_dataset("openai/gsm8k", "main", split="train")
            question_fn = lambda x: x["question"]
            answer_fn = lambda x: x["answer"].split("####")[-1].strip()
            check_fn = _check_numeric_or_substring

        case _:
            raise ValueError(f"Unknown dataset: {dataset_name}")

    return dataset, question_fn, answer_fn, check_fn
