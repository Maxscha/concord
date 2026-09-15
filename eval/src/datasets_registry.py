
from __future__ import annotations

import re
import string
from dataclasses import dataclass
from typing import Any, Callable, Optional

from datasets import load_dataset


_ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)
_BOXED_RE = re.compile(r"\\boxed\{(.+?)\}", re.DOTALL)
_NUMBER_RE = re.compile(r"-?\d+(?:,\d{3})*(?:\.\d+)?")


def _extract_answer_text(prediction: str) -> Optional[str]:
    m = _ANSWER_RE.search(prediction)
    return m.group(1).strip() if m else None


def _extract_last_number(text: str) -> Optional[str]:
    numbers = _NUMBER_RE.findall(text)
    return numbers[-1].replace(",", "") if numbers else None


def _extract_letter(text: str, max_letter: str = "J") -> Optional[str]:
    m = re.search(rf"\b([A-{max_letter}])\b", text, re.IGNORECASE)
    return m.group(1).upper() if m else None


def _normalize_qa(text: str) -> str:
    text = text.lower()
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    text = "".join(ch for ch in text if ch not in set(string.punctuation))
    return " ".join(text.split())


def _match_numeric(model: Optional[str], gold: str) -> bool:
    if not model:
        return False
    model_num = _extract_last_number(model)
    gold_num = _extract_last_number(gold)
    if model_num is None or gold_num is None:
        return False
    try:
        return abs(float(model_num) - float(gold_num)) < 1e-6
    except ValueError:
        return False


def _match_math_verify(model: Optional[str], gold: str) -> bool:
    if not model:
        return False
    inner = model
    boxed = _BOXED_RE.search(inner)
    if boxed:
        inner = boxed.group(1).strip()
    try:
        from math_verify import LatexExtractionConfig, parse, verify
        gold_parsed = parse(f"${gold}$", extraction_config=[LatexExtractionConfig()])
        model_parsed = parse(f"${inner}$", extraction_config=[LatexExtractionConfig()])
        if gold_parsed and model_parsed and verify(gold_parsed, model_parsed):
            return True
    except Exception:
        pass
    normalize = lambda s: s.strip().replace("$", "").replace(" ", "").replace("\\!", "").strip("{}")
    return normalize(inner) == normalize(gold)


def _match_letter(model: Optional[str], gold: str, max_letter: str = "J") -> bool:
    if not model:
        return False
    letter = _extract_letter(model, max_letter=max_letter)
    return letter is not None and letter == gold.upper()


def _match_normalized(model: Optional[str], gold: str) -> bool:
    if not model:
        return False
    return _normalize_qa(model) == _normalize_qa(gold)


def _format_mmlu(example: dict, choices_key: str = "choices") -> str:
    q = f"{example['question']}\n\nChoices:\n"
    for i, choice in enumerate(example[choices_key]):
        q += f"{chr(65 + i)}. {choice}\n"
    return q


def _format_musique(example: dict) -> str:
    paragraphs = "\n\n".join(
        f"Paragraph {i + 1}: {p['paragraph_text']}" for i, p in enumerate(example["paragraphs"])
    )
    return f"Context:\n{paragraphs}\n\nQuestion: {example['question']}"


@dataclass
class DatasetConfig:
    name: str
    hf_id: str
    hf_config: Optional[str]
    default_split: str
    question_fn: Callable[[dict], str]
    gold_answer_fn: Callable[[dict], str]
    match_fn: Callable[[Optional[str], str], bool]


DATASETS: dict[str, DatasetConfig] = {
    "gsm8k": DatasetConfig(
        name="gsm8k",
        hf_id="openai/gsm8k",
        hf_config="main",
        default_split="test",
        question_fn=lambda x: x["question"],
        gold_answer_fn=lambda x: x["answer"].split("####")[-1].strip(),
        match_fn=_match_numeric,
    ),
    "svamp": DatasetConfig(
        name="svamp",
        hf_id="ChilleD/SVAMP",
        hf_config=None,
        default_split="test",
        question_fn=lambda x: x["question_concat"],
        gold_answer_fn=lambda x: str(x["Answer"]),
        match_fn=_match_numeric,
    ),
    "math500": DatasetConfig(
        name="math500",
        hf_id="HuggingFaceH4/MATH-500",
        hf_config=None,
        default_split="test",
        question_fn=lambda x: x["problem"],
        gold_answer_fn=lambda x: x["answer"],
        match_fn=_match_math_verify,
    ),
    "mmlu": DatasetConfig(
        name="mmlu",
        hf_id="cais/mmlu",
        hf_config="all",
        default_split="test",
        question_fn=_format_mmlu,
        gold_answer_fn=lambda x: chr(65 + x["answer"]),
        match_fn=lambda m, g: _match_letter(m, g, max_letter="D"),
    ),
    "mmlu_pro": DatasetConfig(
        name="mmlu_pro",
        hf_id="TIGER-Lab/MMLU-Pro",
        hf_config=None,
        default_split="test",
        question_fn=lambda x: _format_mmlu(x, choices_key="options"),
        gold_answer_fn=lambda x: x["answer"],
        match_fn=lambda m, g: _match_letter(m, g, max_letter="J"),
    ),
    "musique": DatasetConfig(
        name="musique",
        hf_id="dgslibisey/MuSiQue",
        hf_config=None,
        default_split="validation",
        question_fn=_format_musique,
        gold_answer_fn=lambda x: x["answer"],
        match_fn=_match_normalized,
    ),
}


def load(name: str, split: Optional[str] = None):
    cfg = DATASETS[name]
    actual_split = split or cfg.default_split
    if cfg.hf_config:
        ds = load_dataset(cfg.hf_id, cfg.hf_config, split=actual_split)
    else:
        ds = load_dataset(cfg.hf_id, split=actual_split)
    return ds, cfg


def score_results(results: list[dict], cfg: DatasetConfig) -> None:
    for sample in results:
        model_ans = _extract_answer_text(sample["result_text"])
        sample["is_correct"] = cfg.match_fn(model_ans, sample["gold_answer"])
