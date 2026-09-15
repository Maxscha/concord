
from __future__ import annotations

import math
import os
import pickle as pkl
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple

import numpy as np
import torch
from datasets import load_dataset
from simple_parsing import parse
from tqdm import tqdm

# eval/src/ -> ../../train/src is the package root for train code
_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT / "train"))

from src.reward_functions.nli.nli import NLI  # noqa: E402
from src.reward_functions.nli.clusterer import SemanticClusterer  # noqa: E402

# Local eval-side imports
from generate import (  # noqa: E402
    resolve_model_and_adapter_paths,
    load_model_and_tokenizer_from_paths,
)


# Reuse the training-time prompt template (grpo_10) verbatim so generations
# carry the same surface form the paper trained against.
PROMPT_TEMPLATE = """Question: {question}

DIRECT ANSWER PROTOCOL

You are a QA bot. Output format is critical for parsing.

PROTOCOL:
- Reasoning: Max 3 sentences
- <answer>: Exact answer, no extra words
- <confidence>: Value 0-100
- End output immediately after </confidence>

Reasoning: [Your 3 sentences max]

<answer>
[Exact answer value only]
</answer>
<confidence>
[0-100]
</confidence>

REQUIREMENT: Nothing after the </confidence> tag."""


ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)


@dataclass
class Args:
    model_path: Optional[str] = field(default=None, metadata={"help": "HF model id of the unadapted base model."})
    adapter_path: Optional[str] = field(default=None, metadata={"help": "Optional LoRA/PEFT adapter path."})
    task: Literal["gsm8k", "svamp", "math500", "mmlu", "mmlu_pro", "musique"] = field(default="gsm8k")
    split: str = field(default="test")
    num_samples_per_question: int = field(default=10, metadata={"help": "N rollouts per question."})
    gen_chunk_size: int = field(default=10, metadata={"help": "Number of completions generated in one model.generate() call. Lower this if you hit OOM on a large model (e.g. 14B with 96GB). Total samples per question are filled by repeating chunks."})
    max_questions: Optional[int] = field(default=None, metadata={"help": "Truncate dataset (debug)."})
    max_length: int = field(default=2048, metadata={"help": "max_new_tokens per completion."})
    temperature: float = field(default=1.0)
    output_dir: str = field(default="results")
    nli_model_name: str = field(default="microsoft/deberta-large-mnli")
    device: str = field(default="cuda:0")


def _load_task(task: str, split: str):
    if task == "gsm8k":
        ds = load_dataset("openai/gsm8k", "main", split=split)
        q_fn = lambda x: x["question"]
        a_fn = lambda x: x["answer"]
    elif task == "svamp":
        ds = load_dataset("ChilleD/SVAMP", split=split)
        q_fn = lambda x: f"{x['Body']} {x['Question']}"
        a_fn = lambda x: x["Answer"]
    elif task == "math500":
        ds = load_dataset("HuggingFaceH4/MATH-500", split=split)
        q_fn = lambda x: x["problem"]
        a_fn = lambda x: x["answer"]
    elif task == "mmlu":
        ds = load_dataset("cais/mmlu", "all", split=split)
        def _q_mmlu(x):
            t = f"{x['question']}\n\nChoices:\n"
            for i, c in enumerate(x["choices"]):
                t += f"{chr(65 + i)}. {c}\n"
            return t
        q_fn = _q_mmlu
        a_fn = lambda x: chr(65 + x["answer"])
    elif task == "mmlu_pro":
        ds = load_dataset("TIGER-Lab/MMLU-Pro", split=split)
        def _q_mmlu_pro(x):
            t = f"{x['question']}\n\nChoices:\n"
            for i, c in enumerate(x["options"]):
                t += f"{chr(65 + i)}. {c}\n"
            return t
        q_fn = _q_mmlu_pro
        a_fn = lambda x: x["answer"]
    elif task == "musique":
        ds = load_dataset("dgslibisey/MuSiQue", split=split)
        def _q_musique(x):
            paragraphs = "\n\n".join(
                f"Paragraph {i + 1}: {p['paragraph_text']}" for i, p in enumerate(x["paragraphs"])
            )
            return f"Context:\n{paragraphs}\n\nQuestion: {x['question']}"
        q_fn = _q_musique
        a_fn = lambda x: x["answer"]
    else:
        raise ValueError(task)
    return ds, q_fn, a_fn


def _sample_completions(
    model,
    tokenizer,
    prompt: str,
    n: int,
    temperature: float,
    max_new_tokens: int,
    device: str,
    chunk_size: int = 10,
) -> List[str]:
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    original_padding_side = tokenizer.padding_side
    tokenizer.padding_side = "left"

    completions: List[str] = []
    try:
        formatted = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
        enc = tokenizer([formatted], return_tensors="pt", padding=True, truncation=False)
        input_ids = enc["input_ids"].to(device)
        attention_mask = enc["attention_mask"].to(device)
        prompt_len = input_ids.shape[1]

        remaining = n
        while remaining > 0:
            this_chunk = min(chunk_size, remaining)
            with torch.no_grad():
                out = model.generate(
                    input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=max_new_tokens,
                    do_sample=True,
                    temperature=temperature,
                    num_return_sequences=this_chunk,
                    pad_token_id=tokenizer.pad_token_id,
                )
            new_tokens = out[:, prompt_len:]
            completions.extend(tokenizer.batch_decode(new_tokens, skip_special_tokens=True))
            remaining -= this_chunk
        return completions
    finally:
        tokenizer.padding_side = original_padding_side


def _cluster_and_score(answers: List[str], clusterer: SemanticClusterer) -> Tuple[List[List[int]], float, float, int]:
    n = len(answers)
    clustering_inputs = [a if a is not None and a.strip() else "Not Answered" for a in answers]
    _, cluster_indices, _, _ = clusterer.cluster_responses(responses=clustering_inputs)

    sizes = [len(c) for c in cluster_indices]
    largest = max(range(len(sizes)), key=lambda i: sizes[i])
    nli_conf = sizes[largest] / n

    probs = [s / n for s in sizes]
    se = -sum(p * math.log(p) for p in probs if p > 0)
    max_se = math.log(n) if n > 1 else 1.0
    se_conf = 1.0 - (se / max_se if max_se > 0 else 0.0)

    return cluster_indices, nli_conf, se_conf, largest


def _build_result_text(majority_answer: str, confidence_unit: float) -> str:
    pct = int(round(max(0.0, min(1.0, confidence_unit)) * 100))
    return (
        f"<answer>\n{majority_answer}\n</answer>\n"
        f"<confidence>\n{pct}\n</confidence>"
    )


def main(args: Args):
    os.makedirs(args.output_dir, exist_ok=True)

    model_path, adapter_path, run_id = resolve_model_and_adapter_paths(args.model_path, args.adapter_path)
    print(f"Loading model {model_path} (adapter={adapter_path})...")
    model, tokenizer = load_model_and_tokenizer_from_paths(model_path, adapter_path, device=args.device)
    model.eval()

    print(f"Loading NLI clusterer {args.nli_model_name}...")
    nli = NLI(nli_model_name=args.nli_model_name)
    clusterer = SemanticClusterer(nli=nli)

    ds, q_fn, a_fn = _load_task(args.task, args.split)
    if args.max_questions is not None and args.max_questions > 0:
        ds = ds.select(range(min(args.max_questions, len(ds))))
    print(f"Loaded task={args.task} split={args.split} with {len(ds)} questions")

    nli_records: List[Dict[str, Any]] = []
    se_records: List[Dict[str, Any]] = []
    raw_records: List[Dict[str, Any]] = []  # for failure-mode breakdown later

    for idx, ex in enumerate(tqdm(ds, desc=f"TTSC {args.task}")):
        question = q_fn(ex)
        gold = a_fn(ex)
        prompt = PROMPT_TEMPLATE.format(question=question)

        completions = _sample_completions(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            n=args.num_samples_per_question,
            temperature=args.temperature,
            max_new_tokens=args.max_length,
            device=args.device,
            chunk_size=args.gen_chunk_size,
        )

        answers_extracted = []
        for c in completions:
            m = ANSWER_RE.search(c)
            answers_extracted.append(m.group(1).strip() if m else None)

        cluster_indices, nli_conf, se_conf, majority_idx = _cluster_and_score(answers_extracted, clusterer)
        majority_cluster = cluster_indices[majority_idx]
        majority_answer = answers_extracted[majority_cluster[0]] or ""

        common = dict(
            sample_idx=idx,
            question=question,
            gold_answer=gold,
            prompt=prompt,
        )

        nli_records.append({
            **common,
            "result_text": _build_result_text(majority_answer, nli_conf),
        })
        se_records.append({
            **common,
            "result_text": _build_result_text(majority_answer, se_conf),
        })
        raw_records.append({
            **common,
            "completions": completions,
            "answers_extracted": answers_extracted,
            "cluster_indices": cluster_indices,
            "nli_confidence": nli_conf,
            "se_confidence": se_conf,
            "majority_answer": majority_answer,
        })

    base_run_id = run_id.replace("/", "_")

    def _save(name_suffix: str, records: List[Dict[str, Any]]):
        out = {
            "config": {
                "model_path": model_path,
                "dataset": args.task,
                "split": args.split,
                "temperature": args.temperature,
                "method": f"tts_{name_suffix}",
                "num_samples_per_question": args.num_samples_per_question,
            },
            "results": records,
        }
        # File name follows the convention used by save_final_results so existing
        # eval scripts pick it up: <task>_<split>_<run_id>.pkl
        fname = f"{args.task}_{args.split}_{base_run_id}__tts_{name_suffix}.pkl"
        path = os.path.join(args.output_dir, fname)
        with open(path, "wb") as f:
            pkl.dump(out, f)
        print(f"Saved {len(records)} records -> {path}")

    _save("nli", nli_records)
    _save("se", se_records)
    _save("raw", raw_records)


if __name__ == "__main__":
    args = parse(Args)
    main(args)
