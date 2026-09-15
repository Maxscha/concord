
from __future__ import annotations

import json
import os
import pickle
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Optional

from simple_parsing import parse

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from datasets_registry import DATASETS, load, score_results
from eval_calibration_metrics import calculate_calibration_metrics
from generate import (
    load_model_and_tokenizer_from_paths,
    resolve_model_and_adapter_paths,
    run_generation_loop,
    save_final_results,
)


PROMPT_TEMPLATE = """Question: {question}

Please answer the question using the following format by thinking step-by-step, with each step clearly marked:

### Output Format
<step number="1">
[First logical step of reasoning]
</step>
<step number="2">
[Second logical step of reasoning]
</step>
<step number="3">
[Third logical step of reasoning]
</step>
(add more <step number="..."> tags as needed for your reasoning)
<answer>
[Final Answer]
</answer>
<confidence>
[Your confidence level in the answer between 0 and 100%]
</confidence>
"""


@dataclass
class EvalArgs:
    dataset: Literal["gsm8k", "svamp", "math500", "mmlu", "mmlu_pro", "musique"] = field(
        metadata={"help": "Which benchmark to evaluate on."},
    )

    model_path: Optional[str] = field(
        default=None,
        metadata={"help": "HF id or local path of the base model. If omitted and --adapter_path is given, read from adapter_config.json."},
    )
    adapter_path: Optional[str] = field(
        default=None,
        metadata={"help": "Path to a LoRA/PEFT adapter directory. Omit to evaluate an unadapted base model."},
    )

    split: Optional[str] = field(
        default=None,
        metadata={"help": "Dataset split. Defaults to the paper's split for each dataset."},
    )
    max_samples: Optional[int] = field(
        default=None,
        metadata={"help": "Cap on the number of evaluation samples (default: all)."},
    )
    max_length: int = field(
        default=2048,
        metadata={"help": "Maximum new tokens to generate."},
    )
    batch_size: int = field(
        default=1,
        metadata={"help": "Batch size for generation."},
    )
    temperature: float = field(
        default=0.0,
        metadata={"help": "Sampling temperature (0 = greedy)."},
    )

    output_dir: str = field(default="results")
    no_cache: bool = field(
        default=False,
        metadata={"help": "Disable generation caching (otherwise resume from a prior run)."},
    )


def main(args: EvalArgs):
    model_path, adapter_path, run_id = resolve_model_and_adapter_paths(args.model_path, args.adapter_path)

    ds, cfg = load(args.dataset, split=args.split)
    if args.max_samples is not None and args.max_samples > 0:
        ds = ds.select(range(min(args.max_samples, len(ds))))

    print(f"Dataset: {cfg.name} ({cfg.hf_id}, split={args.split or cfg.default_split}, n={len(ds)})")
    print(f"Model: {model_path}" + (f" + adapter {adapter_path}" if adapter_path else ""))

    model, tokenizer = load_model_and_tokenizer_from_paths(model_path, adapter_path)

    results = run_generation_loop(
        model=model,
        tokenizer=tokenizer,
        dataset=ds,
        task_name=cfg.name,
        model_path=run_id,
        max_length=args.max_length,
        prompt_template=PROMPT_TEMPLATE,
        cache_path=f"{args.output_dir}/cache",
        split=args.split or cfg.default_split,
        extract_question_fn=cfg.question_fn,
        extract_answer_fn=cfg.gold_answer_fn,
        batch_size=args.batch_size,
        use_cache=not args.no_cache,
        temperature=args.temperature,
    )
    if results is None:
        print("No results generated.")
        return

    save_final_results(
        results=results,
        task_name=cfg.name,
        model_path=run_id,
        dataset_name=cfg.name,
        split=args.split or cfg.default_split,
        output_dir=args.output_dir,
        temperature=args.temperature,
    )

    model_slug = run_id.replace("/", "_")
    result_file = f"{args.output_dir}/{cfg.name}_{args.split or cfg.default_split}_{model_slug}.pkl"
    with open(result_file, "rb") as f:
        data = pickle.load(f)
    score_results(data["results"], cfg)
    with open(result_file, "wb") as f:
        pickle.dump(data, f)

    metrics = calculate_calibration_metrics(result_file)
    metrics_file = result_file.replace(".pkl", "_metrics.json")
    with open(metrics_file, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\nMetrics saved to {metrics_file}")


if __name__ == "__main__":
    args = parse(EvalArgs)
    main(args)
