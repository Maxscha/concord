# CONCORD: Label-Free Calibration of Verbalized LLM Confidence via Rollout Consistency

[![Paper](https://img.shields.io/badge/paper-OpenReview-b31b1b)](https://openreview.net/pdf?id=ToJd6W2TJc)
[![Venue](https://img.shields.io/badge/COLM-2026-4b44ce)](https://colmweb.org/)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)

**Label-free training for calibrated verbalized confidence via rollout consistency.**

> *CONCORD: Label-Free Calibration of Verbalized LLM Confidence via Rollout Consistency.* Maximilian Schall, Gerard de Melo, Padhraic Smyth. **Accepted at COLM 2026.** · [paper](https://openreview.net/pdf?id=ToJd6W2TJc)

Calibrated confidence — where a model's stated probability matches its actual expected accuracy — is essential whenever confidence scores inform human decisions or feed downstream agents. Existing methods for training models to verbalize calibrated confidence typically require ground-truth correctness labels, which are costly to collect and unavailable for many prompt distributions. Multi-pass consistency methods achieve good calibration but significantly increase inference cost.

**CONCORD** (CONsistency-Calibrated Optimization via Rollout Distributions) is a label-free method that adds no dedicated calibration stage. Adapting a model with GRPO already produces multiple rollouts per prompt; CONCORD repurposes those rollouts as a calibration signal, deriving per-prompt confidence targets from semantic agreement among the sampled answers using either NLI-based clustering (CONCORD-NLI) or semantic entropy (CONCORD-SE). At inference, the model emits a reasoning trace, an answer, and a calibrated confidence score in a single forward pass. Across six benchmarks and four models, CONCORD substantially reduces Expected Calibration Error over unadapted verbalized confidence and label-dependent adaptation baselines, while maintaining competitive AUROC.

This repository contains the exact training and evaluation code used in the paper and is intended to **reproduce the paper's results**.

## Repository layout

The code is split into two subprojects: `train/` for CONCORD training and `eval/` for evaluation.

## Setup

Install [uv](https://docs.astral.sh/uv/) and sync the workspace:

```bash
uv sync
```

This creates one virtualenv shared by `train/` and `eval/`.

## Reproducing the paper

### Training

The paper reports results for four base models trained with identical hyperparameters:

- `Qwen/Qwen2.5-3B-Instruct`
- `Qwen/Qwen2.5-7B-Instruct`
- `meta-llama/Llama-3.2-3B-Instruct`
- `meta-llama/Llama-3.1-8B-Instruct`

To reproduce CONCORD-SE on Qwen-2.5-3B exactly:

```bash
cd train
uv run python train.py \
    --model Qwen/Qwen2.5-3B-Instruct \
    --dataset Maxscha/concord-training-data \
    --num_samples 1600 \
    --reward concord-se \
    --num_generations 10 \
    --temperature 1.0 \
    --start_sigma 0.3 --end_sigma 0.08 \
    --learning_rate 3e-6 --beta 0.1 \
    --output_path models/concord-se_qwen2.5-3b
```

Swap `--reward concord-se` for `--reward concord-nli` to reproduce the NLI variant, and `--model` for any of the other three backbones.

Training arguments (full list in `train/args.py`):

| Flag | Default | Meaning |
|---|---|---|
| `--reward` | `concord-se` | `concord-nli` / `concord-se` (label-free, the paper's methods) or `rlcr` / `rewarding-doubt` / `lovec` (label-based baselines; require a labeled `--dataset`) |
| `--model` | `Qwen/Qwen2.5-3B-Instruct` | HF id of the base model |
| `--dataset` | `Maxscha/concord-training-data` | HF id of the prompt source. The default is the paper's 1600-prompt label-free subset of ODA-Math-460k |
| `--num_samples` | `1600` | Max prompts to use. The default dataset already contains exactly 1600 |
| `--num_generations` | `10` | Rollouts per prompt ($N$) |
| `--temperature` | `1.0` | Rollout sampling temperature |
| `--start_sigma` / `--end_sigma` | `0.3` / `0.08` | Gaussian bandwidth schedule |
| `--use_sigma_annealing` | `True` | Set `False` to hold `--start_sigma` constant |
| `--kernel_type` | `gaussian` | `gaussian`, `laplace`, `cauchy`, or `triangular` |
| `--nli_model_name` | `microsoft/deberta-large-mnli` | 3-class MNLI model for semantic clustering |
| `--beta` | `0.1` | KL regularization coefficient |
| `--learning_rate` | `3e-6` | Constant learning rate |
| `--num_epochs` | `1` | Passes over the subsampled prompts |
| `--per_device_train_batch_size` | `1` | Prompts per GPU step |
| `--generation_batch_size` | `10` | Completions per `model.generate()` call |
| `--use_peft` | `True` | Train with LoRA (`q_proj`, `v_proj`; rank 8, $\alpha=32$) |
| `--gradient_checkpointing` | `False` | Trade throughput for activation memory |
| `--use_format_reward` | `True` | Format reward for well-formed `<answer>`/`<confidence>` |
| `--use_accuracy_reward` | `False` | Label-based accuracy reward (off by default; label-free) |
| `--seed` | `42` | Dataset-shuffle and trainer-init seed |
| `--output_path` | *(auto)* | Where the trained adapter is saved |
| `--wandb_run_name` | `None` | Explicit WandB run name |
| `--wandb_sync_off` | `True` | Run WandB offline and sync at the end |

### Evaluation

```bash
cd eval
uv run python eval.py \
    --dataset gsm8k \
    --adapter_path ../train/models/concord-se_qwen2.5-3b
```

`eval.py` loads the model (optionally with a LoRA adapter), generates one response per prompt, extracts `<answer>` and `<confidence>`, and prints Expected Calibration Error (10 equal-width bins), AUROC, and accuracy. Results are cached to `results/`.

Evaluation arguments (full list in `eval/eval.py`):

| Flag | Default | Meaning |
|---|---|---|
| `--dataset` | *(required)* | `gsm8k`, `svamp`, `math500`, `mmlu`, `mmlu_pro`, or `musique` |
| `--adapter_path` | `None` | Path to a LoRA/PEFT adapter. Omit to evaluate the unadapted base model |
| `--model_path` | *(auto)* | HF id or local path of the base model. If omitted with `--adapter_path`, inferred from `adapter_config.json` |
| `--split` | *(dataset-specific)* | Dataset split (defaults to the paper's split) |
| `--max_samples` | `None` | Cap on evaluation samples (default: all) |
| `--max_length` | `2048` | Max new tokens to generate |
| `--temperature` | `0.0` | Sampling temperature (`0.0` = greedy, matching the paper) |
| `--batch_size` | `1` | Prompts per generation batch |
| `--output_dir` | `results` | Where generations and metrics are written |
| `--no_cache` | `False` | Disable generation caching |

### Multi-pass test-time consistency (TT-NLI, TT-SE)

Reproduce the paper's multi-pass reference baselines (black-box; requires repeated decoding at inference):

```bash
cd eval
uv run python src/run_test_time_consistency.py \
    --model_path Qwen/Qwen2.5-3B-Instruct \
    --task gsm8k \
    --num_samples_per_question 10
```

## Citation

```bibtex
@inproceedings{schall2026concord,
  title     = {{CONCORD}: Label-Free Calibration of Verbalized {LLM} Confidence via Rollout Consistency},
  author    = {Schall, Maximilian and de Melo, Gerard and Smyth, Padhraic},
  booktitle = {Conference on Language Modeling (COLM)},
  year      = {2026}
}
```

## License

See `LICENSE`.
