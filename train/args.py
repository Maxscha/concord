from dataclasses import dataclass
from typing import Literal, Optional

from simple_parsing import field


@dataclass(kw_only=True)
class TrainingArgs:

    model: str = field(
        default="Qwen/Qwen2.5-3B-Instruct",
        metadata={"help": "HF model id or local path of the base model."},
    )

    dataset: str = field(
        default="Maxscha/concord-training-data",
        metadata={"help": "HF dataset id for training prompts. Default is the 1600-prompt uniform-difficulty subset of ODA-Math-460k used in the paper (prompts only; label-free)."},
    )

    num_samples: int = field(
        default=1600,
        metadata={"help": "Max number of prompts to use. The default dataset already contains exactly 1600."},
    )

    reward: Literal["concord-nli", "concord-se", "rlcr", "rewarding-doubt", "lovec"] = field(
        default="concord-se",
        metadata={"help": "Calibration reward: 'concord-nli' / 'concord-se' (label-free, the paper's methods) or 'rlcr' / 'rewarding-doubt' / 'lovec' (label-based baselines; require a dataset with ground-truth answers)."},
    )

    use_sigma_annealing: bool = field(
        default=True,
        metadata={"help": "Linearly anneal the Gaussian kernel bandwidth from start_sigma to end_sigma over training."},
    )

    start_sigma: float = field(
        default=0.3,
        metadata={"help": "Initial Gaussian kernel bandwidth."},
    )

    end_sigma: float = field(
        default=0.08,
        metadata={"help": "Final Gaussian kernel bandwidth."},
    )

    kernel_type: Literal["gaussian", "laplace", "cauchy", "triangular"] = field(
        default="gaussian",
        metadata={"help": "Similarity kernel mapping (confidence - target) to a reward."},
    )

    nli_model_name: str = field(
        default="cross-encoder/nli-deberta-v3-large",
        metadata={"help": "HF id of the MNLI model used for semantic clustering (must expose a 3-class contradiction/neutral/entailment label mapping)."},
    )

    use_format_reward: bool = field(
        default=True,
        metadata={"help": "Add the format reward that encourages well-formed <answer> and <confidence> tags."},
    )

    use_accuracy_reward: bool = field(
        default=False,
        metadata={"help": "Add a label-based accuracy reward. Disabled by default: CONCORD is label-free."},
    )

    gradient_checkpointing: bool = field(
        default=False,
        metadata={"help": "Enable gradient checkpointing to reduce activation memory."},
    )

    learning_rate: float = field(default=3e-6)

    num_epochs: float = field(default=1)

    temperature: float = field(
        default=1.0,
        metadata={"help": "Sampling temperature for rollouts during training."},
    )

    per_device_train_batch_size: int = field(default=1)

    num_generations: int = field(
        default=10,
        metadata={"help": "Number of rollouts per prompt (N in the paper)."},
    )

    generation_batch_size: int = field(default=10)

    scale_rewards: str = field(
        default="group",
        metadata={"help": "'group' (default) scales rewards by within-group std; 'batch' uses batch-wide std; 'none' disables scaling."},
    )

    beta: float = field(
        default=0.1,
        metadata={"help": "KL regularization coefficient."},
    )

    seed: int = field(
        default=42,
        metadata={"help": "Seed controlling dataset shuffle and trainer initialization."},
    )

    use_peft: bool = field(
        default=True,
        metadata={"help": "Train with LoRA adapters rather than full fine-tuning."},
    )

    save_model: bool = field(default=True)
    output_path: Optional[str] = field(default=None)
    output_dir: str = field(default="./trainer_output")
    save_steps: int = field(default=100)
    log_completions: bool = field(default=False)

    wandb_project: str = field(default="concord")
    wandb_run_name: Optional[str] = field(default=None)
    wandb_sync_off: bool = field(
        default=True,
        metadata={"help": "Run wandb offline during training and sync at the end."},
    )
