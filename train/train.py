
import json
import os
import subprocess

import wandb
from peft import LoraConfig
from pprint import pprint
from simple_parsing import parse
from trl import GRPOConfig, GRPOTrainer

from args import TrainingArgs
from src.data import get_dataset_and_lambdas
from src.reward_functions import (
    NLIConsistencyReward,
    SemanticEntropyReward,
    accuracy_reward_func,
    format_reward_func,
    lovec_reward_func,
    rl_doubt_reward_func,
    rlcr_brier_reward_func,
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


def main(args: TrainingArgs):
    os.environ["WANDB_PROJECT"] = args.wandb_project

    if wandb.run is None:
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name,
            mode="offline" if args.wandb_sync_off else "online",
            config=vars(args),
        )

    dataset, template_prompt, template_answer, check_answer_fn = get_dataset_and_lambdas(
        args.dataset, max_samples=args.num_samples
    )
    dataset = dataset.shuffle(seed=args.seed)
    n_samples = min(args.num_samples, len(dataset))
    dataset = dataset.select(range(n_samples))
    print(f"Using dataset: {args.dataset} with {len(dataset)} samples.")

    dataset = dataset.map(
        lambda x: {"prompt": [{"role": "user", "content": PROMPT_TEMPLATE.format(question=template_prompt(x))}]}
    )
    dataset = dataset.map(lambda x: {"output": template_answer(x)})

    peft_config = LoraConfig(
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "v_proj"],
        r=8,
        lora_alpha=32,
        lora_dropout=0.1,
    ) if args.use_peft else None

    reward_funcs = []
    if args.use_format_reward:
        reward_funcs.append(format_reward_func)
    if args.use_accuracy_reward:
        def accuracy_reward_with_checker(prompts, completions, **kwargs):
            kwargs["check_answer_fn"] = check_answer_fn
            return accuracy_reward_func(prompts, completions, **kwargs)
        accuracy_reward_with_checker.__name__ = "accuracy_reward_func"
        reward_funcs.append(accuracy_reward_with_checker)

    if args.reward == "concord-nli":
        reward_funcs.append(
            NLIConsistencyReward(
                nli_model_name=args.nli_model_name,
                use_sigma_annealing=args.use_sigma_annealing,
                start_sigma=args.start_sigma,
                end_sigma=args.end_sigma,
                kernel_type=args.kernel_type,
            )
        )
    elif args.reward == "concord-se":
        reward_funcs.append(
            SemanticEntropyReward(
                nli_model_name=args.nli_model_name,
                use_sigma_annealing=args.use_sigma_annealing,
                start_sigma=args.start_sigma,
                end_sigma=args.end_sigma,
                kernel_type=args.kernel_type,
            )
        )
    elif args.reward in {"rlcr", "rewarding-doubt", "lovec"}:
        if args.dataset == "Maxscha/concord-training-data":
            raise ValueError(
                f"Reward {args.reward!r} is label-based (uses ground-truth answers), but the default "
                "dataset Maxscha/concord-training-data ships prompts only. Pass --dataset openai/gsm8k "
                "(or another labeled dataset) when training the baselines."
            )
        baseline_fns = {
            "rlcr": rlcr_brier_reward_func,
            "rewarding-doubt": rl_doubt_reward_func,
            "lovec": lovec_reward_func,
        }
        reward_funcs.append(baseline_fns[args.reward])
    else:
        raise ValueError(f"Unknown reward: {args.reward}")

    trainer = GRPOTrainer(
        model=args.model,
        reward_funcs=reward_funcs,
        train_dataset=dataset,
        args=GRPOConfig(
            loss_type="dapo",
            seed=args.seed,
            max_prompt_length=2048,
            max_completion_length=2048,
            num_generations=args.num_generations,
            generation_batch_size=args.generation_batch_size,
            num_train_epochs=args.num_epochs,
            scale_rewards=args.scale_rewards,
            temperature=args.temperature,
            bf16=True,
            per_device_train_batch_size=args.per_device_train_batch_size,
            log_completions=args.log_completions,
            gradient_checkpointing=args.gradient_checkpointing,
            gradient_checkpointing_kwargs={"use_reentrant": False},
            learning_rate=args.learning_rate,
            use_vllm=False,
            run_name=args.wandb_run_name,
            beta=args.beta,
            save_only_model=False,
            save_strategy="steps",
            save_steps=args.save_steps,
            output_dir=args.output_dir,
        ),
        peft_config=peft_config,
    )

    trainer.train()

    if args.save_model:
        model_slug = args.model.replace("/", "_")
        output_path = args.output_path or f"models/concord_{args.reward}_{model_slug}_n{args.num_samples}"
        os.makedirs(output_path, exist_ok=True)
        trainer.save_model(output_path)
        with open(os.path.join(output_path, "args.json"), "w") as f:
            json.dump(vars(args), f, indent=4)
        with open(os.path.join(output_path, "trainer_config.json"), "w") as f:
            json.dump(trainer.args.to_dict(), f, indent=4)
        print(f"Saved model to {output_path}")

    if os.getenv("WANDB_MODE") == "offline" and args.wandb_sync_off:
        os.environ["WANDB_MODE"] = "online"
        if wandb.run is not None:
            wandb_path = wandb.run.dir.replace("/files", "")
            wandb.run.finish()
            subprocess.run(["wandb", "sync", wandb_path])


if __name__ == "__main__":
    parsed_args = parse(TrainingArgs, add_config_path_arg=True)
    pprint(vars(parsed_args))
    main(parsed_args)
