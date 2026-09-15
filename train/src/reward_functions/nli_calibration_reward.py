import re
from collections import Counter
from typing import List, Optional
from .accuracy_reward import extract_answer
import wandb
import numpy as np

from src.reward_functions.nli.nli import NLI
from src.reward_functions.nli.clusterer import SemanticClusterer
from src.reward_functions.kernels import apply_kernel

class NLIConsistencyReward:

    def __init__(
        self,
        nli_model_name: str = "cross-encoder/nli-deberta-v3-large",
        use_sigma_annealing: bool = True,
        start_sigma: float = 0.3,
        end_sigma: float = 0.08,
        kernel_type: str = "gaussian",
    ):
        self.nli = NLI(nli_model_name=nli_model_name)
        self.clusterer = SemanticClusterer(nli=self.nli)
        self.use_sigma_annealing = use_sigma_annealing
        self.start_sigma = start_sigma
        self.end_sigma = end_sigma
        self.kernel_type = kernel_type
        self.__name__ = "nli_consistency_reward"
        
    
    def __rich_console__(self):
        return "NLIConsistencyReward"
    
    def __call__(self, prompts: List[str], completions: List[str], **kwargs) -> List[float]:
        # Ensure all prompts in the batch are identical (standard for GRPO/Best-of-N)
        assert len(set([p[0]['content'] for p in prompts])) == 1, "All prompts must be the same for consistency reward."

        # 1. Extract answers and Cluster them
        completion_contents = [c[0]["content"] for c in completions]
        answers_extracted = [extract_answer(c) for c in completion_contents]
        
        # Handle None answers for clustering
        # We treat "No Answer" as a distinct category for clustering purposes
        clustering_inputs = [ans if ans is not None else "Not Answered" for ans in answers_extracted]
        
        # Perform Semantic Clustering
        _, cluster_indices, _, _ = self.clusterer.cluster_responses(responses=clustering_inputs)

        # 2. Determine Training Progress (0.0 to 1.0)
        trainer_state = kwargs.get("trainer_state")
        progress = 0.0
        if trainer_state is not None:
            if hasattr(trainer_state, "max_steps") and trainer_state.max_steps > 0:
                progress = min(1.0, max(0.0, trainer_state.global_step / trainer_state.max_steps))
            elif hasattr(trainer_state, "num_train_epochs") and hasattr(trainer_state, "epoch"):
                progress = min(1.0, max(0.0, trainer_state.epoch / max(trainer_state.num_train_epochs, 1e-8)))

        # 3. Sigma for Gaussian kernel
        # Annealed schedule: start wide (forgiving) -> end narrow (strict)
        if self.use_sigma_annealing:
            current_sigma = self.start_sigma + (self.end_sigma - self.start_sigma) * progress
        else:
            current_sigma = self.start_sigma

        # 4. Calculate Rewards
        rewards = []
        confidences = []
        cluster_confidences = []
        
        # Define the penalty for formatting failure
        MISSING_CONFIDENCE_PENALTY = -1.0 

        for idx, (completion, answer_extracted) in enumerate(zip(completion_contents, answers_extracted)):
            
            # A. Extract Self-Reported Confidence
            raw_confidence = self.extract_confidence(completion)
            
            # --- CRITICAL FIX: Penalize missing tags strictly ---
            if raw_confidence is None:
                rewards.append(MISSING_CONFIDENCE_PENALTY)
                continue 

            # Normalize to [0, 1]
            model_conf = max(0.0, min(100.0, raw_confidence)) / 100.0
            confidences.append(model_conf)

            # B. Calculate Actual Cluster Confidence
            # If the model didn't output an answer text, its cluster confidence is effectively 0
            if answer_extracted is None:
                cluster_conf = 0.0
            else:
                cluster_conf = 0.0
                for cluster in cluster_indices:
                    if idx in cluster:
                        cluster_conf = len(cluster) / len(completion_contents)
                        break
            
            cluster_confidences.append(cluster_conf)

            # C. Kernel-based reward (Gaussian by default; see kernels.apply_kernel)
            reward = apply_kernel(model_conf - cluster_conf, current_sigma, self.kernel_type)

            rewards.append(float(reward))

        # 5. Logging to WandB
        # Only log stats for valid completions (where confidence was found)
        if len(confidences) > 0:
            wandb.log({
                "train/reward/nli_consistency/confidences_mean": np.mean(confidences),
                "train/reward/nli_consistency/confidences_std": np.std(confidences),
                "train/reward/nli_consistency/cluster_conf_mean": np.mean(cluster_confidences),
                "train/reward/nli_consistency/sigma": current_sigma,
                "train/reward/nli_consistency/kernel_type": self.kernel_type,
                "train/reward/nli_consistency/use_sigma_annealing": float(self.use_sigma_annealing),
                "train/reward/nli_consistency/valid_responses": len(confidences),
                "train/reward/nli_consistency/average_reward": np.mean(rewards)
            })
        else:
            # Log that we failed to parse anything
            wandb.log({"train/reward/nli_consistency/valid_responses": 0})

        return rewards

    @staticmethod
    def extract_confidence(completion: str) -> Optional[float]:
        # Matches <confidence>85</confidence> or <confidence> 85% </confidence>
        confidence_pattern = r"<confidence>(.*?)</confidence>"
        match = re.search(confidence_pattern, completion, re.DOTALL)
        if match:
            try:
                text = match.group(1).strip().replace("%", "")
                return float(text)
            except ValueError:
                return None
        return None
    
    # def __call__(self, prompts: List[str], completions: List[str], **kwargs) -> List[float]:
    #     """Reward function for consistency based on NLI (for GRPOTrainer).

    #     Rewards completions where answers are consistent with each other.
    #     This will be logged separately as 'nli_consistency_reward' by the trainer.
    #     """
    #     assert len(set([p[0]['content'] for p in prompts])) == 1, "All prompts must be the same for consistency reward."
        
    #     completions = [c[0]["content"] for c in completions]
    #     answers_before: List[str | None] = [extract_answer(completion) for completion in completions]
        
    #     answers = [ans if ans is not None else "Not Answered" for ans in answers_before] # TODO Come up with better idea
        
    #     clustered_responses, cluster_indices, noncontradiction_scores, entailment_scores = self.clusterer.cluster_responses(responses=answers)
        
    #     trainer_state = kwargs.get("trainer_state")
    #     progress = 0.0
    #     if trainer_state is not None:
    #         max_steps = getattr(trainer_state, "max_steps", None)
    #         if max_steps:
    #             progress = min(1.0, max(0.0, trainer_state.global_step / max_steps))
    #         else:
    #             num_train_epochs = getattr(trainer_state, "num_train_epochs", None)
    #             epoch = getattr(trainer_state, "epoch", None)
    #             if num_train_epochs and epoch is not None:
    #                 progress = min(1.0, max(0.0, epoch / max(num_train_epochs, 1e-8)))

    #     start_allowed_error = 0.30
    #     end_allowed_error = 0.01
    #     allowed_error = start_allowed_error + (end_allowed_error - start_allowed_error) * progress

    #     rewards = []
    #     confidences = []
    #     cluster_confidences = []
    #     confidence_indices = []
        
    #     for idx, (completion, answer_before) in enumerate(zip(completions, answers_before)):
    #         confidence = self.extract_confidence(completion)
    #         if confidence is None:
    #             rewards.append(0.0)
    #             continue
    #         if answer_before is None:
    #             cluster_confidence = 0.0  # If no answer the model should return at least low confidence
    #         else:
    #             cluster_confidence = 0.0
    #             for cluster in cluster_indices:
    #                 if idx in cluster:
    #                     cluster_confidence = len(cluster) / len(answers)
    #                     break
                    
    #         cluster_confidences.append(cluster_confidence)

    #         confidence = max(0.0, min(100.0, confidence)) / 100.0  # Convert to [0, 1]
    #         confidences.append(confidence)
    #         confidence_indices.append(idx)
            
            
    #         diff = abs(confidence - cluster_confidence)  / max(allowed_error, 1e-6)
            
    #         reward = 1 - diff
    #         rewards.append(reward)
            
            
    #     confidences = np.array(confidences)

    #     mean = np.mean(confidences) + 1e-8 if len(confidences) > 0 else 0.0
    #     std = np.std(confidences) + 1e-8 if len(confidences) > 0 else 0.0

        
    #     cluster_confidences = np.array(cluster_confidences)
    #     mean_cluster_confidence = np.mean(cluster_confidences) + 1e-8 if len(cluster_confidences) > 0 else 0.0
    #     std_cluster_confidence = np.std(cluster_confidences) + 1e-8 if len(cluster_confidences) > 0 else 0.0
    #     num_valid_answers = len(cluster_confidences)
        

    #     wandb.log({"train/reward/nli_consistency/confidences/mean": mean})
    #     wandb.log({"train/reward/nli_consistency/confidences/stddev": std})

    #     wandb.log({"train/reward/nli_consistency/allowed_error": allowed_error})
    #     wandb.log({"train/reward/nli_consistency/training_progress": progress})
        
    #     wandb.log({"train/reward/nli_consistency/cluster_confidences/mean": mean_cluster_confidence})
    #     wandb.log({"train/reward/nli_consistency/cluster_confidences/stddev": std_cluster_confidence})
    #     wandb.log({"train/reward/nli_consistency_cluster_confidences/answers": num_valid_answers})
        
    #     return rewards
        

        
    # @staticmethod
    # def extract_confidence(completion: str) -> Optional[str]:
    #     """Extract the confidence level from the completion."""
    #     confidence_pattern = r"<confidence>(.*?)</confidence>"
    #     match = re.search(confidence_pattern, completion, re.DOTALL)
    #     if match:
    #         try:
    #             return float(match.group(1).strip().replace("%", ""))
    #         except ValueError:
    #             return None
    #     return None
