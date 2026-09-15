import re
import math
from typing import List, Optional
from .accuracy_reward import extract_answer
import wandb
import numpy as np

from src.reward_functions.nli.nli import NLI
from src.reward_functions.nli.clusterer import SemanticClusterer
from src.reward_functions.kernels import apply_kernel


class SemanticEntropyReward:

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
        self.__name__ = "semantic_entropy_reward"

    def __rich_console__(self):
        return "SemanticEntropyReward"

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

        # Calculate cluster probabilities for semantic entropy
        cluster_probabilities = self._compute_cluster_probabilities(cluster_indices, len(completion_contents))

        # Calculate semantic entropy and convert to confidence
        semantic_entropy = self._compute_semantic_entropy(cluster_probabilities)
        normalized_entropy = self._normalize_entropy(semantic_entropy, len(completion_contents))
        entropy_based_confidence = 1.0 - normalized_entropy

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
        individual_cluster_sizes = []
        num_clusters = len(cluster_probabilities)

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

            # B. Calculate Actual Cluster Confidence based on semantic entropy
            # If the model didn't output an answer text, its cluster confidence is effectively 0
            if answer_extracted is None:
                cluster_conf = 0.0
                individual_cluster_sizes.append(0)
            else:
                # Find which cluster this response belongs to
                cluster_conf = 0.0
                cluster_size = 0
                for cluster_idx, cluster in enumerate(cluster_indices):
                    if idx in cluster:
                        # Use the entropy-based confidence for this response's cluster
                        # The confidence is based on the overall semantic entropy across all clusters
                        cluster_conf = entropy_based_confidence
                        cluster_size = len(cluster)
                        break
                individual_cluster_sizes.append(cluster_size)

            cluster_confidences.append(cluster_conf)

            # C. Kernel-based reward (Gaussian by default; see kernels.apply_kernel)
            reward = apply_kernel(model_conf - cluster_conf, current_sigma, self.kernel_type)

            rewards.append(float(reward))

        # 5. Logging to WandB
        # Only log stats for valid completions (where confidence was found)
        if len(confidences) > 0:
            wandb.log({
                "train/reward/semantic_entropy/confidences_mean": np.mean(confidences),
                "train/reward/semantic_entropy/confidences_std": np.std(confidences),
                "train/reward/semantic_entropy/cluster_conf_mean": np.mean(cluster_confidences),
                "train/reward/semantic_entropy/entropy_value": semantic_entropy,
                "train/reward/semantic_entropy/normalized_entropy": normalized_entropy,
                "train/reward/semantic_entropy/entropy_based_confidence": entropy_based_confidence,
                "train/reward/semantic_entropy/num_clusters": num_clusters,
                "train/reward/semantic_entropy/sigma": current_sigma,
                "train/reward/semantic_entropy/kernel_type": self.kernel_type,
                "train/reward/semantic_entropy/use_sigma_annealing": float(self.use_sigma_annealing),
                "train/reward/semantic_entropy/valid_responses": len(confidences),
                "train/reward/semantic_entropy/average_reward": np.mean(rewards),
                "train/reward/semantic_entropy/avg_cluster_size": np.mean(individual_cluster_sizes) if individual_cluster_sizes else 0
            })
        else:
            # Log that we failed to parse anything
            wandb.log({"train/reward/semantic_entropy/valid_responses": 0})

        return rewards

    @staticmethod
    def _compute_cluster_probabilities(cluster_indices: List[List[int]], total_responses: int) -> List[float]:
        return [len(cluster) / total_responses for cluster in cluster_indices]

    @staticmethod
    def _compute_semantic_entropy(cluster_probabilities: List[float]) -> float:
        return abs(sum([p * math.log(p) if p > 0.0 else 0 for p in cluster_probabilities]))

    @staticmethod
    def _normalize_entropy(entropy: float, num_responses: int) -> float:
        max_entropy = math.log(num_responses) if num_responses > 0 else 1.0
        return entropy / max_entropy if max_entropy > 0 else 0.0

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
