# Copyright 2025 CVS Health and/or one of its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
import warnings
import torch
from typing import Any, Dict
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import logging

logging.set_verbosity_error()


class NLI:
    def __init__(self, device: Any = None, verbose: bool = False, nli_model_name: str = "cross-encoder/nli-deberta-v3-large", max_length: int = 2000) -> None:
        # Handle device detection
        if device is None:
            device = torch.device('cuda')
        elif isinstance(device, str):
            device = torch.device('cuda')

        self.device = device
        self.verbose = verbose
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(nli_model_name)
        model = AutoModelForSequenceClassification.from_pretrained(nli_model_name)
        self.model = model.to(self.device) if self.device else model
        # Read label ordering from the model so different MNLI checkpoints don't
        # silently swap "entailment" and "neutral" (e.g. cross-encoder/nli-deberta-v3-base).
        id2label = getattr(self.model.config, "id2label", None)
        if not id2label or len(id2label) != 3:
            raise ValueError(
                f"NLI model {nli_model_name!r} does not expose a 3-class id2label "
                f"(got {id2label!r}). Cannot infer contradiction/neutral/entailment indices."
            )
        self.label_mapping = [id2label[i].lower() for i in range(3)]
        expected = {"contradiction", "neutral", "entailment"}
        if set(self.label_mapping) != expected:
            raise ValueError(
                f"NLI model {nli_model_name!r} has unexpected labels {self.label_mapping!r}; "
                f"expected exactly {expected!r}."
            )
        # Index of the entailment class — used by get_nli_results below.
        self._entailment_idx = self.label_mapping.index("entailment")
        self._contradiction_idx = self.label_mapping.index("contradiction")
        self.probabilities = dict()

    def predict(self, premise: str, hypothesis: str) -> Any:
        if len(premise) > self.max_length or len(hypothesis) > self.max_length:
            warnings.warn("Maximum response length exceeded for NLI comparison. Truncation will occur. To adjust, change the value of max_length")
        concat = premise[0 : self.max_length] + " [SEP] " + hypothesis[0 : self.max_length]
        # truncation=True respects the tokenizer's model_max_length — without it,
        # NLI models with fixed positional embeddings (RoBERTa: max 514 tokens) crash
        # with a CUDA device-side assert when char-truncated inputs are still too long
        # in tokens. DeBERTa happens to be forgiving (relative positions); RoBERTa is not.
        encoded_inputs = self.tokenizer(concat, padding=True, truncation=True, return_tensors="pt")
        if self.device:
            encoded_inputs = {name: tensor.to(self.device) for name, tensor in encoded_inputs.items()}
        logits = self.model(**encoded_inputs).logits
        np_logits = logits.detach().cpu().numpy() if self.device else logits.detach().numpy()
        probabilites = np.exp(np_logits) / np.exp(np_logits).sum(axis=-1, keepdims=True)
        return probabilites

    def get_nli_results(self, response1: str, response2: str) -> Dict[str, Any]:
        if response1 == response2:
            avg_noncontradiction_score, entailment, avg_entailment_score = 1, True, 1
        else:
            left = self.predict(premise=response1, hypothesis=response2)
            left_label = self.label_mapping[left.argmax(axis=1)[0]]

            right = self.predict(premise=response2, hypothesis=response1)
            right_label = self.label_mapping[right.argmax(axis=1)[0]]
            s1 = 1 - left[:, self._contradiction_idx]
            s2 = 1 - right[:, self._contradiction_idx]

            entailment = left_label == "entailment" or right_label == "entailment"
            avg_noncontradiction_score = ((s1 + s2) / 2)[0]
            avg_entailment_score = (
                (left[:, self._entailment_idx] + right[:, self._entailment_idx]) / 2
            )[0]
            self.probabilities.update({f"{response1}_{response2}": left, f"{response2}_{response1}": right})
        return {"noncontradiction_score": avg_noncontradiction_score, "entailment": entailment, "entailment_score": avg_entailment_score}
