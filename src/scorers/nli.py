from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification


@dataclass
class NLIScorer:
    model_name: str = "FacebookAI/roberta-large-mnli"
    device: str | None = None

    def __post_init__(self) -> None:
        if self.device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
        self.model.to(self.device)
        self.model.eval()

        # MNLI label order for RoBERTa is typically: contradiction, neutral, entailment
        # We'll handle mapping defensively below.
        self.id2label = {int(k): v for k, v in self.model.config.id2label.items()}

    def _label_probs(self, premise: str, hypothesis: str) -> Tuple[float, float, float]:
        inputs = self.tokenizer(
            premise,
            hypothesis,
            return_tensors="pt",
            truncation=True,
            max_length=256,
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            logits = self.model(**inputs).logits
            probs = torch.softmax(logits, dim=-1).squeeze(0).tolist()

        # Map probs to (contradiction, neutral, entailment) in a robust way.
        # If id2label has names, use them.
        label_map = {self.id2label[i].lower(): probs[i] for i in range(len(probs))}
        c = label_map.get("contradiction", probs[0])
        n = label_map.get("neutral", probs[1] if len(probs) > 1 else 0.0)
        e = label_map.get("entailment", probs[2] if len(probs) > 2 else 0.0)
        return float(c), float(n), float(e)

    def contradiction_prob(self, premise: str, hypothesis: str) -> float:
        c, _, _ = self._label_probs(premise, hypothesis)
        return c