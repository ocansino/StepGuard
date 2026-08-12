from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import List, Optional, Sequence, Tuple

import torch
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
)

from ..execution_metrics import ExecutionMetrics


NLIPair = Tuple[str, str]
LabelProbabilities = Tuple[float, float, float]


@dataclass
class NLIScorer:
    model_name: str = "FacebookAI/roberta-large-mnli"
    device: str | None = None
    metrics: Optional[ExecutionMetrics] = None
    batch_size: int = 32

    def __post_init__(self) -> None:
        if (
            not isinstance(self.batch_size, int)
            or isinstance(self.batch_size, bool)
            or self.batch_size < 1
        ):
            raise ValueError(
                "batch_size must be a positive integer"
            )

        if self.device is None:
            self.device = (
                "cuda"
                if torch.cuda.is_available()
                else "cpu"
            )

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name
        )
        self.model = (
            AutoModelForSequenceClassification
            .from_pretrained(self.model_name)
        )
        self.model.to(self.device)
        self.model.eval()

        self.id2label = {
            int(key): value
            for key, value
            in self.model.config.id2label.items()
        }

    def _label_probs_batch(
        self,
        pairs: Sequence[NLIPair],
    ) -> List[LabelProbabilities]:
        if not pairs:
            return []

        premises = [
            premise
            for premise, _ in pairs
        ]
        hypotheses = [
            hypothesis
            for _, hypothesis in pairs
        ]

        inputs = self.tokenizer(
            premises,
            hypotheses,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=256,
        )
        inputs = {
            key: value.to(self.device)
            for key, value in inputs.items()
        }

        with torch.no_grad():
            logits = self.model(**inputs).logits
            probability_rows = torch.softmax(
                logits,
                dim=-1,
            ).tolist()

        results: List[LabelProbabilities] = []

        for probabilities in probability_rows:
            label_map = {
                self.id2label.get(
                    index,
                    str(index),
                ).lower(): probabilities[index]
                for index in range(len(probabilities))
            }

            contradiction = label_map.get(
                "contradiction",
                probabilities[0],
            )
            neutral = label_map.get(
                "neutral",
                (
                    probabilities[1]
                    if len(probabilities) > 1
                    else 0.0
                ),
            )
            entailment = label_map.get(
                "entailment",
                (
                    probabilities[2]
                    if len(probabilities) > 2
                    else 0.0
                ),
            )

            results.append(
                (
                    float(contradiction),
                    float(neutral),
                    float(entailment),
                )
            )

        return results

    def contradiction_probs(
        self,
        pairs: Sequence[NLIPair],
        *,
        batch_size: Optional[int] = None,
    ) -> List[float]:
        normalized_pairs = list(pairs)

        if not normalized_pairs:
            return []

        resolved_batch_size = (
            self.batch_size
            if batch_size is None
            else batch_size
        )

        if (
            not isinstance(resolved_batch_size, int)
            or isinstance(resolved_batch_size, bool)
            or resolved_batch_size < 1
        ):
            raise ValueError(
                "batch_size must be a positive integer"
            )

        contradictions: List[float] = []

        for batch_start in range(
            0,
            len(normalized_pairs),
            resolved_batch_size,
        ):
            batch = normalized_pairs[
                batch_start:
                batch_start + resolved_batch_size
            ]
            started = perf_counter()

            try:
                batch_probabilities = (
                    self._label_probs_batch(batch)
                )
            except Exception as error:
                if self.metrics is not None:
                    self.metrics.record_operation(
                        provider="local_roberta",
                        operation=(
                            "contradiction_scoring"
                        ),
                        elapsed_seconds=(
                            perf_counter() - started
                        ),
                        success=False,
                        error_type=type(error).__name__,
                    )
                raise

            if self.metrics is not None:
                self.metrics.record_operation(
                    provider="local_roberta",
                    operation="contradiction_scoring",
                    elapsed_seconds=(
                        perf_counter() - started
                    ),
                    success=True,
                )

            contradictions.extend(
                contradiction
                for contradiction, _, _
                in batch_probabilities
            )

        return contradictions

    def contradiction_prob(
        self,
        premise: str,
        hypothesis: str,
    ) -> float:
        return self.contradiction_probs(
            [(premise, hypothesis)],
            batch_size=1,
        )[0]