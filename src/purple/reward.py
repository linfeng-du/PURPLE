from collections.abc import Callable

import evaluate
import torch

from lamp import LABELS


RewardFn = Callable[[list[str], list[str]], torch.Tensor]


def create_reward_fn(task: str) -> RewardFn:
    if task in {"LaMP-1", "LaMP-2"}:
        return _classification_reward_fn
    elif task in {"LaMP-3"}:
        return _create_regression_reward_fn(LABELS[task])
    elif task in {
        "LaMP-4", "LaMP-5", "LaMP-7", "LongLaMP-2", "LongLaMP-3", "LongLaMP-4"
    }:
        return _create_generation_reward_fn()
    else:
        raise ValueError(f"Invalid task: {task}")


def _classification_reward_fn(
    predictions: list[str],
    references: list[str]
) -> torch.Tensor:
    rewards = [
        float(p.strip() == r.strip())
        for p, r in zip(predictions, references, strict=True)
    ]
    return torch.tensor(rewards, dtype=torch.float32)


def _create_regression_reward_fn(labels: tuple[str, ...]) -> RewardFn:
    min_value = min(float(l) for l in labels)
    max_value = max(float(l) for l in labels)

    def to_float(prediction: str, reference: str) -> float:
        try:
            return float(prediction)
        except ValueError:
            reference = float(reference)

            # Map to the most distant label value
            if abs(reference - min_value) > abs(reference - max_value):
                return min_value
            else:
                return max_value

    def _regression_reward_fn(
        predictions: list[str],
        references: list[str]
    ) -> torch.Tensor:
        rewards = [
            -abs(to_float(p, r) - float(r))
            for p, r in zip(predictions, references, strict=True)
        ]
        return torch.tensor(rewards, dtype=torch.float32)

    return _regression_reward_fn


def _create_generation_reward_fn() -> RewardFn:
    rouge_metric = evaluate.load("rouge")

    def generation_reward_fn(
        predictions: list[str],
        references: list[str]
    ) -> torch.Tensor:
        predictions = [p.strip() for p in predictions]
        references = [[r.strip()] for r in references]

        rouge_results = rouge_metric.compute(
            predictions=predictions,
            references=references,
            rouge_types=["rouge1"],
            use_aggregator=False
        )
        return torch.tensor(rouge_results["rouge1"], dtype=torch.float32)

    return generation_reward_fn
