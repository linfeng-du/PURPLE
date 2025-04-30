from typing import Callable

import torch
import evaluate


def create_reward(task: str) -> Callable[[list[str], list[str]], torch.Tensor]:
    """Creates reward function for the specified task."""
    if task in {'LaMP-1', 'LaMP-2'}:
        return _classification_reward
    elif task in {'LaMP-3'}:
        return _regression_reward
    elif task in {'LaMP-4', 'LaMP-5', 'LaMP-6', 'LaMP-7'}:
        return _create_generation_reward()
    else:
        raise ValueError(f'Invalid task: {task}')


def _classification_reward(predictions: list[str], targets: list[str]) -> torch.Tensor:
    """Computes classification rewards based on prediction and target sequences."""
    rewards = []

    for prediction, target in zip(predictions, targets):
        reward = float(prediction.strip() == target.strip())
        rewards.append(reward)

    return torch.tensor(rewards, dtype=torch.float)


def _regression_reward(predictions: list[str], targets: list[str]) -> torch.Tensor:
    """Computes regression rewards based on prediction and target sequences."""
    rewards = []

    for prediction, target in zip(predictions, targets):
        target_float = float(target)

        try:
            prediction_float = float(prediction)
        except ValueError:
            if abs(1 - target_float) > abs(5 - target_float):
                prediction_float = 1.
            else:
                prediction_float = 5.

        reward = -abs(prediction_float - target_float)
        rewards.append(reward)

    return torch.tensor(rewards, dtype=torch.float)


def _create_generation_reward() -> Callable[[list[str], list[str]], torch.Tensor]:
    """Creates reward function for generation tasks."""
    rouge_metric = evaluate.load('rouge')

    def generation_reward(predictions: list[str], targets: list[str]) -> torch.Tensor:
        """Computes generation rewards based on prediction and target sequences."""
        rouge_results = rouge_metric.compute(
            predictions=[prediction.strip() for prediction in predictions],
            references=[[target.strip()] for target in targets],
            rouge_types=['rouge1'],
            use_aggregration=False
        )
        return torch.tensor(rouge_results['rouge1'], dtype=torch.float)

    return generation_reward
