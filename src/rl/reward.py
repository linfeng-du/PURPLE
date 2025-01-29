from typing import Callable

import torch
import evaluate


def create_reward_function(task: str) -> (
    Callable[[list[str], list[str], torch.device], torch.Tensor]
):
    """Create the reward function for the specified task.

    Args:
        task (str): The LaMP task.

    Returns:
        Callable[[list[str], list[str], torch.device], torch.Tensor]:
            The reward function corresponding to the task.
    """
    task_fn = {
        'LaMP-1': classification_reward_function,
        'LaMP-2': classification_reward_function,
        'LaMP-3': regression_reward_function,
        'LaMP-4': create_generation_reward_function(),
        'LaMP-5': create_generation_reward_function(),
        'LaMP-6': create_generation_reward_function(),
        'LaMP-7': create_generation_reward_function()
    }
    return task_fn[task]


def classification_reward_function(
    predictions: list[str],
    targets: list[str],
    device: torch.device
) -> torch.Tensor:
    """Compute the classification reward based on prediction and target sequences.

    Args:
        predictions (list[str]): Prediction sequences.
        targets (list[str]): Target sequences.
        device (torch.device): Device for the reward tensor.

    Returns:
        reward (torch.Tensor):
            Whether the prediction matches the target. Shape (batch_size,)
    """
    comparisons = []

    for prediction, target in zip(predictions, targets):
        comparison = prediction.strip() == target.strip()
        comparisons.append(comparison)

    reward = torch.tensor(comparisons, dtype=torch.float, device=device)
    return reward


def regression_reward_function(
    predictions: list[str],
    targets: list[str],
    device: torch.device
) -> torch.Tensor:
    """Compute the regression reward based on prediction and label sequences.

    Args:
        predictions (list[str]): Prediction sequences.
        targets (list[str]): Target sequences.
        device (torch.device): Device for the reward tensor.

    Returns:
        reward (torch.Tensor):
            The negative L1 distance between the prediction and the target. Shape (batch_size,)
    """
    negative_distances = []

    for prediction, target in zip(predictions, targets):
        target_value = float(target)

        try:
            prediction_value = float(prediction)
        except ValueError:
            if abs(1 - target_value) > abs(5 - target_value):
                prediction_value = 1.
            else:
                prediction_value = 5.

        negative_distance = -abs(prediction_value - target_value)
        negative_distances.append(negative_distance)

    reward = torch.tensor(negative_distances, dtype=torch.float, device=device)
    return reward


def create_generation_reward_function() -> (
    Callable[[list[str], list[str], torch.device], torch.Tensor]
):
    """Wrapper function to initialize the ROUGE metric.

    Returns:
        generation_reward (Callable): Function that computes the generation reward.
    """
    rouge_metric = evaluate.load('rouge')

    def generation_reward_function(
        predictions: list[str],
        targets: list[str],
        device: torch.device
    ) -> torch.Tensor:
        """Compute the generation reward based on the ROUGE-1 score.

        Args:
            predictions (list[str]): Prediction sequences.
            targets (list[str]): Target sequences.
            device (torch.device): Device for the reward tensor.

        Returns:
            reward (torch.Tensor):
                The ROUGE-1 score of the prediction. Shape (batch_size,)
        """
        rouge_scores = []

        for prediction, target in zip(predictions, targets):
            prediction = [prediction.strip()]
            target = [[target.strip()]]
            rouge_results = rouge_metric.compute(predictions=prediction, references=target)
            rouge_scores.append(rouge_results['rouge1'])

        reward = torch.tensor(rouge_scores, dtype=torch.float, device=device)
        return reward

    return generation_reward_function
