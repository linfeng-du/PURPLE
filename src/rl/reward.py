from typing import Callable

import evaluate


def create_reward_function(task: str) -> Callable[[list[str], list[str]], list[float]]:
    """Create the reward function for the specified task.

    Args:
        task (str): The LaMP task.

    Returns:
        Callable[[list[str], list[str], torch.device], torch.Tensor]:
            The reward function corresponding to the task.
    """
    if task in {'LaMP-1', 'LaMP-2'}:
        return _classification_reward_function
    elif task in {'LaMP-3'}:
        return _regression_reward_function
    elif task in {'LaMP-4', 'LaMP-5', 'LaMP-6', 'LaMP-7'}:
        return _create_generation_reward_function()
    else:
        raise ValueError(f'Unsupported task: {task}')


def _classification_reward_function(predictions: list[str], targets: list[str]) -> list[float]:
    """Compute classification rewards based on prediction and target sequences.

    Args:
        predictions (list[str]): Prediction sequences.
        targets (list[str]): Target sequences.

    Returns:
        rewards (list[float]):
            Rewards indicating whether each prediction matches its target.
    """
    rewards = []

    for prediction, target in zip(predictions, targets):
        reward = float(prediction.strip() == target.strip())
        rewards.append(reward)

    return rewards


def _regression_reward_function(predictions: list[str], targets: list[str]) -> list[float]:
    """Compute regression rewards based on prediction and target sequences.

    Args:
        predictions (list[str]): Prediction sequences.
        targets (list[str]): Target sequences.

    Returns:
        reward (list[float]):
            Rewards computed as the negative L1 distance between predictions and targets.
    """
    rewards = []

    for prediction, target in zip(predictions, targets):
        target_value = float(target)

        try:
            prediction_value = float(prediction)
        except ValueError:
            if abs(1 - target_value) > abs(5 - target_value):
                prediction_value = 1.
            else:
                prediction_value = 5.

        reward = -abs(prediction_value - target_value)
        rewards.append(reward)

    return rewards


def _create_generation_reward_function() -> Callable[[list[str], list[str]], list[float]]:
    """Wrapper function to initialize the ROUGE metric.

    Returns:
        generation_reward_function (Callable[[list[str], list[str]], list[float]]):
            Function that computes the generation reward.
    """
    rouge_metric = evaluate.load('rouge')

    def generation_reward_function(predictions: list[str], targets: list[str]) -> list[float]:
        """Compute the generation reward based on prediction and target sequences.

        Args:
            predictions (list[str]): Prediction sequences.
            targets (list[str]): Target sequences.

        Returns:
            rewards (list[float]):
                Rewards computed as ROUGE-1 scores for prediction-target pairs.
        """
        rewards = []

        for prediction, target in zip(predictions, targets):
            prediction = [prediction.strip()]
            target = [[target.strip()]]
            rouge_results = rouge_metric.compute(predictions=prediction, references=target)
            rewards.append(rouge_results['rouge1'])

        return rewards

    return generation_reward_function
