from typing import Callable

import evaluate


def create_metric_function(task: str) -> Callable[[list[str], list[str]], list[float]]:
    if task in {'LaMP-1', 'LaMP-2'}:
        return _create_classification_metric_function(_load_all_labels(task))
    elif task in {'LaMP-3'}:
        return _create_regression_metric_function()
    elif task in {'LaMP-4', 'LaMP-5', 'LaMP-6', 'LaMP-7'}:
        return _create_generation_metric_function()
    else:
        raise ValueError(f'Unsupported task: {task}')


def _load_all_labels(task: str) -> list[str]:
    task_labels = {
        'LaMP-1': ['[1]', '[2]'],
        'LaMP-2': [
            'sci-fi', 'based on a book', 'comedy', 'action',
            'twist ending', 'dystopia', 'dark comedy', 'classic',
            'psychology', 'fantasy', 'romance', 'thought-provoking',
            'social commentary', 'violence', 'true story'
        ],
        'LaMP-3': ['1', '2', '3', '4', '5'],
        'LaMP-4': [],
        'LaMP-5': [],
        'LaMP-6': [],
        'LaMP-7': []
    }
    return task_labels[task]


def _create_classification_metric_function(all_labels: list[str]) -> (
    Callable[[list[str], list[str]], dict[str, float]]
):
    accuracy_metric = evaluate.load('accuracy')
    f1_metric = evaluate.load('f1')

    def classification_metric_function(
        predictions: list[str],
        targets: list[str]
    ) -> dict[str, float]:
        prediction_indices = [map_to_index(prediction) for prediction in predictions]
        target_indices = [map_to_index(target) for target in targets]

        accuracy_results = accuracy_metric.compute(
            predictions=prediction_indices,
            references=target_indices
        )
        f1_results = f1_metric.compute(
            predictions=prediction_indices,
            references=target_indices,
            labels=list(range(len(all_labels))),
            average='macro'
        )
        return {'accuracy': accuracy_results['accuracy'], 'f1': f1_results['f1']}

    def map_to_index(string: str) -> int:
        try:
            return all_labels.index(string.strip())
        except ValueError:
            return -1

    return classification_metric_function


def _create_regression_metric_function() -> (
    Callable[[list[str], list[str]], dict[str, float]]
):
    mae_metric = evaluate.load('mae')
    mse_metric = evaluate.load('mse')

    def regression_metric_function(
        predictions: list[str],
        targets: list[str]
    ) -> dict[str, float]:
        prediction_values = [
            map_to_value(prediction, target)
            for prediction, target in zip(predictions, targets)
        ]
        target_values = [map_to_value(target, target) for target in targets]

        mae_results = mae_metric.compute(
            predictions=prediction_values,
            references=target_values
        )
        rmse_results = mse_metric.compute(
            predictions=prediction_values,
            references=target_values,
            squared=False
        )
        return {'mae': mae_results['mae'], 'rmse': rmse_results['mse']}

    def map_to_value(string: str, target: str) -> float:
        try:
            return float(string)
        except ValueError:
            target_value = float(target)

            if abs(1 - target_value) > abs(5 - target_value):
                return 1.
            else:
                return 5.

    return regression_metric_function


def _create_generation_metric_function() -> (
    Callable[[list[str], list[str]], dict[str, float]]
):
    bleu_metric = evaluate.load('sacrebleu')
    rouge_metric = evaluate.load('rouge')
    meteor_metric = evaluate.load('meteor')

    def generation_metric_function(
        predictions: list[str],
        targets: list[str]
    ) -> dict[str, float]:
        stripped_predictions = [prediction.strip() for prediction in predictions]
        stripped_targets = [[target.strip()] for target in targets]

        bleu_results = bleu_metric.compute(
            predictions=stripped_predictions,
            references=stripped_targets
        )
        rouge_results = rouge_metric.compute(
            predictions=stripped_predictions,
            references=stripped_targets
        )
        meteor_results = meteor_metric.compute(
            predictions=stripped_predictions,
            references=stripped_targets
        )
        return {
            'bleu': bleu_results['score'],
            'rouge-1': rouge_results['rouge1'],
            'rouge-2': rouge_results['rouge2'],
            'rouge-L': rouge_results['rougeL'],
            'rouge-LSum': rouge_results['rougeLsum'],
            'meteor': meteor_results['meteor']
        }

    return generation_metric_function
