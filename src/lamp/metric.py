# Adapted from https://github.com/LaMP-Benchmark/LaMP/blob/main/LaMP/data/datasets.py
# and https://github.com/LaMP-Benchmark/LaMP/blob/main/LaMP/metrics/classification_metrics.py
# and https://github.com/LaMP-Benchmark/LaMP/blob/main/LaMP/metrics/generation_metrics.py
import evaluate

from .data_types import Metric


def get_labels(task: str) -> list[str]:
    if task == 'LaMP-1':
        return ['[1]', '[2]']
    elif task == 'LaMP-2':
        return [
            'sci-fi', 'based on a book', 'comedy', 'action',
            'twist ending', 'dystopia', 'dark comedy', 'classic',
            'psychology', 'fantasy', 'romance', 'thought-provoking',
            'social commentary', 'violence', 'true story'
        ]
    elif task == 'LaMP-3':
        return ['1', '2', '3', '4', '5']
    else:
        raise ValueError(f'Not a classification or regression task: {task}')


def create_metric(task: str, average: bool = True) -> Metric:
    if task in {'LaMP-1', 'LaMP-2'}:
        return _create_classification_metric(get_labels(task), average)
    elif task in {'LaMP-3'}:
        return _create_regression_metric(average)
    elif task in {'LaMP-4', 'LaMP-5', 'LaMP-6', 'LaMP-7'}:
        return _create_generation_metric(average)
    elif task in {'LongLaMP-1', 'LongLaMP-2', 'LongLaMP-3', 'LongLaMP-4'}:
        return _create_generation_metric(average)
    else:
        raise ValueError(f'Invalid task: {task}')


def _create_classification_metric(labels: list[str], average: bool) -> Metric:
    accuracy_metric = evaluate.load('accuracy')
    f1_metric = evaluate.load('f1')

    def map_to_label_index(string: str) -> int:
        try:
            return labels.index(string.strip())
        except ValueError:
            return -1

    def classification_metric(predictions: list[str], targets: list[str]) -> (
        dict[str, float] | dict[str, list[int]]
    ):
        predictions = [map_to_label_index(prediction) for prediction in predictions]
        targets = [map_to_label_index(target) for target in targets]

        if average:
            accuracy_results = accuracy_metric.compute(predictions=predictions, references=targets)
            f1_results = f1_metric.compute(
                predictions=predictions,
                references=targets,
                labels=list(range(len(labels))),
                average='macro'
            )
            return {'accuracy': accuracy_results['accuracy'], 'f1': f1_results['f1']}
        else:
            accuracy = [int(prediction == target) for prediction, target in zip(predictions, targets)]
            return {'accuracy': accuracy}

    return classification_metric


def _create_regression_metric(average: bool) -> Metric:
    mae_metric = evaluate.load('mae')
    mse_metric = evaluate.load('mse')

    def map_to_float(prediction: str, target: str) -> float:
        try:
            return float(prediction)
        except ValueError:
            target = float(target)

            if abs(1 - target) > abs(5 - target):
                return 1.
            else:
                return 5.

    def regression_metric(predictions: list[str], targets: list[str]) -> (
        dict[str, float] | dict[str, list[float]]
    ):
        predictions = [map_to_float(prediction, target) for prediction, target in zip(predictions, targets)]
        targets = [float(target) for target in targets]

        if average:
            mae_results = mae_metric.compute(predictions=predictions, references=targets)
            rmse_results = mse_metric.compute(predictions=predictions, references=targets, squared=False)
            return {'mae': mae_results['mae'], 'rmse': rmse_results['mse']}
        else:
            return {'mae': [abs(prediction - target) for prediction, target in zip(predictions, targets)]}

    return regression_metric


def _create_generation_metric(average: bool) -> Metric:
    rouge_metric = evaluate.load('rouge')
    meteor_metric = evaluate.load('meteor')

    def generation_metric(predictions: list[str], targets: list[str]) -> (
        dict[str, float] | dict[str, list[float]]
    ):
        predictions = [prediction.strip() for prediction in predictions]
        targets = [[target.strip()] for target in targets]

        rouge_results = rouge_metric.compute(
            predictions=predictions,
            references=targets,
            rouge_types=['rouge1', 'rougeL'],
            use_aggregator=average
        )

        if average:
            meteor_results = meteor_metric.compute(predictions=predictions, references=targets)
        else:
            meteor_results = {'meteor': [
                meteor_metric.compute(predictions=[prediction], references=[target])['meteor']
                for prediction, target in zip(predictions, targets)
            ]}

        return {
            'rouge-1': rouge_results['rouge1'],
            'rouge-L': rouge_results['rougeL'],
            'meteor': meteor_results['meteor']
        }

    return generation_metric
