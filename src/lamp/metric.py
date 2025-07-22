# Adapted from https://github.com/LaMP-Benchmark/LaMP/blob/main/LaMP/metrics/classification_metrics.py
# and https://github.com/LaMP-Benchmark/LaMP/blob/main/LaMP/metrics/generation_metrics.py
import evaluate

from .data_types import Metric


def get_labels(task: str) -> list[str]:
    if task == 'LaMP-1':
        return ['[1]', '[2]']
    elif task == 'LaMP-2':
        return [
            'sci-fi', 'based on a book', 'comedy', 'action', 'twist ending', 'dystopia', 'dark comedy', 'classic',
            'psychology', 'fantasy', 'romance', 'thought-provoking', 'social commentary', 'violence', 'true story'
        ]
    elif task == 'LaMP-3':
        return ['1', '2', '3', '4', '5']
    else:
        raise ValueError(f'Not a classification or regression task: {task}')


def create_metric(task: str) -> Metric:
    if task in {'LaMP-1', 'LaMP-2'}:
        return _create_classification_metric(get_labels(task))
    elif task in {'LaMP-3'}:
        return _create_regression_metric()
    elif task in {'LaMP-4', 'LaMP-5', 'LaMP-6', 'LaMP-7'}:
        return _create_generation_metric()
    elif task in {'LongLaMP-1', 'LongLaMP-2', 'LongLaMP-3', 'LongLaMP-4'}:
        return _create_generation_metric()
    else:
        raise ValueError(f'Invalid task: {task}')


def _create_classification_metric(labels: list[str]) -> Metric:
    accuracy_metric = evaluate.load('accuracy')
    f1_metric = evaluate.load('f1')

    def map_to_label_index(string: str) -> int:
        try:
            return labels.index(string.strip())
        except ValueError:
            return -1

    def classification_metric(predictions: list[str], targets: list[str]) -> dict[str, float]:
        prediction_indices = [map_to_label_index(prediction) for prediction in predictions]
        target_indices = [map_to_label_index(target) for target in targets]

        accuracy_results = accuracy_metric.compute(predictions=prediction_indices, references=target_indices)
        f1_results = f1_metric.compute(
            predictions=prediction_indices,
            references=target_indices,
            labels=list(range(len(labels))),
            average='macro'
        )
        return {'accuracy': accuracy_results['accuracy'], 'f1': f1_results['f1']}

    return classification_metric


def _create_regression_metric() -> Metric:
    mae_metric = evaluate.load('mae')
    mse_metric = evaluate.load('mse')

    def map_to_float(prediction: str, target: str) -> float:
        try:
            return float(prediction)
        except ValueError:
            target_float = float(target)

            if abs(1 - target_float) > abs(5 - target_float):
                return 1.
            else:
                return 5.

    def regression_metric(predictions: list[str], targets: list[str]) -> dict[str, float]:
        prediction_floats = [map_to_float(prediction, target) for prediction, target in zip(predictions, targets)]
        target_floats = [float(target) for target in targets]

        mae_results = mae_metric.compute(predictions=prediction_floats, references=target_floats)
        rmse_results = mse_metric.compute(predictions=prediction_floats, references=target_floats, squared=False)
        return {'mae': mae_results['mae'], 'rmse': rmse_results['mse']}

    return regression_metric


def _create_generation_metric() -> Metric:
    rouge_metric = evaluate.load('rouge')
    meteor_metric = evaluate.load('meteor')

    def generation_metric(predictions: list[str], targets: list[str]) -> dict[str, float]:
        stripped_predictions = [prediction.strip() for prediction in predictions]
        stripped_targets = [target.strip() for target in targets]
        target_lists = [[target] for target in stripped_targets]

        rouge_results = rouge_metric.compute(
            predictions=stripped_predictions,
            references=target_lists,
            rouge_types=['rouge1', 'rougeL']
        )
        meteor_results = meteor_metric.compute(predictions=stripped_predictions, references=target_lists)
        return {
            'rouge-1': rouge_results['rouge1'],
            'rouge-L': rouge_results['rougeL'],
            'meteor': meteor_results['meteor']
        }

    return generation_metric
