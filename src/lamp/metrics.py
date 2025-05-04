from typing import Callable

import evaluate


def create_metric(task: str) -> Callable[[list[str], list[str]], dict[str, float]]:
    if task in {'LaMP-1', 'LaMP-2'}:
        labels = get_labels(task)
        classification_metric = _create_classification_metric(labels)
        return classification_metric
    elif task in {'LaMP-3'}:
        regression_metric = _create_regression_metric()
        return regression_metric
    elif task in {'LaMP-4', 'LaMP-5', 'LaMP-6', 'LaMP-7'}:
        generation_metric = _create_generation_metric()
        return generation_metric
    else:
        raise ValueError(f'Invalid task: {task}')


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


def _create_classification_metric(labels: list[str]) -> Callable[[list[str], list[str]], dict[str, float]]:
    accuracy_metric = evaluate.load('accuracy')
    f1_metric = evaluate.load('f1')

    def map_to_label_index(input_: str) -> int:
        try:
            return labels.index(input_.strip())
        except ValueError:
            return -1

    def classification_metric(predictions: list[str], targets: list[str]) -> dict[str, float]:
        prediction_indices = [map_to_label_index(prediction) for prediction in predictions]
        target_indices = [map_to_label_index(target) for target in targets]

        compute_kwargs = {'predictions': prediction_indices, 'references': target_indices}
        accuracy_results = accuracy_metric.compute(**compute_kwargs)
        f1_results = f1_metric.compute(**compute_kwargs, labels=list(range(len(labels))), average='macro')
        return {'accuracy': accuracy_results['accuracy'], 'f1': f1_results['f1']}

    return classification_metric


def _create_regression_metric() -> Callable[[list[str], list[str]], dict[str, float]]:
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

        compute_kwargs = {'predictions': prediction_floats, 'references': target_floats}
        mae_results = mae_metric.compute(**compute_kwargs)
        rmse_results = mse_metric.compute(**compute_kwargs, squared=False)
        return {'mae': mae_results['mae'], 'rmse': rmse_results['mse']}

    return regression_metric


def _create_generation_metric() -> Callable[[list[str], list[str]], dict[str, float]]:
    rouge_metric = evaluate.load('rouge')

    def generation_metric(predictions: list[str], targets: list[str]) -> dict[str, float]:
        rouge_results = rouge_metric.compute(
            predictions=[prediction.strip() for prediction in predictions],
            references=[[target.strip()] for target in targets],
            rouge_types=['rouge1', 'rougeL']
        )
        return {'rouge-1': rouge_results['rouge1'], 'rouge-L': rouge_results['rougeL']}

    return generation_metric
