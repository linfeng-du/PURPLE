# Adapted from:
#   https://github.com/LaMP-Benchmark/LaMP/blob/main/LaMP/data/datasets.py
#   https://github.com/LaMP-Benchmark/LaMP/blob/main/LaMP/metrics/classification_metrics.py
#   https://github.com/LaMP-Benchmark/LaMP/blob/main/LaMP/metrics/generation_metrics.py
from types import MappingProxyType
from typing import Callable, TypeAlias

import evaluate


MetricFn: TypeAlias = Callable[
    [list[str], list[str]], dict[str, float | list[float]]
]


LABELS = MappingProxyType({
    "LaMP-1": ("[1]", "[2]"),
    "LaMP-2": (
        "sci-fi",
        "based on a book",
        "comedy",
        "action",
        "twist ending",
        "dystopia",
        "dark comedy",
        "classic",
        "psychology",
        "fantasy",
        "romance",
        "thought-provoking",
        "social commentary",
        "violence",
        "true story"
    ),
    "LaMP-3": ("1", "2", "3", "4", "5")
})


def create_metric_fn(task: str, aggregate: bool = True) -> MetricFn:
    if task in {"LaMP-1", "LaMP-2"}:
        return _create_classification_metric_fn(LABELS[task], aggregate)
    elif task in {"LaMP-3"}:
        return _create_regression_metric_fn(LABELS[task], aggregate)
    elif task in {
        "LaMP-4", "LaMP-5", "LaMP-7", "LongLaMP-2", "LongLaMP-3", "LongLaMP-4"
    }:
        return _create_generation_metric_fn(aggregate)
    else:
        raise ValueError(f"Invalid task: {task}")


def _create_classification_metric_fn(
    labels: tuple[str, ...],
    aggregate: bool
) -> MetricFn:
    accuracy_metric = evaluate.load("accuracy")
    f1_metric = evaluate.load("f1")
    label_to_index = {l: i for i, l in enumerate(labels)}

    def classification_metric_fn(
        predictions: list[str],
        references: list[str]
    ) -> dict[str, float | list[float]]:
        predictions = [label_to_index.get(p.strip(), -1) for p in predictions]
        references = [label_to_index.get(r.strip(), -1) for r in references]

        if aggregate:
            accuracy_results = accuracy_metric.compute(
                predictions=predictions, references=references
            )
            f1_results = f1_metric.compute(
                predictions=predictions,
                references=references,
                labels=list(label_to_index.values()),
                average="macro"
            )
            return {
                "accuracy": accuracy_results["accuracy"],
                "f1": f1_results["f1"]
            }
        else:
            correctness = [
                float(p == r) for p, r in zip(predictions, references)
            ]
            return {"correctness": correctness}

    return classification_metric_fn


def _create_regression_metric_fn(
    labels: tuple[str, ...],
    aggregate: bool
) -> MetricFn:
    mae_metric = evaluate.load("mae")
    mse_metric = evaluate.load("mse")

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

    def regression_metric_fn(
        predictions: list[str],
        references: list[str]
    ) -> dict[str, float | list[float]]:
        predictions = [to_float(p, r) for p, r in zip(predictions, references)]
        references = [float(r) for r in references]

        if aggregate:
            mae_results = mae_metric.compute(
                predictions=predictions, references=references
            )
            mse_results = mse_metric.compute(
                predictions=predictions, references=references, squared=False
            )
            return {"mae": mae_results["mae"], "rmse": mse_results["mse"]}
        else:
            error = [abs(p - r) for p, r in zip(predictions, references)]
            return {"error": error}

    return regression_metric_fn


def _create_generation_metric_fn(aggregate: bool) -> MetricFn:
    rouge_metric = evaluate.load("rouge")
    meteor_metric = evaluate.load("meteor")

    def generation_metric_fn(
        predictions: list[str],
        references: list[str]
    ) -> dict[str, float | list[float]]:
        predictions = [p.strip() for p in predictions]
        references = [[r.strip()] for r in references]

        rouge_results = rouge_metric.compute(
            predictions=predictions,
            references=references,
            rouge_types=["rouge1", "rougeL"],
            use_aggregator=aggregate
        )

        if aggregate:
            meteor_results = meteor_metric.compute(
                predictions=predictions, references=references
            )
        else:
            meteor = [
                meteor_metric.compute(predictions=[p], references=[r])["meteor"]
                for p, r in zip(predictions, references)
            ]
            meteor_results = {"meteor": meteor}

        return {
            "rouge-1": rouge_results["rouge1"],
            "rouge-L": rouge_results["rougeL"],
            "meteor": meteor_results["meteor"]
        }

    return generation_metric_fn
