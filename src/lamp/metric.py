# https://github.com/LaMP-Benchmark/LaMP/blob/main/LaMP/data/datasets.py
# https://github.com/LaMP-Benchmark/LaMP/blob/main/LaMP/metrics/classification_metrics.py
# https://github.com/LaMP-Benchmark/LaMP/blob/main/LaMP/metrics/generation_metrics.py
import os
import re
from collections.abc import Callable
from types import MappingProxyType

import nltk

if os.getenv("HF_EVALUATE_OFFLINE") == "1":
    nltk.download = lambda *args, **kwargs: None

import evaluate
import numpy as np


MetricFn = Callable[[list[str], list[str]], dict[str, float | list[float]]]


LABELS = MappingProxyType({
    "lamp1": ("[1]", "[2]"),
    "lamp2": (
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
    "lamp3": ("1", "2", "3", "4", "5")
})


def create_metric_fn(task: str, aggregate: bool = True) -> MetricFn:
    if task in {"lamp1", "lamp2"}:
        return _create_classification_metric_fn(LABELS[task], aggregate)
    elif task in {"lamp3"}:
        return _create_regression_metric_fn(LABELS[task], aggregate)
    elif task in {
        "lamp4", "lamp5", "lamp7", "longlamp2", "longlamp3", "longlamp4"
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

    def metric_fn(
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
                float(p == r)
                for p, r in zip(predictions, references, strict=True)
            ]
            return {"correctness": correctness}

    return metric_fn


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

    def metric_fn(
        predictions: list[str],
        references: list[str]
    ) -> dict[str, float | list[float]]:
        predictions = [
            to_float(p, r)
            for p, r in zip(predictions, references, strict=True)
        ]
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
            error = [
                abs(p - r)
                for p, r in zip(predictions, references, strict=True)
            ]
            return {"error": error}

    return metric_fn


def _create_generation_metric_fn(aggregate: bool) -> MetricFn:
    rouge_metric = evaluate.load("rouge")
    meteor_metric = evaluate.load("meteor")

    # METEOR crashes on tokens with too many trailing "y"s
    invalid_pattern = r"[yY]{100,}"

    def to_float(
        result: float | np.float64 | list[float] | list[np.float64]
    ) -> float | list[float]:
        if isinstance(result, list):
            return [float(r) for r in result]

        return float(result)

    def metric_fn(
        predictions: list[str],
        references: list[str]
    ) -> dict[str, float | list[float]]:
        predictions = [
            "" if re.search(invalid_pattern, p) else p.strip()
            for p in predictions
        ]
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
                meteor_metric
                .compute(predictions=[p], references=[r])["meteor"]
                for p, r in zip(predictions, references, strict=True)
            ]
            meteor_results = {"meteor": meteor}

        return {
            "rouge-1": to_float(rouge_results["rouge1"]),
            "rouge-L": to_float(rouge_results["rougeL"]),
            "meteor": to_float(meteor_results["meteor"])
        }

    return metric_fn
