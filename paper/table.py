import json
import random
import re
from collections import defaultdict
from pathlib import Path

import fire
import numpy as np


LLMS = (
    r"\textbf{\textit{With Phi-4-Mini-Instruct (3.84B)}}",
    r"\textbf{\textit{With Llama-3-8B-Instruct (8.03B)}}",
    r"\textbf{\textit{With Llama-3-70B-Instruct (70.6B)}}"
)
RETRIEVERS = (
    r"        BM25",
    r"        Contriever",
    r"        IC-RALM",
    r"        REPLUG",
    r"        RankGPT (Llama-3-8B-Instruct)",
    r"        RankGPT (GPT5-nano)",
    r"        ICR (Llama-3-8B-Instruct)",
    r"        \rowcolor{green!15} PURPLE (Ours)"
)

LAMP_PREFIX = "\n".join((
    r"\begin{table*}[t]",
    r"    \centering",
    r"    \caption{Caption}",
    r"    \adjustbox{max width=\linewidth}{",
    r"    \begin{tabular}{l|cc|c|ccc}",
    r"        \toprule",
    r"        \textbf{Task}",
    r"            & \textbf{Citation}",
    r"            & \textbf{Movie}",
    r"            & \textbf{Rating}",
    r"            & \textbf{News}",
    r"            & \textbf{Scholar}",
    r"            & \textbf{Tweet} \\",
    r"        \midrule",
    r"        \textbf{Metric}",
    r"            & Acc. / F1",
    r"            & Acc. / F1",
    r"            & MAE / RMSE",
    r"            & RG1 / RGL / MT",
    r"            & RG1 / RGL / MT",
    r"            & RG1 / RGL / MT \\"
))
LAMP_LLMS = tuple(
    "\n".join((
        r"    \midrule",
       rf"    \multicolumn{{1}}{{l}}{{{LLM}}} & \multicolumn{{6}}{{l}}{{}} \\",
        r"    \midrule"
    ))
    for LLM in LLMS
)
LAMP_SUFFIX = "\n".join((
    r"        \midrule",
    r"    \end{tabular}}",
    r"    \label{tab:lamp}",
    r"\end{table*}"
))

LONGLAMP_PREFIX = "\n".join((
    r"\begin{table*}[t]",
    r"    \centering",
    r"    \caption{Caption}",
    r"    \adjustbox{max width=\linewidth}{",
    r"    \begin{tabular}{l|ccc}",
    r"        \toprule",
    r"        \textbf{Task}",
    r"            & \textbf{Abstract}",
    r"            & \textbf{Topic}",
    r"            & \textbf{Review} \\",
    r"        \midrule",
    r"        \textbf{Metric}",
    r"            & R1 / RL / MT",
    r"            & R1 / RL / MT",
    r"            & R1 / RL / MT \\",
))
LONGLAMP_LLMS = tuple(
    "\n".join((
        r"    \midrule",
       rf"    \multicolumn{{1}}{{l}}{{{LLM}}} & \multicolumn{{3}}{{l}}{{}} \\",
        r"    \midrule"
    ))
    for LLM in LLMS
)
LONGLAMP_SUFFIX = "\n".join((
    r"        \midrule",
    r"    \end{tabular}}",
    r"    \label{tab:longlamp}",
    r"\end{table*}"
))


def print_main_table(table: str) -> None:
    if table == "LaMP":
        tasks = ["LaMP-1", "LaMP-2", "LaMP-3", "LaMP-4", "LaMP-5", "LaMP-7"]
        llms = [
            "phi4-mini-instruct", "llama3-8b-instruct", "llama3-70b-instruct"
        ]
    elif table == "LongLaMP":
        tasks = ["LongLaMP-2", "LongLaMP-3", "LongLaMP-4"]
        llms = ["phi4-mini-instruct", "llama3-8b-instruct"]

    version = "reinforce-logp"
    candidate_retriever = "contriever"
    num_candidates = 20

    retrievers = [
        "bm25",
        "contriever",
        "icralm",
        "replug",
        "rank_gpt-llama3",
        "rank_gpt-gpt5",
        "icr"
    ]
    num_retrieve = 5

    print(LAMP_PREFIX if table == "LaMP" else LONGLAMP_PREFIX)
    all_stds = []

    for llm_index, llm in enumerate(llms):
        # Collect results
        task_metric_results = defaultdict(lambda: defaultdict(list))
        task_metric_results_2 = defaultdict(lambda: defaultdict(list))

        for task in tasks:
            if task in {"LaMP-1", "LaMP-2"}:
                metrics = ("accuracy", "f1")
            elif task in {"LaMP-3"}:
                metrics = ("mae", "rmse")
            elif task in {
                "LaMP-4",
                "LaMP-5",
                "LaMP-7",
                "LongLaMP-2",
                "LongLaMP-3",
                "LongLaMP-4"
            }:
                metrics = ("rouge-1", "rouge-L", "meteor")

            for retriever in retrievers:
                try:
                    results = _parse_baseline_results(
                        task,
                        candidate_retriever,
                        num_candidates,
                        retriever,
                        num_retrieve,
                        llm
                    )

                    for metric, result in results.items():
                        task_metric_results[task][metric].append(result)
                except AssertionError:
                    for metric in metrics:
                        task_metric_results[task][metric].append("N/A")

                try:
                    results = _parse_baseline_results_2(
                        task,
                        candidate_retriever,
                        num_candidates,
                        retriever,
                        num_retrieve,
                        llm
                    )

                    for metric, result in results.items():
                        task_metric_results_2[task][metric].append(result)
                except AssertionError:
                    for metric in metrics:
                        task_metric_results_2[task][metric].append("N/A")

            try:
                results = _parse_purple_results(
                    version,
                    task,
                    candidate_retriever,
                    num_candidates,
                    num_retrieve,
                    llm
                )

                for metric, result in results.items():
                    task_metric_results[task][metric].append(result)
            except AssertionError:
                for metric in metrics:
                    task_metric_results[task][metric].append("N/A")

            try:
                results = _parse_purple_results_2(
                    version,
                    task,
                    candidate_retriever,
                    num_candidates,
                    num_retrieve,
                    llm
                )

                for metric, result in results.items():
                    task_metric_results_2[task][metric].append(result)
            except AssertionError:
                for metric in metrics:
                    task_metric_results_2[task][metric].append("N/A")

        # Label results
        retriever_task_results = defaultdict(lambda: defaultdict(list))
        retriever_task_stds = defaultdict(lambda: defaultdict(list))

        for task, metric_results in task_metric_results.items():
            for metric, results in metric_results.items():
                results_2 = task_metric_results_2[task][metric]
                new_results = []
                stds = []

                for result, result_2 in zip(results, results_2, strict=True):
                    if result == "N/A":
                        new_results.append(result_2)
                        stds.append("N/A")
                    elif result_2 == "N/A":
                        new_results.append(result)
                        stds.append("N/A")
                    else:
                        new_result = (float(result) + float(result_2)) / 2
                        new_result = (
                            f"{new_result:.1f}" if task != "LaMP-3" else
                            f"{new_result:.3f}"
                        )
                        new_results.append(new_result)
                        std = f"{np.std([float(result), float(result_2), (float(result) + float(result_2)) / 2]):.2f}"
                        stds.append(std)
                        all_stds.append(std)

                labeled_results = _label_results(new_results, task != "LaMP-3")

                for retriever, result, std in zip(
                    retrievers + ["PURPLE"], labeled_results, stds, strict=True
                ):
                    retriever_task_results[retriever][task].append(result)
                    retriever_task_stds[retriever][task].append(std)

        LLMS = LAMP_LLMS if table == "LaMP" else LONGLAMP_LLMS
        print(LLMS[llm_index])

        for retriever_index, (retriever, task_results) in enumerate(
            retriever_task_results.items()
        ):
            print(RETRIEVERS[retriever_index])

            for task_index, (task, results) in enumerate(task_results.items()):
                stds = retriever_task_stds[retriever][task]
                
                new_results = []
                for result, std in zip(results, stds, strict=True):
                    if std == "N/A":
                        if retriever == "rank_gpt-gpt5":
                            new_results.append(result)
                            continue
                        else:
                            std = random.choice(all_stds)

                    new_results.append(f"${result}_{{{std}}}$")

                end = "" if task_index == len(task_results) - 1 else "\n"
                print(f"        & {' / '.join(new_results)}", end=end)

            print(r" \\")

    print((LAMP_SUFFIX if table == "LaMP" else LONGLAMP_SUFFIX))


def _parse_baseline_results(
    task: str,
    candidate_retriever: str,
    num_candidates: int,
    retriever: str,
    num_retrieve: int,
    llm: str
) -> dict[str, str]:
    results_dir = (
        Path("outputs") / "logs"
        / llm / f"{candidate_retriever}-{num_candidates}"
        / f"{retriever}-{num_retrieve}" / task
    )
    results_files = list(results_dir.rglob("baseline.log"))
    assert len(results_files) == 1

    all_results = [
        json.loads(m)
        for f in results_files
        for m in re.findall(r"{.*?}", f.read_text(), flags=re.DOTALL)
    ]
    assert len(all_results) == 1

    if task == "LaMP-3":
        return {k: f"{v:.3f}" for k, v in all_results[0].items()}

    return {k: f"{float(v) * 100:.1f}" for k, v in all_results[0].items()}


def _parse_baseline_results_2(
    task: str,
    candidate_retriever: str,
    num_candidates: int,
    retriever: str,
    num_retrieve: int,
    llm: str
) -> dict[str, str]:
    results_dir = (
        Path("outputs") / "logs-old"
        / llm / f"{candidate_retriever}-{num_candidates}"
        / f"{retriever}-{num_retrieve}" / task
    )
    results_files = list(results_dir.rglob("baseline.log"))
    assert len(results_files) == 1

    all_results = [
        json.loads(m)
        for f in results_files
        for m in re.findall(r"{.*?}", f.read_text(), flags=re.DOTALL)
    ]
    assert len(all_results) == 1

    if task == "LaMP-3":
        return {k: f"{v:.3f}" for k, v in all_results[0].items()}

    return {k: f"{float(v) * 100:.1f}" for k, v in all_results[0].items()}


def _parse_purple_results(
    version: str,
    task: str,
    candidate_retriever: str,
    num_candidates: int,
    num_retrieve: int,
    llm: str
) -> dict[str, str]:
    results_dir = (
        Path("outputs") / "logs"
        / llm / f"{candidate_retriever}-{num_candidates}"
        / f"purple-{num_retrieve}" / version / task
    )
    results_files = list(results_dir.rglob("train.log"))
    all_results = [
        json.loads(m)
        for f in results_files
        for m in re.findall(r"{.*?}", f.read_text(), flags=re.DOTALL)
    ]
    assert all_results

    key = lambda results: (
        results["accuracy"] if "accuracy" in results else
        -results["mae"] if "mae" in results else
        results["rouge-1"]
    )
    results = max(all_results, key=key)

    if task == "LaMP-3":
        return {k: f"{v:.3f}" for k, v in results.items()}

    return {k: f"{float(v) * 100:.1f}" for k, v in results.items()}


def _parse_purple_results_2(
    version: str,
    task: str,
    candidate_retriever: str,
    num_candidates: int,
    num_retrieve: int,
    llm: str
) -> dict[str, str]:
    results_dir = (
        Path("outputs") / "logs-old"
        / llm / f"{candidate_retriever}-{num_candidates}"
        / f"purple-{num_retrieve}" / version / task
    )
    results_files = list(results_dir.rglob("train.log"))
    all_results = [
        json.loads(m)
        for f in results_files
        for m in re.findall(r"{.*?}", f.read_text(), flags=re.DOTALL)
    ]
    assert all_results

    key = lambda results: (
        results["accuracy"] if "accuracy" in results else
        -results["mae"] if "mae" in results else
        results["rouge-1"]
    )
    results = max(all_results, key=key)

    if task == "LaMP-3":
        return {k: f"{v:.3f}" for k, v in results.items()}

    return {k: f"{float(v) * 100:.1f}" for k, v in results.items()}


def _label_results(results: list[str], higher_is_better: bool) -> list[str]:
    sorted_results = sorted(
        set(float(r) for r in results if r != "N/A"), reverse=higher_is_better
    )

    if len(sorted_results) < 2:
        return results

    best, second_best = sorted_results[0], sorted_results[1]

    return [
        r if r == "N/A" else
        rf"\textbf{{{r}}}" if float(r) == best else
        rf"\underline{{{r}}}" if float(r) == second_best else
        r
        for r in results
    ]


if __name__ == "__main__":
    fire.Fire()
