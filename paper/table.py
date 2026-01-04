import json
import re
from collections import defaultdict
from pathlib import Path

import fire


def print_table(table: int, version: str = "reinforce-logp") -> None:
    if table == 1:
        tasks = ["LaMP-1", "LaMP-2", "LaMP-3", "LaMP-4", "LaMP-5", "LaMP-7"]
        llms = [
            "phi4-mini-instruct", "llama3-8b-instruct", "llama3-70b-instruct"
        ]
    elif table == 2:
        tasks = ["LongLaMP-2", "LongLaMP-3", "LongLaMP-4"]
        llms = ["phi4-mini-instruct", "llama3-8b-instruct"]

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

    print(TABLE_1_PREFIX if table == 1 else TABLE_2_PREFIX)

    for llm_index, llm in enumerate(llms):
        # Collect results
        task_metric_results = defaultdict(lambda: defaultdict(list))

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

        # Label results
        retriever_task_results = defaultdict(lambda: defaultdict(list))

        for task, metric_results in task_metric_results.items():
            for metric, results in metric_results.items():
                labeled_results = _label_results(results, task != "LaMP-3")

                for retriever, result in zip(
                    retrievers + ["PURPLE"], labeled_results
                ):
                    retriever_task_results[retriever][task].append(result)

        print(
            TABLE_1_LLMS[llm_index] if table == 1 else TABLE_2_LLMS[llm_index]
        )

        for retriever_index, (retriever, task_results) in enumerate(
            retriever_task_results.items()
        ):
            print(RETRIEVERS[retriever_index])

            for task_index, (task, results) in enumerate(task_results.items()):
                end = "" if task_index == len(task_results) - 1 else "\n"
                print(f"        & {' / '.join(results)}", end=end)

            print(r" \\")

    print((TABLE_1_SUFFIX if table == 1 else TABLE_2_SUFFIX))


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

TABLE_1_PREFIX = "\n".join((
    r"\begin{table}[t]",
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
TABLE_1_LLMS = tuple(
    "\n".join((
        r"    \midrule",
       rf"    \multicolumn{1}{{l}}{{{LLM}}} & \multicolumn{6}{{l}}{{}} \\",
        r"    \midrule"
    ))
    for LLM in LLMS
)
TABLE_1_SUFFIX = "\n".join((
    r"        \midrule",
    r"    \end{tabular}}",
    r"    \label{tab:lamp}",
    r"\end{table}"
))

TABLE_2_PREFIX = "\n".join((
    r"\begin{table}[t]",
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
    r"            & R1 / RL / M",
    r"            & R1 / RL / M",
    r"            & R1 / RL / M \\",
))
TABLE_2_LLMS = tuple(
    "\n".join((
        r"    \midrule",
       rf"    \multicolumn{1}{{l}}{{{LLM}}} & \multicolumn{3}{{l}}{{}} \\",
        r"    \midrule"
    ))
    for LLM in LLMS
)
TABLE_2_SUFFIX = "\n".join((
    r"        \midrule",
    r"    \end{tabular}}",
    r"    \label{tab:longlamp}",
    r"\end{table}"
))


if __name__ == "__main__":
    fire.Fire()
