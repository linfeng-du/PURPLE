import json
import re
from collections import defaultdict
from pathlib import Path

import fire


def table(table: int) -> None:
    llms = (
        ['phi-4-mini-instruct', 'llama-3-8b-instruct', 'llama-3-70b-instruct'] if table == 1 else
        ['phi-4-mini-instruct', 'llama-3-8b-instruct']
    )
    tasks = (
        ['LaMP-1', 'LaMP-2', 'LaMP-3', 'LaMP-4', 'LaMP-5', 'LaMP-7'] if table == 1 else
        ['LongLaMP-2', 'LongLaMP-3', 'LongLaMP-4']
    )
    rerankers = ['bm25', 'contriever', 'icralm', 'replug', 'rank_gpt-llama3', 'rank_gpt-gpt5', 'icr']

    print((TABLE_1_PREFIX if table == 1 else TABLE_2_PREFIX))

    for llm_index, llm in enumerate(llms):
        # Collect all results
        task_metric_results = defaultdict(lambda: defaultdict(list))

        for task in tasks:
            metrics = (
                ('accuracy', 'f1') if task in {'LaMP-1', 'LaMP-2'} else
                ('mae', 'rmse') if task == 'LaMP-3' else
                ('rouge-1', 'rouge-L', 'meteor')
            )

            for reranker in rerankers:
                try:
                    results = baseline_results(task, llm, 'bm25', 20, reranker, 5)

                    for metric, result in results.items():
                        task_metric_results[task][metric].append(result)
                except Exception:
                    for metric in metrics:
                        result = ('100' if task == 'LaMP-3' else '0')
                        task_metric_results[task][metric].append(result)

            try:
                results = bandit_ramp_results(task, llm, 'contriever', 20, 5, 'cross_attn-12')

                for metric, result in results.items():
                    task_metric_results[task][metric].append(result)
            except Exception:
                for metric in metrics:
                    result = ('100' if task == 'LaMP-3' else '0')
                    task_metric_results[task][metric].append(result)

        # Label best and second best results
        reranker_task_results = defaultdict(lambda: defaultdict(list))

        for task, metric_results in task_metric_results.items():
            for metric, results in metric_results.items():
                labeled_results = label_results(task, llm, results)

                for reranker, result in zip(rerankers + ['BanditRAMP'], labeled_results):
                    reranker_task_results[reranker][task].append(result)

        # Print results
        print((TABLE_1_LLMS[llm_index] if table == 1 else TABLE_2_LLMS[llm_index]))

        for reranker_index, (reranker, task_results) in enumerate(reranker_task_results.items()):
            print(TABLE_RERANKER_STRINGS[reranker_index])

            for task_index, (task, results) in enumerate(task_results.items()):
                end = ('' if task_index == len(task_results) - 1 else '\n')
                print(f'        & {" / ".join(results)}', end=end)

            print(r' \\')

    print((TABLE_1_SUFFIX if table == 1 else TABLE_2_SUFFIX))


def bandit_ramp_results(
    task: str, llm: str,
    retriever: str, num_candidates: int,
    num_rerank: int, version: str
) -> dict[str, str]:
    exp_name = f'{llm}/{retriever}-{num_candidates}/bandit_ramp-{num_rerank}/{version}/{task}'
    results = []

    for result_dir in (Path('./logs') / exp_name).iterdir():
        if not result_dir.is_dir():
            continue

        with open(result_dir / 'train.log', 'r') as file:
            text = file.read()

        results.extend([json.loads(match) for match in re.findall(r'\{.*?\}', text, flags=re.DOTALL)])

    key = lambda x: (x['accuracy'] if 'accuracy' in x else -x['mae'] if 'mae' in x else x['rouge-1'])
    best_results = max(results, key=key)
    return {
        key: f'{value:.3f}'
        for key, value in best_results.items()
        if key in {'accuracy', 'f1', 'mae', 'rmse', 'rouge-1', 'rouge-L', 'meteor'}
    }


def baseline_results(
    task: str, llm: str,
    retriever: str, num_candidates: int,
    reranker: str, num_rerank: int
) -> dict[str, str]:
    exp_name = f'{llm}/{retriever}-{num_candidates}/{reranker}-{num_rerank}/{task}'
    result_dirs = [result_dir for result_dir in (Path('./logs') / exp_name).iterdir() if result_dir.is_dir()]
    result_dir = result_dirs[0]

    with open(result_dir / 'baseline.log', 'r') as file:
        text = file.read()

    results_list = [json.loads(match) for match in re.findall(r'\{.*?\}', text, flags=re.DOTALL)]
    assert len(results_list) == 1

    return {
        key: f'{value:.3f}'
        for key, value in results_list[0].items()
        if key in {'accuracy', 'f1', 'mae', 'rmse', 'rouge-1', 'rouge-L', 'meteor'}
    }


def label_results(task: str, llm: str, results: list[str]) -> list[str]:
    if task != 'LaMP-3':
        results = [f'{float(result) * 100:.1f}' for result in results]

    best, second_best = sorted(set(map(float, results)), reverse=(task != 'LaMP-3'))[:2]
    labeled_results = [
        rf'\textbf{{{result}}}' if float(result) == best else
        rf'\underline{{{result}}}' if float(result) == second_best else
        result
        for result in results
    ]

    if llm == 'llama-3-70b-instruct' and task == 'LaMP-3':
        # REPLUG OOM
        labeled_results[3] = '-'
    elif llm == 'llama-3-70b-instruct' and task in {'LongLaMP-2', 'LongLaMP-3', 'LongLaMP-4'}:
        labeled_results = ['-'] * len(results)

    return labeled_results


MID_RULE = r'    \midrule'

TABLE_1_PREFIX = (
    r'\begin{table}[ht]' + '\n'
    r'    \centering' + '\n'
    r'    \adjustbox{max width=\linewidth}{' + '\n'
    r'    \begin{tabular}{l|cc|c|ccc}' + '\n'
    r'    \toprule' + '\n'
    r'    \textbf{Task}' + '\n'
    r'        & \textbf{Citation}' + '\n'
    r'        & \textbf{Movie}' + '\n'
    r'        & \textbf{Rating}' + '\n'
    r'        & \textbf{News}' + '\n'
    r'        & \textbf{Scholar}' + '\n'
    r'        & \textbf{Tweet} \\' + '\n'
    rf'{MID_RULE}' + '\n'
    r'    \textbf{Metric}' + '\n'
    r'        & Acc. / F1' + '\n'
    r'        & Acc. / F1' + '\n'
    r'        & MAE / RMSE' + '\n'
    r'        & RG1 / RGL / MT' + '\n'
    r'        & RG1 / RGL / MT' + '\n'
    r'        & RG1 / RGL / MT \\'
)
TABLE_2_PREFIX = (
    r'\begin{table}[ht]' + '\n'
    r'    \centering' + '\n'
    r'    \adjustbox{max width=\linewidth}{' + '\n'
    r'    \begin{tabular}{l|ccc}' + '\n'
    r'    \toprule' + '\n'
    r'    \textbf{Task}' + '\n'
    r'        & \textbf{Abstract}' + '\n'
    r'        & \textbf{Topic}' + '\n'
    r'        & \textbf{Review} \\' + '\n'
    rf'{MID_RULE}' + '\n'
    r'    \textbf{Metric}' + '\n'
    r'        & R1 / RL / M' + '\n'
    r'        & R1 / RL / M' + '\n'
    r'        & R1 / RL / M \\'
)

TABLE_1_SUFFIX = (
    rf'{MID_RULE}' + '\n'
    r'    \end{tabular}}' + '\n'
    r'    \caption{Caption}' + '\n'
    r'    \label{tab:lamp_results}' + '\n'
    r'\end{table}'
)
TABLE_2_SUFFIX = (
    rf'{MID_RULE}' + '\n'
    r'    \end{tabular}}' + '\n'
    r'    \caption{Caption}' + '\n'
    r'    \label{tab:longlamp_results}' + '\n'
    r'\end{table}'
)

TABLE_1_LLMS = [
    rf'{MID_RULE}' + '\n    '
        + r'\multicolumn{1}{l}{\textbf{\textit{With Phi-4-Mini-Instruct (3.84B)}}} & \multicolumn{6}{l}{} \\'
        + '\n' + rf'{MID_RULE}',
    rf'{MID_RULE}' + '\n    '
        + r'\multicolumn{1}{l}{\textbf{\textit{With Llama-3-8B-Instruct (8.03B)}}} & \multicolumn{6}{l}{} \\'
        + '\n' + rf'{MID_RULE}',
    rf'{MID_RULE}' + '\n    '
        + r'\multicolumn{1}{l}{\textbf{\textit{With Llama-3-70B-Instruct (70.6B)}}} & \multicolumn{6}{l}{} \\'
        + '\n' + rf'{MID_RULE}',
]
TABLE_2_LLMS = [
    rf'{MID_RULE}' + '\n    '
        + r'\multicolumn{1}{l}{\textbf{\textit{With Phi-4-Mini-Instruct (3.84B)}}} & \multicolumn{3}{l}{} \\'
        + '\n' + rf'{MID_RULE}',
    rf'{MID_RULE}' + '\n    '
        + r'\multicolumn{1}{l}{\textbf{\textit{With Llama-3-8B-Instruct (8.03B)}}} & \multicolumn{3}{l}{} \\'
        + '\n' + rf'{MID_RULE}',
    rf'{MID_RULE}' + '\n    '
        + r'\multicolumn{1}{l}{\textbf{\textit{With Llama-3-70B-Instruct (70.6B)}}} & \multicolumn{3}{l}{} \\'
        + '\n' + rf'{MID_RULE}',
]

TABLE_RERANKER_STRINGS = [
    r'    BM25',
    r'    Contriever',
    r'    IC-RALM-Llama-3-8B-Instruct',
    r'    REPLUG-LSR',
    r'    RankGPT-Llama-3-8B-Instruct',
    r'    RankGPT-GPT5-nano',
    r'    ICR-Llama-3-8B-Instruct',
    r'    \rowcolor{green!15} BASEP (Ours)'
]


if __name__ == '__main__':
    fire.Fire(table)
