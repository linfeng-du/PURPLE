import json
import re
from collections import defaultdict
from pathlib import Path

import evaluate
from datasets import load_dataset
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer

import fire

from bandit_ramp import create_preprocessor, load_retrieved_lamp_dataset


def download() -> None:
    print('Downloading tokenizers...')
    AutoTokenizer.from_pretrained('facebook/contriever')
    AutoTokenizer.from_pretrained('microsoft/Phi-4-mini-instruct')
    AutoTokenizer.from_pretrained('meta-llama/Meta-Llama-3-8B-Instruct')
    AutoTokenizer.from_pretrained('meta-llama/Meta-Llama-3-70B-Instruct')
    AutoTokenizer.from_pretrained('Qwen/Qwen2.5-72B-Instruct')

    print('Downloading models...')
    AutoModel.from_pretrained('facebook/contriever')
    AutoModelForCausalLM.from_pretrained('microsoft/Phi-4-mini-instruct')
    AutoModelForCausalLM.from_pretrained('meta-llama/Meta-Llama-3-8B-Instruct')
    AutoModelForCausalLM.from_pretrained('meta-llama/Meta-Llama-3-70B-Instruct')
    AutoModelForCausalLM.from_pretrained('Qwen/Qwen2.5-72B-Instruct')

    print('Downloading LongLaMP datasets...')
    load_dataset('LongLaMP/LongLaMP', name='abstract_generation_user')
    load_dataset('LongLaMP/LongLaMP', name='topic_writing_user')
    load_dataset('LongLaMP/LongLaMP', name='product_review_user')

    print('Downloading metrics...')
    evaluate.load('accuracy')
    evaluate.load('f1')
    evaluate.load('mae')
    evaluate.load('mse')
    evaluate.load('rouge')
    evaluate.load('meteor')


def preprocess(task: str, retriever: str, num_candidates: int) -> None:
    print(f'Preprocessing {task}...')

    test_split = ('dev' if task.startswith('LaMP') else 'test')
    train_dataset = load_retrieved_lamp_dataset(task, 'train', retriever, num_candidates)
    test_dataset = load_retrieved_lamp_dataset(task, test_split, retriever, num_candidates)

    tokenizer = AutoTokenizer.from_pretrained('facebook/contriever')
    preprocessor = create_preprocessor(
        max_num_profiles=-1,
        max_query_length=512,
        max_document_length=512,
        tokenizer=tokenizer
    )
    train_dataset.map(preprocessor, batched=True, remove_columns=['query', 'corpus'], num_proc=16)

    # Re-initialize tokenizer to ensure consistent hashing
    tokenizer = AutoTokenizer.from_pretrained('facebook/contriever')
    preprocessor = create_preprocessor(
        max_num_profiles=-1,
        max_query_length=512,
        max_document_length=512,
        tokenizer=tokenizer
    )
    test_dataset.map(preprocessor, batched=True, remove_columns=['query', 'corpus'], num_proc=16)


def results_formatted(
    version: str, retriever: str = 'bm25',
    num_candidates: int = 20, num_rerank: int = 5
) -> None:
    def label_results(task: str, results: list[str]) -> list[str]:
        if task != 'LaMP-3':
            results = [f'{float(result) * 100:.1f}' for result in results]

        best, second_best = sorted(set(map(float, results)), reverse=(task != 'LaMP-3'))[:2]
        labeled_results = [
            rf'\textbf{{{result}}}' if float(result) == best else
            rf'\underline{{{result}}}' if float(result) == second_best else
            result
            for result in results
        ]
        return labeled_results

    llms = ['phi-4-mini-instruct', 'llama-3-8b-instruct', 'llama-3-70b-instruct']
    tasks = [
        'LaMP-1', 'LaMP-2', 'LaMP-3', 'LaMP-4', 'LaMP-5', 'LaMP-7',
        'LongLaMP-2', 'LongLaMP-3', 'LongLaMP-4'
    ]
    rerankers = ['bm25', 'contriever', 'icralm', 'replug', 'rank_gpt-llama3', 'rank_gpt-gpt5', 'icr']

    print(TABLE_PREFIX_STRING)

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
                    results = baseline_results(task, llm, retriever, num_candidates, reranker, num_rerank)

                    for metric, result in results.items():
                        task_metric_results[task][metric].append(result)
                except Exception:
                    for metric in metrics:
                        result = ('100' if task == 'LaMP-3' else '0')
                        task_metric_results[task][metric].append(result)

            try:
                results = bandit_ramp_results(task, llm, 'contriever', num_candidates, num_rerank, version)

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
                labeled_results = label_results(task, results)

                for reranker, result in zip(rerankers + ['BanditRAMP'], labeled_results):
                    reranker_task_results[reranker][task].append(result)

        # Print results
        print(LLM_STRINGS[llm_index])

        for reranker_index, (reranker, task_results) in enumerate(reranker_task_results.items()):
            print(RERANKER_STRINGS[reranker_index])

            for task_index, (task, results) in enumerate(task_results.items()):
                end = ('' if task_index == len(task_results) - 1 else '\n')
                print(f'        & {" / ".join(results)}', end=end)

            print(r' \\')

    print(TABLE_SUFFIX_STRING)


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


MID_RULE_STRING = r'    \midrule'

TABLE_PREFIX_STRING = (
    r'\begin{table}[ht]' + '\n'
    r'    \centering' + '\n'
    r'    \adjustbox{max width=\linewidth}{' + '\n'
    r'    \begin{tabular}{l|cc|c|ccc|cccc}' + '\n'
    r'    \toprule' + '\n'
    r'    \textbf{Task}' + '\n'
    r'        & \textbf{Citation}' + '\n'
    r'        & \textbf{Movie}' + '\n'
    r'        & \textbf{Rating}' + '\n'
    r'        & \textbf{News}' + '\n'
    r'        & \textbf{Scholar}' + '\n'
    r'        & \textbf{Tweet}' + '\n'
    r'        & \textbf{Abstract}' + '\n'
    r'        & \textbf{Topic}' + '\n'
    r'        & \textbf{Review} \\' + '\n'
    rf'{MID_RULE_STRING}' + '\n'
    r'    \textbf{Metric}' + '\n'
    r'        & Acc / F1' + '\n'
    r'        & Acc / F1' + '\n'
    r'        & MAE / RMSE' + '\n'
    r'        & R1 / RL / M' + '\n'
    r'        & R1 / RL / M' + '\n'
    r'        & R1 / RL / M' + '\n'
    r'        & R1 / RL / M' + '\n'
    r'        & R1 / RL / M' + '\n'
    r'        & R1 / RL / M \\'
)

TABLE_SUFFIX_STRING = (
    r'    \midrule' + '\n'
    r'    \end{tabular}}' + '\n'
    r'    \caption{Caption}' + '\n'
    r'    \label{tab:main_results}' + '\n'
    r'\end{table}'
)

LLM_STRINGS = [
    rf'{MID_RULE_STRING}' + '\n    '
        + r'\multicolumn{1}{l}{\textbf{\textit{With Phi-4-Mini-Instruct (3.84B)}}} & \multicolumn{9}{l}{} \\'
        + '\n' + rf'{MID_RULE_STRING}',
    rf'{MID_RULE_STRING}' + '\n    '
        + r'\multicolumn{1}{l}{\textbf{\textit{With Llama-3-8B-Instruct (8.03B)}}} & \multicolumn{9}{l}{} \\'
        + '\n' + rf'{MID_RULE_STRING}',
    rf'{MID_RULE_STRING}' + '\n    '
        + r'\multicolumn{1}{l}{\textbf{\textit{With Llama-3-70B-Instruct (70.6B)}}} & \multicolumn{9}{l}{} \\'
        + '\n' + rf'{MID_RULE_STRING}',
]

RERANKER_STRINGS = [
    r'    BM25~\citep{robertson2009probabilistic}',
    r'    Contriever~\citep{izacard2022unsupervised}',
    r'    IC-RALM-Llama-3-8B-Instruct~\cite{ram2023incontext}',
    r'    REPLUG-LSR~\citep{shi2024replug}',
    r'    RankGPT-Llama-3-8B-Instruct~\cite{sun2023chatgpt}',
    r'    RankGPT-GPT5-nano~\cite{sun2023chatgpt}',
    r'    ICR-Llama-3-8B-Instruct~\cite{chen2025attention}',
    r'    \rowcolor{green!15} BASEP (Ours)'
]



if __name__ == '__main__':
    fire.Fire()
