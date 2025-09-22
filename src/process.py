import itertools
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
    def label_results(results: list[str], higher_is_better: bool = True) -> str:
        best, second_best = sorted(set(map(float, results)), reverse=higher_is_better)[:2]
        labeled_results = [
            f'\\textbf{{{result}}}'
            if float(result) == best else
            f'\\underline{{{result}}}'
            if float(result) == second_best else
            result
            for result in results
        ]
        return labeled_results

    for task in [
        'LaMP-1', 'LaMP-2', 'LaMP-3', 'LaMP-4', 'LaMP-5', 'LaMP-7',
        'LongLaMP-2', 'LongLaMP-3', 'LongLaMP-4'
    ]:
        metric_results = defaultdict(lambda: defaultdict(list))

        for llm in ['phi-4-mini-instruct', 'llama-3-8b-instruct']:
            results = bandit_ramp_results(task, llm, retriever, num_candidates, num_rerank, version)

            for metric, result in results.items():
                metric_results[metric][llm].append(result)

            for reranker in ['icr', 'rank_gpt', 'replug', 'icralm', 'contriever', 'bm25']:
                try:
                    results = baseline_results(task, llm, retriever, num_candidates, reranker, num_rerank)

                    for metric, result in results.items():
                        metric_results[metric][llm].append(result)
                except IndexError:
                    print(f'{reranker} not found')
                    continue

        for metric, llm_results in metric_results.items():
            for llm, results in llm_results.items():
                metric_results[metric][llm] = label_results(results, higher_is_better=(task != 'LaMP-3'))

        print(f'{task}')

        for metric, llm_results in metric_results.items():
            results = list(itertools.chain.from_iterable(zip(*llm_results.values())))
            fmetric = (
                'Accuracy $\\uparrow$' if metric == 'accuracy' else
                'F1 $\\uparrow$' if metric == 'f1' else
                'MAE $\\downarrow$' if metric == 'mae' else
                'RMSE $\\downarrow$' if metric == 'rmse' else
                'ROUGE-1 $\\uparrow$' if metric == 'rouge-1' else
                'ROUGE-L $\\uparrow$' if metric == 'rouge-L' else
                'METEOR $\\uparrow$' if metric == 'meteor' else
                None
            )
            fresults = f'\n{" " * 12}& '.join([' & '.join(results[i:i+2]) for i in range(0, len(results), 2)])
            print(f'{" " * 8}& {fmetric}')
            print(f'{" " * 12}& {fresults} \\\\')

        print('-' * 100)


def baseline_results(
    task: str, llm: str,
    retriever: str, num_candidates: int,
    reranker: str, num_rerank: int
) -> dict[str, str]:
    exp_name = f'{llm}/{retriever}-{num_candidates}/{reranker}-{num_rerank}/{task}'
    result_file = list((Path('./logs') / exp_name).rglob('*.out'))[0]

    with open(result_file, 'r') as file:
        text = file.read()

    results_list = [json.loads(match) for match in re.findall(r'\{.*?\}', text, flags=re.DOTALL)]
    assert len(results_list) == 1

    return {
        key: f'{value:.3f}'
        for key, value in results_list[0].items()
        if key in {'accuracy', 'f1', 'mae', 'rmse', 'rouge-1', 'rouge-L', 'meteor'}
    }


def bandit_ramp_results(
    task: str, llm: str,
    retriever: str, num_candidates: int,
    num_rerank: int, version: str
) -> dict[str, str]:
    exp_name = f'{llm}/{retriever}-{num_candidates}/bandit_ramp-{num_rerank}/{version}/{task}'
    results = []

    for result_file in (Path('./logs') / exp_name).rglob('*.out'):
        with open(result_file, 'r') as file:
            text = file.read()

        results.extend([json.loads(match) for match in re.findall(r'\{.*?\}', text, flags=re.DOTALL)])

    key = lambda x: (x['accuracy'] if 'accuracy' in x else -x['mae'] if 'mae' in x else x['rouge-1'])
    best_results = max(results, key=key)
    return {
        key: f'{value:.3f}'
        for key, value in best_results.items()
        if key in {'accuracy', 'f1', 'mae', 'rmse', 'rouge-1', 'rouge-L', 'meteor'}
    }


if __name__ == '__main__':
    fire.Fire()
