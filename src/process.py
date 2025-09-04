import re
import json
from pathlib import Path
from collections import defaultdict

import fire


def download() -> None:
    import evaluate
    from datasets import load_dataset
    from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM

    print('Downloading tokenizers...')
    AutoTokenizer.from_pretrained('facebook/contriever')
    AutoTokenizer.from_pretrained('meta-llama/Meta-Llama-3-8B-Instruct')
    AutoTokenizer.from_pretrained('microsoft/Phi-4-mini-instruct')

    print('Downloading models...')
    AutoModel.from_pretrained('facebook/contriever')
    AutoModelForCausalLM.from_pretrained('meta-llama/Meta-Llama-3-8B-Instruct')
    AutoModelForCausalLM.from_pretrained('microsoft/Phi-4-mini-instruct')

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


def preprocess() -> None:
    from transformers import AutoTokenizer
    from bandit_pr import load_retrieved_lamp_dataset, create_preprocessor

    for task in [
        'LaMP-1', 'LaMP-2', 'LaMP-3', 'LaMP-4', 'LaMP-5', 'LaMP-7',
        'LongLaMP-2', 'LongLaMP-3', 'LongLaMP-4'
    ]:
        print(f'Preprocessing {task}...')
        test_split = 'dev' if task.startswith('LaMP') else 'test'
        train_dataset = load_retrieved_lamp_dataset(task, 'train', num_candidates=20)
        test_dataset = load_retrieved_lamp_dataset(task, test_split, num_candidates=20)

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


def baseline_results_formatted() -> None:
    for task in [
        'LaMP-1', 'LaMP-2', 'LaMP-3', 'LaMP-4', 'LaMP-5', 'LaMP-7',
        'LongLaMP-2', 'LongLaMP-3', 'LongLaMP-4'
    ]:
        metric_results = defaultdict(list)

        for retriever in ['icr', 'rank_gpt', 'contriever', 'bm25']:
            for llm in ['phi-4-mini-instruct', 'llama-3-8b-instruct']:
                results = baseline_results(task, retriever, llm)

                for metric, result in results.items():
                    metric_results[metric].append(result)

        for metric, results in metric_results.items():
            print(task, metric)
            print('& ' + '\n& '.join([' & '.join(results[i : i + 2]) for i in range(0, len(results), 2)]) + ' \\\\')
            print('-' * 100)


def baseline_results(task: str, retriever: str, llm: str) -> None:
    result_dir = Path(f'logs/{llm}/{retriever}-5/{task}')
    result_file = list(result_dir.rglob('*.out'))[0]

    with open(result_file, 'r') as file:
        text = file.read()

    results_list = [json.loads(match) for match in re.findall(r'\{.*?\}', text, flags=re.DOTALL)]
    assert len(results_list) == 1

    results = {
        key: f'{value:.3f}'
        for key, value in results_list[0].items()
        if key in ['accuracy', 'f1', 'mae', 'rmse', 'rouge-1', 'rouge-L', 'meteor']
    }
    return results


def bandit_pr_results_formatted(version: str) -> None:
    for task in [
        'LaMP-1', 'LaMP-2', 'LaMP-3', 'LaMP-4', 'LaMP-5', 'LaMP-7',
        'LongLaMP-2', 'LongLaMP-3', 'LongLaMP-4'
    ]:
        metric_results = defaultdict(list)

        for llm in ['phi-4-mini-instruct', 'llama-3-8b-instruct']:
            results = bandit_pr_results(task, version, llm)

            for metric, result in results.items():
                metric_results[metric].append(result)

        for metric, results in metric_results.items():
            print(task, metric)
            print('& ' + '\n& '.join([' & '.join(results[i : i + 2]) for i in range(0, len(results), 2)]))
            print('-' * 100)


def bandit_pr_results(task: str, version: str, llm: str) -> None:
    results = []
    result_dir = Path(f'logs/{llm}/bandit_pr-5/{version}/{task}')

    for result_file in result_dir.rglob('*.out'):
        with open(result_file, 'r') as file:
            text = file.read()

        results.extend([json.loads(match) for match in re.findall(r'\{.*?\}', text, flags=re.DOTALL)])

    key = lambda x: (
        x['accuracy'] if 'accuracy' in x else
        -x['mae'] if 'mae' in x else x['rouge-1']
    )
    best_results = max(results, key=key)
    best_results = {
        key: f'{value:.3f}'
        for key, value in best_results.items()
        if key in ['accuracy', 'f1', 'mae', 'rmse', 'rouge-1', 'rouge-L', 'meteor']
    }
    return best_results


if __name__ == '__main__':
    fire.Fire()
