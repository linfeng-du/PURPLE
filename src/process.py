import re
import json
from pathlib import Path

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


def baseline_results(task: str, retriever: str, llm: str) -> None:
    result_dir = Path(f'logs/{llm}/{retriever}-5/{task}')
    result_file = list(result_dir.rglob('*.out'))[0]

    with open(result_file, 'r') as file:
        text = file.read()

    results = [json.loads(match) for match in re.findall(r'\{.*?\}', text, flags=re.DOTALL)]
    assert len(results) == 1

    result = {
        key: f'{value:.3f}'
        for key, value in results[0].items()
        if key in ['accuracy', 'f1', 'mae', 'rmse', 'rouge-1', 'rouge-L']
    }
    print(json.dumps(result, indent=4))


def bandit_pr_results(task: str, version: str, llm: str) -> None:
    result_dir = Path(f'logs/{llm}/bandit_pr-5/{version}/{task}')
    result_file = list(result_dir.rglob('*.out'))[0]

    with open(result_file, 'r') as file:
        text = file.read()

    results = [json.loads(match) for match in re.findall(r'\{.*?\}', text, flags=re.DOTALL)]
    key = lambda x: (
        x['accuracy'] if 'accuracy' in x else
        x['mae'] if 'mae' in x else x['rouge-1']
    )
    best_result = max(results, key=key)
    best_result = {
        key: f'{value:.3f}'
        for key, value in best_result.items()
        if key in ['accuracy', 'f1', 'mae', 'rmse', 'rouge-1', 'rouge-L']
    }
    print(json.dumps(best_result, indent=4))


if __name__ == '__main__':
    fire.Fire()
