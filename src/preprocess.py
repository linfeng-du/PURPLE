import os

import evaluate
from datasets import load_dataset
from transformers import AutoTokenizer

import fire

from bandit_pr import load_retrieved_lamp_dataset, create_preprocessor


def download() -> None:
    print('Downloading models...')
    os.system('huggingface-cli download facebook/contriever')
    os.system('huggingface-cli download meta-llama/Meta-Llama-3-8B-Instruct')
    os.system('huggingface-cli download microsoft/Phi-4-mini-instruct')

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


if __name__ == '__main__':
    fire.Fire()
