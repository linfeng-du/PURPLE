import evaluate
from datasets import load_dataset
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer

import fire

from purple import create_preprocessor, load_retrieved_lamp_dataset


def download() -> None:
    print('Downloading tokenizers...')
    AutoTokenizer.from_pretrained('facebook/contriever')
    AutoTokenizer.from_pretrained('microsoft/Phi-4-mini-instruct')
    AutoTokenizer.from_pretrained('meta-llama/Meta-Llama-3-8B-Instruct')
    AutoTokenizer.from_pretrained('meta-llama/Meta-Llama-3-70B-Instruct')

    print('Downloading models...')
    AutoModel.from_pretrained('facebook/contriever')
    AutoModelForCausalLM.from_pretrained('microsoft/Phi-4-mini-instruct')
    AutoModelForCausalLM.from_pretrained('meta-llama/Meta-Llama-3-8B-Instruct')
    AutoModelForCausalLM.from_pretrained('meta-llama/Meta-Llama-3-70B-Instruct')

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


if __name__ == '__main__':
    fire.Fire()
