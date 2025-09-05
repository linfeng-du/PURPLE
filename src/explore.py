import os
import math
import itertools

import matplotlib.pyplot as plt

import torch
from torch.utils.data import ConcatDataset
import evaluate
from transformers import AutoTokenizer

from llm import LLM
from lamp.retrievers import Contriever
from lamp import load_lamp_dataset, create_prompt_generator
from bandit_pr import load_retrieved_lamp_dataset

import fire
from tqdm import tqdm


def dataset_stats() -> None:
    tokenizer = AutoTokenizer.from_pretrained('facebook/contriever')

    for task in [
        'LaMP-1', 'LaMP-2', 'LaMP-3', 'LaMP-4', 'LaMP-5', 'LaMP-7',
        'LongLaMP-2', 'LongLaMP-3', 'LongLaMP-4'
    ]:
        query_lengths = []
        document_lengths = []
        target_lengths = []

        test_split = 'dev' if task.startswith('LaMP') else 'test'
        dataset = ConcatDataset([
            load_lamp_dataset(task, 'train'),
            load_lamp_dataset(task, test_split)
        ])

        for example in tqdm(dataset, desc=task):
            query_tokens = tokenizer(example['query'])['input_ids']
            corpus_tokens = tokenizer(example['corpus'])['input_ids']
            target_tokens = tokenizer(example['target'])['input_ids']

            query_lengths.append(len(query_tokens))
            document_lengths.extend(list(map(len, corpus_tokens)))
            target_lengths.append(len(target_tokens))

        print(task)
        print(f'Average query length: {sum(query_lengths) / len(query_lengths)}')
        print(f'Max query length: {max(query_lengths)}')
        print(f'Average document length: {sum(document_lengths) / len(document_lengths)}')
        print(f'Max document length: {max(document_lengths)}')
        print(f'Average target length: {sum(target_lengths) / len(target_lengths)}')
        print(f'Max target length: {max(target_lengths)}')
        print('-' * 100)


def performance_range(task: str, num_retrieve: int) -> None:
    test_dataset = load_lamp_dataset(task, 'dev')
    contriever = Contriever(torch.device('cuda'))

    llm = LLM(
        task,
        model='microsoft/Phi-4-mini-instruct',
        provider='local',
        generate_config={
            'batch_size': 1,
            'max_new_tokens': 256,
            'do_sample': True,
            'num_beams': 1,
            'temperature': 0.7,
            'top_p': 0.8
        }
    )
    prompt_generator = create_prompt_generator(
        task,
        'first_k',
        num_retrieve,
        max_length=2048,
        tokenizer=AutoTokenizer.from_pretrained('microsoft/Phi-4-mini-instruct')
    )
    rouge_metric = evaluate.load('rouge')

    save_dir = f'logs/debug/{task}-{num_retrieve}'
    os.makedirs(save_dir, exist_ok=True)

    for index, example in enumerate(test_dataset):
        sources = []
        targets = []
        retrieved_profiles = contriever(
            example['query'],
            example['corpus'],
            example['profiles'],
            num_retrieve=20
        )

        for profiles in tqdm(
            itertools.combinations(retrieved_profiles, r=num_retrieve),
            desc=f'Generating Prompts: {index + 1}/{len(test_dataset)}',
            total=math.comb(len(retrieved_profiles), num_retrieve)
        ):
            source = prompt_generator(example['source'], profiles)
            target = example['target']

            sources.append(source)
            targets.append(target)

        predictions = llm.generate(sources, verbose=True)
        rouge_results = rouge_metric.compute(
            predictions=[prediction.strip() for prediction in predictions],
            references=[[target.strip()] for target in targets],
            rouge_types=['rouge1'],
            use_aggregator=False
        )

        plt.scatter(range(len(rouge_results['rouge1'])), rouge_results['rouge1'], s=4)
        plt.savefig(f'{save_dir}/{index}.png')
        plt.clf()


def marginalization(llm: str, task: str, num_retrieve: int) -> None:
    model = (
        'microsoft/Phi-4-mini-instruct'
        if llm == 'phi-4-mini-instruct' else
        'meta-llama/Meta-Llama-3-8B-Instruct'
    )
    contriever = Contriever(torch.device('cuda'))
    prompt_generator = create_prompt_generator(
        task, retriever='first_k', num_retrieve=1, max_length=2048,
        tokenizer=AutoTokenizer.from_pretrained(model)
    )
    llm = LLM(
        task, model, provider='local',
        generate_config={
            'batch_size': 1,
            'max_new_tokens': 256,
            'do_sample': False,
            'num_beams': 4
        }
    )

    test_split = 'dev' if task.startswith('LaMP') else 'test'
    test_dataset = load_retrieved_lamp_dataset(task, test_split, num_retrieve)

    for index, example in enumerate(test_dataset):
        profiles, probs = contriever(
            example['query'],
            example['corpus'],
            example['profiles'],
            num_retrieve,
            return_probs=True
        )

        for profile, prob in zip(profiles, probs):
            source = prompt_generator(example['source'], [profile])
            predictions = llm.generate([source])

        print(probs)


if __name__ == '__main__':
    fire.Fire()
