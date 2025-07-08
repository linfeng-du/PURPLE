import os
import math
import itertools

import evaluate
import matplotlib.pyplot as plt
from transformers import AutoTokenizer

import fire
from tqdm import tqdm

from llm import LLM
from lamp import Contriever, load_lamp_dataset, create_prompt_generator
from bandit_pr import load_retrieved_lamp_dataset


def cache_retrieved_dataset(task: str, num_candidates: int) -> None:
    load_retrieved_lamp_dataset(task, 'train', num_candidates)
    load_retrieved_lamp_dataset(task, 'dev', num_candidates)


def inspect_token_length() -> None:
    max_query_length = 512
    max_document_length = 512
    tokenizer = AutoTokenizer.from_pretrained('facebook/contriever')

    for task in [
        'LaMP-1', 'LaMP-2', 'LaMP-3', 'LaMP-4', 'LaMP-5', 'LaMP-7',
        'LongLaMP-2', 'LongLaMP-3', 'LongLaMP-4'
    ]:
        query_cnt = 0
        question_cnt = 0
        document_cnt = 0
        train_dataset = load_lamp_dataset(task, 'train')

        for example in tqdm(train_dataset, desc=task):
            query_tokens = tokenizer(example['query'])['input_ids']
            corpus_tokens = tokenizer(example['corpus'])['input_ids']

            if len(query_tokens) > max_query_length:
                query_cnt += 1

            flag = False

            for document_tokens in corpus_tokens:
                if len(document_tokens) > max_document_length:
                    if not flag:
                        question_cnt += 1
                        flag = True

                    document_cnt += 1

        print(
            f'{task}: Query count: {query_cnt} | '
            f'Question count: {question_cnt} | '
            f'Document count: {document_cnt}'
        )


def inspect_performance_range(task: str, num_retrieve: int) -> None:
    test_dataset = load_lamp_dataset(task, 'dev')

    contriever = Contriever()
    contriever.to('cuda')

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
        },
        verbose=True
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

        predictions = llm.generate(sources)
        rouge_results = rouge_metric.compute(
            predictions=[prediction.strip() for prediction in predictions],
            references=[[target.strip()] for target in targets],
            rouge_types=['rouge1'],
            use_aggregator=False
        )

        plt.scatter(range(len(rouge_results['rouge1'])), rouge_results['rouge1'], s=4)
        plt.savefig(f'{save_dir}/{index}.png')
        plt.clf()


if __name__ == '__main__':
    fire.Fire()
