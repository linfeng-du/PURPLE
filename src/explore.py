import itertools
import math
import os

from torch.utils.data import ConcatDataset
from transformers import AutoTokenizer

import fire
import matplotlib.pyplot as plt
from tqdm import tqdm

from bandit_ramp import create_reward, load_retrieved_lamp_dataset
from lamp import create_prompt_generator, load_lamp_dataset
from llm import LLM


def dataset_stats() -> None:
    tokenizer = AutoTokenizer.from_pretrained('facebook/contriever')

    for task in [
        'LaMP-1', 'LaMP-2', 'LaMP-3', 'LaMP-4', 'LaMP-5', 'LaMP-7',
        'LongLaMP-2', 'LongLaMP-3', 'LongLaMP-4'
    ]:
        query_lengths = []
        document_lengths = []
        target_lengths = []

        dataset = ConcatDataset([
            load_lamp_dataset(task, 'train'),
            load_lamp_dataset(task, ('dev' if task.startswith('LaMP') else 'test'))
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


def performance_range(llm: str, task: str, num_rerank: int) -> None:
    model = (
        'microsoft/Phi-4-mini-instruct'
        if llm == 'phi-4-mini-instruct' else
        'meta-llama/Meta-Llama-3-8B-Instruct'
    )
    llm = LLM(
        task,
        model=model,
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

    test_split = ('dev' if task.startswith('LaMP') else 'test')
    test_dataset = load_retrieved_lamp_dataset(task, test_split, retriever='contriever', num_candidates=20)

    prompt_generator = create_prompt_generator(
        task, 'first_k', num_rerank,
        max_length=2048, tokenizer=AutoTokenizer.from_pretrained(model)
    )
    reward_fn = create_reward(task)

    save_dir = f'./logs/explore/{llm}/{task}-5/'
    os.makedirs(save_dir, exist_ok=True)

    for index, example in enumerate(test_dataset):
        prompts = []
        targets = []

        for profiles in tqdm(
            itertools.combinations(example['profiles'], r=num_rerank),
            desc=f'Generating Prompts: {index + 1}/{len(test_dataset)}',
            total=math.comb(len(example['profiles']), num_rerank)
        ):
            prompt = prompt_generator(example['source'], profiles)
            target = example['target']
            prompts.append(prompt)
            targets.append(target)

        predictions = llm.generate(prompts, verbose=True)
        reward = reward_fn(predictions, targets)
        reward = reward.cpu().numpy()

        plt.scatter(range(len(reward)), reward, s=4)
        plt.savefig(f'{save_dir}/{index}.png')
        plt.clf()


if __name__ == '__main__':
    fire.Fire()
