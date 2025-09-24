import json
import random
import sys
from collections import defaultdict
from pathlib import Path
from time import perf_counter

import fire
from tqdm import tqdm

from transformers import AutoTokenizer

sys.path.append(str(Path(__file__).parent.parent.absolute() / 'src'))
from bandit_ramp import ScoreModel, load_retrieved_lamp_dataset
from lamp import create_metric, create_prompt_generator
from llm import LLM

from table import bandit_ramp_results, baseline_results


def correct_rate(task: str, llm: str, reranker: str) -> None:
    metric = ('accuracy' if task in {'LaMP-1', 'LaMP-2'} else 'mae' if task == 'LaMP-3' else 'rouge-1')
    model = (
        'microsoft/Phi-4-mini-instruct' if llm == 'phi-4-mini-instruct' else
        'meta-llama/Meta-Llama-3-8B-Instruct'
    )

    if reranker == 'bandit_ramp':
        performance = bandit_ramp_results(task, llm, 'contriever', 20, 5, 'cross_attn-12')[metric]
    else:
        performance = baseline_results(task, llm, 'bm25', 20, reranker, 5)[metric]

    print(performance)

    tokenizer = AutoTokenizer.from_pretrained(model)

    if reranker == 'bandit_ramp':
        exp_name = f'{llm}/contriever-20/bandit_ramp-5/cross_attn-12/{task}'
        score_model = ScoreModel.from_pretrained(f'./models/{exp_name}')
        score_model.to('cuda')
    else:
        retrieval_prompt_generator = create_prompt_generator(
            task, reranker, num_retrieve=5,
            max_length=2048, tokenizer=tokenizer
        )

    prompt_generator = create_prompt_generator(
        task, retriever='first_k', num_retrieve=5,
        max_length=2048, tokenizer=tokenizer
    )
    metric_fn = create_metric(task, average=False)
    
    llm = LLM(
        task,
        model=model,
        provider='local', endpoint=None,
        generate_config={
            'batch_size': 4,
            'max_new_tokens': 256,
            'do_sample': True,
            'num_beams': 1,
            'temperature': 0.7,
            'top_p': 0.8
        }
    )

    test_split = ('dev' if task.startswith('LaMP') else 'test')
    test_dataset = load_retrieved_lamp_dataset(task, test_split, retriever='contriever', num_candidates=20)

    correct_cnt = 0
    total_time = 0
    results = defaultdict(list)

    for example in tqdm(test_dataset, desc='Testing'):
        prompts = []

        start_time = perf_counter()

        if reranker == 'bandit_ramp':
            retrieved_profiles = score_model.rerank(
                example['query'], example['corpus'], example['profiles'], num_rerank=5
            )
            prompt = prompt_generator(example['source'], retrieved_profiles)
        else:
            prompt, retrieved_profiles = retrieval_prompt_generator(
                example['source'], example['profiles'],
                example['query'], example['corpus'],
                return_retrieved=True
            )

        end_time = perf_counter()
        total_time += end_time - start_time

        prompts.append(prompt)

        for _ in range(5):
            profiles = retrieved_profiles[:]
            random.shuffle(profiles)
            prompt = prompt_generator(example['source'], profiles)
            prompts.append(prompt)

        predictions = llm.generate(prompts)
        all_results = metric_fn(predictions, [example['target']] * len(prompts))

        if (
            (task == 'LaMP-3' and all_results[metric][0] == min(all_results[metric]))
            or (task != 'LaMP-3' and all_results[metric][0] == max(all_results[metric]))
        ):
            correct_cnt += 1

        for key, value in all_results.items():
            results[key].append(value[0])

    results = {
        'time': total_time,
        'num_examples': len(test_dataset),
        'throughput': len(test_dataset) / total_time,
        'performance': performance,
        'ranking_accuracy': correct_cnt / len(test_dataset)
    }
    save_dir = Path('./logs/plot') / llm / task
    save_dir.mkdir(parents=True, exist_ok=True)

    with open(save_dir / f'{reranker}.json', 'w') as file:
        json.dump(results, file, indent=2)

    with open(save_dir / f'{reranker}-results.json', 'w') as file:
        json.dump(results, file)


if __name__ == '__main__':
    fire.Fire(correct_rate)
