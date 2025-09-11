import itertools
import json

import torch
from transformers import AutoTokenizer

import fire
from tqdm import tqdm

from bandit_pr import load_retrieved_lamp_dataset
from lamp import create_metric, create_prompt_generator
from lamp.retrievers import Contriever
from llm import LLM


def replug(llm: str, task: str, num_retrieve: int) -> None:
    contriever = Contriever(torch.device('cuda'))
    model = (
        'microsoft/Phi-4-mini-instruct'
        if llm == 'phi-4-mini-instruct' else
        'meta-llama/Meta-Llama-3-8B-Instruct'
    )
    llm = LLM(
        task, model, provider='local',
        generate_config={
            'batch_size': 1,
            'max_new_tokens': 256,
            'do_sample': False,
            'num_beams': 4,
            'temperature': None,
            'top_p': None,
            'num_return_sequences': 4
        }
    )

    prompt_generator = create_prompt_generator(
        task, retriever='first_k', num_retrieve=1,
        max_length=2048, tokenizer=AutoTokenizer.from_pretrained(model)
    )

    test_split = ('dev' if task.startswith('LaMP') else 'test')
    test_dataset = load_retrieved_lamp_dataset(task, test_split, retriever='bm25', num_candidates=20)

    predictions = []
    targets = []

    for example in tqdm(test_dataset, desc='Evaluating marginalization'):
        profiles, retriever_logps = contriever(
            example['query'], example['corpus'], example['profiles'],
            num_retrieve, return_logps=True
        )

        sources = [prompt_generator(example['source'], [profile]) for profile in profiles]
        prediction_beams = llm.generate(sources)
        all_predictions = list(itertools.chain.from_iterable(prediction_beams))

        expanded_sources = [source for _ in range(len(all_predictions)) for source in sources]
        expanded_predictions = [prediction for prediction in all_predictions for _ in range(len(sources))]
        llm_logps = llm.compute_target_logps(expanded_sources, expanded_predictions)

        logps = llm_logps.view(len(all_predictions), len(sources)) + retriever_logps
        marginal_logps = torch.logsumexp(logps, dim=1)
        index = marginal_logps.argmax().item()
        prediction = all_predictions[index]

        predictions.append(prediction)
        targets.append(example['target'])

    metric_fn = create_metric(task)
    test_results = metric_fn(predictions, targets)
    print(f'Evaluation results:\n{json.dumps(test_results, indent=2)}')


if __name__ == '__main__':
    fire.Fire()
