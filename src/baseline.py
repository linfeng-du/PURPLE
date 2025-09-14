import itertools
import json
import logging
import os
import random
import time

import nltk
import numpy as np
from datasets import Dataset

import torch
from transformers import AutoTokenizer

import hydra
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

from bandit_ramp import load_retrieved_lamp_dataset
from lamp import create_metric, create_prompt_generator
from lamp.data_types import PromptGenerator
from lamp.retrievers import Contriever
from llm import LLM


if os.getenv('HF_EVALUATE_OFFLINE') == '1':
    nltk.download = lambda *args, **kwargs: None


logging.getLogger('absl').setLevel(logging.WARNING)
logging.getLogger('httpx').setLevel(logging.WARNING)
logger = logging.getLogger(__name__)


@hydra.main(config_path='../conf', config_name='baseline', version_base=None)
def main(cfg: DictConfig) -> None:
    # Check for missing keys
    missing_keys = OmegaConf.missing_keys(cfg)

    if missing_keys:
        raise ValueError(f'Missing keys in config:\n{missing_keys}')

    # Seed everything for reproducibility
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    torch.cuda.manual_seed(cfg.seed)

    # Prepare dataset
    test_split = ('dev' if cfg.task.startswith('LaMP') else 'test')
    test_dataset = load_retrieved_lamp_dataset(cfg.task, test_split, cfg.retriever, cfg.num_candidates)

    # Prepare LaMP components
    if cfg.reranker in {'replug', 'icralm'}:
        retriever = 'first_k'
        num_retrieve = 1
    else:
        retriever = cfg.reranker
        num_retrieve = cfg.num_rerank

    tokenizer = (
        AutoTokenizer.from_pretrained(cfg.llm.model)
        if cfg.llm.provider == 'local' else
        AutoTokenizer.from_pretrained('gpt2')
    )
    prompt_generator = create_prompt_generator(
        cfg.task, retriever, num_retrieve,
        cfg.prompt_generator.max_length, tokenizer
    )
    metric_fn = create_metric(cfg.task)

    start_time = time.perf_counter()

    if cfg.reranker == 'replug':
        predictions, targets = replug(cfg, test_dataset, prompt_generator)
    elif cfg.reranker == 'icralm':
        predictions, targets = icralm(cfg, test_dataset, prompt_generator)
    else:
        predictions, targets = reranker(cfg, test_dataset, prompt_generator)

    elapsed_time = time.perf_counter() - start_time

    test_results = metric_fn(predictions, targets)
    logger.info(f'Evaluation results:\n{json.dumps(test_results, indent=2)}')
    logger.info(
        f'{len(test_dataset)} examples | {elapsed_time} seconds | '
        f'{elapsed_time / len(test_dataset)} seconds per example'
    )


def replug(cfg: DictConfig, dataset: Dataset, prompt_generator: PromptGenerator) -> (
    tuple[list[str], list[str]]
):
    contriever = Contriever(torch.device('cuda'))
    llm = LLM(
        cfg.task, cfg.llm.model, provider='local',
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

    predictions = []
    targets = []

    for example in tqdm(dataset, desc='Generating responses'):
        profiles, retriever_logps = contriever(
            example['query'], example['corpus'], example['profiles'],
            cfg.num_rerank, return_logps=True
        )
        target = example['target']

        sources = [prompt_generator(example['source'], [profile]) for profile in profiles]
        all_predictions = llm.generate(sources)
        all_predictions = list(itertools.chain.from_iterable(all_predictions))

        expanded_sources = [source for _ in range(len(all_predictions)) for source in sources]
        expanded_predictions = [prediction for prediction in all_predictions for _ in range(len(sources))]
        llm_logps = llm.compute_target_logps(expanded_sources, expanded_predictions)
        llm_logps = llm_logps.to(retriever_logps.device).view(len(all_predictions), len(sources))

        logps = llm_logps + retriever_logps
        marginal_logps = torch.logsumexp(logps, dim=1)
        best_index = marginal_logps.argmax().item()
        prediction = all_predictions[best_index]

        predictions.append(prediction)
        targets.append(target)

    return predictions, targets


def icralm(cfg: DictConfig, dataset: Dataset, prompt_generator: PromptGenerator) -> (
    tuple[list[str], list[str]]
):
    rerank_stride = 5
    rerank_length = 5

    contriever = Contriever(torch.device('cuda'))
    llm = LLM(cfg.task, **cfg.llm)

    predictions = []
    targets = []

    for example in tqdm(dataset, desc='Generating responses'):
        profiles = contriever(example['query'], example['corpus'], example['profiles'], cfg.num_rerank)
        prompts = [prompt_generator(example['source'], [profile]) for profile in profiles]
        target = example['target']

        prompts_ids = llm.apply_chat_template(prompts)
        cur_prompt_ids = prompts_ids[0]
        new_ids = []

        while (
            (not new_ids) or (
                len(new_ids) < llm.generate_config['max_new_tokens']
                and new_ids[-1] != llm.pipeline.tokenizer.eos_token_id
            )
        ):
            if new_ids and len(new_ids) % rerank_stride == 0:
                concats_ids = [prompt_ids + new_ids for prompt_ids in prompts_ids]
                inputs_ids = [concat_ids[:-rerank_length] for concat_ids in concats_ids]
                targets_ids = [concat_ids[-rerank_length:] for concat_ids in concats_ids]

                logps = llm.compute_target_id_logps(inputs_ids, targets_ids)
                best_index = logps.argmax().item()
                cur_prompt_ids = prompts_ids[best_index]

            cur_input_ids = cur_prompt_ids + new_ids
            cur_input_ids = torch.tensor([cur_input_ids], dtype=torch.long, device=llm.device)

            with torch.no_grad():
                outputs = llm.pipeline.model(input_ids=cur_input_ids)
                logits = outputs.logits[:, -1, :]

            logits /= llm.generate_config['temperature']
            probs = torch.softmax(logits, dim=1)
            sorted_probs, sorted_indices = probs.sort(dim=1, descending=True)
            cum_probs = torch.cumsum(sorted_probs, dim=1)

            mask = (cum_probs <= llm.generate_config['top_p'])
            mask[:, 0] = 1
            mask[:, 1:] = mask[:, :-1].clone()

            probs = torch.where(mask, sorted_probs, torch.zeros_like(sorted_probs))
            probs = probs / probs.sum(dim=1, keepdim=True)
            index = torch.multinomial(probs, num_samples=1)
            next_token_id = sorted_indices.gather(dim=1, index=index)

            cur_input_ids = torch.cat([cur_input_ids, next_token_id], dim=1)
            new_ids.append(next_token_id.item())

        prediction = llm.pipeline.tokenizer.decode(new_ids, skip_special_tokens=True)
        predictions.append(prediction)
        targets.append(target)

    return predictions, targets


def reranker(cfg: DictConfig, dataset: Dataset, prompt_generator: PromptGenerator) -> (
    tuple[list[str], list[str]]
):
    prompts = []
    targets = []

    for example in tqdm(dataset, desc='Generating Prompts'):
        prompt = prompt_generator(
            example['source'], example['profiles'],
            example['query'], example['corpus']
        )
        target = example['target']
        prompts.append(prompt)
        targets.append(target)

    llm = LLM(cfg.task, **cfg.llm)
    predictions = llm.generate(prompts, verbose=True)
    return predictions, targets


if __name__ == '__main__':
    main()
