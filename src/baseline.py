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
logging.getLogger('openai._base_client').setLevel(logging.WARNING)
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

    prompt_generator = create_prompt_generator(
        cfg.task, retriever, num_retrieve,
        cfg.prompt_generator.max_length, AutoTokenizer.from_pretrained(cfg.llm.model)
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
    contriever = Contriever()

    if cfg.llm.provider == 'local':
        OmegaConf.set_struct(cfg.llm.generate_config, False)
        cfg.llm.generate_config.update({
            'batch_size': 1,
            'do_sample': False,
            'num_beams': 4,
            'temperature': None,
            'top_p': None,
            'num_return_sequences': 4
        })
        OmegaConf.set_struct(cfg.llm.generate_config, True)
    elif cfg.llm.provider == 'vllm':
        # Use sampling instead of beam search
        OmegaConf.set_struct(cfg.llm.generate_config, False)
        cfg.llm.generate_config.update({'n': 4})
        OmegaConf.set_struct(cfg.llm.generate_config, True)

    llm = LLM(cfg.task, **cfg.llm)

    predictions = []
    targets = []

    for example in tqdm(dataset, desc='Generating responses'):
        profiles, retriever_logps = contriever(
            example['query'], example['corpus'], example['profiles'], cfg.num_rerank,
            return_logps=True
        )
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
        targets.append(example['target'])

    return predictions, targets


def icralm(cfg: DictConfig, dataset: Dataset, prompt_generator: PromptGenerator) -> (
    tuple[list[str], list[str]]
):
    rerank_stride = 5
    rerank_length = 5

    contriever = Contriever()

    # Get the original maximum new tokens and set it to 1
    if cfg.llm.provider == 'local':
        max_new_tokens = cfg.llm.generate_config.max_new_tokens
        OmegaConf.set_struct(cfg.llm.generate_config, False)
        cfg.llm.generate_config.update({'max_new_tokens': 1})
        OmegaConf.set_struct(cfg.llm.generate_config, True)
    elif cfg.llm.provider == 'vllm':
        max_new_tokens = cfg.llm.generate_config.max_completion_tokens
        del cfg.llm.generate_config.max_completion_tokens
        OmegaConf.set_struct(cfg.llm.generate_config, False)
        cfg.llm.generate_config.update({'max_tokens': 1})
        OmegaConf.set_struct(cfg.llm.generate_config, True)

    llm = LLM(cfg.task, **cfg.llm)

    predictions = []
    targets = []

    for example in tqdm(dataset, desc='Generating responses'):
        profiles = contriever(example['query'], example['corpus'], example['profiles'], cfg.num_rerank)
        prompts = [prompt_generator(example['source'], [profile]) for profile in profiles]

        prompts_ids = llm.apply_chat_template(prompts)
        cur_prompt_ids = prompts_ids[0]
        response_ids = []

        while (
            (not response_ids)
            or (len(response_ids) < max_new_tokens and response_ids[-1] not in llm.end_token_ids)
        ):
            if response_ids and len(response_ids) % rerank_stride == 0:
                concats_ids = [prompt_ids + response_ids for prompt_ids in prompts_ids]
                prefixes_ids = [concat_ids[:-rerank_length] for concat_ids in concats_ids]
                suffixes_ids = [concat_ids[-rerank_length:] for concat_ids in concats_ids]

                prefixes = llm.tokenizer.batch_decode(prefixes_ids, skip_special_tokens=False)
                suffixes = llm.tokenizer.batch_decode(suffixes_ids, skip_special_tokens=False)

                logps = llm.compute_target_logps(prefixes, suffixes, apply_template=False)
                best_index = logps.argmax().item()
                cur_prompt_ids = prompts_ids[best_index]

            cur_input_ids = cur_prompt_ids + response_ids
            cur_inputs = llm.tokenizer.batch_decode([cur_input_ids], skip_special_tokens=False)

            outputs = llm.generate(cur_inputs, apply_template=False)

            if outputs[0] == '':
                # The API returns an empty string when the EOS token is reached
                break

            response_id = llm.tokenizer.encode(outputs[0], add_special_tokens=False)[0]
            response_ids.append(response_id)

        prediction = llm.tokenizer.decode(response_ids, skip_special_tokens=True)
        predictions.append(prediction)
        targets.append(example['target'])

    return predictions, targets


def reranker(cfg: DictConfig, dataset: Dataset, prompt_generator: PromptGenerator) -> (
    tuple[list[str], list[str]]
):
    prompts = []
    targets = []

    for example in tqdm(dataset, desc='Generating Prompts'):
        prompt = prompt_generator(example['source'], example['profiles'], example['query'], example['corpus'])
        prompts.append(prompt)
        targets.append(example['target'])

    llm = LLM(cfg.task, **cfg.llm)
    predictions = llm.generate(prompts, verbose=True)
    return predictions, targets


if __name__ == '__main__':
    main()
