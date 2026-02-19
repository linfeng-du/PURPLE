import json
import logging
import random
import time

logging.getLogger("absl").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)

import hydra
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

import numpy as np
from datasets import Dataset

import torch
from transformers import AutoTokenizer

from lamp import (
    create_chat_prompt_fn,
    create_metric_fn,
    create_prompt_fn,
    create_retrieval_fn
)
from llm import create_llm
from purple import load_retrieved_lamp_dataset


logger = logging.getLogger(__name__)


@hydra.main(config_path="../conf", config_name="baseline", version_base=None)
def main(cfg: DictConfig) -> None:
    # Seed everything for reproducibility
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    torch.cuda.manual_seed(cfg.seed)

    # Prepare dataset
    test_split = "dev" if cfg.task.startswith("lamp") else "test"
    test_dataset = load_retrieved_lamp_dataset(
        cfg.task, test_split, cfg.candidate_retriever, cfg.num_candidates
    )

    start_time = time.perf_counter()

    if cfg.retriever == "icralm":
        predictions, references = icralm(cfg, test_dataset)
    elif cfg.retriever == "replug":
        predictions, references = replug(cfg, test_dataset)
    else:
        prompt_fn = create_prompt_fn(**cfg.create_prompt_fn)
        chat_prompt_fn = create_chat_prompt_fn(cfg.task)

        chat_prompts = []
        references = []

        for example in tqdm(test_dataset, desc="Generating Prompts"):
            chat_prompt = chat_prompt_fn(
                prompt_fn(
                    example["source"],
                    example["profile"],
                    example["query"],
                    example["corpus"]
                )
            )
            chat_prompts.append(chat_prompt)
            references.append(example["target"])

        llm = create_llm(**cfg.llm)
        predictions = [
            comps[0] for comps in llm.generate(chat_prompts, verbose=True)
        ]

    elapsed_time = time.perf_counter() - start_time

    metric_fn = create_metric_fn(cfg.task)
    test_results = metric_fn(predictions, references)

    logger.info(f"Evaluation results:\n{json.dumps(test_results, indent=2)}")
    logger.info(
        f"{len(test_dataset)} examples | {elapsed_time} seconds | "
        f"{elapsed_time / len(test_dataset)} seconds per example"
    )


def icralm(
    cfg: DictConfig,
    dataset: Dataset,
    retrieve_stride: int = 5,
    retrieve_length: int = 5
) -> tuple[list[str], list[str]]:
    contriever = create_retrieval_fn(retriever="contriever")

    cfg.retriever = "first_k"
    cfg.num_retrieve = 1
    prompt_fn = create_prompt_fn(**cfg.create_prompt_fn)
    chat_prompt_fn = create_chat_prompt_fn(cfg.task)

    if cfg.llm.backend == "hf":
        max_completion_length = cfg.llm.generation_kwargs.max_new_tokens
        OmegaConf.update(
            cfg.llm.generation_kwargs, key="max_new_tokens", value=1
        )
    elif cfg.llm.backend == "vllm":
        max_completion_length = cfg.llm.generation_kwargs.max_completion_tokens
        OmegaConf.update(
            cfg.llm.generation_kwargs, key="max_completion_tokens", value=1
        )

    llm = create_llm(**cfg.llm)

    predictions = []
    references = []

    for example in tqdm(dataset, desc="Generating completions"):
        profile = contriever(
            example["query"],
            example["corpus"],
            example["profile"],
            cfg.num_retrieve
        )
        prompts = [
            prompt_fn(
                example["source"], [rec], example["query"], example["corpus"]
            )
            for rec in profile
        ]

        cur_prompt = prompts[0]
        completion_tokens = []

        for _ in range(max_completion_length):
            if (
                len(completion_tokens) > retrieve_length
                and len(completion_tokens) % retrieve_stride == 0
            ):
                completion_prefixes = [
                    ''.join(completion_tokens[:-retrieve_length])
                    for _ in range(len(prompts))
                ]
                chat_prompts = [
                    chat_prompt_fn(p, c)
                    for p, c in zip(prompts, completion_prefixes, strict=True)
                ]
                completions = [
                    ''.join(completion_tokens[-retrieve_length:])
                    for _ in range(len(prompts))
                ]

                logprobs = llm.compute_completion_logprobs(
                    chat_prompts, completions
                )
                cur_prompt = prompts[logprobs.argmax()]

            if completion_tokens:
                completion_prefix = ''.join(completion_tokens)
            else:
                completion_prefix = None

            chat_prompts = [chat_prompt_fn(cur_prompt, completion_prefix)]
            completion = llm.generate(chat_prompts)[0][0]

            if completion == "":
                break

            completion_tokens.append(completion)

        predictions.append("".join(completion_tokens))
        references.append(example["target"])

    return predictions, references


def replug(cfg: DictConfig, dataset: Dataset) -> tuple[list[str], list[str]]:
    contriever = create_retrieval_fn(retriever="contriever")

    cfg.retriever = "first_k"
    cfg.num_retrieve = 1
    prompt_fn = create_prompt_fn(**cfg.create_prompt_fn)
    chat_prompt_fn = create_chat_prompt_fn(cfg.task)

    if cfg.llm.backend == "hf":
        OmegaConf.update(
            cfg.llm.generation_kwargs,
            key="num_return_sequences",
            value=4,
            force_add=True
        )
    elif cfg.llm.backend == "vllm":
        OmegaConf.update(
            cfg.llm.generation_kwargs, key="n", value=4, force_add=True
        )

    llm = create_llm(**cfg.llm)

    predictions = []
    references = []

    for example in tqdm(dataset, desc="Generating completions"):
        profile, retriever_logprobs = contriever.retrieve_with_logprobs(
            example["query"],
            example["corpus"],
            example["profile"],
            cfg.num_retrieve
        )
        chat_prompts = [
            chat_prompt_fn(
                prompt_fn(
                    example["source"],
                    [rec],
                    example["query"],
                    example["corpus"]
                )
            )
            for rec in profile
        ]
        completions = [
            c
            for completions in llm.generate(chat_prompts)
            for c in completions
        ]

        all_chat_prompts = [
            c for _ in range(len(completions)) for c in chat_prompts
        ]
        all_completions = [
            c for c in completions for _ in range(len(chat_prompts))
        ]
        llm_logprobs = llm.compute_completion_logprobs(
            all_chat_prompts, all_completions
        )
        llm_logprobs = (
            llm_logprobs.to(retriever_logprobs.device)
            .view(len(completions), len(chat_prompts))
        )

        logprobs = llm_logprobs + retriever_logprobs
        marginal_logprobs = torch.logsumexp(logprobs, dim=-1)

        predictions.append(completions[marginal_logprobs.argmax()])
        references.append(example["target"])

    return predictions, references


if __name__ == "__main__":
    main()
