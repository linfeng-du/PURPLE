import json
import logging
import os
import random
import time

logging.getLogger("absl").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("openai._base_client").setLevel(logging.WARNING)

import hydra
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

import nltk
import numpy as np
from datasets import Dataset

if os.getenv("HF_EVALUATE_OFFLINE") == "1":
    nltk.download = lambda *args, **kwargs: None

import torch
from transformers import AutoTokenizer

from lamp import (
    create_metric_fn,
    create_prompt_fn,
    create_retrieval_fn
)
from llm import LLM
from purple import load_retrieved_lamp_dataset


logger = logging.getLogger(__name__)


@hydra.main(config_path="../conf", config_name="baseline", version_base=None)
def main(cfg: DictConfig) -> None:
    # Check config validity
    missing_keys = OmegaConf.missing_keys(cfg)

    if missing_keys:
        raise ValueError(f"Missing keys in config:\n{missing_keys}")

    # Seed everything for reproducibility
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    torch.cuda.manual_seed(cfg.seed)

    # Prepare dataset
    test_split = "dev" if cfg.task.startswith("LaMP-") else "test"
    test_dataset = load_retrieved_lamp_dataset(
        cfg.task, test_split, cfg.candidate_retriever, cfg.num_candidates
    )

    start_time = time.perf_counter()

    if cfg.retriever == "icralm":
        predictions, references = icralm(cfg, test_dataset)
    elif cfg.retriever == "replug":
        predictions, references = replug(cfg, test_dataset)
    else:
        prompt_fn = create_prompt_fn(
            retriever=cfg.retriever,
            num_retrieve=cfg.num_retrieve,
            tokenizer=AutoTokenizer.from_pretrained(cfg.llm.model),
            **cfg.prompt_fn
        )

        prompts = []
        references = []

        for example in tqdm(test_dataset, desc="Generating Prompts"):
            prompt = prompt_fn(
                example["source"],
                example["profile"],
                example["query"],
                example["corpus"]
            )
            reference = example["target"]
            prompts.append(prompt)
            references.append(reference)

        llm = LLM(cfg.task, **cfg.llm)
        predictions = llm.generate(prompts, verbose=True)

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
    prompt_fn = create_prompt_fn(
        retriever="first_k",
        num_retrieve=1,
        tokenizer=AutoTokenizer.from_pretrained(cfg.llm.model),
        **cfg.prompt_fn
    )
    contriever = create_retrieval_fn(retriever="contriever")

    llm_kwargs = OmegaConf.to_container(cfg.llm, resolve=True)

    if llm_kwargs["backend"] == "hf":
        max_completion_length = (
            llm_kwargs["generation_kwargs"]["max_new_tokens"]
        )
        llm_kwargs["generation_kwargs"]["max_new_tokens"] = 1
    elif llm_kwargs["backend"] == "vllm":
        max_completion_length = (
            llm_kwargs["generation_kwargs"]["max_completion_tokens"]
        )
        llm_kwargs["generation_kwargs"]["max_completion_tokens"] = 1

    llm = LLM(cfg.task, **llm_kwargs)

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
            prompt_fn(example["source"], [rec], None, None) for rec in profile
        ]

        cur_prompt = prompts[0]
        completion_tokens = []

        for _ in range(max_completion_length):
            if (
                len(completion_tokens) > retrieve_length
                and len(completion_tokens) % retrieve_stride == 0
            ):
                completions = [
                    ''.join(completion_tokens[-retrieve_length:])
                    for _ in range(len(prompts))
                ]
                assistant_prompts = [
                    ''.join(completion_tokens[:-retrieve_length])
                    for _ in range(len(prompts))
                ]
                logps = llm.compute_completion_logps(
                    prompts, completions, assistant_prompts=assistant_prompts
                )
                cur_prompt = prompts[logps.argmax()]

            if completion_tokens:
                assistant_prompts = [''.join(completion_tokens)]
            else:
                assistant_prompts = None

            completion = (
                llm
                .generate([cur_prompt], assistant_prompts=assistant_prompts)[0]
            )

            if completion == "":
                break

            completion_tokens.append(completion)

        prediction = "".join(completion_tokens)
        reference = example["target"]
        predictions.append(prediction)
        references.append(reference)

    return predictions, references


def replug(cfg: DictConfig, dataset: Dataset) -> tuple[list[str], list[str]]:
    prompt_fn = create_prompt_fn(
        retriever="first_k",
        num_retrieve=1,
        tokenizer=AutoTokenizer.from_pretrained(cfg.llm.model),
        **cfg.prompt_fn
    )
    contriever = create_retrieval_fn(retriever="contriever")

    llm_kwargs = OmegaConf.to_container(cfg.llm, resolve=True)

    if llm_kwargs["backend"] == "hf":
        llm_kwargs["generation_kwargs"].update({"num_return_sequences": 4})
    elif llm_kwargs["backend"] == "vllm":
        llm_kwargs["generation_kwargs"].update({"n": 4})

    llm = LLM(cfg.task, **llm_kwargs)

    predictions = []
    references = []

    for example in tqdm(dataset, desc="Generating completions"):
        profile, retriever_logps = contriever.retrieve_with_logps(
            example["query"],
            example["corpus"],
            example["profile"],
            cfg.num_retrieve
        )
        prompts = [
            prompt_fn(example["source"], [rec], None, None) for rec in profile
        ]
        completions = [
            c for completions in llm.generate(prompts) for c in completions
        ]

        all_prompts = [
            p for _ in range(len(completions)) for p in prompts
        ]
        all_completions = [
            c for c in completions for _ in range(len(prompts))
        ]
        llm_logps = llm.compute_completion_logps(all_prompts, all_completions)
        llm_logps = (
            llm_logps.to(retriever_logps.device)
            .view(len(completions), len(prompts))
        )

        logps = llm_logps + retriever_logps
        marginal_logps = torch.logsumexp(logps, dim=-1)

        prediction = completions[marginal_logps.argmax()]
        reference = example["target"]
        predictions.append(prediction)
        references.append(reference)

    return predictions, references


if __name__ == "__main__":
    main()
