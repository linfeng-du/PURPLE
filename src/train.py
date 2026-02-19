import logging
import random

logging.getLogger("absl").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)

import hydra
from omegaconf import DictConfig

import numpy as np
import torch

from lamp import create_chat_prompt_fn, create_metric_fn, create_prompt_fn
from llm import create_llm
from purple import (
    ScoreModel,
    Trainer,
    create_collate_fn,
    create_pretokenize_fn,
    create_reward_fn,
    load_retrieved_lamp_dataset
)


@hydra.main(config_path="../conf", config_name="purple", version_base=None)
def main(cfg: DictConfig) -> None:
    cfg.create_prompt_fn.retriever = "first_k"

    # Seed everything for reproducibility
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    torch.cuda.manual_seed_all(cfg.seed)

    # Prepare models
    score_model = ScoreModel(**cfg.score_model)
    llm = create_llm(**cfg.llm)

    # Prepare datasets
    train_split = "train"
    train_dataset = load_retrieved_lamp_dataset(
        cfg.task, train_split, cfg.candidate_retriever, cfg.num_candidates
    )
    train_dataset = train_dataset.map(
        create_pretokenize_fn(**cfg.create_pretokenize_fn),
        batched=True,
        remove_columns=["query", "corpus"],
        num_proc=4
    )

    test_split = "dev" if cfg.task.startswith("lamp") else "test"
    test_dataset = load_retrieved_lamp_dataset(
        cfg.task, test_split, cfg.candidate_retriever, cfg.num_candidates
    )
    test_dataset = test_dataset.map(
        create_pretokenize_fn(**cfg.create_pretokenize_fn),
        batched=True,
        remove_columns=["query", "corpus"],
        num_proc=4
    )

    # Create trainer and start training
    trainer = Trainer(
        score_model,
        llm,
        cfg.trainer_args,
        create_collate_fn(cfg.score_model.encoder_model),
        train_dataset,
        test_dataset,
        create_prompt_fn(**cfg.create_prompt_fn),
        create_chat_prompt_fn(cfg.task),
        create_reward_fn(cfg.task),
        create_metric_fn(cfg.task)
    )
    trainer.train()


if __name__ == "__main__":
    main()
