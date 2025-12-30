import logging
import os
import random
from pathlib import Path

logging.getLogger("absl").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("openai._base_client").setLevel(logging.WARNING)

import hydra
from omegaconf import DictConfig, OmegaConf

import nltk
import numpy as np

if os.getenv("HF_EVALUATE_OFFLINE") == "1":
    nltk.download = lambda *args, **kwargs: None

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from lamp import create_metric_fn, create_prompt_fn
from llm import LLM
from purple import (
    ScoreModel,
    Trainer,
    create_collate_fn,
    create_preprocess_fn,
    create_reward_fn,
    load_retrieved_lamp_dataset
)


logger = logging.getLogger(__name__)


@hydra.main(config_path="../conf", config_name="purple", version_base=None)
def main(cfg: DictConfig) -> None:
    # Check config validity
    missing_keys = OmegaConf.missing_keys(cfg)

    if missing_keys:
        raise ValueError(f"Missing keys in config:\n{missing_keys}")

    if cfg.eval_every % cfg.batch_size != 0:
        raise ValueError(f"`eval_every` not divisible by `batch_size`")

    # Seed everything for reproducibility
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    torch.cuda.manual_seed_all(cfg.seed)

    # Prepare score model and LLM
    score_model = ScoreModel(**cfg.score_model)

    if cfg.resume:
        model_dir = Path(cfg.run_dir).parent / "model"
        score_model.from_pretrained(model_dir)
        logger.info(f"Loaded model from {model_dir}")

    llm = LLM(cfg.task, **cfg.llm)

    # Prepare dataset
    train_split = "train"
    test_split = "dev" if cfg.task.startswith("LaMP-") else "test"

    train_dataset = load_retrieved_lamp_dataset(
        cfg.task, train_split, cfg.candidate_retriever, cfg.num_candidates
    )
    test_dataset = load_retrieved_lamp_dataset(
        cfg.task, test_split, cfg.candidate_retriever, cfg.num_candidates
    )

    tokenizer = AutoTokenizer.from_pretrained(cfg.score_model.encoder_model)
    preprocess_fn = create_preprocess_fn(
        tokenizer=tokenizer, **cfg.preprocess_fn
    )
    train_dataset = train_dataset.map(
        preprocess_fn,
        batched=True,
        remove_columns=["query", "corpus"],
        num_proc=4
    )

    # Re-create tokenizer to keep the `.map()` fingerprint deterministic
    tokenizer = AutoTokenizer.from_pretrained(cfg.score_model.encoder_model)
    preprocess_fn = create_preprocess_fn(
        tokenizer=tokenizer, **cfg.preprocess_fn
    )
    test_dataset = test_dataset.map(
        preprocess_fn,
        batched=True,
        remove_columns=["query", "corpus"],
        num_proc=4
    )

    collate_fn = create_collate_fn(tokenizer)
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        drop_last=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=cfg.eval_batch_size, collate_fn=collate_fn
    )

    # Prepare LaMP components
    prompt_fn = create_prompt_fn(
        retriever="first_k",
        tokenizer=AutoTokenizer.from_pretrained(cfg.llm.model),
        **cfg.prompt_fn
    )
    reward_fn = create_reward_fn(cfg.task)
    metric_fn = create_metric_fn(cfg.task)

    # Create trainer and start training
    trainer = Trainer(
        cfg,
        score_model,
        llm,
        train_loader,
        test_loader,
        prompt_fn,
        reward_fn,
        metric_fn,
        cfg.resume
    )
    trainer.train()


if __name__ == "__main__":
    main()
