import logging
import os
import random

import nltk
import numpy as np

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

import hydra
from omegaconf import DictConfig, OmegaConf

from bandit_pr import (
    create_collator,
    create_preprocessor,
    create_reward,
    load_retrieved_lamp_dataset,
    ScoreModel,
    Trainer
)
from lamp import create_metric, create_prompt_generator
from llm import LLM


if os.getenv('HF_EVALUATE_OFFLINE') == '1':
    nltk.download = lambda *args, **kwargs: None


logging.getLogger('absl').setLevel(logging.WARNING)
logging.getLogger('httpx').setLevel(logging.WARNING)
logger = logging.getLogger(__name__)


@hydra.main(config_path='../conf', config_name='bandit_pr', version_base=None)
def main(cfg: DictConfig) -> None:
    # Check for missing keys
    missing_keys = OmegaConf.missing_keys(cfg)

    if missing_keys:
        raise ValueError(f'Missing keys in config:\n{missing_keys}')

    # Check config validity
    effective_batch_size = cfg.batch_size * cfg.gradient_accumulation_steps

    if cfg.eval_every % effective_batch_size != 0:
        raise ValueError(f'eval_every must be divisible by effective batch size')

    # Seed everything for reproducibility
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    torch.cuda.manual_seed_all(cfg.seed)

    # Prepare models
    score_model = ScoreModel(**cfg.score_model)

    if cfg.from_pretrained:
        score_model.from_pretrained(f'./models/{cfg.exp_name}')
        logger.info(f'Loaded model from {f"./models/{cfg.exp_name}"}')

    llm = LLM(cfg.task, **cfg.llm)

    # Prepare datasets
    test_split = ('dev' if cfg.task.startswith('LaMP') else 'test')
    train_dataset = load_retrieved_lamp_dataset(cfg.task, 'train', cfg.num_candidates)
    test_dataset = load_retrieved_lamp_dataset(cfg.task, test_split, cfg.num_candidates)

    tokenizer = AutoTokenizer.from_pretrained(cfg.score_model.encoder_model)
    preprocessor = create_preprocessor(tokenizer=tokenizer, **cfg.preprocessor)
    train_dataset = train_dataset.map(
        preprocessor, batched=True,
        remove_columns=['query', 'corpus'], num_proc=16
    )

    # Re-initialize tokenizer to ensure consistent hashing
    tokenizer = AutoTokenizer.from_pretrained(cfg.score_model.encoder_model)
    preprocessor = create_preprocessor(tokenizer=tokenizer, **cfg.preprocessor)
    test_dataset = test_dataset.map(
        preprocessor, batched=True,
        remove_columns=['query', 'corpus'], num_proc=16
    )

    collate_fn = create_collator(tokenizer)
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        drop_last=True
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=cfg.eval_batch_size,
        collate_fn=collate_fn
    )

    # Prepare LaMP components
    tokenizer = (
        AutoTokenizer.from_pretrained(cfg.llm.model)
        if cfg.llm.provider == 'local' else
        AutoTokenizer.from_pretrained('gpt2')
    )
    prompt_generator = create_prompt_generator(
        cfg.task, 'first_k', cfg.num_retrieve,
        cfg.prompt_generator.max_length, tokenizer
    )
    reward_fn = create_reward(cfg.task)
    metric_fn = create_metric(cfg.task)

    # Initialize trainer and start training
    trainer = Trainer(
        cfg, score_model, llm,
        train_loader, test_loader,
        prompt_generator, reward_fn, metric_fn,
        cfg.from_pretrained
    )
    trainer.train()


if __name__ == '__main__':
    main()
