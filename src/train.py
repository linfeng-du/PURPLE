import random
import logging

import numpy as np

import torch
from transformers import AutoTokenizer

import hydra
from omegaconf import OmegaConf, DictConfig

from llm import LLM
from bandit_pr import ScoreModel, Trainer, create_preprocessor, create_collator, create_reward
from lamp import load_lamp_dataset, create_prompt_generator, create_metric


logging.getLogger('absl').setLevel(logging.WARNING)
logging.getLogger('httpx').setLevel(logging.WARNING)
logger = logging.getLogger(__name__)


@hydra.main(config_path='../conf', config_name='bandit_pr', version_base=None)
def main(config: DictConfig):
    # Checks config validity
    missing_keys = OmegaConf.missing_keys(config)

    if missing_keys:
        raise ValueError(f'Missing keys in config:\n{missing_keys}')

    effective_batch_size = config.batch_size * config.gradient_accumulation_steps

    if config.eval_every % effective_batch_size != 0:
        config.eval_every = config.eval_every - config.eval_every % effective_batch_size
        logger.warning(f'eval_every changed to {config.eval_every} to be divisible by effective batch size')

    # Seeds everything for reproducibility
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)

    # Prepares models
    score_model = ScoreModel(**config.score_model)

    if config.from_pretrained is not None:
        score_model.from_pretrained(config.from_pretrained)

    llm = LLM(config.task, **config.llm)

    # Prepares datasets
    tokenizer = AutoTokenizer.from_pretrained(config.score_model.encoder_model)
    preprocessor = create_preprocessor(tokenizer=tokenizer, **config.preprocessor)
    train_dataset = load_lamp_dataset(config.task, split='train').map(
        preprocessor,
        batched=True,
        remove_columns=['query', 'corpus'],
        num_proc=16
    )

    # Re-initializes tokenizer to ensure consistent hashing
    tokenizer = AutoTokenizer.from_pretrained(config.score_model.encoder_model)
    preprocessor = create_preprocessor(tokenizer=tokenizer, **config.preprocessor)
    test_dataset = load_lamp_dataset(config.task, split='dev').map(
        preprocessor,
        batched=True,
        remove_columns=['query', 'corpus'],
        num_proc=16
    )

    collate_fn = create_collator(tokenizer)

    # Prepares LaMP components
    tokenizer = (
        AutoTokenizer.from_pretrained(config.llm.model)
        if config.llm.provider == 'local'
        else AutoTokenizer.from_pretrained('gpt2')
    )
    prompt_generator = create_prompt_generator(
        config.task,
        'first_k',
        config.num_retrieve,
        config.prompt_generator.max_length,
        tokenizer
    )
    reward_fn = create_reward(config.task)
    metric_fn = create_metric(config.task)

    # Initializes trainer and starts training
    trainer = Trainer(
        config,
        score_model, llm,
        train_dataset, test_dataset, collate_fn,
        prompt_generator, reward_fn, metric_fn
    )
    trainer.train()


if __name__ == '__main__':
    main()
