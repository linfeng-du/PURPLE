import random

import numpy as np
import torch
from transformers import AutoTokenizer

import hydra
from omegaconf import OmegaConf, DictConfig

from bandit_pr import ScoreModel
from llm import LLM
from lamp import RetrieverTrainingDataset, RetrieverTrainingCollator
from trainer import RetrieverTrainer


@hydra.main(config_path='../conf', config_name='bandit_pr', version_base=None)
def train(config: DictConfig):
    # Checks config validity
    missing_keys = OmegaConf.missing_keys(config)

    if missing_keys:
        raise ValueError(f'Missing keys in config:\n{missing_keys}')

    if config.eval_every % config.batch_size != 0:
        raise ValueError(f'eval_every ({config.eval_every}) not divisble by batch_size ({config.batch_size})')

    # Seeds everything for reproducibility
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)

    # Prepares models
    score_model = ScoreModel(**config.score_model)
    llm = LLM(**config.llm)

    # Prepares dataset and metric
    train_dataset = RetrieverTrainingDataset(config.task, split='train')
    test_dataset = RetrieverTrainingDataset(config.task, split='dev')
    collate_fn = RetrieverTrainingCollator(
        tokenizer=AutoTokenizer.from_pretrained(config.score_model.encoder_model),
        **config.collator
    )

    # Initializes trainer and starts training
    trainer = RetrieverTrainer(config, score_model, llm, train_dataset, test_dataset, collate_fn)
    trainer.train()


if __name__ == '__main__':
    train()
