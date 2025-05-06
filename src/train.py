import random
import logging

import numpy as np
import torch
from transformers import AutoTokenizer

import hydra
from omegaconf import OmegaConf, DictConfig

from bandit_pr import ScoreModel
from llm import LLM
from lamp import RetrieverTrainingDataset, RetrieverTrainingCollator
from trainer import RetrieverTrainer


logging.getLogger('absl').setLevel(logging.WARNING)
logging.getLogger('httpx').setLevel(logging.WARNING)
logger = logging.getLogger(__name__)


@hydra.main(config_path='../conf', config_name='bandit_pr', version_base=None)
def main(config: DictConfig):
    # Checks config validity
    missing_keys = OmegaConf.missing_keys(config)

    if missing_keys:
        raise ValueError(f'Missing keys in config:\n{missing_keys}')

    if config.eval_every % config.batch_size != 0:
        config.eval_every = config.eval_every - config.eval_every % config.batch_size
        logger.warning(f'eval_every changed to {config.eval_every} to be divisible by batch size')

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

    # Prepares dataset and metric
    train_dataset = RetrieverTrainingDataset(config.task, 'train')
    test_dataset = RetrieverTrainingDataset(config.task, 'dev')
    collate_fn = RetrieverTrainingCollator(
        tokenizer=AutoTokenizer.from_pretrained(config.score_model.encoder_model),
        **config.collator
    )

    # Initializes trainer and starts training
    trainer = RetrieverTrainer(config, score_model, llm, train_dataset, test_dataset, collate_fn)
    trainer.train()


if __name__ == '__main__':
    main()
