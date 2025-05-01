import json
import random
import logging

import numpy as np
import torch
from transformers import AutoTokenizer

import hydra
from omegaconf import OmegaConf, DictConfig
from tqdm import tqdm

from llm import LLM
from lamp import LaMPDataset, create_prompt_generator, create_metric


logger = logging.getLogger(__name__)
logging.getLogger('httpx').setLevel(logging.WARNING)


@hydra.main(config_path='../conf', config_name='baseline', version_base=None)
def baseline(config: DictConfig):
    # Checks for missing keys
    missing_keys = OmegaConf.missing_keys(config)

    if missing_keys:
        raise ValueError(f'Missing keys in config:\n{missing_keys}')

    # Seeds everything for reproducibility
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed(config.seed)

    # Prepares dataset
    prompt_generator = create_prompt_generator(
        tokenizer=AutoTokenizer.from_pretrained('meta-llama/Meta-Llama-3-8B-Instruct'),
        device=('cuda' if torch.cuda.is_available() else 'cpu'),
        **config.prompt_generator
    )
    test_dataset = LaMPDataset(config.task, split='dev', prompt_generator=prompt_generator)

    # Collects sources and targets
    sources = []
    targets = []

    for example in tqdm(test_dataset, desc='Generating Prompt'):
        sources.append(example['source'])
        targets.append(example['target'])

    # Generates predictions
    llm = LLM(verbose=True, **config.llm)
    predictions = llm(sources)

    # Computes metrics
    metric_fn = create_metric(config.task)
    test_results = metric_fn(predictions, targets)
    logger.info(f'Test set results:\n{json.dumps(test_results, indent=2)}')


if __name__ == '__main__':
    baseline()
