import os
import json
import random
import logging

import nltk
import numpy as np

import torch
from transformers import AutoTokenizer

import hydra
from omegaconf import OmegaConf, DictConfig
from tqdm import tqdm

from llm import LLM
from lamp import load_lamp_dataset, create_prompt_generator, create_metric


if os.getenv('HF_EVALUATE_OFFLINE') == '1':
    nltk.download = lambda *args, **kwargs: None


logging.getLogger('absl').setLevel(logging.WARNING)
logging.getLogger('httpx').setLevel(logging.WARNING)
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
    tokenizer = (
        AutoTokenizer.from_pretrained(cfg.llm.model)
        if cfg.llm.provider == 'local'
        else AutoTokenizer.from_pretrained('gpt2')
    )
    prompt_generator = create_prompt_generator(
        cfg.task, cfg.retriever, cfg.num_retrieve,
        cfg.prompt_generator.max_length, tokenizer,
        device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    )

    test_split = 'dev' if cfg.task.startswith('LaMP') else 'test'
    test_dataset = load_lamp_dataset(cfg.task, test_split)

    # Collect sources and targets
    sources = []
    targets = []

    for example in tqdm(test_dataset, desc='Generating Prompts'):
        source = prompt_generator(example['source'], example['profiles'], example['query'], example['corpus'])
        target = example['target']
        sources.append(source)
        targets.append(target)

    # Generate predictions
    llm = LLM(cfg.task, **cfg.llm)
    predictions = llm.generate(sources, verbose=True)

    # Compute metrics
    metric_fn = create_metric(cfg.task)
    test_results = metric_fn(predictions, targets)
    logger.info(f'Evaluation results:\n{json.dumps(test_results, indent=2)}')


if __name__ == '__main__':
    main()
