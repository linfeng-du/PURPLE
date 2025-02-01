import json
import random
import logging

import hydra
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

import torch
from transformers import AutoTokenizer

from LaMP import LaMPDataset, create_retrieval_prompt_generator, create_metric_function
from openai_api import request_completions


logger = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path='../conf', config_name='baseline_config')
def baseline(config: DictConfig):
    # Check config validity
    missing_keys = OmegaConf.missing_keys(config)

    if missing_keys:
        raise ValueError(f'Missing keys in config:\n{missing_keys}')

    # Seed everything for reproducibility
    random.seed(config.seed)

    # Detect device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # The tokenizer is used solely to control the tokenized length of prompts
    prompt_generator = create_retrieval_prompt_generator(
        tokenizer=AutoTokenizer.from_pretrained('gpt2'),
        device=device,
        **config.prompt_generator
    )

    test_dataset = LaMPDataset(
        data_path=f'./dataset/{config.task}/dev_questions.json',
        label_path=f'./dataset/{config.task}/dev_outputs.json',
        prompt_generator=prompt_generator
    )
    metric_fn = create_metric_function(config.task)

    sources = []
    targets = []

    for example in tqdm(test_dataset, desc='Generating Prompt'):
        sources.append(example['source'])
        targets.append(example['target'])

    predictions = request_completions(sources, **config.generation)

    test_results = metric_fn(predictions, targets)
    logger.info(f'Test set results:\n{json.dumps(test_results, indent=4)}')


if __name__ == '__main__':
    baseline()
