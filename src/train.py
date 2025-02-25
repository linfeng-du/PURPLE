import hydra
from omegaconf import DictConfig, OmegaConf

import torch

from rl import ProfileScoreModel
from trainer import RetrieverTrainer
from openai_api import initialize_openai_client


@hydra.main(version_base=None, config_path='../conf', config_name='train_config')
def train(config: DictConfig):
    # Check config validity
    missing_keys = OmegaConf.missing_keys(config)

    if missing_keys:
        raise ValueError(f'Missing keys in config:\n{missing_keys}')

    if config.eval_every % config.batch_size != 0:
        raise ValueError(f'eval_every must be divisble by batch_size')

    # Seed everything for reproducibility
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)

    # Detect device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Prepare score model for retriever training
    score_model = ProfileScoreModel(**config.score_model)
    score_model.to(device)

    for param in score_model.bert_encoder.parameters():
        param.requires_grad = False

    # Initialize OpenAI client
    response_generator = initialize_openai_client(**config.generation)

    # Initialize trainer and start training
    trainer = RetrieverTrainer(config, score_model, response_generator, device)
    trainer.train()


if __name__ == '__main__':
    train()
