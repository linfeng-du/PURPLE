import os
import json
import logging

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import wandb
from tqdm import tqdm
from omegaconf import DictConfig

from llm import LLM
from lamp.data_types import PromptGenerator, Metric

from . import reinforce
from .score_model import ScoreModel
from .data_types import Reward


logger = logging.getLogger(__name__)


class Trainer:

    def __init__(
        self,
        cfg: DictConfig,
        score_model: ScoreModel,
        llm: LLM,
        train_loader: DataLoader,
        test_loader: DataLoader,
        prompt_generator: PromptGenerator,
        reward_fn: Reward,
        metric_fn: Metric,
        from_pretrained: bool
    ) -> None:
        self.cfg = cfg
        self.score_model = score_model
        self.llm = llm
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.prompt_generator = prompt_generator
        self.reward_fn = reward_fn
        self.metric_fn = metric_fn

        # Trainer states
        self.example_cnt = 0
        self.best_eval_metric = None
        self.optimizer = torch.optim.Adam(
            [param for param in self.score_model.parameters() if param.requires_grad],
            lr=self.cfg.lr
        )

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.score_model.to(self.device)

        if from_pretrained:
            self._load_states(f'./models/{self.cfg.exp_name}')

        self.wandb = wandb.init(project='BanditPR', dir='logs', name=f'{self.cfg.exp_name}')

    def train(self) -> None:
        self.score_model.train()
        example_cnt = 0

        for epoch in range(self.cfg.num_epochs):
            for step, batch in enumerate(tqdm(self.train_loader, desc=f'Epoch {epoch}')):
                if (example_cnt > 0) and (example_cnt % self.cfg.eval_every == 0):
                    eval_results = self.evaluate()
                    self.score_model.train()

                    metric = (
                        'accuracy' if 'accuracy' in eval_results else
                        'mae' if 'mae' in eval_results else 'rouge-1'
                    )
                    eval_metric = eval_results[metric]
                    self.wandb.log({'eval_metric': eval_metric})
                    logger.info(
                        f'Evaluation results after {example_cnt} training examples:\n'
                        f'{json.dumps(eval_results, indent=2)}'
                    )

                    if (
                        (not self.best_eval_metric)
                        or (metric == 'mae' and eval_metric < self.best_eval_metric)
                        or (eval_metric > self.best_eval_metric)
                    ):
                        logger.info(f'Best evaluation metric achieved: {eval_metric}')
                        self.best_eval_metric = eval_metric
                        self._save_states()

                likelihoods = self.score_model(
                    batch['query_inputs'].to(self.device),
                    [
                        [document_inputs.to(self.device) for document_inputs in document_subbatches]
                        for document_subbatches in batch['corpus_inputs']
                    ],
                    batch['profile_mask'].to(self.device)
                )
                retrieved_indices, logps = reinforce.sample(
                    likelihoods,
                    batch['profile_mask'].to(self.device),
                    self.cfg.reinforce.num_samples,
                    self.cfg.num_retrieve
                )

                prompts = []
                targets = []

                for batch_index, batch_retrieved_indices in enumerate(retrieved_indices):
                    for sample_retrieved_indices in batch_retrieved_indices:
                        profiles = [
                            batch['profiles'][batch_index][retrieved_index]
                            for retrieved_index in sample_retrieved_indices
                        ]
                        prompt = self.prompt_generator(batch['source'][batch_index], profiles)
                        target = batch['target'][batch_index]

                        prompts.append(prompt)
                        targets.append(target)

                if self.cfg.reinforce.reward == 'metric':
                    responses = self.llm.generate(prompts)
                    rewards = self.reward_fn(responses, targets)
                elif self.cfg.reinforce.reward == 'logp':
                    rewards = self.llm.compute_target_logps(prompts, targets)

                rewards = rewards.to(self.device).view_as(logps)

                loss = reinforce.compute_loss(logps, rewards, self.cfg.reinforce.loss)
                loss = loss / self.cfg.gradient_accumulation_steps
                loss.backward()

                if (step + 1) % self.cfg.gradient_accumulation_steps == 0:
                    nn.utils.clip_grad_norm_(self.score_model.parameters(), self.cfg.max_grad_norm)
                    self.optimizer.step()
                    self.optimizer.zero_grad()

                example_cnt += len(batch['source'])
                self.wandb.log({'reward': rewards.mean().item(), 'loss': loss.item()})

        self.wandb.finish()

    @torch.no_grad()
    def evaluate(self) -> dict[str, float]:
        self.score_model.eval()
        prompts = []
        targets = []

        for batch in tqdm(self.test_loader, desc='Evaluating'):
            likelihoods = self.score_model(
                batch['query_inputs'].to(self.device),
                [
                    [document_inputs.to(self.device) for document_inputs in document_subbatches]
                    for document_subbatches in batch['corpus_inputs']
                ],
                batch['profile_mask'].to(self.device)
            )
            num_retrieve = min(self.cfg.num_retrieve, likelihoods.shape[1])
            _, retrieved_indices = likelihoods.topk(num_retrieve, dim=1)

            for batch_index, batch_retrieved_indices in enumerate(retrieved_indices):
                retrieved_profiles = [
                    batch['profiles'][batch_index][retrieved_index]
                    for retrieved_index in batch_retrieved_indices
                    if batch['profile_mask'][batch_index][retrieved_index]
                ]
                prompt = self.prompt_generator(batch['source'][batch_index], retrieved_profiles)
                target = batch['target'][batch_index]

                prompts.append(prompt)
                targets.append(target)

        predictions = self.llm.generate(prompts, verbose=True)
        return self.metric_fn(predictions, targets)

    def _load_states(self, ckpt_dir: str) -> None:
        ckpt = torch.load(f'{ckpt_dir}/trainer.pt', map_location=self.device, weights_only=False)
        self.example_cnt = ckpt['example_cnt']
        self.best_eval_metric = ckpt['best_eval_metric']
        self.optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        logger.info(f'Loaded trainer states from {ckpt_dir}')

    def _save_states(self) -> None:
        ckpt_dir = f'./models/{self.cfg.exp_name}'
        os.makedirs(f'./{ckpt_dir}', exist_ok=True)
        self.score_model.save_pretrained(ckpt_dir)
        torch.save({
            'example_cnt': self.example_cnt,
            'best_eval_metric': self.best_eval_metric,
            'optimizer_state_dict': self.optimizer.state_dict()
        }, f'{ckpt_dir}/trainer.pt')
