import json
import logging

from datasets import Dataset

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
from .data_types import Collator, Reward


logger = logging.getLogger(__name__)


class Trainer:

    def __init__(
        self,
        config: DictConfig,
        score_model: ScoreModel,
        llm: LLM,
        train_dataset: Dataset,
        test_dataset: Dataset,
        collate_fn: Collator,
        prompt_generator: PromptGenerator,
        reward_fn: Reward,
        metric_fn: Metric
    ) -> None:
        self.config = config
        self.score_model = score_model
        self.llm = llm
        self.prompt_generator = prompt_generator
        self.reward_fn = reward_fn
        self.metric_fn = metric_fn

        self.wandb = wandb.init(project='BanditPR', dir='logs', name=f'{config.experiment}')
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.score_model.to(self.device)

        self.train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            collate_fn=collate_fn,
            drop_last=True
        )
        self.test_loader = DataLoader(
            test_dataset,
            batch_size=self.config.eval_batch_size,
            collate_fn=collate_fn
        )
        self.optimizer = torch.optim.Adam(
            [param for param in self.score_model.parameters() if param.requires_grad],
            lr=self.config.lr
        )

    def train(self) -> None:
        self.score_model.train()
        example_cnt = 0
        best_eval_reward = float('-inf')

        for epoch in range(self.config.num_epochs):
            for step, batch in enumerate(tqdm(self.train_loader, desc=f'Epoch {epoch}')):
                if (example_cnt > 0) and (example_cnt % self.config.eval_every == 0):
                    eval_results = self.evaluate()
                    self.score_model.train()
                    self.wandb.log({'eval_reward': eval_results['reward']})
                    logger.info(
                        f'Evaluation results after {example_cnt} training examples:\n'
                        f'{json.dumps(eval_results, indent=2)}'
                    )

                    if eval_results['reward'] > best_eval_reward:
                        logger.info(f'Best evaluation reward achieved: {eval_results["reward"]}')
                        best_eval_reward = eval_results['reward']
                        self.score_model.save_pretrained(f'{self.config.experiment}')

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
                    self.config.reinforce.num_samples,
                    self.config.num_retrieve,
                    self.config.reinforce.epsilon
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

                if self.config.reinforce.reward == 'metric':
                    responses = self.llm.generate(prompts)
                    rewards = self.reward_fn(responses, targets)
                if self.config.reinforce.reward == 'logp':
                    rewards = self.llm.compute_target_logps(prompts, targets)

                rewards = rewards.to(self.device).view_as(logps)

                loss = reinforce.compute_loss(logps, rewards, self.config.reinforce.loss)
                loss = loss / self.config.gradient_accumulation_steps
                loss.backward()

                if (step + 1) % self.config.gradient_accumulation_steps == 0:
                    nn.utils.clip_grad_norm_(self.score_model.parameters(), self.config.max_grad_norm)
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
            num_retrieve = min(self.config.num_retrieve, likelihoods.shape[1])
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
        rewards = self.reward_fn(predictions, targets)
        results = self.metric_fn(predictions, targets)
        results.update({'reward': rewards.mean().item()})
        return results
