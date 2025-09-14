import json
import logging
import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import wandb
from omegaconf import DictConfig
from tqdm import tqdm

from llm import LLM
from lamp.data_types import Metric, PromptGenerator

from . import reinforce
from .data_types import Batch, Reward
from .score_model import ScoreModel


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
        self.epoch = 0
        self.example_cnt = 0
        self.best_eval_result = None
        self.optimizer = torch.optim.Adam(
            [param for param in self.score_model.parameters() if param.requires_grad],
            lr=self.cfg.lr
        )

        self.device = torch.device(('cuda' if torch.cuda.is_available() else 'cpu'))
        self.score_model.to(self.device)

        if from_pretrained:
            self._load_states(f'./models/{self.cfg.exp_name}')
            logger.info(f'Loaded trainer states from {f"./models/{self.cfg.exp_name}"}')

        self.wandb = wandb.init(project='BanditPR', dir='logs', name=f'{self.cfg.exp_name}')

    def train(self) -> None:
        self.score_model.train()
        start_flag = True

        for _ in range(self.cfg.num_epochs):
            for step, batch in enumerate(tqdm(self.train_loader, desc=f'Epoch {self.epoch}')):
                if (not start_flag) and (self.example_cnt % self.cfg.eval_every == 0):
                    eval_results = self.evaluate()
                    self.score_model.train()

                    metric = (
                        'accuracy' if 'accuracy' in eval_results else
                        'mae' if 'mae' in eval_results else 'rouge-1'
                    )
                    eval_result = eval_results[metric]
                    self.wandb.log({f'eval_{metric}': eval_result})
                    logger.info(
                        f'Evaluation results after {self.example_cnt} training examples:\n'
                        f'{json.dumps(eval_results, indent=2)}'
                    )

                    if (
                        (not self.best_eval_result)
                        or (metric == 'mae' and eval_result < self.best_eval_result)
                        or (metric in {'accuracy', 'rouge-1'} and eval_result > self.best_eval_result)
                    ):
                        logger.info(f'Best evaluation {metric} achieved: {eval_result}')
                        self.best_eval_result = eval_result
                        self._save_states()

                batch = self._move_to_device(batch)
                likelihoods = self.score_model(
                    batch['query_inputs'],
                    batch['corpus_inputs'],
                    batch['profile_mask']
                )
                reranked_indices, logps = reinforce.sample(
                    likelihoods,
                    self.cfg.reinforce.num_samples,
                    self.cfg.num_rerank
                )

                prompts = []
                targets = []

                for batch_index, batch_reranked_indices in enumerate(reranked_indices):
                    for sample_reranked_indices in batch_reranked_indices:
                        profiles = [
                            batch['profiles'][batch_index][reranked_index]
                            for reranked_index in sample_reranked_indices
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
                else:
                    raise ValueError(f'Invalid reward: {self.cfg.reinforce.reward}')

                rewards = rewards.to(self.device).view_as(logps)
                loss = reinforce.compute_loss(logps, rewards)
                loss = loss / self.cfg.gradient_accumulation_steps
                loss.backward()

                if (step + 1) % self.cfg.gradient_accumulation_steps == 0:
                    nn.utils.clip_grad_norm_(self.score_model.parameters(), self.cfg.max_grad_norm)
                    self.optimizer.step()
                    self.optimizer.zero_grad()

                start_flag = False
                self.example_cnt += len(batch['source'])
                self.wandb.log({'reward': rewards.mean().item(), 'loss': loss.item()})

            self.epoch += 1

        self.wandb.finish()

    @torch.no_grad()
    def evaluate(self) -> dict[str, float]:
        self.score_model.eval()

        prompts = []
        targets = []

        for batch in tqdm(self.test_loader, desc='Evaluating'):
            batch = self._move_to_device(batch)
            likelihoods = self.score_model(
                batch['query_inputs'],
                batch['corpus_inputs'],
                batch['profile_mask']
            )
            num_rerank = min(self.cfg.num_rerank, likelihoods.shape[1])
            _, reranked_indices = likelihoods.topk(num_rerank, dim=1)

            for batch_index, batch_reranked_indices in enumerate(reranked_indices):
                reranked_profiles = [
                    batch['profiles'][batch_index][reranked_index]
                    for reranked_index in batch_reranked_indices
                    if batch['profile_mask'][batch_index][reranked_index]
                ]
                prompt = self.prompt_generator(batch['source'][batch_index], reranked_profiles)
                target = batch['target'][batch_index]

                prompts.append(prompt)
                targets.append(target)

        predictions = self.llm.generate(prompts, verbose=True)
        return self.metric_fn(predictions, targets)

    def _move_to_device(self, batch: Batch) -> Batch:
        batch['query_inputs'] = batch['query_inputs'].to(self.device)
        batch['corpus_inputs'] = [
            [document_inputs.to(self.device) for document_inputs in document_subbatches]
            for document_subbatches in batch['corpus_inputs']
        ]
        batch['profile_mask'] = batch['profile_mask'].to(self.device)
        return batch

    def _load_states(self, ckpt_dir: str) -> None:
        ckpt = torch.load(f'{ckpt_dir}/trainer.pt', map_location=self.device, weights_only=False)
        self.epoch = ckpt['epoch'] + 1
        self.example_cnt = ckpt['example_cnt']
        self.best_eval_result = ckpt['best_eval_result']
        self.optimizer.load_state_dict(ckpt['optimizer_state_dict'])

    def _save_states(self) -> None:
        ckpt_dir = f'./models/{self.cfg.exp_name}'
        os.makedirs(f'./{ckpt_dir}', exist_ok=True)
        self.score_model.save_pretrained(ckpt_dir)
        torch.save({
            'epoch': self.epoch,
            'example_cnt': self.example_cnt,
            'best_eval_result': self.best_eval_result,
            'optimizer_state_dict': self.optimizer.state_dict()
        }, f'{ckpt_dir}/trainer.pt')
