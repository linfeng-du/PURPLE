import json
import logging

import wandb
from tqdm import tqdm
from omegaconf import DictConfig

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from bandit_pr import ScoreModel, reinforce, create_reward
from llm import LLM
from lamp import RetrieverTrainingDataset, RetrieverTrainingCollator, create_prompt_generator, create_metric


logger = logging.getLogger(__name__)
logging.getLogger('httpx').setLevel(logging.WARNING)


class RetrieverTrainer:

    def __init__(
        self,
        config: DictConfig,
        score_model: ScoreModel,
        llm: LLM,
        train_dataset: RetrieverTrainingDataset,
        test_dataset: RetrieverTrainingDataset,
        collate_fn: RetrieverTrainingCollator
    ) -> None:
        self.config = config
        self.score_model = score_model
        self.llm = llm

        self.wandb = wandb.init(project='BanditPR', dir='logs', name=f'{config.experiment}_{config.task}')
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.score_model.to(self.device)

        self.train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            collate_fn=collate_fn,
            drop_last=True
        )
        self.test_loader = DataLoader(test_dataset, batch_size=self.config.eval_batch_size, collate_fn=collate_fn)

        self.prompt_generator = create_prompt_generator(
            tokenizer=AutoTokenizer.from_pretrained('meta-llama/Meta-Llama-3-8B-Instruct'),
            **self.config.prompt_generator
        )
        self.reward_fn = create_reward(self.config.task)
        self.metric_fn = create_metric(self.config.task)
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
                    eval_metrics = self.evaluate()
                    self.score_model.train()
                    logger.info(
                        f'Evaluation metrics after {example_cnt} training examples:\n'
                        f'{json.dumps(eval_metrics, indent=2)}'
                    )

                    if eval_metrics['reward'] > best_eval_reward:
                        logger.info(f'Best evaluation reward achieved: {eval_metrics["reward"]}')
                        best_eval_reward = eval_metrics['reward']
                        self.score_model.save_pretrained('score_model')

                candidate_likelihoods, candidate_mask, candidate_indices = self.score_model(
                    batch['query_inputs'].to(self.device),
                    [document_inputs.to(self.device) for document_inputs in batch['corpus_inputs']],
                    batch['profile_mask'].to(self.device)
                )
                retrieved_indices, log_probs = reinforce.sample(
                    candidate_likelihoods,
                    candidate_mask,
                    **self.config.reinforce
                )

                prompts = []
                targets = []

                for batch_index, batch_retrieved_indices in enumerate(retrieved_indices):
                    for sample_retrieved_indices in batch_retrieved_indices:
                        profiles = [
                            batch['profile'][batch_index][candidate_indices[batch_index][retrieved_index]]
                            for retrieved_index in sample_retrieved_indices
                        ]
                        prompt = self.prompt_generator(batch['source'][batch_index], profiles)
                        target = batch['target'][batch_index]

                        prompts.append(prompt)
                        targets.append(target)

                predictions = self.llm(prompts)
                rewards = self.reward_fn(predictions, targets)
                rewards = rewards.to(self.device).view_as(log_probs)

                loss = reinforce.compute_loss(log_probs, rewards)
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
        predictions = []
        targets = []

        for batch in tqdm(self.test_loader, desc='Evaluating'):
            candidate_likelihoods, candidate_mask, candidate_indices = self.score_model(
                batch['query_inputs'].to(self.device),
                [document_inputs.to(self.device) for document_inputs in batch['corpus_inputs']],
                batch['profile_mask'].to(self.device)
            )
            num_retrieve = min(self.config.num_retrieve, candidate_likelihoods.size(dim=-1))
            _, retrieved_indices = candidate_likelihoods.topk(num_retrieve, dim=-1)

            prompts = []

            for batch_index, batch_retrieved_indices in enumerate(retrieved_indices):
                retrieved_profiles = [
                    batch['profile'][batch_index][candidate_indices[batch_index][retrieved_index]]
                    for retrieved_index in batch_retrieved_indices
                    if candidate_mask[batch_index][retrieved_index]
                ]
                prompt = self.prompt_generator(batch['source'][batch_index], retrieved_profiles)
                prompts.append(prompt)

            batch_predictions = self.llm(prompts)
            batch_targets = batch['target']
            predictions.extend(batch_predictions)
            targets.extend(batch_targets)

        rewards = self.reward_fn(predictions, targets)
        metrics = self.metric_fn(predictions, targets)
        metrics.update({'reward': rewards.mean().item()})
        return metrics
