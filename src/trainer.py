import os
import json
import logging
from typing import Callable

from tqdm import tqdm
from omegaconf import DictConfig

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from transformers import AutoTokenizer

from LaMP import (
    RetrieverTrainingDataset,
    RetrieverTrainingCollator,
    create_metric_function,
    create_query_corpus_generator,
    create_retrieval_prompt_generator
)
from rl import ProfileScoreModel, Reinforce, create_reward_function


logger = logging.getLogger(__name__)
logging.getLogger('httpx').setLevel(logging.WARNING)


class RetrieverTrainer:

    def __init__(
        self,
        config: DictConfig,
        score_model: ProfileScoreModel,
        response_generator: Callable[[list[str]], list[str]],
        device: torch.device
    ) -> None:
        self.config = config
        self.score_model = score_model
        self.response_generator = response_generator
        self.device = device

        self.reinforce = Reinforce()
        self.reward_fn = create_reward_function(self.config.task)
        self.metric_fn = create_metric_function(self.config.task)

        # The tokenizer is used solely to control the tokenized length of prompts
        self.prompt_generator = create_retrieval_prompt_generator(
            tokenizer=AutoTokenizer.from_pretrained('gpt2'),
            **self.config.prompt_generator
        )

        self.train_loader, self.val_loader, self.test_loader = self._load_dataset_splits()
        self.optimizer = torch.optim.Adam(self.score_model.parameters(), lr=self.config.lr)

    def _load_dataset_splits(self) -> (
        tuple[DataLoader, DataLoader, DataLoader]
    ):
        train_val_dataset = RetrieverTrainingDataset(
            data_path=f'./dataset/{self.config.task}/train_questions.json',
            label_path=f'./dataset/{self.config.task}/train_outputs.json',
            query_corpus_generator=create_query_corpus_generator(self.config.task)
        )
        train_dataset, val_dataset = random_split(
            train_val_dataset,
            lengths=[0.8, 0.2],
            generator=torch.Generator().manual_seed(self.config.dataset_seed)
        )
        test_dataset = RetrieverTrainingDataset(
            data_path=f'./dataset/{self.config.task}/dev_questions.json',
            label_path=f'./dataset/{self.config.task}/dev_outputs.json',
            query_corpus_generator=create_query_corpus_generator(self.config.task)
        )

        collate_fn = RetrieverTrainingCollator(
            AutoTokenizer.from_pretrained(self.config.score_model.bert_encoder),
            **self.config.collator
        )

        loader_kwargs = {'batch_size': self.config.batch_size, 'collate_fn': collate_fn}
        train_loader = DataLoader(train_dataset, shuffle=True, **loader_kwargs)

        loader_kwargs.update({'batch_size': self.config.eval_batch_size})
        val_loader = DataLoader(val_dataset, **loader_kwargs)
        test_loader = DataLoader(test_dataset, **loader_kwargs)

        return train_loader, val_loader, test_loader

    def train(self) -> None:
        example_cnt = 0
        best_val_avg_reward = 0

        for epoch in range(self.config.n_epochs):
            for batch in tqdm(self.train_loader, desc=f'Epoch {epoch}'):
                if example_cnt > 0 and example_cnt % self.config.eval_every == 0:
                    val_avg_reward = self.validate()
                    logger.info(
                        f'Average validation reward after {example_cnt} '
                        f'training examples: {val_avg_reward}'
                    )

                    if val_avg_reward > best_val_avg_reward:
                        logger.info(f'Best validation average reward: {val_avg_reward}')
                        best_val_avg_reward = val_avg_reward
                        model_path = os.path.join(self.config.run_dir, 'model.pt')
                        torch.save(self.score_model.state_dict(), model_path)

                self.score_model.train()

                sources = batch['source']
                profiles = batch['profile']
                query_inputs = batch['query_inputs'].to(self.device)
                all_corpus_inputs = [
                    corpus_inputs.to(self.device)
                    for corpus_inputs in batch['all_corpus_inputs']
                ]
                profile_mask = batch['profile_mask'].to(self.device)
                targets = batch['target']

                candidate_likelihoods, candidate_mask, candidate_indices = (
                    self.score_model(query_inputs, all_corpus_inputs, profile_mask)
                )
                retrieved_indices, log_probs = self.reinforce.sample(
                    candidate_likelihoods,
                    candidate_mask,
                    **self.config.reinforce
                )

                sample_prompts = []
                sample_targets = []

                for batch_index, batch_retrieved_indices in enumerate(retrieved_indices):
                    for sample_retrieved_indices in batch_retrieved_indices:
                        sample_profiles = [
                            profiles[batch_index][candidate_indices[batch_index][retrieved_index]]
                            for retrieved_index in sample_retrieved_indices
                        ]
                        sample_prompt = self.prompt_generator(sources[batch_index], sample_profiles)
                        sample_target = targets[batch_index]

                        sample_prompts.append(sample_prompt)
                        sample_targets.append(sample_target)

                sample_predictions = self.response_generator(sample_prompts)

                rewards = self.reward_fn(sample_predictions, sample_targets)
                rewards = torch.tensor(rewards, device=self.device).view_as(log_probs)

                loss = self.reinforce.compute_loss(log_probs, rewards)
                loss.backward()
                nn.utils.clip_grad_norm_(self.score_model.parameters(), self.config.max_grad_norm)

                self.optimizer.step()
                self.optimizer.zero_grad()

                example_cnt += len(sources)

        model_path = os.path.join(self.config.run_dir, 'model.pt')

        if os.path.exists(model_path):
            model_state = torch.load(model_path, map_location=self.device, weights_only=True)
            self.score_model.load_state_dict(model_state)

        test_results = self.test()
        logger.info(f'Test set results:\n{json.dumps(test_results, indent=4)}')

    @torch.no_grad()
    def validate(self) -> float:
        self.score_model.eval()

        all_rewards = []

        for batch in tqdm(self.val_loader, desc='Validating'):
            sources = batch['source']
            profiles = batch['profile']
            query_inputs = batch['query_inputs'].to(self.device)
            all_corpus_inputs = [
                corpus_inputs.to(self.device)
                for corpus_inputs in batch['all_corpus_inputs']
            ]
            profile_mask = batch['profile_mask'].to(self.device)
            targets = batch['target']

            candidate_likelihoods, candidate_mask, candidate_indices = (
                self.score_model(query_inputs, all_corpus_inputs, profile_mask)
            )
            n_retrieve = min(self.config.n_retrieve, candidate_likelihoods.size(dim=1))
            _, retrieved_indices = candidate_likelihoods.topk(n_retrieve, dim=-1)

            prompts = []

            for batch_index, batch_retrieved_indices in enumerate(retrieved_indices):
                retrieved_profiles = [
                    profiles[batch_index][candidate_indices[batch_index][retrieved_index]]
                    for retrieved_index in batch_retrieved_indices
                    if candidate_mask[batch_index][retrieved_index]
                ]
                prompt = self.prompt_generator(sources[batch_index], retrieved_profiles)
                prompts.append(prompt)

            predictions = self.response_generator(prompts)

            rewards = self.reward_fn(predictions, targets)
            all_rewards.extend(rewards)

        avg_reward = sum(all_rewards) / len(all_rewards)
        return avg_reward

    @torch.no_grad()
    def test(self) -> dict[str, float]:
        self.score_model.eval()

        all_predictions = []
        all_targets = []

        for batch in tqdm(self.test_loader, desc='Testing'):
            sources = batch['source']
            profiles = batch['profile']
            query_inputs = batch['query_inputs'].to(self.device)
            all_corpus_inputs = [
                corpus_inputs.to(self.device)
                for corpus_inputs in batch['all_corpus_inputs']
            ]
            profile_mask = batch['profile_mask'].to(self.device)
            targets = batch['target']

            candidate_likelihoods, candidate_mask, candidate_indices = (
                self.score_model(query_inputs, all_corpus_inputs, profile_mask)
            )
            n_retrieve = min(self.config.n_retrieve, candidate_likelihoods.size(dim=1))
            _, retrieved_indices = candidate_likelihoods.topk(n_retrieve, dim=-1)

            prompts = []

            for batch_index, batch_retrieved_indices in enumerate(retrieved_indices):
                retrieved_profiles = [
                    profiles[batch_index][candidate_indices[batch_index][retrieved_index]]
                    for retrieved_index in batch_retrieved_indices
                    if candidate_mask[batch_index][retrieved_index]
                ]
                prompt = self.prompt_generator(sources[batch_index], retrieved_profiles)
                prompts.append(prompt)

            predictions = self.response_generator(prompts)

            all_predictions.extend(predictions)
            all_targets.extend(targets)

        return self.metric_fn(all_predictions, all_targets)
