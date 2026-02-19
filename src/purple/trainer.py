import json
import logging
from pathlib import Path

import wandb
from omegaconf import DictConfig
from tqdm import tqdm

from datasets import Dataset

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import BatchEncoding

from lamp import ChatPromptFn, MetricFn, PromptFn
from llm import HFLLM, VLLMClient

from . import rl
from .dataset import CollateFn, LaMPBatch
from .reward import RewardFn
from .score_model import ScoreModel


logger = logging.getLogger(__name__)


class Trainer:
    def __init__(
        self,
        score_model: ScoreModel,
        llm: HFLLM | VLLMClient,
        args: DictConfig,
        collate_fn: CollateFn,
        train_dataset: Dataset,
        test_dataset: Dataset,
        prompt_fn: PromptFn,
        chat_prompt_fn: ChatPromptFn,
        reward_fn: RewardFn,
        metric_fn: MetricFn
    ) -> None:
        if args.eval_steps % args.train_batch_size != 0:
            raise ValueError(f"eval_steps not divisible by train_batch_size")

        self.args = args
        self.score_model = score_model
        self.llm = llm

        self.prompt_fn = prompt_fn
        self.chat_prompt_fn = chat_prompt_fn
        self.reward_fn = reward_fn
        self.metric_fn = metric_fn

        self.train_loader = DataLoader(
            train_dataset,
            batch_size=args.train_batch_size,
            shuffle=True,
            collate_fn=collate_fn,
            drop_last=True
        )
        self.test_loader = DataLoader(
            test_dataset,
            batch_size=args.eval_batch_size,
            collate_fn=collate_fn
        )

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.score_model.to(self.device)

        # Trainer states
        self.epoch = 0
        self.examples_seen = 0
        self.best_eval_result = None
        self.optimizer = torch.optim.Adam(
            [
                param
                for param in self.score_model.parameters()
                if param.requires_grad
            ],
            lr=self.args.learning_rate
        )

        if self.args.resume_from_checkpoint:
            self._load_states()

        self.wandb = wandb.init(
            project="PURPLE",
            dir="outputs",
            name=self.args.run_name
        )

    def train(self) -> None:
        self.score_model.train()

        for _ in range(self.args.num_train_epochs):
            for batch in tqdm(self.train_loader, desc=f"Epoch {self.epoch}"):
                batch = move_to_device(batch, self.device)

                if self.args.loss_type == "reinforce":
                    self._train_reinforce(batch)
                elif self.args.loss_type == "grpo":
                    self._train_grpo(batch)
                else:
                    raise ValueError(
                        f"Invalid loss type: {self.args.loss_type}"
                    )

                self.examples_seen += len(batch["source"])

                if self.examples_seen % self.args.eval_steps == 0:
                    eval_results = self.evaluate()
                    self.score_model.train()

                    if "accuracy" in eval_results:
                        metric = "accuracy"
                        higher_is_better = True
                    elif "mae" in eval_results:
                        metric = "mae"
                        higher_is_better = False
                    elif "rouge-1" in eval_results:
                        metric = "rouge-1"
                        higher_is_better = True

                    eval_result = eval_results[metric]
                    self.wandb.log({f"eval/{metric}": eval_result})
                    logger.info(
                        "Evaluation results "
                        f"after {self.examples_seen} training examples:\n"
                        f"{json.dumps(eval_results, indent=2)}"
                    )

                    if self.best_eval_result is None:
                        is_best = True
                    elif higher_is_better:
                        is_best = eval_result > self.best_eval_result
                    else:
                        is_best = eval_result < self.best_eval_result

                    if is_best:
                        logger.info(f"New best eval {metric}: {eval_result}")
                        self.best_eval_result = eval_result
                        self._save_states()

            self.epoch += 1

        self.wandb.finish()

    def _train_reinforce(self, batch: LaMPBatch) -> None:
        _, rollout_logprobs, rewards = self._sample_rollouts(batch)
        loss = rl.compute_reinforce_loss(rollout_logprobs, rewards)
        self.wandb.log({"train/loss": loss.item()})

        loss.backward()
        nn.utils.clip_grad_norm_(
            self.score_model.parameters(), self.args.max_grad_norm
        )
        self.optimizer.step()
        self.optimizer.zero_grad()

    def _train_grpo(self, batch: LaMPBatch) -> None:
        self.score_model.eval()

        with torch.no_grad():
            rollout_indices, rollout_logprobs, rewards = (
                self._sample_rollouts(batch)
            )

        # Prepare rollout dataset
        dataset = rl.RolloutDataset(
            batch["query_inputs"],
            batch["corpus_inputs"],
            batch["record_mask"],
            rollout_indices,
            rollout_logprobs,
            rewards
        )
        rollout_loader = DataLoader(
            dataset,
            batch_size=self.args.mini_batch_size,
            shuffle=True,
            collate_fn=rl.rollout_collate_fn,
            drop_last=True
        )

        self.score_model.train()

        for rollout_batch in rollout_loader:
            likelihoods = self.score_model(
                rollout_batch["query_inputs"],
                rollout_batch["corpus_inputs"],
                rollout_batch["record_mask"]
            )
            rollout_logprobs = rl.compute_rollout_logprobs(
                likelihoods,
                rollout_batch["rollout_indices"]
            )
            loss = rl.compute_grpo_loss(
                rollout_logprobs, rollout_batch, self.args.epsilon
            )
            self.wandb.log({"train/loss": loss.item()})

            loss.backward()
            nn.utils.clip_grad_norm_(
                self.score_model.parameters(), self.args.max_grad_norm
            )
            self.optimizer.step()
            self.optimizer.zero_grad()

    def _sample_rollouts(
        self,
        batch: LaMPBatch
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Sample rollouts for each query
        likelihoods = self.score_model(
            batch["query_inputs"],
            batch["corpus_inputs"],
            batch["record_mask"]
        )
        rollout_indices, rollout_logprobs = rl.sample_rollouts(
            likelihoods,
            self.args.num_rollouts,
            self.args.num_retrieve
        )

        # Gather prompt and reference for each rollout
        chat_prompts = []
        references = []

        for index, query_rollout_indices in enumerate(rollout_indices):
            for record_indices in query_rollout_indices:
                chat_prompt = self.chat_prompt_fn(
                    self.prompt_fn(
                        batch["source"][index],
                        [batch["profile"][index][i] for i in record_indices],
                        None,
                        None
                    )
                )
                chat_prompts.append(chat_prompt)
                references.append(batch["target"][index])

        # Compute reward for each rollout
        if self.args.reward_type == "metric":
            predictions = [
                comps[0] for comps in self.llm.generate(chat_prompts)
            ]
            rewards = self.reward_fn(predictions, references)
        elif self.args.reward_type == "logprob":
            rewards = self.llm.compute_completion_logprobs(
                chat_prompts, references
            )
        else:
            raise ValueError(f"Invalid reward type: {self.args.reward_type}")

        rewards = rewards.to(self.device).view_as(rollout_logprobs)
        self.wandb.log({"train/reward": rewards.mean().item()})

        return rollout_indices, rollout_logprobs, rewards

    @torch.no_grad()
    def evaluate(self) -> dict[str, float]:
        self.score_model.eval()

        chat_prompts = []
        references = []

        for batch in tqdm(self.test_loader, desc="Evaluating"):
            batch = move_to_device(batch, self.device)
            likelihoods = self.score_model(
                batch["query_inputs"],
                batch["corpus_inputs"],
                batch["record_mask"]
            )
            max_num_retrieve = batch["record_mask"].sum(dim=-1).min().item()
            num_retrieve = min(self.args.num_retrieve, max_num_retrieve)
            _, retrieved_indices = likelihoods.topk(num_retrieve)

            for index, query_retrieved_indices in enumerate(retrieved_indices):
                record_indices = [
                    i for i in query_retrieved_indices
                    if batch["record_mask"][index][i]
                ]
                chat_prompt = self.chat_prompt_fn(
                    self.prompt_fn(
                        batch["source"][index],
                        [batch["profile"][index][i] for i in record_indices],
                        None,
                        None
                    )
                )
                chat_prompts.append(chat_prompt)
                references.append(batch["target"][index])

        predictions = [
            comps[0] for comps in self.llm.generate(chat_prompts, verbose=True)
        ]
        return self.metric_fn(predictions, references)

    def _load_states(self) -> None:
        output_dir = Path(self.args.output_dir)

        self.score_model.load_pretrained(output_dir)
        logger.info(f"Loaded model weights from {output_dir}")

        states = torch.load(
            output_dir / "trainer.pt",
            map_location=self.device,
            weights_only=False
        )

        self.epoch = states["epoch"] + 1
        self.examples_seen = states["examples_seen"]
        self.best_eval_result = states["best_eval_result"]
        self.optimizer.load_state_dict(states["optimizer_state_dict"])
        logger.info(f"Loaded trainer states from {output_dir}")

    def _save_states(self) -> None:
        output_dir = Path(self.args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        self.score_model.save_pretrained(output_dir)
        torch.save(
            {
                "epoch": self.epoch,
                "examples_seen": self.examples_seen,
                "best_eval_result": self.best_eval_result,
                "optimizer_state_dict": self.optimizer.state_dict()
            },
            output_dir / "trainer.pt"
        )


def move_to_device(batch: LaMPBatch, device: str) -> LaMPBatch:
    for key, value in batch.items():
        if isinstance(value, (BatchEncoding, torch.Tensor)):
            batch[key] = value.to(device)

    batch["corpus_inputs"] = [
        [doc_inputs.to(device) for doc_inputs in doc_batches]
        for doc_batches in batch["corpus_inputs"]
    ]
    return batch
