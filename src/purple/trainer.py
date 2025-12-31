import json
import logging
from pathlib import Path

import wandb
from omegaconf import DictConfig
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import BatchEncoding

from lamp import MetricFn, PromptFn
from llm import LLM

from . import grpo
from .dataset import LaMPBatch
from .reward import RewardFn
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
        prompt_fn: PromptFn,
        reward_fn: RewardFn,
        metric_fn: MetricFn,
        resume: bool
    ) -> None:
        self.cfg = cfg
        self.score_model = score_model
        self.llm = llm
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.prompt_fn = prompt_fn
        self.reward_fn = reward_fn
        self.metric_fn = metric_fn

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
            lr=self.cfg.lr
        )

        if resume:
            self._load_states()

        self.wandb = wandb.init(
            dir="./outputs",
            project="PURPLE",
            name=f"{self.cfg.run_name}"
        )

    def train(self) -> None:
        self.score_model.train()

        for _ in range(self.cfg.num_epochs):
            for batch in tqdm(self.train_loader, desc=f"Epoch {self.epoch}"):
                rollout_loader = self._sample_rollouts(batch)
                self.score_model.train()

                for rollout_batch in rollout_loader:
                    likelihoods = self.score_model(
                        rollout_batch["query_inputs"],
                        rollout_batch["corpus_inputs"],
                        rollout_batch["record_mask"]
                    )
                    rollout_logps = grpo.compute_rollout_logps(
                        likelihoods,
                        rollout_batch["rollout_indices"]
                    )
                    loss = grpo.compute_loss(
                        rollout_logps, rollout_batch, self.cfg.grpo.epsilon
                    )

                    loss.backward()
                    nn.utils.clip_grad_norm_(
                        self.score_model.parameters(), self.cfg.max_grad_norm
                    )
                    self.optimizer.step()
                    self.optimizer.zero_grad()

                    self.wandb.log({"train/loss": loss.item()})

                self.examples_seen += len(batch["source"])

                if self.examples_seen % self.cfg.eval_every == 0:
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

    @torch.no_grad()
    def _sample_rollouts(self, batch: LaMPBatch) -> DataLoader:
        self.score_model.eval()
        batch = move_to_device(batch, self.device)

        # Sample rollouts for each query
        likelihoods = self.score_model(
            batch["query_inputs"],
            batch["corpus_inputs"],
            batch["record_mask"]
        )
        rollout_indices, rollout_logps = grpo.sample_rollouts(
            likelihoods,
            self.cfg.grpo.num_rollouts,
            self.cfg.num_retrieve
        )

        # Gather prompt and reference for each rollout
        prompts = []
        references = []

        for index, query_rollout_indices in enumerate(rollout_indices):
            for record_indices in query_rollout_indices:
                profile = [batch["profile"][index][i] for i in record_indices]
                prompt = self.prompt_fn(
                    batch["source"][index], profile, None, None
                )
                reference = batch["target"][index]

                prompts.append(prompt)
                references.append(reference)

        # Compute reward for each rollout
        if self.cfg.grpo.reward == "metric":
            predictions = self.llm.generate(prompts)
            rewards = self.reward_fn(predictions, references)
        elif self.cfg.grpo.reward == "logp":
            rewards = self.llm.compute_completion_logps(
                prompts, references
            )
        else:
            raise ValueError(f"Invalid reward: {self.cfg.grpo.reward}")

        rewards = rewards.to(self.device).view_as(rollout_logps)
        self.wandb.log({"train/reward": rewards.mean().item()})

        # Compute advantage for each rollout
        advantages = (
            (rewards - rewards.mean(dim=-1, keepdim=True))
            / (rewards.std(dim=-1, keepdim=True) + 1e-8)
        )

        # Prepare rollout dataset
        dataset = grpo.RolloutDataset(
            batch["query_inputs"],
            batch["corpus_inputs"],
            batch["record_mask"],
            rollout_indices,
            rollout_logps,
            advantages
        )
        return DataLoader(
            dataset,
            batch_size=self.cfg.grpo.batch_size,
            shuffle=True,
            collate_fn=grpo.collate_fn,
            drop_last=True
        )

    @torch.no_grad()
    def evaluate(self) -> dict[str, float]:
        self.score_model.eval()

        prompts = []
        references = []

        for batch in tqdm(self.test_loader, desc="Evaluating"):
            batch = move_to_device(batch, self.device)
            likelihoods = self.score_model(
                batch["query_inputs"],
                batch["corpus_inputs"],
                batch["record_mask"]
            )
            max_num_retrieve = batch["record_mask"].sum(dim=-1).min().item()
            num_retrieve = min(self.cfg.num_retrieve, max_num_retrieve)
            _, retrieved_indices = likelihoods.topk(num_retrieve)

            for index, query_retrieved_indices in enumerate(retrieved_indices):
                retrieved_profile = [
                    batch["profile"][index][i]
                    for i in query_retrieved_indices
                    if batch["record_mask"][index][i]
                ]
                prompt = self.prompt_fn(
                    batch["source"][index], retrieved_profile, None, None
                )
                reference = batch["target"][index]

                prompts.append(prompt)
                references.append(reference)

        predictions = self.llm.generate(prompts, verbose=True)
        return self.metric_fn(predictions, references)

    def _load_states(self) -> None:
        model_dir = Path("outputs") / "models" / self.cfg.run_name

        self.score_model.from_pretrained(model_dir)
        logger.info(f"Loaded model weights from {model_dir}")

        states = torch.load(
            model_dir / "trainer.pt",
            map_location=self.device,
            weights_only=False
        )

        self.epoch = states["epoch"] + 1
        self.examples_seen = states["examples_seen"]
        self.best_eval_result = states["best_eval_result"]
        self.optimizer.load_state_dict(states["optimizer_state_dict"])
        logger.info(f"Loaded trainer states from {model_dir}")

    def _save_states(self) -> None:
        model_dir = Path("outputs") / "models" / self.cfg.run_name
        model_dir.mkdir(parents=True, exist_ok=True)

        self.score_model.save_pretrained(model_dir)
        torch.save(
            {
                "epoch": self.epoch,
                "examples_seen": self.examples_seen,
                "best_eval_result": self.best_eval_result,
                "optimizer_state_dict": self.optimizer.state_dict()
            },
            model_dir / "trainer.pt"
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
