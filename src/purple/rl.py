from typing import TypedDict

import torch
from torch.utils.data import Dataset
from transformers import BatchEncoding


class RolloutGroup(TypedDict):
    query_inputs: dict[str, torch.Tensor]
    corpus_inputs: list[BatchEncoding]
    record_mask: torch.Tensor
    rollout_indices: torch.Tensor
    rollout_logprobs: torch.Tensor
    rewards: torch.Tensor


class RolloutGroupBatch(TypedDict):
    query_inputs: BatchEncoding
    corpus_inputs: list[list[BatchEncoding]]
    record_mask: torch.Tensor
    rollout_indices: torch.Tensor
    rollout_logprobs: torch.Tensor
    rewards: torch.Tensor


class RolloutDataset(Dataset):
    def __init__(
        self,
        query_inputs: BatchEncoding,
        corpus_inputs: list[list[BatchEncoding]],
        record_mask: torch.Tensor,
        rollout_indices: torch.Tensor,
        rollout_logprobs: torch.Tensor,
        rewards: torch.Tensor
    ) -> None:
        self.examples = []

        for index in range(len(corpus_inputs)):
            self.examples.append(
                RolloutGroup(
                    query_inputs={
                        k: v[index] for k, v in query_inputs.items()
                    },
                    corpus_inputs=corpus_inputs[index],
                    record_mask=record_mask[index],
                    rollout_indices=rollout_indices[index],
                    rollout_logprobs=rollout_logprobs[index],
                    rewards=rewards[index]
                )
            )

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> RolloutGroup:
        return self.examples[index]


def sample_rollouts(
    likelihoods: torch.Tensor,
    num_rollouts: int,
    num_retrieve: int
) -> torch.Tensor:
    max_num_retrieve = (likelihoods > 0).sum(dim=-1).min().item()
    num_retrieve = min(num_retrieve, max_num_retrieve)

    rollout_indices = []
    rollout_logprobs = []

    for _ in range(num_rollouts):
        indices = torch.full_like(
            likelihoods[:, :num_retrieve], fill_value=-1, dtype=torch.long
        )
        logprobs = torch.zeros_like(likelihoods[:, :num_retrieve])

        likelihoods_ = likelihoods

        for index in range(num_retrieve):
            probs = likelihoods_ / likelihoods_.sum(dim=-1, keepdim=True)
            chosen_indices = torch.multinomial(probs, num_samples=1)
            chosen_probs = probs.gather(dim=-1, index=chosen_indices)

            indices[:, index] = chosen_indices.squeeze(dim=-1)
            logprobs[:, index] = chosen_probs.log().squeeze(dim=-1)

            likelihoods_ = likelihoods_.scatter(
                dim=-1, index=chosen_indices, value=0.0
            )

        rollout_indices.append(indices)
        rollout_logprobs.append(logprobs.sum(dim=-1))

    rollout_indices = torch.stack(rollout_indices, dim=1)
    rollout_logprobs = torch.stack(rollout_logprobs, dim=1)
    return rollout_indices, rollout_logprobs


def compute_rollout_logprobs(
    likelihoods: torch.Tensor,
    rollout_indices: torch.Tensor
) -> torch.Tensor:
    _, num_rollouts, num_retrieve = rollout_indices.shape

    rollout_logprobs = torch.zeros_like(
        rollout_indices[:, :, 0], dtype=torch.float32
    )
    likelihoods = likelihoods.unsqueeze(dim=1).expand(-1, num_rollouts, -1)

    for index in range(num_retrieve):
        chosen_indices = rollout_indices[:, :, index].unsqueeze(dim=-1)
        chosen_likelihoods = (
            likelihoods.gather(dim=-1, index=chosen_indices).squeeze(dim=-1)
        )

        rollout_logprobs += (
            chosen_likelihoods.log() - likelihoods.sum(dim=-1).log()
        )
        likelihoods = likelihoods.scatter(
            dim=-1, index=chosen_indices, value=0.0
        )

    return rollout_logprobs


def rollout_collate_fn(examples: list[RolloutGroup]) -> RolloutGroupBatch:
    query_inputs = {}

    for key in examples[0]["query_inputs"].keys():
        query_inputs[key] = torch.stack(
            [e["query_inputs"][key] for e in examples]
        )

    query_inputs = BatchEncoding(query_inputs)
    corpus_inputs = [e["corpus_inputs"] for e in examples]
    record_mask = torch.stack([e["record_mask"] for e in examples])
    rollout_indices = torch.stack([e["rollout_indices"] for e in examples])
    rollout_logprobs = torch.stack([e["rollout_logprobs"] for e in examples])
    rewards = torch.stack([e["rewards"] for e in examples])

    return RolloutGroupBatch(
        query_inputs=query_inputs,
        corpus_inputs=corpus_inputs,
        record_mask=record_mask,
        rollout_indices=rollout_indices,
        rollout_logprobs=rollout_logprobs,
        rewards=rewards
    )


def compute_reinforce_loss(
    rollout_logprobs: torch.Tensor,
    rewards: torch.Tensor
) -> torch.Tensor:
    advantages = (
        (rewards - rewards.mean(dim=-1, keepdim=True))
        / (rewards.std(dim=-1, keepdim=True) + 1e-8)
    )
    return -(rollout_logprobs * advantages).mean()


def compute_grpo_loss(
    rollout_logprobs: torch.Tensor,
    rollout_batch: RolloutGroupBatch,
    epsilon: float
) -> torch.Tensor:
    old_rollout_logprobs = rollout_batch["rollout_logprobs"]
    rewards = rollout_batch["rewards"]

    advantages = (
        (rewards - rewards.mean(dim=-1, keepdim=True))
        / (rewards.std(dim=-1, keepdim=True) + 1e-8)
    )

    ratio = (rollout_logprobs - old_rollout_logprobs).exp()
    surrogate_1 = ratio * advantages
    surrogate_2 = ratio.clamp(min=1 - epsilon, max=1 + epsilon) * advantages

    return -torch.min(surrogate_1, surrogate_2).mean()
