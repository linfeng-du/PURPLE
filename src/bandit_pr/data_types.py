from typing import Callable, TypeAlias, TypedDict

import torch
from transformers import BatchEncoding

from lamp.data_types import Profile


class Example(TypedDict):

    source: str
    profiles: list[Profile]
    target: str
    query_inputs: dict[str, list[int]]
    corpus_inputs: list[dict[str, list[int]]]


class Batch(TypedDict):

    source: list[str]
    profiles: list[list[Profile]]
    target: list[str]
    query_inputs: BatchEncoding
    corpus_inputs: list[BatchEncoding]
    profile_mask: torch.Tensor


Collator: TypeAlias = Callable[[list[Example]], Batch]
Reward: TypeAlias = Callable[[list[str], list[str]], torch.Tensor]
