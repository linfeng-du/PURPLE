from typing import Callable, TypeAlias, TypedDict

import torch
from transformers import BatchEncoding

from lamp.data_types import Profile


class Batch(TypedDict):

    source: list[str]
    profiles: list[list[Profile]]
    target: list[str]
    query_inputs: BatchEncoding
    corpus_inputs: list[BatchEncoding]
    profile_mask: torch.Tensor


Example: TypeAlias = dict[str, str | list[Profile] | dict[str, list[int]] | list[dict[str, list[int]]]]
Collator: TypeAlias = Callable[[list[Example]], Batch]
Reward: TypeAlias = Callable[[list[str], list[str]], torch.Tensor]
