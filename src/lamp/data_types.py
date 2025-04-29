from typing import Callable, TypeAlias, TypedDict

import torch
from transformers import BatchEncoding


Profile: TypeAlias = dict[str, str]
PromptGenerator: TypeAlias = Callable[[str, list[Profile], float], str]
QueryCorpusGenerator: TypeAlias = Callable[[str, list[Profile]], tuple[str, list[str]]]


class LaMPExample(TypedDict):

    id: str
    source: str
    target: str


class RetrieverTrainingExample(TypedDict):

    id: str
    source: str
    profile: list[Profile]
    query: str
    corpus: list[str]
    target: str


class BatchedRetrieverTrainingExamples(TypedDict):

    id: list[str]
    source: list[str]
    profile: list[list[Profile]]
    query_inputs: BatchEncoding
    corpus_inputs: list[BatchEncoding]
    profile_mask: torch.Tensor
    targets: list[str]
