import json
from typing import TypedDict

import torch
from transformers import PreTrainedTokenizerBase, BatchEncoding
from datasets import Dataset, DatasetDict
from datasets.formatting.formatting import LazyBatch

from lamp import create_query_corpus_generator
from lamp.data_types import Profile


def load_lamp_dataset(task: str, split: str) -> Dataset:
    with open(f'./dataset/{task}/{split}_questions.json', 'r') as file:
        examples = json.load(file)

    with open(f'./dataset/{task}/{split}_outputs.json', 'r') as file:
        targets = json.load(file)


class RetrieverTrainingExample(TypedDict):

    id: str
    source: str
    profile: list[Profile]
    query: str
    corpus: list[str]
    target: str


class RetrieverTrainingDataset(Dataset):

    def __init__(self, task: str, split: str) -> None:
        with open(f'./dataset/{task}/{split}_questions.json', 'r') as file:
            self.examples = json.load(file)

        with open(f'./dataset/{task}/{split}_outputs.json', 'r') as file:
            outputs = json.load(file)
            self.targets = {gold['id']: gold['output'] for gold in outputs['golds']}

        self.query_corpus_generator = create_query_corpus_generator(task)

    def __getitem__(self, index: int) -> RetrieverTrainingExample:
        example = self.examples[index]

        id_ = example['id']
        source = example['input']
        profile = example['profile']
        target = self.targets[id_]
        query, corpus = self.query_corpus_generator(source, profile)

        return RetrieverTrainingExample(
            id=id_,
            source=source,
            profile=profile,
            query=query,
            corpus=corpus,
            target=target
        )

    def __len__(self) -> int:
        return len(self.examples)


def create_preprocessor(tokenizer: PreTrainedTokenizerBase, max_query_length: int, max_document_length: int):
    def preprocessor(batch: LazyBatch) -> LazyBatch:
        query_inputs = tokenizer(batch['query'], truncation=True, max_length=max_query_length)
        batch['query_inputs'] = [
            {key: value[index] for key, value in query_inputs.items()}
            for index in range(len(batch['query']))
        ]
        batch['corpus_inputs'] = [
            tokenizer(corpus, truncation=True, max_length=max_document_length)
            for corpus in batch['corpus']
        ]
        return batch

    return preprocessor


class BatchedRetrieverTrainingExamples(TypedDict):

    id: list[str]
    source: list[str]
    profile: list[list[Profile]]
    query_inputs: BatchEncoding
    corpus_inputs: list[BatchEncoding]
    profile_mask: torch.Tensor
    targets: list[str]


class RetrieverTrainingCollator:

    def __init__(
        self,
        max_num_profiles: int,
        max_query_length: int,
        max_document_length: int,
        tokenizer: PreTrainedTokenizerBase
    ) -> None:
        self.max_num_profiles = max_num_profiles
        self.max_query_length = max_query_length
        self.max_document_length = max_document_length
        self.tokenizer = tokenizer

    def __call__(self, examples: list[RetrieverTrainingExample]) -> BatchedRetrieverTrainingExamples:
        ids = [example['id'] for example in examples]
        sources = [example['source'] for example in examples]
        profiles = [example['profile'] for example in examples]
        queries = [example['query'] for example in examples]
        corpuses = [example['corpus'] for example in examples]
        targets = [example['target'] for example in examples]

        # Keep only `self.max_num_profiles` profiles for each example
        max_num_profiles = max(len(profile) for profile in profiles)

        if self.max_num_profiles > 0:
            max_num_profiles = min(max_num_profiles, self.max_num_profiles)

        profile_mask = torch.ones(len(examples), max_num_profiles, dtype=torch.bool)

        for index, corpus in enumerate(corpuses):
            if len(corpus) < max_num_profiles:
                profile_mask[index, len(corpus):] = 0
            elif len(corpus) > max_num_profiles:
                corpus[max_num_profiles:] = []
                profiles[index][max_num_profiles:] = []

        query_inputs = self.tokenizer(
            queries,
            padding=True,
            truncation=True,
            max_length=self.max_query_length,
            return_tensors='pt'
        )

        # Split documents into batches of 100 to save memory
        corpus_inputs = []
        documents = [document for corpus in corpuses for document in corpus]
        document_batches = [documents[i : i + 100] for i in range(0, len(documents), 100)]

        for document_batch in document_batches:
            document_batch_inputs = self.tokenizer(
                document_batch,
                padding=True,
                truncation=True,
                max_length=self.max_document_length,
                return_tensors='pt'
            )
            corpus_inputs.append(document_batch_inputs)

        return BatchedRetrieverTrainingExamples(
            id=ids,
            source=sources,
            profile=profiles,
            # query_inputs=query_inputs,
            # corpus_inputs=corpus_inputs,
            profile_mask=profile_mask,
            target=targets
        )
