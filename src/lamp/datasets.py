import json

import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerBase

from .data_types import (
    PromptGenerator,
    QueryCorpusGenerator,
    LaMPExample,
    RetrieverTrainingExample,
    BatchedRetrieverTrainingExamples
)


class LaMPDataset(Dataset):

    def __init__(self, task: str, split: str, prompt_generator: PromptGenerator | None = None) -> None:
        with open(f'./dataset/{task}/{split}_questions.json', 'r') as file:
            self.examples= json.load(file)

        with open(f'./dataset/{task}/{split}_outputs.json', 'r') as file:
            outputs = json.load(file)
            self.targets = {gold['id']: gold['output'] for gold in outputs['golds']}

        self.prompt_generator = prompt_generator

    def __getitem__(self, index: int) -> LaMPExample:
        example = self.examples[index]

        id_ = example['id']
        source = example['input']
        target = self.targets[id_]

        if self.prompt_generator is not None:
            source = self.prompt_generator(source, example['profile'])

        return LaMPExample(
            id=id_,
            source=source,
            target=target
        )

    def __len__(self) -> int:
        return len(self.examples)


class RetrieverTrainingDataset(Dataset):

    def __init__(self, task: str, split: str, query_corpus_generator: QueryCorpusGenerator) -> None:
        with open(f'./dataset/{task}/{split}_questions.json', 'r') as file:
            self.examples = json.load(file)

        with open(f'./dataset/{task}/{split}_outputs.json', 'r') as file:
            outputs = json.load(file)
            self.targets = {gold['id']: gold['output'] for gold in outputs['golds']}

        self.query_corpus_generator = query_corpus_generator

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
            query_inputs=query_inputs,
            corpus_inputs=corpus_inputs,
            profile_mask=profile_mask,
            targets=targets
        )
