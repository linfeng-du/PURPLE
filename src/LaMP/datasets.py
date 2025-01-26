import json
from typing import Callable

import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerBase, BatchEncoding


def load_all_labels(task: str) -> list[str]:
    task_labels = {
        'LaMP-1': ['[1]', '[2]'],
        'LaMP-2': [
            'sci-fi', 'based on a book', 'comedy', 'action',
            'twist ending', 'dystopia', 'dark comedy', 'classic',
            'psychology', 'fantasy', 'romance', 'thought-provoking',
            'social commentary', 'violence', 'true story'
        ],
        'LaMP-3': ['1', '2', '3', '4', '5'],
        'LaMP-4': [],
        'LaMP-5': [],
        'LaMP-6': [],
        'LaMP-7': []
    }
    return task_labels[task]


class LaMPDataset(Dataset):

    def __init__(
        self,
        data: list[dict[str, str | list[str]]],
        labels: list[dict[str, str]],
        prompt_generator: Callable | None = None
    ) -> None:
        self.data = data
        self.labels = labels
        self.prompt_generator = prompt_generator

    @classmethod
    def from_disk(
        cls,
        data_path: str,
        label_path: str,
        prompt_generator: Callable | None = None
    ) -> 'LaMPDataset':
        with open(data_path, 'r') as file:
            data = json.load(file)

        with open(label_path, 'r') as file:
            labels = {label['id']: label['output'] for label in json.load(file)['golds']}

        return cls(data, labels, prompt_generator)

    @classmethod
    def from_batch_profile_indices(
        cls,
        batch_profile_indices: torch.Tensor,
        batch: dict[str, list[str] | list[list[str]] | BatchEncoding | torch.Tensor],
        prompt_generator: Callable | None = None
    ) -> 'LaMPDataset':
        """Create the dataset based on a batch of sampled profiles.

        Args:
            batch_profile_indices: Tensor of shape (batch_size, sample_size, n_retrieve)
                Indices of a batch of sampled profiles.
        """
        data = []
        labels = {}

        for batch_index, sample_profile_indices in enumerate(batch_profile_indices):
            data_id = batch['id'][batch_index]
            labels[data_id] = batch['target'][batch_index]

            for profile_indices in sample_profile_indices:
                data.append({
                    'id': data_id,
                    'input': batch['source'][batch_index],
                    'profile': [
                        batch['profile'][batch_index][profile_index]
                        for profile_index in profile_indices
                    ]
                })

        return cls(data, labels, prompt_generator)

    def __getitem__(self, index: int) -> dict[str, str]:
        example = self.data[index]
        source = example['input']

        if self.prompt_generator is not None:
            source = self.prompt_generator(source, example['profile'])

        return {
            'id': example['id'],
            'source': source,
            'target': self.labels[example['id']] 
        }

    def __len__(self) -> int:
        return len(self.data)


class LaMPCollator:

    def __init__(self, tokenizer: PreTrainedTokenizerBase, max_length: int) -> None:
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(self, examples: list[dict[str, str]]) -> BatchEncoding:
        sources = []
        targets = []

        for example in examples:
            sources.append(example['source'])
            targets.append(example['target'])

        return self.tokenizer(
            sources,
            text_target=targets,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )


class RetrieverTrainingDataset(Dataset):

    def __init__(self, data_path: str, label_path: str, query_corpus_generator: Callable) -> None:
        super().__init__()

        with open(data_path, 'r') as file:
            self.data = json.load(file)

        with open(label_path, 'r') as file:
            self.labels = {label['id']: label['output'] for label in json.load(file)['golds']}

        self.query_corpus_generator = query_corpus_generator

    def __getitem__(self, index: int) -> dict[str, str | list[str]]:
        example = self.data[index]

        source = example['input']
        profile = example['profile']
        query, corpus = self.query_corpus_generator(source, profile)

        return {
            'id': example['id'],
            'source': source,
            'profile': profile,
            'query': query,
            'corpus': corpus,
            'target': self.labels[example['id']]
        }

    def __len__(self) -> int:
        return len(self.data)


class RetrieverTrainingCollator:

    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        max_n_profiles: int,
        max_query_length: int,
        max_document_length: int
    ) -> None:
        self.tokenizer = tokenizer
        self.max_n_profiles = max_n_profiles
        self.max_query_length = max_query_length
        self.max_document_length = max_document_length

    def __call__(self, examples: list[dict[str, str | list[str]]]) -> (
        dict[str, list[str] | list[list[str]] | BatchEncoding | torch.Tensor]
    ):
        ids = []
        sources = []
        profiles = []
        queries = []
        corpuses = []
        targets = []

        for example in examples:
            ids.append(example['id'])
            sources.append(example['source'])
            profiles.append(example['profile'])
            queries.append(example['query'])
            corpuses.append(example['corpus'])
            targets.append(example['target'])

        # Keep only `self.max_n_profiles` profiles for each example
        profile_mask = torch.ones(len(examples), self.max_n_profiles, dtype=torch.bool)

        for batch_index, corpus in enumerate(corpuses):
            if len(corpus) < self.max_corpus_size:
                profile_mask[batch_index, len(corpus):] = 0
                corpus.extend([''] * (self.max_corpus_size - len(corpus)))
            elif len(corpus) > self.max_corpus_size:
                corpus[self.max_corpus_size:] = []
                profiles[batch_index][self.max_corpus_size:] = []

        query_inputs = self.tokenizer(
            queries,
            padding=True,
            truncation=True,
            max_length=self.max_query_length,
            return_tensors='pt'
        )
        corpus_inputs = self.tokenizer(
            [document for corpus in corpuses for document in corpus],
            padding=True,
            truncation=True,
            max_length=self.max_document_length,
            return_tensors='pt'
        )

        return {
            'id': ids,
            'source': sources,
            'profile': profiles,
            'query_inputs': query_inputs,
            'corpus_inputs': corpus_inputs,
            'profile_mask': profile_mask,
            'target': targets
        }
