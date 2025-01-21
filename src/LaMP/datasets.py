import json

import torch
from torch.utils.data import Dataset


def load_all_labels(task):
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

    def __init__(self, data, labels, prompt_generator):
        self.data = data
        self.labels = labels
        self.prompt_generator = prompt_generator

    @classmethod
    def from_disk(cls, data_path, label_path, prompt_generator=None):
        with open(data_path, 'r') as file:
            data = json.load(file)

        with open(label_path, 'r') as file:
            labels = {label['id']: label['output'] for label in json.load(file)['golds']}

        return cls(data, labels, prompt_generator)

    @classmethod
    def from_batch_profile_indices(cls, batch, batch_profile_indices, prompt_generator):
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

    def __getitem__(self, index):
        example = self.data[index]

        source = example['input']

        if self.prompt_generator is not None:
            source = self.prompt_generator(source, example['profile'])

        return {
            'id': example['id'],
            'source': source,
            'target': self.labels[example['id']] 
        }

    def __len__(self):
        return len(self.data)


class LaMPCollator:

    def __init__(self, tokenizer, max_length):
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(self, examples):
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

    def __init__(self, data_path, label_path, query_corpus_generator):
        super().__init__()

        with open(data_path, 'r') as file:
            self.data = json.load(file)

        with open(label_path, 'r') as file:
            self.labels = {label['id']: label['output'] for label in json.load(file)['golds']}

        self.query_corpus_generator = query_corpus_generator

    def __getitem__(self, index):
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

    def __len__(self):
        return len(self.data)


class RetrieverTrainingCollator:

    def __init__(self, tokenizer, max_corpus_size, max_query_length, max_document_length):
        self.tokenizer = tokenizer
        self.max_corpus_size = max_corpus_size
        self.max_query_length = max_query_length
        self.max_document_length = max_document_length

    def __call__(self, examples):
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

        profile_mask = torch.ones(len(examples), self.max_corpus_size, dtype=torch.bool)

        for batch_index, corpus in enumerate(corpuses):
            if len(corpus) < self.max_corpus_size:
                profile_mask[batch_index, len(corpus):] = 0
                corpus.extend([''] * (self.max_corpus_size - len(corpus)))
            elif len(corpus) > self.max_corpus_size:
                corpus[self.max_corpus_size:] = []
                profiles[batch_index][self.max_corpus_size:] = []

        tokenized_queries = self.tokenizer(
            queries,
            padding=True,
            truncation=True,
            max_length=self.max_query_length,
            return_tensors='pt'
        )
        tokenized_corpora = self.tokenizer(
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
            'query': tokenized_queries,
            'corpus': tokenized_corpora,
            'profile_mask': profile_mask,
            'target': targets,
        }
