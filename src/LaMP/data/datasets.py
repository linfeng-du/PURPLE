"""Dataset classes and a function for loading task labels.

- GeneralSeq2SeqDataset:
  A dataset class for retrieval-augmented generation that integrates retrieval for each example.

- Seq2SeqRetrieverTrainingDataset:
    A dataset class for training the retriever, providing a query and corpus for each example.
"""

import json

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


class GeneralSeq2SeqDataset(Dataset):

    def __init__(self, data_path, label_path, prompt_generator=None):
        super().__init__()
        with open(data_path, 'r') as file:
            self.data = json.load(file)

        with open(label_path, 'r') as file:
            labels = json.load(file)
            self.labels = {label['id']: label['output'] for label in labels['golds']}

        self.prompt_generator = prompt_generator

    def __getitem__(self, index):
        data_idx = self.data[index]

        data_id = data_idx['id']
        source = data_idx['input']
        target = self.labels[data_idx['id']]

        if self.prompt_generator is not None:
            source = self.prompt_generator(source, data_idx['profile'])

        item = {
            'id': data_id,
            'source': source,
            'target': target
        }
        return item

    def __len__(self):
        return len(self.data)


class Seq2SeqRetrieverTrainingDataset(Dataset):

    def __init__(self, data_path, label_path, query_corpus_generator):
        super().__init__()
        with open(data_path, 'r') as file:
            self.data = json.load(file)

        with open(label_path, 'r') as file:
            labels = json.load(file)
            self.labels = {label['id']: label['output'] for label in labels['golds']}

        self.query_corpus_generator = query_corpus_generator

    def __getitem__(self, index):
        data_idx = self.data[index]

        data_id = data_idx['id']
        source = data_idx['input']
        profile = data_idx['profile']
        query, corpus = self.query_corpus_generator(source, profile)
        target = self.labels[data_idx['id']]

        item = {
            'id': data_id,
            'source': source,
            'profile': profile,
            'query': query,
            'corpus': corpus,
            'target': target
        }
        return item

    def __len__(self):
        return len(self.data)
