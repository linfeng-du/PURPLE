"""Dataset classes and preprocessors.

- GeneralSeq2SeqDataset:
  A dataset class for retrieval-augmented generation that integrates retrieval for each example.

- Seq2SeqRetrieverTrainingDataset:
  A dataset class for training the retriever, providing a query and corpus for each example.
"""

import json

from torch.utils.data import Dataset


class GeneralSeq2SeqDataset(Dataset):

    def __init__(self, data, labels, prompt_generator):
        super().__init__()
        self.data = data
        self.labels = labels
        self.prompt_generator = prompt_generator

    @classmethod
    def from_disk(cls, data_path, label_path, prompt_generator=None):
        with open(data_path, 'r') as file:
            data = json.load(file)

        with open(label_path, 'r') as file:
            labels = json.load(file)
            labels = {label['id']: label['output'] for label in labels['golds']}

        return cls(data, labels, prompt_generator)

    @classmethod
    def from_batch_sample_indices(cls, batch, sample_indices, prompt_generator):
        data = []
        labels = {}
        for batch_idx, sample_indices_ in enumerate(sample_indices):
            data_id = batch['id'][batch_idx]
            labels[data_id] = batch['target'][batch_idx]
            for sample_indices__ in sample_indices_:
                data.append({
                    'id': data_id,
                    'input': batch['source'][batch_idx],
                    'profile': [batch['profile'][batch_idx][i] for i in sample_indices__]
                })

        return cls(data, labels, prompt_generator)

    def __getitem__(self, index):
        data_idx = self.data[index]

        source = data_idx['input']
        if self.prompt_generator is not None:
            source = self.prompt_generator(source, data_idx['profile'])

        item = {
            'id': data_idx['id'],
            'source': source,
            'target': self.labels[data_idx['id']] 
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

        source = data_idx['input']
        profile = data_idx['profile']
        query, corpus = self.query_corpus_generator(source, profile)

        item = {
            'id': data_idx['id'],
            'source': source,
            'profile': profile,
            'query': query,
            'corpus': corpus,
            'target': self.labels[data_idx['id']]
        }
        return item

    def __len__(self):
        return len(self.data)


def create_preprocessor(tokenizer, max_length):
    def preprocess_dataset(examples):
        sources = [example for example in examples['source']]
        targets = [example for example in examples['target']]
        model_inputs = tokenizer(sources, text_target=targets, max_length=max_length, truncation=True)
        return model_inputs

    return preprocess_dataset
