"""Collators for batching."""

import torch


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

        tokenized = self.tokenizer(
            sources,
            text_target=targets,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )
        return tokenized


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
        corpora = []
        targets = []
        for example in examples:
            ids.append(example['id'])
            sources.append(example['source'])
            profiles.append(example['profile'])
            queries.append(example['query'])
            corpora.append(example['corpus'])
            targets.append(example['target'])

        corpora_mask = torch.ones(len(examples), self.max_corpus_size, dtype=torch.bool)
        for idx, corpus in enumerate(corpora):
            if len(corpus) < self.max_corpus_size:
                corpora_mask[idx, len(corpus):] = 0
                corpus.extend([''] * (self.max_corpus_size - len(corpus)))
            elif len(corpus) > self.max_corpus_size:
                corpus[self.max_corpus_size:] = []
                profiles[idx][self.max_corpus_size:] = []

        tokenized_queries = self.tokenizer(
            queries,
            padding=True,
            truncation=True,
            max_length=self.max_query_length,
            return_tensors='pt'
        )
        tokenized_corpora = self.tokenizer(
            [document for corpus in corpora for document in corpus],
            padding=True,
            truncation=True,
            max_length=self.max_document_length,
            return_tensors='pt'
        )

        batch = {
            'id': ids,
            'source': sources,
            'profile': profiles,
            'query': tokenized_queries,
            'corpus': tokenized_corpora,
            'corpus_mask': corpora_mask,
            'target': targets,
        }
        return batch
