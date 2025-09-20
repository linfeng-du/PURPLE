from pathlib import Path
from typing import Callable

from datasets import Dataset, load_from_disk
from datasets.formatting.formatting import LazyBatch
from rank_bm25 import BM25Okapi

import torch
from transformers import PreTrainedTokenizerBase

from tqdm import tqdm

from lamp import load_lamp_dataset
from lamp.retrievers import Contriever
from .data_types import Batch, Collator, Example


def load_retrieved_lamp_dataset(task: str, split: str, retriever: str, num_candidates: int) -> Dataset:
    dataset_dir = Path('./dataset') / task / f'{retriever}-{num_candidates}' / split

    if not dataset_dir.exists():
        if retriever == 'contriever':
            contriever = Contriever()

        examples = []
        dataset = load_lamp_dataset(task, split)

        for example in tqdm(dataset, desc='Retrieving'):
            query = example['query']
            corpus = example['corpus']
            profiles = example['profiles']

            if retriever == 'bm25':
                bm25 = BM25Okapi([document.split() for document in corpus])
                retrieved_indices = bm25.get_top_n(query.split(), range(len(corpus)), n=num_candidates)
            elif retriever == 'contriever':
                retrieved_indices = contriever(query, corpus, range(len(corpus)), num_candidates)
            else:
                raise ValueError(f'Invalid retriever: {retriever}')

            example['corpus'] = [corpus[index] for index in retrieved_indices]
            example['profiles'] = [profiles[index] for index in retrieved_indices]
            examples.append(example)

        Dataset.from_list(examples).save_to_disk(dataset_dir)

    return load_from_disk(dataset_dir)


def create_preprocessor(
    max_num_profiles: int,
    max_query_length: int,
    max_document_length: int,
    tokenizer: PreTrainedTokenizerBase
) -> Callable[[LazyBatch], LazyBatch]:
    def preprocessor(batch: LazyBatch) -> LazyBatch:
        if max_num_profiles > 0:
            batch['profiles'] = [profiles[:max_num_profiles] for profiles in batch['profiles']]
            batch['corpus'] = [corpus[:max_num_profiles] for corpus in batch['corpus']]

        query_inputs = tokenizer(batch['query'], truncation=True, max_length=max_query_length)
        batch['query_inputs'] = [
            {key: value[index] for key, value in query_inputs.items()}
            for index in range(len(batch['query']))
        ]

        batch['corpus_inputs'] = []

        for corpus in batch['corpus']:
            corpus_inputs = tokenizer(corpus, truncation=True, max_length=max_document_length)
            corpus_inputs = [
                {key: value[index] for key, value in corpus_inputs.items()}
                for index in range(len(corpus))
            ]
            batch['corpus_inputs'].append(corpus_inputs)

        return batch

    return preprocessor


def create_collator(tokenizer: PreTrainedTokenizerBase) -> Collator:
    def collator(examples: list[Example]) -> Batch:
        sources = [example['source'] for example in examples]
        profiles = [example['profiles'] for example in examples]
        targets = [example['target'] for example in examples]
        query_inputs = [example['query_inputs'] for example in examples]
        corpus_inputs = [example['corpus_inputs'] for example in examples]

        # Create profile mask
        max_num_profiles = max([len(profile) for profile in profiles])
        profile_mask = torch.ones(len(profiles), max_num_profiles, dtype=torch.bool)

        for index, example_profiles in enumerate(profiles):
            profile_mask[index, len(example_profiles):] = 0

        # Pad query inputs
        query_inputs = tokenizer.pad(query_inputs, return_tensors='pt')

        # Split corpus into batches of 128 documents to save memory
        subbatched_corpus_inputs = []

        for example_corpus_inputs in corpus_inputs:
            document_subbatches = []

            for document_inputs in [
                example_corpus_inputs[i:i+128]
                for i in range(0, len(example_corpus_inputs), 128)
            ]:
                document_inputs = tokenizer.pad(document_inputs, return_tensors='pt')
                document_subbatches.append(document_inputs)

            subbatched_corpus_inputs.append(document_subbatches)

        return Batch(
            source=sources,
            profiles=profiles,
            target=targets,
            query_inputs=query_inputs,
            corpus_inputs=subbatched_corpus_inputs,
            profile_mask=profile_mask
        )

    return collator
