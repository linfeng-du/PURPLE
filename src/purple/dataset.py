from collections.abc import Callable
from pathlib import Path
from typing import TypedDict

from tqdm import tqdm

from datasets import Dataset, load_from_disk
from datasets.formatting.formatting import LazyBatch

import torch
from transformers import AutoTokenizer, BatchEncoding

from lamp import create_retrieval_fn, load_lamp_dataset


class LaMPExample(TypedDict):
    source: str
    profile: list[dict[str, str]]
    target: str
    query_inputs: dict[str, list[int]]
    corpus_inputs: list[dict[str, list[int]]]


class LaMPBatch(TypedDict):
    source: list[str]
    profile: list[list[dict[str, str]]]
    target: list[str]
    query_inputs: BatchEncoding
    corpus_inputs: list[list[BatchEncoding]]
    record_mask: torch.Tensor


CollateFn = Callable[[list[LaMPExample]], LaMPBatch]


def load_retrieved_lamp_dataset(
    task: str,
    split: str,
    candidate_retriever: str,
    num_candidates: int
) -> Dataset:
    dataset_dir = (
        Path("data") / task / f"{candidate_retriever}-{num_candidates}" / split
    )

    if not dataset_dir.exists():
        dataset = load_lamp_dataset(task, split)
        retrieval_fn = create_retrieval_fn(candidate_retriever)
        examples = []

        for example in tqdm(dataset, desc="Retrieving"):
            query = example["query"]
            corpus = example["corpus"]
            profile = example["profile"]

            retrieved_indices = retrieval_fn(
                query, corpus, list(range(len(corpus))), num_candidates
            )
            example["corpus"] = [corpus[i] for i in retrieved_indices]
            example["profile"] = [profile[i] for i in retrieved_indices]
            examples.append(example)

        Dataset.from_list(examples).save_to_disk(dataset_dir)

    return load_from_disk(dataset_dir)


def create_pretokenize_fn(
    encoder_model: str,
    max_query_length: int,
    max_document_length: int
) -> Callable[[LazyBatch], LazyBatch]:
    tokenizer = AutoTokenizer.from_pretrained(encoder_model)

    def preprocess_fn(batch: LazyBatch) -> LazyBatch:
        queries = batch["query"]
        corpora = batch["corpus"]

        query_inputs = tokenizer(
            queries, truncation=True, max_length=max_query_length
        )
        batch["query_inputs"] = [
            {k: v[i] for k, v in query_inputs.items()}
            for i in range(len(queries))
        ]

        batch["corpus_inputs"] = []

        for corpus in corpora:
            corpus_inputs = tokenizer(
                corpus, truncation=True, max_length=max_document_length
            )
            batch["corpus_inputs"].append([
                {k: v[i] for k, v in corpus_inputs.items()}
                for i in range(len(corpus))
            ])

        return batch

    return preprocess_fn


def create_collate_fn(encoder_model: str) -> CollateFn:
    tokenizer = AutoTokenizer.from_pretrained(encoder_model)

    def collate_fn(examples: list[LaMPExample]) -> LaMPBatch:
        sources = [e["source"] for e in examples]
        profiles = [e["profile"] for e in examples]
        targets = [e["target"] for e in examples]

        # Pad `query_inputs` without truncation (done in `preprocess_fn`)
        query_inputs = tokenizer.pad(
            [e["query_inputs"] for e in examples], return_tensors="pt"
        )

        # Split each corpus into batches of 128 documents to save memory
        batched_corpus_inputs = []

        for corpus_inputs in [e["corpus_inputs"] for e in examples]:
            document_batches = []

            for document_inputs in [
                corpus_inputs[i : i + 128]
                for i in range(0, len(corpus_inputs), 128)
            ]:
                document_inputs = tokenizer.pad(
                    document_inputs, return_tensors="pt"
                )
                document_batches.append(document_inputs)

            batched_corpus_inputs.append(document_batches)

        # Create record mask to indicate non-padding records
        max_num_records = max(len(p) for p in profiles)
        record_mask = torch.zeros(
            len(profiles), max_num_records, dtype=torch.bool
        )

        for index, profile in enumerate(profiles):
            record_mask[index, :len(profile)] = 1

        return LaMPBatch(
            source=sources,
            profile=profiles,
            target=targets,
            query_inputs=query_inputs,
            corpus_inputs=batched_corpus_inputs,
            record_mask=record_mask
        )

    return collate_fn
