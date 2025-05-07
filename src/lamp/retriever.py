import random
from typing import Callable

from rank_bm25 import BM25Okapi

import torch
from transformers import AutoTokenizer, AutoModel

from .data_types import Profile


def create_retriever(retriever: str, device: torch.device | None = None) -> (
    Callable[[str, list[str], list[Profile], int], list[Profile]]
):
    if retriever == 'first_k':
        return _first_k_retriever
    elif retriever == 'random':
        return _random_retriever
    elif retriever == 'bm25':
        return _bm25_retriever
    elif retriever == 'contriever':
        contriever = _ContrieverRetriever()
        contriever.to(device)
        return contriever
    else:
        raise ValueError(f'Invalid retriever: {retriever}')


def _first_k_retriever(
    query: str,
    corpus: list[str],
    profiles: list[Profile],
    num_retrieve: int
) -> list[Profile]:
    num_retrieve = min(num_retrieve, len(profiles))
    return profiles[:num_retrieve]


def _random_retriever(
    query: str,
    corpus: list[str],
    profiles: list[Profile],
    num_retrieve: int
) -> list[Profile]:
    num_retrieve = min(num_retrieve, len(profiles))
    return random.choices(profiles, k=num_retrieve)


def _bm25_retriever(
    query: str,
    corpus: list[str],
    profiles: list[Profile],
    num_retrieve: int
) -> list[Profile]:
    num_retrieve = min(num_retrieve, len(profiles))
    tokenized_query = query.split()
    tokenized_corpus = [document.split() for document in corpus]
    return BM25Okapi(tokenized_corpus).get_top_n(tokenized_query, profiles, n=num_retrieve)


class _ContrieverRetriever:

    def __init__(self) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained('facebook/contriever')
        self.contriever = AutoModel.from_pretrained('facebook/contriever')
        self.contriever.eval()

    def to(self, device: str) -> None:
        self.contriever.to(device)

    @torch.no_grad()
    def __call__(
        self,
        query: str,
        corpus: list[str],
        profiles: list[Profile],
        num_retrieve: int
    ) -> list[Profile]:
        num_retrieve = min(num_retrieve, len(profiles))

        query_embedding = self._compute_sentence_embeddings([query])
        corpus_embeddings = self._compute_sentence_embeddings(corpus)
        scores = (query_embedding @ corpus_embeddings.T).squeeze(dim=0)

        _, indices = scores.topk(num_retrieve, dim=0)
        return [profiles[index] for index in indices]

    def _compute_sentence_embeddings(self, sentences: list[str]) -> torch.Tensor:
        inputs = self.tokenizer(sentences, padding=True, truncation=True, return_tensors='pt')
        inputs = inputs.to(self.contriever.device)

        outputs = self.contriever(**inputs)
        token_embeddings = outputs.last_hidden_state
        attention_mask = inputs['attention_mask'].unsqueeze(dim=2)

        token_embeddings.masked_fill_(attention_mask == 0, value=0.)
        sentence_embeddings = token_embeddings.sum(dim=1) / attention_mask.sum(dim=1)
        return sentence_embeddings
