from typing import Callable

from rank_bm25 import BM25Okapi

import torch
from transformers import AutoTokenizer, AutoModel


def create_retriever(retriever: str, device: str | None = None) -> Callable[
    [str, list[dict[str, str]], int, Callable[[str, list[dict[str, str]]], tuple[str, list[str]]]],
    list[str]
]:
    if retriever == 'contriever':
        contriever = _ContrieverRetriever()
        contriever.to(device)
        return contriever

    retriever_fns = {
        'first_k': _first_k_retriever,
        'bm25': _bm25_retriever
    }
    return retriever_fns[retriever]


def _first_k_retriever(input_, profiles, n_retrieve, query_corpus_generator):
    n_retrieve = min(n_retrieve, len(profiles))
    return profiles[:n_retrieve]


def _bm25_retriever(input_, profiles, n_retrieve, query_corpus_generator):
    n_retrieve = min(n_retrieve, len(profiles))
    query, corpus = query_corpus_generator(input_, profiles)

    tokenized_query = query.split()
    tokenized_corpus = [document.split() for document in corpus]
    return BM25Okapi(tokenized_corpus).get_top_n(tokenized_query, profiles, n=n_retrieve)


class _ContrieverRetriever:

    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained('facebook/contriever')
        self.contriever = AutoModel.from_pretrained('facebook/contriever')
        self.contriever.eval()

    def to(self, device):
        self.contriever.to(device)

    @torch.no_grad()
    def __call__(self, input_, profiles, n_retrieve, query_corpus_generator):
        n_retrieve = min(n_retrieve, len(profiles))
        query, corpus = query_corpus_generator(input_, profiles)

        scores = []
        query_embedding = self._compute_sentence_embeddings(query)

        for batch_corpus in [corpus[i : i + 4] for i in range(0, len(corpus), 4)]:
            batch_corpus_embeddings = self._compute_sentence_embeddings(batch_corpus)
            batch_scores = (query_embedding @ batch_corpus_embeddings.T).squeeze(dim=0)
            scores.append(batch_scores)

        scores = torch.cat(scores, dim=0)
        _, indices = torch.topk(scores, n_retrieve, dim=0)
        return [profiles[index] for index in indices]

    def _compute_sentence_embeddings(self, sentences):
        inputs = self.tokenizer(sentences, padding=True, truncation=True, return_tensors='pt')
        inputs = inputs.to(self.contriever.device)

        outputs = self.contriever(**inputs)
        token_embeddings = outputs.last_hidden_state
        mask = inputs['attention_mask'].unsqueeze(dim=-1)

        token_embeddings = token_embeddings.masked_fill(~mask.bool(), 0.)
        sentence_embeddings = token_embeddings.sum(dim=1) / mask.sum(dim=1)
        return sentence_embeddings
