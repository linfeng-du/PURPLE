"""Retrievers for the retrieval-augmented generation pipeline."""

from rank_bm25 import BM25Okapi

import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel


def create_retriever(retriever_name, device=None):
    if retriever_name == 'contriever':
        contriever = _ContrieverRetriever()
        contriever.to(device)
        contriever.eval()
        return contriever.contriever_retriever

    retriever_fns = {
        'first_k': _first_k_retriever,
        'bm25': _bm25_retriever,
    }
    return retriever_fns[retriever_name]


def _first_k_retriever(inp, profile, num_retrieve, *args):
    num_retrieve = min(num_retrieve, len(profile))
    return profile[:num_retrieve]


def _bm25_retriever(inp, profile, num_retrieve, query_corpus_generator):
    num_retrieve = min(num_retrieve, len(profile))
    query, corpus = query_corpus_generator(inp, profile)

    tokenized_query = query.split()
    tokenized_corpus = [d.split() for d in corpus]
    bm25 = BM25Okapi(tokenized_corpus)
    return bm25.get_top_n(tokenized_query, profile, n=num_retrieve)


class _ContrieverRetriever(nn.Module):

    def __init__(self):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained('facebook/contriever')
        self.contriever = AutoModel.from_pretrained('facebook/contriever')

    @torch.no_grad()
    def contriever_retriever(self, inp, profile, num_retrieve, query_corpus_generator):
        num_retrieve = min(num_retrieve, len(profile))
        query, corpus = query_corpus_generator(inp, profile)

        scores = []
        query_embedding = self._compute_sentence_embeddings(query)
        corpus_embeddings = self._compute_sentence_embeddings(corpus)
        scores = (query_embedding @ corpus_embeddings.T).squeeze(dim=0)

        _, indices = torch.topk(scores, num_retrieve)
        return [profile[i] for i in indices]

    def _compute_sentence_embeddings(self, sentences):
        device = next(self.parameters()).device

        inputs = self.tokenizer(sentences, padding=True, truncation=True, return_tensors='pt')
        inputs = inputs.to(device)
        outputs = self.contriever(**inputs)

        token_embeddings = outputs.last_hidden_state
        mask = inputs['attention_mask'].unsqueeze(dim=-1)
        token_embeddings = token_embeddings.masked_fill(~mask.bool(), 0.)
        sentence_embeddings = token_embeddings.sum(dim=1) / mask.sum(dim=1)
        return sentence_embeddings

    @staticmethod
    def _batchify(lst, batch_size):
        return [lst[i:i + batch_size] for i in range(0, len(lst), batch_size)]
