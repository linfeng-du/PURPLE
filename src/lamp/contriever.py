# Adapted from https://github.com/LaMP-Benchmark/LaMP/blob/main/LaMP/prompts/contriever_retriever.py
import torch
from transformers import AutoTokenizer, AutoModel

from .data_types import Profile


class Contriever:

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

        query_embed = self._compute_sentence_embedding([query])
        corpus_embeds = self._compute_sentence_embedding(corpus)
        scores = (query_embed @ corpus_embeds.T).squeeze(dim=0)

        _, indices = scores.topk(num_retrieve, dim=0)
        return [profiles[index] for index in indices]

    def _compute_sentence_embedding(self, sentences: list[str]) -> torch.Tensor:
        inputs = self.tokenizer(sentences, padding=True, truncation=True, return_tensors='pt')
        inputs = inputs.to(self.contriever.device)
        attention_mask = inputs['attention_mask'].unsqueeze(dim=2)

        token_embeds = self.contriever(**inputs).last_hidden_state
        token_embeds.masked_fill_(attention_mask == 0, value=0.)
        return token_embeds.sum(dim=1) / attention_mask.sum(dim=1)
