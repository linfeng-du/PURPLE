# Adapted from https://github.com/LaMP-Benchmark/LaMP/blob/main/LaMP/prompts/contriever_retriever.py
import torch
from transformers import AutoModel, AutoTokenizer

from ..data_types import Profile


class Contriever:

    def __init__(self) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained('facebook/contriever')
        self.contriever = AutoModel.from_pretrained('facebook/contriever')
        self.contriever.to('cuda')
        self.contriever.eval()

    @torch.no_grad()
    def __call__(
        self,
        query: str,
        corpus: list[str],
        profiles: list[Profile],
        num_retrieve: int,
        return_logps: bool = False
    ) -> list[Profile] | tuple[list[Profile], torch.Tensor]:
        num_retrieve = min(num_retrieve, len(profiles))

        scores = []
        query_embed = self._compute_sentence_embedding([query])

        for batch_corpus in [corpus[i:i+128] for i in range(0, len(corpus), 128)]:
            batch_corpus_embeds = self._compute_sentence_embedding(batch_corpus)
            batch_scores = (query_embed @ batch_corpus_embeds.T).squeeze(dim=0)
            scores.append(batch_scores)

        scores = torch.cat(scores, dim=0)
        values, indices = scores.topk(num_retrieve, dim=0)
        retrieved_profiles = [profiles[index] for index in indices]

        if return_logps:
            logps = torch.log_softmax(values, dim=0)
            return retrieved_profiles, logps

        return retrieved_profiles

    def _compute_sentence_embedding(self, sentences: list[str]) -> torch.Tensor:
        inputs = self.tokenizer(sentences, padding=True, truncation=True, return_tensors='pt')
        inputs = inputs.to(self.contriever.device)
        attention_mask = inputs['attention_mask'].unsqueeze(dim=2)

        token_embeds = self.contriever(**inputs).last_hidden_state
        token_embeds.masked_fill_(attention_mask == 0, value=0.)
        return token_embeds.sum(dim=1) / attention_mask.sum(dim=1)
