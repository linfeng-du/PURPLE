# Adapted from: https://github.com/LaMP-Benchmark/LaMP/blob/main/LaMP/prompts/contriever_retriever.py
import torch
from transformers import AutoModel, AutoTokenizer


class Contriever:

    def __init__(self) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained("facebook/contriever")
        self.contriever = AutoModel.from_pretrained("facebook/contriever")
        self.contriever.to("cuda" if torch.cuda.is_available() else "cpu")
        self.contriever.eval()

    def __call__(
        self,
        query: str,
        corpus: list[str],
        profile: list[dict[str, str]],
        num_retrieve: int
    ) -> list[dict[str, str]]:
        retrieved_profile, _ = self.retrieve_with_logps(
            query, corpus, profile, num_retrieve
        )
        return retrieved_profile

    @torch.no_grad()
    def retrieve_with_logps(
        self,
        query: str,
        corpus: list[str],
        profile: list[dict[str, str]],
        num_retrieve: int
    ) -> tuple[list[dict[str, str]], torch.Tensor]:
        assert len(corpus) == len(profile) != 0
        num_retrieve = min(num_retrieve, len(profile))

        query_emb = self._compute_sentence_embedding([query])
        scores = []

        for corpus_batch in [
            corpus[i:i+128] for i in range(0, len(corpus), 128)
        ]:
            corpus_embs = self._compute_sentence_embedding(corpus_batch)
            scores_batch = (query_emb @ corpus_embs.T).squeeze(dim=0)
            scores.append(scores_batch)

        scores = torch.cat(scores)
        logits, indices = scores.topk(num_retrieve)

        logps = logits.log_softmax(dim=-1)
        retrieved_profile = [profile[i] for i in indices]
        return retrieved_profile, logps

    def _compute_sentence_embedding(self, sentences: list[str]) -> torch.Tensor:
        inputs = self.tokenizer(
            sentences, padding=True, truncation=True, return_tensors="pt"
        )
        inputs = inputs.to(self.contriever.device)
        mask = inputs["attention_mask"].unsqueeze(dim=-1)

        token_embs = self.contriever(**inputs).last_hidden_state
        sentence_embs = (token_embs * mask).sum(dim=1) / mask.sum(dim=1)
        return sentence_embs
