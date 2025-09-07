import torch
from rank_bm25 import BM25Okapi

from ...data_types import Profile
from .in_context_reranker import InContextReranker


class ICR:

    def __init__(self, device: torch.device) -> None:
        self.icr = InContextReranker(
            base_llm_name='meta-llama/Meta-Llama-3-8B-Instruct',
            scoring_strategy='masked_NA_calibration',
            retrieval_type='IE',
            sliding_window_size=10
        )
        self.icr.llm.to(device)

    def __call__(
        self,
        query: str,
        corpus: list[str],
        profiles: list[Profile],
        num_retrieve: int
    ) -> list[Profile]:
        # Retrieve 20 profiles using BM25
        bm25 = BM25Okapi([document.split() for document in corpus])
        retrieved_indices = bm25.get_top_n(query.split(), range(len(profiles)), n=min(20, len(profiles)))

        retrieved_corpus = [corpus[index].strip() for index in retrieved_indices]
        retrieved_profiles = [profiles[index] for index in retrieved_indices]

        # Rerank retrieved profiles
        (ranking, _), _ = self.icr.rerank(query, retrieved_corpus)
        return [retrieved_profiles[rank] for rank in ranking[:num_retrieve]]
