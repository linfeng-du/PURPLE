from .in_context_reranker import InContextReranker
from ..data_types import Profile


class ICR:

    def __init__(self) -> None:
        self.icr = InContextReranker(
            base_llm_name='meta-llama/Meta-Llama-3-8B-Instruct',
            scoring_strategy='masked_NA_calibration',
            retrieval_type='IE',
            sliding_window_size=10
        )

    def __call__(self, query: str, corpus: list[str], profiles: list[Profile], num_rerank: int) -> (
        list[Profile]
    ):
        corpus = [document.strip() for document in corpus]
        (ranking, _), _ = self.icr.rerank(query, corpus)
        return [profiles[rank] for rank in ranking[:num_rerank]]
