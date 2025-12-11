from .in_context_reranker.in_context_reranker import InContextReranker


class ICR:

    def __init__(self, sliding_window_size: int = 10) -> None:
        self.icr = InContextReranker(
            "meta-llama/Meta-Llama-3-8B-Instruct",
            scoring_strategy="masked_NA_calibration",
            retrieval_type="IE",
            sliding_window_size=sliding_window_size
        )

    def __call__(
        self,
        query: str,
        corpus: list[str],
        profile: list[dict[str, str]],
        num_retrieve: int
    ) -> list[dict[str, str]]:
        assert len(corpus) == len(profile) != 0
        num_retrieve = min(num_retrieve, len(profile))

        corpus = [doc.strip() for doc in corpus]
        (ranking, _), _ = self.icr.rerank(query, corpus)
        retrieved_profile = [profile[idx] for idx in ranking[:num_retrieve]]
        return retrieved_profile
