from collections.abc import Callable
from typing import Any

from .rerankers import ICR, RankGPT
from .retrievers import Contriever, bm25_retriever, first_k_retriever


RetrievalFn = Callable[
    [str, list[str], list[dict[str, str]], int], list[dict[str, str]]
]


def create_retrieval_fn(retriever: str, **kwargs: Any) -> RetrievalFn:
    if retriever == "first_k":
        return first_k_retriever
    elif retriever == "bm25":
        return bm25_retriever
    elif retriever == "contriever":
        return Contriever()
    elif retriever == "rank_gpt-llama3":
        return RankGPT(
            model="meta-llama/Meta-Llama-3-8B-Instruct", backend="hf", **kwargs
        )
    elif retriever == "rank_gpt-gpt5":
        return RankGPT(model="gpt-5-nano", backend="openai", **kwargs)
    elif retriever == "icr":
        return ICR(**kwargs)
    else:
        raise ValueError(f"Invalid retriever: {retriever}")
