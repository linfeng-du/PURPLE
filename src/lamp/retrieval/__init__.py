from collections.abc import Callable
from typing import Any


RetrievalFn = Callable[
    [str, list[str], list[dict[str, str]], int], list[dict[str, str]]
]


def create_retrieval_fn(retriever: str, **kwargs: Any) -> RetrievalFn:
    if retriever == "first_k":
        from .retrievers import first_k_retriever
        return first_k_retriever

    elif retriever == "random":
        from .retrievers import random_retriever
        return random_retriever

    elif retriever == "bm25":
        from .retrievers import bm25_retriever
        return bm25_retriever

    elif retriever == "contriever":
        from .retrievers import Contriever
        return Contriever()

    elif retriever == "rank_gpt-llama3":
        from .rerankers import RankGPT
        return RankGPT("meta-llama/Meta-Llama-3-8B-Instruct", "hf", **kwargs)

    elif retriever == "rank_gpt-gpt5":
        from .rerankers import RankGPT
        return RankGPT("gpt-5-nano", "openai", **kwargs)

    elif retriever == "icr":
        from .rerankers import ICR
        return ICR(**kwargs)

    else:
        raise ValueError(f"Invalid retriever: {retriever}")
