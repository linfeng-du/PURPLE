import random
from typing import Any, Callable, TypeAlias

from rank_bm25 import BM25Okapi

from .contriever import Contriever
from .icr import ICR
from .rank_gpt import RankGPT


RetrieverFn: TypeAlias = Callable[
    [str | None, list[str] | None, list[dict[str, str]], int],
    list[dict[str, str]]
]


def create_retriever_fn(retriever: str, **kwargs: Any) -> RetrieverFn:
    if retriever == "first_k":
        return lambda _query, _corpus, profile, num_retrieve: (
            profile[:num_retrieve]
        )

    elif retriever == "random":
        return lambda _query, _corpus, profile, num_retrieve: (
            random.sample(profile, min(num_retrieve, len(profile)))
        )

    elif retriever == "bm25":
        return lambda query, corpus, profile, num_retrieve: (
            BM25Okapi([doc.split() for doc in corpus])
            .get_top_n(query.split(), profile, n=num_retrieve)
        )
            
    elif retriever == "contriever":
        return Contriever(**kwargs)

    elif retriever == "rank_gpt-llama3":
        return RankGPT("meta-llama/Meta-Llama-3-8B-Instruct", "hf", **kwargs)

    elif retriever == "rank_gpt-gpt5":
        return RankGPT("gpt-5-nano", "openai", **kwargs)

    elif retriever == "icr":
        return ICR(**kwargs)

    else:
        raise ValueError(f"Invalid retriever: {retriever}")
