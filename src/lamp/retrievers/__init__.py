import random
from collections.abc import Callable
from typing import Any

from rank_bm25 import BM25Okapi

from .contriever import Contriever
from .icr import ICR
from .rank_gpt import RankGPT


RetrieverFn = (
    Callable[[str, list[str], list[dict[str, str]], int], list[dict[str, str]]]
    | Callable[[None, None, list[dict[str, str]], int], list[dict[str, str]]]
)


def create_retriever_fn(retriever: str, **kwargs: Any) -> RetrieverFn:
    if retriever == "first_k":
        return _first_k_retriever

    elif retriever == "random":
        return _random_retriever

    elif retriever == "bm25":
        return _bm25_retriever

    elif retriever == "contriever":
        return Contriever()

    elif retriever == "rank_gpt-llama3":
        return RankGPT("meta-llama/Meta-Llama-3-8B-Instruct", "hf", **kwargs)

    elif retriever == "rank_gpt-gpt5":
        return RankGPT("gpt-5-nano", "openai", **kwargs)

    elif retriever == "icr":
        return ICR(**kwargs)

    else:
        raise ValueError(f"Invalid retriever: {retriever}")


def _first_k_retriever(
    _query: None,
    _corpus: None,
    profile: list[dict[str, str]],
    num_retrieve: int
) -> list[dict[str, str]]:
    assert profile
    num_retrieve = min(num_retrieve, len(profile))
    return profile[:num_retrieve]


def _random_retriever(
    _query: None,
    _corpus: None,
    profile: list[dict[str, str]],
    num_retrieve: int
) -> list[dict[str, str]]:
    assert profile
    num_retrieve = min(num_retrieve, len(profile))
    return random.sample(profile, num_retrieve)


def _bm25_retriever(
    query: str,
    corpus: list[str],
    profile: list[dict[str, str]],
    num_retrieve: int
) -> list[dict[str, str]]:
    assert len(corpus) == len(profile) != 0
    num_retrieve = min(num_retrieve, len(profile))
    bm25 = BM25Okapi([doc.split() for doc in corpus])
    return bm25.get_top_n(query.split(), profile, n=num_retrieve)
