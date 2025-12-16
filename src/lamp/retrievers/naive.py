import random


def first_k_retriever(
    _query: str,
    _corpus: list[str],
    profile: list[dict[str, str]],
    num_retrieve: int
) -> list[dict[str, str]]:
    assert profile
    num_retrieve = min(num_retrieve, len(profile))
    return profile[:num_retrieve]


def random_retriever(
    _query: str,
    _corpus: list[str],
    profile: list[dict[str, str]],
    num_retrieve: int
) -> list[dict[str, str]]:
    assert profile
    num_retrieve = min(num_retrieve, len(profile))
    return random.sample(profile, num_retrieve)
