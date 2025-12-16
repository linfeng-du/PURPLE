from rank_bm25 import BM25Okapi


def bm25_retriever(
    query: str,
    corpus: list[str],
    profile: list[dict[str, str]],
    num_retrieve: int
) -> list[dict[str, str]]:
    assert len(corpus) == len(profile) != 0
    num_retrieve = min(num_retrieve, len(profile))
    bm25 = BM25Okapi([doc.split() for doc in corpus])
    return bm25.get_top_n(query.split(), profile, n=num_retrieve)
