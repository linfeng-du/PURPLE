# https://github.com/LaMP-Benchmark/LaMP/blob/main/LaMP/data/datasets.py
# https://github.com/LaMP-Benchmark/LaMP/blob/main/LaMP/prompts/prompts.py
# https://github.com/LongLaMP-benchmark/LongLaMP-Benchmark/blob/main/longLaMP/prompts/prompts.py
import json
from pathlib import Path

from datasets import Dataset, load_dataset, load_from_disk


def load_lamp_dataset(task: str, split: str) -> Dataset:
    dataset_dir = Path("data") / task / split

    if not dataset_dir.exists():
        if task in {"lamp1", "lamp2", "lamp3", "lamp4", "lamp5", "lamp7"}:
            _prepare_lamp_dataset(task, split, dataset_dir)
        elif task in {"longlamp2", "longlamp3", "longlamp4"}:
            _prepare_longlamp_dataset(task, split, dataset_dir)
        else:
            raise ValueError(f"Invalid task: {task}")

    return load_from_disk(dataset_dir)


def _prepare_lamp_dataset(task: str, split: str, dataset_dir: Path) -> None:
    questions_file = Path("data") / task / f"{split}_questions.json"
    outputs_file = Path("data") / task / f"{split}_outputs.json"

    with questions_file.open() as f:
        questions = json.load(f)

    with outputs_file.open() as f:
        outputs = {g["id"]: g["output"] for g in json.load(f)["golds"]}

    query_corpus_fn = QUERY_CORPUS_FNS[task]
    examples = []

    for question in questions:
        source = question["input"]
        profile = question["profile"]
        target = outputs[question["id"]]
        query, corpus = query_corpus_fn(source, profile)

        examples.append({
            "source": source,
            "profile": profile,
            "query": query,
            "corpus": corpus,
            "target": target
        })

    Dataset.from_list(examples).save_to_disk(dataset_dir)


LONGLAMP_USER_SUBSETS = {
    "longlamp2": "abstract_generation_user",
    "longlamp3": "product_review_user",
    "longlamp4": "topic_writing_user"
}


def _prepare_longlamp_dataset(
    task: str,
    split: str,
    dataset_dir: Path
) -> None:
    dataset = load_dataset(
        "LongLaMP/LongLaMP", name=LONGLAMP_USER_SUBSETS[task], split=split
    )

    query_corpus_fn = QUERY_CORPUS_FNS[task]
    examples = []

    for row in dataset:
        source = row["input"]
        profile = row["profile"]
        target = row["output"]
        query, corpus = query_corpus_fn(source, profile)

        examples.append({
            "source": source,
            "profile": profile,
            "query": query,
            "corpus": corpus,
            "target": target
        })

    Dataset.from_list(examples).save_to_disk(dataset_dir)


# [LaMP-1] Personalized Citation Identification
def _classification_citation_query_corpus_fn(
    source: str,
    profile: list[dict[str, str]]
) -> tuple[str, list[str]]:
    reference_1, reference_2 = _extract_references(source)
    query = f"{reference_1} {reference_2}"
    corpus = [f"{rec['title']} {rec['abstract']}" for rec in profile]
    return query, corpus


def _extract_references(source: str) -> tuple[str, str]:
    delimiter_1 = 'Just answer with [1] or [2] without explanation. [1]: "'
    delimiter_2 = '" [2]: "'

    index_1 = source.find(delimiter_1)
    index_2 = source.find(delimiter_2)
    assert index_1 != -1 and index_2 != -1 and source.endswith('"')

    reference_1 = source[index_1 + len(delimiter_1) : index_2]
    reference_2 = source[index_2 + len(delimiter_2) : -1]
    return reference_1, reference_2


# [LaMP-2] Personalized Movie Tagging
def _classification_movie_query_corpus_fn(
    source: str,
    profile: list[dict[str, str]]
) -> tuple[str, list[str]]:
    query = _extract_after(source, "description: ")
    corpus = [rec["description"] for rec in profile]
    return query, corpus


# [LaMP-3] Personalized Product Rating
def _regression_rating_query_corpus_fn(
    source: str,
    profile: list[dict[str, str]]
) -> tuple[str, list[str]]:
    query = _extract_after(source, "review: ")
    corpus = [rec["text"] for rec in profile]
    return query, corpus


# [LaMP-4] Personalized News Headline Generation
def _generation_news_query_corpus_fn(
    source: str,
    profile: list[dict[str, str]]
) -> tuple[str, list[str]]:
    query = _extract_after(source, "article: ")
    corpus = [f"{rec['title']} {rec['text']}" for rec in profile]
    return query, corpus


# [LaMP-5] Personalized Scholarly Title Generation
def _generation_scholar_query_corpus_fn(
    source: str,
    profile: list[dict[str, str]]
) -> tuple[str, list[str]]:
    query = _extract_after(source, "paper: ")
    corpus = [f"{rec['title']} {rec['abstract']}" for rec in profile]
    return query, corpus


# [LaMP-7] Personalized Tweet Paraphrasing
def _generation_tweet_query_corpus_fn(
    source: str,
    profile: list[dict[str, str]]
) -> tuple[str, list[str]]:
    query = _extract_after(source, ": ")
    corpus = [rec["text"] for rec in profile]
    return query, corpus


# [LongLaMP-2] Personalized Abstract Generation
def _generation_abstract_query_corpus_fn(
    source: str,
    profile: list[dict[str, str]]
) -> tuple[str, list[str]]:
    query = _extract_after(source, "items: ")
    corpus = [f"{rec['title']} {rec['abstract']}" for rec in profile]
    return query, corpus


# [LongLaMP-3] Personalized Product Review Generation
def _generation_review_query_corpus_fn(
    source: str,
    profile: list[dict[str, str]]
) -> tuple[str, list[str]]:
    corpus = [
        (
            f"{rec['overall']} {rec['summary']} "
            f"{rec['description']} {rec['reviewText']}"
        )
        for rec in profile
    ]
    return source, corpus


# [LongLaMP-4] Personalized Topic Generation
def _generation_topic_query_corpus_fn(
    source: str,
    profile: list[dict[str, str]]
) -> tuple[str, list[str]]:
    corpus = [f"{rec['summary']} {rec['content']}" for rec in profile]
    return source, corpus


def _extract_after(source: str, delimiter: str) -> str:
    index = source.find(delimiter)
    assert index != -1
    return source[index + len(delimiter):]


QUERY_CORPUS_FNS = {
    "lamp1": _classification_citation_query_corpus_fn,
    "lamp2": _classification_movie_query_corpus_fn,
    "lamp3": _regression_rating_query_corpus_fn,
    "lamp4": _generation_news_query_corpus_fn,
    "lamp5": _generation_scholar_query_corpus_fn,
    "lamp7": _generation_tweet_query_corpus_fn,
    "longlamp2": _generation_abstract_query_corpus_fn,
    "longlamp3": _generation_review_query_corpus_fn,
    "longlamp4": _generation_topic_query_corpus_fn
}
