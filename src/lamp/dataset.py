# Adapted from https://github.com/LaMP-Benchmark/LaMP/blob/main/LaMP/data/datasets.py
# and https://github.com/LaMP-Benchmark/LaMP/blob/main/LaMP/prompts/prompts.py
import json
import os

from datasets import Dataset, load_dataset, load_from_disk

from .data_types import Profile


def load_lamp_dataset(task: str, split: str) -> Dataset:
    dataset_dir = f'./dataset/{task}/{split}'

    if not os.path.exists(dataset_dir):
        if task.startswith('LaMP'):
            _process_lamp_dataset(task, split, dataset_dir)
        elif task.startswith('LongLaMP'):
            _process_long_lamp_dataset(task, split, dataset_dir)
        else:
            raise ValueError(f'Invalid task: {task}')

    return load_from_disk(dataset_dir)


def _process_lamp_dataset(task: str, split: str, cache_dir: str) -> None:
    with open(f'./dataset/{task}/{split}_questions.json', 'r') as file:
        questions = json.load(file)

    with open(f'./dataset/{task}/{split}_outputs.json', 'r') as file:
        outputs = {gold['id']: gold['output'] for gold in json.load(file)['golds']}

    query_corpus_generator = {
        'LaMP-1': _generate_query_corpus_classification_citation,
        'LaMP-2': _generate_query_corpus_classification_movies,
        'LaMP-3': _generate_query_corpus_regression_review,
        'LaMP-4': _generate_query_corpus_generation_news,
        'LaMP-5': _generate_query_corpus_generation_paper,
        'LaMP-6': _generate_query_corpus_generation_avocado,
        'LaMP-7': _generate_query_corpus_generation_tweet
    }[task]
    examples = []

    for question in questions:
        source = question['input']
        profiles = question['profile']
        target = outputs[question['id']]
        query, corpus = query_corpus_generator(source, profiles)

        example = {
            'source': source,
            'profiles': profiles,
            'query': query,
            'corpus': corpus,
            'target': target
        }
        examples.append(example)

    Dataset.from_list(examples).save_to_disk(cache_dir)


def _process_long_lamp_dataset(task: str, split: str, cache_dir: str) -> None:
    name = {
        'LongLaMP-2': 'abstract_generation_user',
        'LongLaMP-3': 'topic_writing_user',
        'LongLaMP-4': 'product_review_user'
    }[task]
    dataset = load_dataset('LongLaMP/LongLaMP', name=name, split=split)

    query_corpus_generator = {
        'LongLaMP-1': _generate_query_corpus_generation_email,
        'LongLaMP-2': _generate_query_corpus_generation_abstract,
        'LongLaMP-3': _generate_query_corpus_generation_topic,
        'LongLaMP-4': _generate_query_corpus_generation_review
    }[task]
    examples = []

    for row in dataset:
        source = row['input']
        profiles = row['profile']
        target = row['output']
        query, corpus = query_corpus_generator(source, profiles)

        example = {
            'source': source,
            'profiles': profiles,
            'query': query,
            'corpus': corpus,
            'target': target
        }
        examples.append(example)

    Dataset.from_list(examples).save_to_disk(cache_dir)


# ==================================   LaMP 1: Personalized Citation Identification   ==================================
def _generate_query_corpus_classification_citation(source: str, profiles: list[Profile]) -> tuple[str, list[str]]:
    reference_1, reference_2 = _extract_references(source)
    query = f'{reference_1} {reference_2}'
    corpus = [f'{profile["title"]} {profile["abstract"]}' for profile in profiles]
    return query, corpus


# ==================================        LaMP 2: Personalized Movie Tagging        ==================================
def _generate_query_corpus_classification_movies(source: str, profiles: list[Profile]) -> tuple[str, list[str]]:
    query = _extract_string_after_keyword(source, 'description: ')
    corpus = [profile['description'] for profile in profiles]
    return query, corpus


# ==================================       LaMP 3: Personalized Product Rating       ==================================
def _generate_query_corpus_regression_review(source: str, profiles: list[Profile]) -> tuple[str, list[str]]:
    query = _extract_string_after_keyword(source, 'review: ')
    corpus = [profile['text'] for profile in profiles]
    return query, corpus


# ==================================  LaMP 4: Personalized News Headline Generation  ==================================
def _generate_query_corpus_generation_news(source: str, profiles: list[Profile]) -> tuple[str, list[str]]:
    query = _extract_string_after_keyword(source, 'article: ')
    corpus = [f'{profile["title"]} {profile["text"]}' for profile in profiles]
    return query, corpus


# ================================== LaMP 5: Personalized Scholarly Title Generation ==================================
def _generate_query_corpus_generation_paper(source: str, profiles: list[Profile]) -> tuple[str, list[str]]:
    query = _extract_string_after_keyword(source, 'paper: ')
    corpus = [f'{profile["title"]} {profile["abstract"]}' for profile in profiles]
    return query, corpus


# ==================================  LaMP 6: Personalized Email Subject Generation  ==================================
def _generate_query_corpus_generation_avocado(source: str, profiles: list[Profile]) -> tuple[str, list[str]]:
    query = _extract_string_after_keyword(source, ': ')
    corpus = [profile['text'] for profile in profiles]
    return query, corpus


# ==================================     LaMP 7: Personalized Tweet Paraphrasing     ==================================
def _generate_query_corpus_generation_tweet(source: str, profiles: list[Profile]) -> tuple[str, list[str]]:
    query = _extract_string_after_keyword(source, ': ')
    corpus = [profile['text'] for profile in profiles]
    return query, corpus


# =================================     LongLaMP 1: Personalized Email Completion     =================================
def _generate_query_corpus_generation_email(source: str, profiles: list[Profile]) -> tuple[str, list[str]]:
    query = _extract_string_after_keyword(source, ': ')
    corpus = [profile['text'] for profile in profiles]
    return query, corpus


# ================================     LongLaMP 2: Personalized Abstract Generation     ================================
def _generate_query_corpus_generation_abstract(source: str, profiles: list[Profile]) -> tuple[str, list[str]]:
    query = _extract_string_after_keyword(source, 'items: ')
    corpus = [f'{profile["title"]} {profile["abstract"]}' for profile in profiles]
    return query, corpus


# =================================     LongLaMP 3: Personalized Topic Generation     =================================
def _generate_query_corpus_generation_topic(source: str, profiles: list[Profile]) -> tuple[str, list[str]]:
    corpus = [f'{profile["content"]} {profile["summary"]}' for profile in profiles]
    return source, corpus


# =============================     LongLaMP 4: Personalized Product Review Generation     =============================
def _generate_query_corpus_generation_review(source: str, profiles: list[Profile]) -> tuple[str, list[str]]:
    corpus = [
        f'{profile["overall"]} {profile["summary"]} {profile["description"]} {profile["reviewText"]}'
        for profile in profiles
    ]
    return source, corpus


# ==================================                Utility Functions                ==================================
def _extract_references(source: str) -> tuple[str, str]:
    template_1 = 'Just answer with [1] or [2] without explanation. [1]: "'
    template_2 = '" [2]: "'

    template_1_start = source.find(template_1)
    template_2_start = source.find(template_2)
    assert template_1_start != -1 and template_2_start != -1 and source.endswith('"')

    reference_1 = source[template_1_start + len(template_1) : template_2_start]
    reference_2 = source[template_2_start + len(template_2) : -1]
    return reference_1, reference_2


def _extract_string_after_keyword(source: str, keyword: str) -> str:
    keyword_start = source.find(keyword)

    if keyword_start == -1:
        raise ValueError(f'Keyword "{keyword}" not found in input')

    return source[keyword_start + len(keyword):]
