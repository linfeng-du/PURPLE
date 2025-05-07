import os
import json
from typing import Callable

from datasets import Dataset, load_from_disk

from .data_types import Profile


def load_lamp_dataset(task: str, split: str) -> Dataset:
    dataset_dir = f'./dataset/{task}/{split}'

    if not os.path.exists(dataset_dir):
        with open(f'./dataset/{task}/{split}_questions.json', 'r') as file:
            questions = json.load(file)

        with open(f'./dataset/{task}/{split}_outputs.json', 'r') as file:
            outputs = {gold['id']: gold['output'] for gold in json.load(file)['golds']}

        examples = []
        query_corpus_generator = _create_query_corpus_generator(task)

        for question in questions:
            source = question['input']
            profiles = question['profile']
            target = outputs[question['id']]
            query, corpus = query_corpus_generator(source, profiles)

            example = {'source': source, 'profiles': profiles, 'query': query, 'corpus': corpus, 'target': target}
            examples.append(example)

        Dataset.from_list(examples).save_to_disk(dataset_dir)

    return load_from_disk(dataset_dir)


def _create_query_corpus_generator(task: str) -> Callable[[str, list[Profile]], tuple[str, list[str]]]:
    task_fns = {
        'LaMP-1': _generate_query_corpus_classification_citation,
        'LaMP-2': _generate_query_corpus_classification_movies,
        'LaMP-3': _generate_query_corpus_regression_review,
        'LaMP-4': _generate_query_corpus_generation_news,
        'LaMP-5': _generate_query_corpus_generation_paper,
        'LaMP-6': _generate_query_corpus_generation_avocado,
        'LaMP-7': _generate_query_corpus_paraphrase_tweet
    }
    return task_fns[task]


# ==================================   LaMP 1: Personalized Citation Identification   ==================================
def _generate_query_corpus_classification_citation(input_: str, profiles: list[Profile]) -> tuple[str, list[str]]:
    reference_1, reference_2 = _extract_references(input_)
    query = f'{reference_1} {reference_2}'
    corpus = [f'{profile["title"]} {profile["abstract"]}' for profile in profiles]
    return query, corpus


# ==================================        LaMP 2: Personalized Movie Tagging        ==================================
def _generate_query_corpus_classification_movies(input_: str, profiles: list[Profile]) -> tuple[str, list[str]]:
    query = _extract_string_after_keyword(input_, 'description: ')
    corpus = [profile['description'] for profile in profiles]
    return query, corpus


# ==================================       LaMP 3: Personalized Product Rating       ==================================
def _generate_query_corpus_regression_review(input_: str, profiles: list[Profile]) -> tuple[str, list[str]]:
    query = _extract_string_after_keyword(input_, 'review: ')
    corpus = [profile['text'] for profile in profiles]
    return query, corpus


# ==================================  LaMP 4: Personalized News Headline Generation  ==================================
def _generate_query_corpus_generation_news(input_: str, profiles: list[Profile]) -> tuple[str, list[str]]:
    query = _extract_string_after_keyword(input_, 'article: ')
    corpus = [f'{profile["title"]} {profile["text"]}' for profile in profiles]
    return query, corpus


# ================================== LaMP 5: Personalized Scholarly Title Generation ==================================
def _generate_query_corpus_generation_paper(input_: str, profiles: list[Profile]) -> tuple[str, list[str]]:
    query = _extract_string_after_keyword(input_, 'paper: ')
    corpus = [f'{profile["title"]} {profile["abstract"]}' for profile in profiles]
    return query, corpus


# ==================================  LaMP 6: Personalized Email Subject Generation  ==================================
def _generate_query_corpus_generation_avocado(input_: str, profiles: list[Profile]) -> tuple[str, list[str]]:
    query = _extract_string_after_keyword(input_, ': ')
    corpus = [profile['text'] for profile in profiles]
    return query, corpus


# ==================================     LaMP 7: Personalized Tweet Paraphrasing     ==================================
def _generate_query_corpus_paraphrase_tweet(input_: str, profiles: list[Profile]) -> tuple[str, list[str]]:
    query = _extract_string_after_keyword(input_, ': ')
    corpus = [profile['text'] for profile in profiles]
    return query, corpus


# ==================================                Utility Functions                ==================================
def _extract_references(input_: str) -> tuple[str, str]:
    template_1 = 'Just answer with [1] or [2] without explanation. [1]: "'
    template_2 = '" [2]: "'

    template_1_start = input_.find(template_1)
    template_2_start = input_.find(template_2)
    assert template_1_start != -1 and template_2_start != -1 and input_.endswith('"')

    reference_1 = input_[template_1_start + len(template_1) : template_2_start]
    reference_2 = input_[template_2_start + len(template_2) : -1]
    return reference_1, reference_2


def _extract_string_after_keyword(input_: str, keyword: str) -> str:
    keyword_start = input_.find(keyword)

    if keyword_start == -1:
        raise ValueError(f'Keyword "{keyword}" not found in input')

    return input_[keyword_start + len(keyword):]
