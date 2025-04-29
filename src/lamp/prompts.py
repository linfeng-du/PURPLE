# Adapted from https://github.com/LaMP-Benchmark/LaMP/blob/main/LaMP/prompts/prompts.py
import logging
from typing import Callable

from transformers import PreTrainedTokenizerBase

from .retrievers import create_retriever
from .data_types import Profile, PromptGenerator, QueryCorpusGenerator


logger = logging.getLogger(__name__)


def create_prompt_generator(
    task: str,
    retriever: str,
    num_retrieve: int,
    max_prompt_length: int,
    tokenizer: PreTrainedTokenizerBase,
    device: str | None = None
) -> PromptGenerator:
    retriever = create_retriever(retriever, device=device)
    query_corpus_generator = create_query_corpus_generator(task)
    prompt_generator = _create_prompt_generator(task)

    def retrieval_augmented_prompt_generator(input_: str, profiles: list[Profile], factor: float = 0.6) -> str:
        retrieved_profiles = retriever(input_, profiles, num_retrieve, query_corpus_generator)
        input_length = len(tokenizer.encode(input_, truncation=True, max_length=max_prompt_length))

        while True:
            try:
                reserved_length = min(input_length, int(factor * max_prompt_length))
                max_profile_length = max_prompt_length - reserved_length
                prompt = prompt_generator(input_, retrieved_profiles, max_profile_length, tokenizer)
                return prompt
            except OverflowError:
                factor -= 0.1

                if factor < 0:
                    logger.warning(f'Encountered long input ({input_length} tokens), returning as is')
                    return input_

    return retrieval_augmented_prompt_generator


def create_query_corpus_generator(task: str) -> QueryCorpusGenerator:
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


def _create_prompt_generator(task: str) -> Callable[[str, list[Profile], int, PreTrainedTokenizerBase], str]:
    task_fns = {
        'LaMP-1': _generate_prompt_classification_citation,
        'LaMP-2': _generate_prompt_classification_movies,
        'LaMP-3': _generate_prompt_regression_review,
        'LaMP-4': _generate_prompt_generation_news,
        'LaMP-5': _generate_prompt_generation_paper,
        'LaMP-6': _generate_prompt_generation_avocado,
        'LaMP-7': _generate_prompt_paraphrase_tweet
    }
    return task_fns[task]


# ==================================   LaMP 1: Personalized Citation Identification   ==================================
def _generate_query_corpus_classification_citation(input_: str, profiles: list[Profile]) -> tuple[str, list[str]]:
    reference_1, reference_2 = _extract_references(input_)
    query = f'{reference_1} {reference_2}'
    corpus = [f'{profile["title"]} {profile["abstract"]}' for profile in profiles]
    return query, corpus


def _generate_prompt_classification_citation(
    input_: str,
    profiles: list[Profile],
    max_length: int,
    tokenizer: PreTrainedTokenizerBase
) -> str:
    template_length = 2 * len(profiles)
    max_length_per_profile = (max_length - template_length) // len(profiles)

    profile_prompts = []
    saved_length = 0

    for profile in profiles:
        profile_template_length = 2
        max_profile_length = max_length_per_profile + saved_length - profile_template_length

        input_ids = tokenizer.encode(
            profile['title'],
            add_special_tokens=False,
            truncation=True,
            max_length=max_profile_length
        )
        new_title = tokenizer.decode(input_ids, skip_special_tokens=True)
        profile_prompt = f'"{new_title}"'

        profile_prompts.append(profile_prompt)
        saved_length += max_length_per_profile - profile_template_length - len(input_ids)

    title_start = input_.find('title')
    prompt = f'{input_[:title_start + 5]}, and {", and ".join(profile_prompts)}{input_[title_start + 5:]}'
    return prompt


# ==================================        LaMP 2: Personalized Movie Tagging        ==================================
def _generate_query_corpus_classification_movies(input_: str, profiles: list[Profile]) -> tuple[str, list[str]]:
    query = _extract_string_after_keyword(input_, 'description: ')
    corpus = [profile['description'] for profile in profiles]
    return query, corpus


def _generate_prompt_classification_movies(
    input_: str,
    profiles: list[Profile],
    max_length: int,
    tokenizer: PreTrainedTokenizerBase
) -> str:
    template_length = 2 * (len(profiles) - 1) + 1
    max_length_per_profile = (max_length - template_length) // len(profiles)

    profile_prompts = []
    saved_length = 0

    for profile in profiles:
        profile_template = f'the tag for the movie: " " is "{profile["tag"]}" '
        profile_template_length = len(tokenizer.encode(profile_template, add_special_tokens=False))
        max_profile_length = max_length_per_profile + saved_length - profile_template_length

        input_ids = tokenizer.encode(
            profile['description'],
            add_special_tokens=False,
            truncation=True,
            max_length=max_profile_length
        )
        new_description = tokenizer.decode(input_ids, skip_special_tokens=True)
        profile_prompt = f'the tag for the movie: "{new_description}" is "{profile["tag"]}" '

        profile_prompts.append(profile_prompt)
        saved_length += max_length_per_profile - profile_template_length - len(input_ids)

    prompt = f'{", and ".join(profile_prompts)}. {input_}'
    return prompt


# ==================================       LaMP 3: Personalized Product Rating       ==================================
def _generate_query_corpus_regression_review(input_: str, profiles: list[Profile]) -> tuple[str, list[str]]:
    query = _extract_string_after_keyword(input_, 'review: ')
    corpus = [profile['text'] for profile in profiles]
    return query, corpus


def _generate_prompt_regression_review(
    input_: str,
    profiles: list[Profile],
    max_length: int,
    tokenizer: PreTrainedTokenizerBase
) -> str:
    template_length = 2 * (len(profiles) - 1) + 1
    max_length_per_profile = (max_length - template_length) // len(profiles)

    profile_prompts = []
    saved_length = 0

    for profile in profiles:
        profile_template = f'{profile["score"]} is the score for " " '
        profile_template_length = len(tokenizer.encode(profile_template, add_special_tokens=False))
        max_profile_length = max_length_per_profile + saved_length - profile_template_length

        input_ids = tokenizer.encode(
            profile['text'],
            add_special_tokens=False,
            truncation=True,
            max_length=max_profile_length
        )
        new_text = tokenizer.decode(input_ids, skip_special_tokens=True)
        profile_prompt = f'{profile["score"]} is the score for "{new_text}" '

        profile_prompts.append(profile_prompt)
        saved_length += max_length_per_profile - profile_template_length - len(input_ids)

    prompt = f'{", and ".join(profile_prompts)}. {input_}'
    return prompt


# ==================================  LaMP 4: Personalized News Headline Generation  ==================================
def _generate_query_corpus_generation_news(input_: str, profiles: list[Profile]) -> tuple[str, list[str]]:
    query = _extract_string_after_keyword(input_, 'article: ')
    corpus = [f'{profile["title"]} {profile["text"]}' for profile in profiles]
    return query, corpus


def _generate_prompt_generation_news(
    input_: str,
    profiles: list[Profile],
    max_length: int,
    tokenizer: PreTrainedTokenizerBase
) -> str:
    template_length = 2 * (len(profiles) - 1) + 1
    max_length_per_profile = (max_length - template_length) // len(profiles)

    profile_prompts = []
    saved_length = 0

    for profile in profiles:
        profile_template = f'"{profile["title"]}" is the title for " " '
        profile_template_length = len(tokenizer.encode(profile_template, add_special_tokens=False))
        max_profile_length = max_length_per_profile + saved_length - profile_template_length

        input_ids = tokenizer.encode(
            profile['text'],
            add_special_tokens=False,
            truncation=True,
            max_length=max_profile_length
        )
        new_text = tokenizer.decode(input_ids, skip_special_tokens=True)
        profile_prompt = f'"{profile["title"]}" is the title for "{new_text}" '

        profile_prompts.append(profile_prompt)
        saved_length += max_length_per_profile - profile_template_length - len(input_ids)

    prompt = f'{", and ".join(profile_prompts)}. {input_}'
    return prompt


# ================================== LaMP 5: Personalized Scholarly Title Generation ==================================
def _generate_query_corpus_generation_paper(input_: str, profiles: list[Profile]) -> tuple[str, list[str]]:
    query = _extract_string_after_keyword(input_, 'paper: ')
    corpus = [f'{profile["title"]} {profile["abstract"]}' for profile in profiles]
    return query, corpus


def _generate_prompt_generation_paper(
    input_: str,
    profiles: list[Profile],
    max_length: int,
    tokenizer: PreTrainedTokenizerBase
) -> str:
    template = 'Following the given patterns'
    template_length = (
        2 * (len(profiles) - 1) + 1
        + len(tokenizer.encode(template, add_special_tokens=False))
    )
    max_length_per_profile = (max_length - template_length) // len(profiles)

    profile_prompts = []
    saved_length = 0

    for profile in profiles:
        profile_template = f'"{profile["title"]}" is a title for " " '
        profile_template_length = len(tokenizer.encode(profile_template, add_special_tokens=False))
        max_profile_length = max_length_per_profile + saved_length - profile_template_length

        input_ids = tokenizer.encode(
            profile['abstract'],
            add_special_tokens=False,
            truncation=True,
            max_length=max_profile_length
        )
        new_abstract = tokenizer.decode(input_ids, skip_special_tokens=True)
        profile_prompt = f'"{profile["title"]}" is a title for "{new_abstract}" '

        profile_prompts.append(profile_prompt)
        saved_length += max_length_per_profile - profile_template_length - len(input_ids)

    prompt = f'{", and ".join(profile_prompts)}. Following the given patterns {input_}'
    return prompt


# ==================================  LaMP 6: Personalized Email Subject Generation  ==================================
def _generate_query_corpus_generation_avocado(input_: str, profiles: list[Profile]) -> tuple[str, list[str]]:
    query = _extract_string_after_keyword(input_, ': ')
    corpus = [profile['text'] for profile in profiles]
    return query, corpus


def _generate_prompt_generation_avocado(
    input_: str,
    profiles: list[Profile],
    max_length: int,
    tokenizer: PreTrainedTokenizerBase
) -> str:
    template_length = 2 * (len(profiles) - 1) + 1
    max_length_per_profile = (max_length - template_length) // len(profiles)

    profile_prompts = []
    saved_length = 0

    for profile in profiles:
        profile_template = f'"{profile["title"]}" is the title for " " '
        profile_template_length = len(tokenizer.encode(profile_template, add_special_tokens=False))
        max_profile_length = max_length_per_profile + saved_length - profile_template_length

        input_ids = tokenizer.encode(
            profile['text'],
            add_special_tokens=False,
            truncation=True,
            max_length=max_profile_length
        )
        new_text = tokenizer.decode(input_ids, skip_special_tokens=True)
        profile_prompt = f'"{profile["title"]}" is the title for "{new_text}" '

        profile_prompts.append(profile_prompt)
        saved_length += max_length_per_profile - profile_template_length - len(input_ids)

    prompt = f'{", and ".join(profile_prompts)}. {input_}'
    return prompt


# ==================================     LaMP 7: Personalized Tweet Paraphrasing     ==================================
def _generate_query_corpus_paraphrase_tweet(input_: str, profiles: list[Profile]) -> tuple[str, list[str]]:
    query = _extract_string_after_keyword(input_, ': ')
    corpus = [profile['text'] for profile in profiles]
    return query, corpus


def _generate_prompt_paraphrase_tweet(
    input_: str,
    profiles: list[Profile],
    max_length: int,
    tokenizer: PreTrainedTokenizerBase
) -> str:
    template = 'are written by a person. Following the given patterns'
    template_length = (
        2 * (len(profiles) - 1) + 1
        + len(tokenizer.encode(template, add_special_tokens=False))
    )
    max_length_per_profile = (max_length - template_length) // len(profiles)

    profile_prompts = []
    saved_length = 0

    for profile in profiles:
        profile_template_length = 2
        max_profile_length = max_length_per_profile + saved_length - profile_template_length

        input_ids = tokenizer.encode(
            profile['text'],
            add_special_tokens=False,
            truncation=True,
            max_length=max_profile_length
        )
        new_text = tokenizer.decode(input_ids, skip_special_tokens=True)
        profile_prompt = f'"{new_text}" '

        profile_prompts.append(profile_prompt)
        saved_length += max_length_per_profile - profile_template_length - len(input_ids)

    prompt = f'{", and ".join(profile_prompts)} are written by a person. Following the given patterns {input_}'
    return prompt


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

    extracted = input_[keyword_start + len(keyword):]
    return extracted
