# Adapted from https://github.com/LaMP-Benchmark/LaMP/blob/main/LaMP/prompts/prompts.py
import logging
from typing import Callable
import torch
from transformers import PreTrainedTokenizerBase

from .retriever import create_retriever
from .data_types import Profile, PromptGenerator


logger = logging.getLogger(__name__)


def create_prompt_generator(
    task: str,
    retriever: str,
    num_retrieve: int,
    max_length: int,
    tokenizer: PreTrainedTokenizerBase,
    device: torch.device | None = None
) -> PromptGenerator:
    retriever = create_retriever(retriever, device=device)
    prompt_generator = _create_prompt_generator(task)

    def retrieval_augmented_prompt_generator(
        source: str,
        profiles: list[Profile],
        query: str,
        corpus: list[str],
        factor: float = 0.6
    ) -> str:
        retrieved_profiles = retriever(query, corpus, profiles, num_retrieve)
        source_length = len(tokenizer.encode(source, truncation=True, max_length=max_length))

        while True:
            try:
                reserved_length = min(source_length, int(factor * max_length))
                max_profile_length = max_length - reserved_length
                return prompt_generator(source, retrieved_profiles, max_profile_length, tokenizer)
            except OverflowError:
                factor -= 0.1

                if factor < 0:
                    logger.warning(f'Returning question as is')
                    return source

    return retrieval_augmented_prompt_generator


def _create_prompt_generator(task: str) -> (
    Callable[[str, list[Profile], int, PreTrainedTokenizerBase], str]
):
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
def _generate_prompt_classification_citation(
    source: str,
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

    title_start = source.find('title')
    return f'{source[:title_start + 5]}, and {", and ".join(profile_prompts)}{source[title_start + 5:]}'


# ==================================        LaMP 2: Personalized Movie Tagging        ==================================
def _generate_prompt_classification_movies(
    source: str,
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

    return f'{", and ".join(profile_prompts)}. {source}'


# ==================================       LaMP 3: Personalized Product Rating       ==================================
def _generate_prompt_regression_review(
    source: str,
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

    return f'{", and ".join(profile_prompts)}. {source}'


# ==================================  LaMP 4: Personalized News Headline Generation  ==================================
def _generate_prompt_generation_news(
    source: str,
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

    return f'{", and ".join(profile_prompts)}. {source}'


# ================================== LaMP 5: Personalized Scholarly Title Generation ==================================
def _generate_prompt_generation_paper(
    source: str,
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

    return f'{", and ".join(profile_prompts)}. Following the given patterns {source}'


# ==================================  LaMP 6: Personalized Email Subject Generation  ==================================
def _generate_prompt_generation_avocado(
    source: str,
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

    return f'{", and ".join(profile_prompts)}. {source}'


# ==================================     LaMP 7: Personalized Tweet Paraphrasing     ==================================
def _generate_prompt_paraphrase_tweet(
    source: str,
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

    return f'{", and ".join(profile_prompts)} are written by a person. Following the given patterns {source}'
