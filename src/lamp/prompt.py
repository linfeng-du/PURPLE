# Adapted from:
#   https://github.com/LaMP-Benchmark/LaMP/blob/main/LaMP/prompts/prompts.py
#   https://github.com/LongLaMP-benchmark/LongLaMP-Benchmark/blob/main/longLaMP/prompts/prompts.py
import logging
from typing import Callable, TypeAlias

from transformers import PreTrainedTokenizerBase

from .retrievers import get_retriever_fn


logger = logging.getLogger(__name__)


PromptGenerator: TypeAlias = Callable[
    [str, list[dict[str, str]], str | None, list[str] | None],
    str | tuple[str, list[dict[str, str]]]
]


def create_prompt_fn(
    task: str,
    retriever: str,
    num_retrieve: int,
    max_prompt_length: int,
    tokenizer: PreTrainedTokenizerBase,
    factor: float = 0.6,
    return_retrieved: bool = False
) -> PromptGenerator:
    retriever_fn = get_retriever_fn(retriever)
    prompt_fn = PROMPT_FNS[task]

    def retrieval_augmented_prompt_fn(
        source: str,
        profile: list[dict[str, str]],
        query: str | None = None,
        corpus: list[str] | None = None
    ) -> str | tuple[str, list[dict[str, str]]]:
        local_factor = factor
        retrieved_profile = retriever_fn(query, corpus, profile, num_retrieve)
        source_length = len(tokenizer.encode(source, truncation=True, max_length=max_prompt_length))

        while True:
            try:
                reserved_length = min(source_length, int(local_factor * max_prompt_length))
                max_profile_length = max_prompt_length - reserved_length
                prompt = prompt_fn(source, retrieved_profile, max_profile_length, tokenizer)

                if return_retrieved:
                    return prompt, retrieved_profile

                return prompt
            except OverflowError:
                local_factor -= 0.1

                if local_factor < 0:
                    logger.warning(f"Returning question as is")
                    return source

    return retrieval_augmented_prompt_fn


# [LaMP-1] Personalized Citation Identification
def _classification_citation_prompt_fn(
    source: str,
    profile: list[dict[str, str]],
    max_length: int,
    tokenizer: PreTrainedTokenizerBase
) -> str:
    template_length = 2 * len(profile)
    max_length_per_record = (max_length - template_length) // len(profile)

    prompts = []
    saved_length = 0

    for record in profile:
        record_template_length = 2
        max_record_length = max_length_per_record + saved_length - record_template_length

        input_ids = tokenizer.encode(
            record["title"],
            add_special_tokens=False,
            truncation=True,
            max_length=max_record_length
        )
        new_title = tokenizer.decode(input_ids, skip_special_tokens=True)
        prompt = f"'{new_title}'"

        prompts.append(prompt)
        saved_length += max_length_per_record - record_template_length - len(input_ids)

    index = source.find("title")
    return f"{source[:index + 5]}, and {', and '.join(prompts)}{source[index + 5:]}"


# [LaMP-2] Personalized Movie Tagging
def _classification_movies_prompt_fn(
    source: str,
    profile: list[dict[str, str]],
    max_length: int,
    tokenizer: PreTrainedTokenizerBase
) -> str:
    template_length = 2 * (len(profile) - 1) + 1
    max_length_per_record = (max_length - template_length) // len(profile)

    prompts = []
    saved_length = 0

    for record in profile:
        record_template = f"the tag for the movie: \" \" is \"{record['tag']}\" "
        record_template_length = len(tokenizer.encode(record_template, add_special_tokens=False))
        max_record_length = max_length_per_record + saved_length - record_template_length

        input_ids = tokenizer.encode(
            record["description"],
            add_special_tokens=False,
            truncation=True,
            max_length=max_record_length
        )
        new_description = tokenizer.decode(input_ids, skip_special_tokens=True)
        prompt = f"the tag for the movie: \"{new_description}\" is \"{record['tag']}\" "

        prompts.append(prompt)
        saved_length += max_length_per_record - record_template_length - len(input_ids)

    return f"{', and '.join(prompts)}. {source}"


# [LaMP-3] Personalized Product Rating
def _regression_review_prompt_fn(
    source: str,
    profile: list[dict[str, str]],
    max_length: int,
    tokenizer: PreTrainedTokenizerBase
) -> str:
    template_length = 2 * (len(profile) - 1) + 1
    max_length_per_record = (max_length - template_length) // len(profile)

    prompts = []
    saved_length = 0

    for record in profile:
        record_template = f"{record['score']} is the score for " " "
        record_template_length = len(tokenizer.encode(record_template, add_special_tokens=False))
        max_record_length = max_length_per_record + saved_length - record_template_length

        input_ids = tokenizer.encode(
            record["text"],
            add_special_tokens=False,
            truncation=True,
            max_length=max_record_length
        )
        new_text = tokenizer.decode(input_ids, skip_special_tokens=True)
        prompt = f"{record['score']} is the score for \"{new_text}\" "

        prompts.append(prompt)
        saved_length += max_length_per_record - record_template_length - len(input_ids)

    return f"{', and '.join(prompts)}. {source}"


# [LaMP-4] Personalized News Headline Generation
def _generation_news_prompt_fn(
    source: str,
    profile: list[dict[str, str]],
    max_length: int,
    tokenizer: PreTrainedTokenizerBase
) -> str:
    template_length = 2 * (len(profile) - 1) + 1
    max_length_per_record = (max_length - template_length) // len(profile)

    prompts = []
    saved_length = 0

    for record in profile:
        record_template = f"\"{record['title']}\" is the title for \" \" "
        record_template_length = len(tokenizer.encode(record_template, add_special_tokens=False))
        max_record_length = max_length_per_record + saved_length - record_template_length

        input_ids = tokenizer.encode(
            record["text"],
            add_special_tokens=False,
            truncation=True,
            max_length=max_record_length
        )
        new_text = tokenizer.decode(input_ids, skip_special_tokens=True)
        prompt = f"\"{record['title']}\" is the title for \"{new_text}\" "

        prompts.append(prompt)
        saved_length += max_length_per_record - record_template_length - len(input_ids)

    return f"{', and '.join(prompts)}. {source}"


# [LaMP-5] Personalized Scholarly Title Generation
def _generation_paper_prompt_fn(
    source: str,
    profile: list[dict[str, str]],
    max_length: int,
    tokenizer: PreTrainedTokenizerBase
) -> str:
    template = "Following the given patterns"
    template_length = (
        2 * (len(profile) - 1) + 1
        + len(tokenizer.encode(template, add_special_tokens=False))
    )
    max_length_per_record = (max_length - template_length) // len(profile)

    prompts = []
    saved_length = 0

    for record in profile:
        record_template = f"\"{record['title']}\" is a title for \" \" "
        record_template_length = len(tokenizer.encode(record_template, add_special_tokens=False))
        max_record_length = max_length_per_record + saved_length - record_template_length

        input_ids = tokenizer.encode(
            record["abstract"],
            add_special_tokens=False,
            truncation=True,
            max_length=max_record_length
        )
        new_abstract = tokenizer.decode(input_ids, skip_special_tokens=True)
        prompt = f"\"{record['title']}\" is a title for \"{new_abstract}\" "

        prompts.append(prompt)
        saved_length += max_length_per_record - record_template_length - len(input_ids)

    return f"{', and '.join(prompts)}. Following the given patterns {source}"


# [LaMP-7] Personalized Tweet Paraphrasing
def _generation_tweet_prompt_fn(
    source: str, profile: list[dict[str, str]],
    max_length: int, tokenizer: PreTrainedTokenizerBase
) -> str:
    template = "are written by a person. Following the given patterns"
    template_length = (
        2 * (len(profile) - 1) + 1
        + len(tokenizer.encode(template, add_special_tokens=False))
    )
    max_length_per_record = (max_length - template_length) // len(profile)

    prompts = []
    saved_length = 0

    for record in profile:
        record_template_length = 2
        max_record_length = max_length_per_record + saved_length - record_template_length

        input_ids = tokenizer.encode(
            record["text"],
            add_special_tokens=False,
            truncation=True,
            max_length=max_record_length
        )
        new_text = tokenizer.decode(input_ids, skip_special_tokens=True)
        prompt = f"\"{new_text}\" "

        prompts.append(prompt)
        saved_length += max_length_per_record - record_template_length - len(input_ids)

    return (
        f"{', and '.join(prompts)} are written by a person. "
        f"Following the given patterns {source}"
    )


# [LongLaMP-2] Personalized Abstract Generation
def _generation_abstract_prompt_fn(
    source: str,
    profile: list[dict[str, str]],
    max_length: int,
    tokenizer: PreTrainedTokenizerBase
) -> str:
    prompts = []

    for record in profile:
        # Truncates abstract to 750 words
        input_ids = tokenizer.encode(
            " ".join(record["abstract"].split()[:750]),
            add_special_tokens=False,
            truncation=True,
            max_length=max_length
        )
        new_abstract = tokenizer.decode(input_ids, skip_special_tokens=True)
        prompt = f"\"{new_abstract}\" is the abstract for the title \"{record['title']}\""
        prompts.append(prompt)

    return (
        f"{', and '.join(prompts)}. Use the above abstracts as context to "
        f"understand the style and language of the user and, {source}"
    )


# [LongLaMP-3] Personalized Product Review Generation
def _generation_review_prompt_fn(
    source: str,
    profile: list[dict[str, str]],
    max_length: int,
    tokenizer: PreTrainedTokenizerBase
) -> str:
    prompts = []

    for record in profile:
        input_ids = tokenizer.encode(
            record["reviewText"],
            add_special_tokens=False,
            truncation=True,
            max_length=max_length
        )
        new_review_text = tokenizer.decode(input_ids, skip_special_tokens=True)
        prompt = (
            f"\"{record['overall']}\" is a rating for "
            f"the product with description \"{record['description']}\". "
            f"\"{record['summary']}\" is summary for \"{new_review_text}\" "
        )
        prompts.append(prompt)

    return f"{', and '.join(prompts)}. Following the given patterns {source}"


# [LongLaMP-4] Personalized Topic Generation
def _generation_topic_prompt_fn(
    source: str,
    profile: list[dict[str, str]],
    max_length: int,
    tokenizer: PreTrainedTokenizerBase
) -> str:
    prompts = []

    for record in profile:
        input_ids = tokenizer.encode(
            record["summary"],
            add_special_tokens=False,
            truncation=True,
            max_length=max_length
        )
        new_summary = tokenizer.decode(input_ids, skip_special_tokens=True)
        prompt = f"\"{record['content']}\" is a summary for \"{new_summary}\" "
        prompts.append(prompt)

    return f"{', and '.join(prompts)}. Following the given patterns, {source}"


PROMPT_FNS = {
    "LaMP-1": _classification_citation_prompt_fn,
    "LaMP-2": _classification_movies_prompt_fn,
    "LaMP-3": _regression_review_prompt_fn,
    "LaMP-4": _generation_news_prompt_fn,
    "LaMP-5": _generation_paper_prompt_fn,
    "LaMP-7": _generation_tweet_prompt_fn,
    "LongLaMP-2": _generation_abstract_prompt_fn,
    "LongLaMP-3": _generation_review_prompt_fn,
    "LongLaMP-4": _generation_topic_prompt_fn
}
