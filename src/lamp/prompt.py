# Adapted from:
#   https://github.com/LaMP-Benchmark/LaMP/blob/main/LaMP/prompts/prompts.py
#   https://github.com/LongLaMP-benchmark/LongLaMP-Benchmark/blob/main/longLaMP/prompts/prompts.py
import logging
from collections.abc import Callable

from transformers import PreTrainedTokenizerBase

from .retrieval import create_retriever_fn


logger = logging.getLogger(__name__)


PromptFn = Callable[[str, list[dict[str, str]], str, list[str]], str]


def create_prompt_fn(
    task: str,
    retriever: str,
    num_retrieve: int,
    max_prompt_length: int,
    tokenizer: PreTrainedTokenizerBase,
    factor: float = 0.6
) -> PromptFn:
    retriever_fn = create_retriever_fn(retriever)
    prompt_fn = PROMPT_FNS[task]

    def retrieval_augmented_prompt_fn(
        source: str,
        profile: list[dict[str, str]],
        query: str,
        corpus: list[str]
    ) -> str:
        local_factor = factor
        retrieved_profile = retriever_fn(query, corpus, profile, num_retrieve)
        source_length = len(tokenizer.encode(source))

        while True:
            try:
                reserved_length = min(
                    source_length, int(local_factor * max_prompt_length)
                )
                max_profile_length = max_prompt_length - reserved_length
                return prompt_fn(
                    source, retrieved_profile, max_profile_length, tokenizer
                )
            except OverflowError:
                local_factor -= 0.1

                if local_factor < 0:
                    logger.warning("Returning source as is")
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
        max_record_length = (
            max_length_per_record + saved_length - record_template_length
        )
        title, record_length = _truncate_text(
            record["title"], max_record_length, tokenizer
        )

        prompts.append(f'"{title}"')
        saved_length += (
            max_length_per_record - record_template_length - record_length
        )

    index = source.find("title")
    assert index != -1
    return (
        f"{source[:index + 5]}, and {', and '.join(prompts)}"
        f"{source[index + 5:]}"
    )


# [LaMP-2] Personalized Movie Tagging
def _classification_movie_prompt_fn(
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
        record_template = f'the tag for the movie: " " is "{record["tag"]}" '
        record_template_length = len(
            tokenizer.encode(record_template, add_special_tokens=False)
        )
        max_record_length = (
            max_length_per_record + saved_length - record_template_length
        )
        description, record_length = _truncate_text(
            record["description"], max_record_length, tokenizer
        )

        prompts.append(
            f'the tag for the movie: "{description}" is "{record["tag"]}" '
        )
        saved_length += (
            max_length_per_record - record_template_length - record_length
        )

    return f"{', and '.join(prompts)}. {source}"


# [LaMP-3] Personalized Product Rating
def _regression_rating_prompt_fn(
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
        record_template = f'{record["score"]} is the score for " " '
        record_template_length = len(
            tokenizer.encode(record_template, add_special_tokens=False)
        )
        max_record_length = (
            max_length_per_record + saved_length - record_template_length
        )
        text, record_length = _truncate_text(
            record["text"], max_record_length, tokenizer
        )

        prompts.append(f'{record["score"]} is the score for "{text}" ')
        saved_length += (
            max_length_per_record - record_template_length - record_length
        )

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
        record_template = f'"{record["title"]}" is the title for " " '
        record_template_length = len(
            tokenizer.encode(record_template, add_special_tokens=False)
        )
        max_record_length = (
            max_length_per_record + saved_length - record_template_length
        )
        text, record_length = _truncate_text(
            record["text"], max_record_length, tokenizer
        )

        prompts.append(f'"{record["title"]}" is the title for "{text}" ')
        saved_length += (
            max_length_per_record - record_template_length - record_length
        )

    return f"{', and '.join(prompts)}. {source}"


# [LaMP-5] Personalized Scholarly Title Generation
def _generation_scholar_prompt_fn(
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
        record_template = f'"{record["title"]}" is a title for " " '
        record_template_length = len(
            tokenizer.encode(record_template, add_special_tokens=False)
        )
        max_record_length = (
            max_length_per_record + saved_length - record_template_length
        )
        abstract, record_length = _truncate_text(
            record["abstract"], max_record_length, tokenizer
        )

        prompts.append(f'"{record["title"]}" is a title for "{abstract}" ')
        saved_length += (
            max_length_per_record - record_template_length - record_length
        )

    return f"{', and '.join(prompts)}. Following the given patterns {source}"


# [LaMP-7] Personalized Tweet Paraphrasing
def _generation_tweet_prompt_fn(
    source: str,
    profile: list[dict[str, str]],
    max_length: int,
    tokenizer: PreTrainedTokenizerBase
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
        max_record_length = (
            max_length_per_record + saved_length - record_template_length
        )
        text, record_length = _truncate_text(
            record["text"], max_record_length, tokenizer
        )

        prompts.append(f'"{text}" ')
        saved_length += (
            max_length_per_record - record_template_length - record_length
        )

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
        # Truncate abstract to 750 words
        abstract, _ = _truncate_text(
            " ".join(record["abstract"].split()[:750]), max_length, tokenizer
        )
        prompts.append(
            f'"{abstract}" is the abstract for the title "{record["title"]}"'
        )

    return (
        f"{', and '.join(prompts)}. Use the above abstracts as context "
        f"to understand the style and language of the user and, {source}"
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
        review_text, _ = _truncate_text(
            record["reviewText"], max_length, tokenizer
        )
        prompts.append(
            f'"{record["overall"]}" is a rating for the product '
            f'with description "{record["description"]}". '
            f'"{record["summary"]}" is summary for "{review_text}" '
        )

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
        summary, _ = _truncate_text(record["summary"], max_length, tokenizer)
        prompts.append(f'"{record["content"]}" is a summary for "{summary}" ')

    return f"{', and '.join(prompts)}. Following the given patterns, {source}"


def _truncate_text(
    text: str,
    max_length: int,
    tokenizer: PreTrainedTokenizerBase
) -> tuple[str, int]:
    input_ids = tokenizer.encode(
        text, add_special_tokens=False, truncation=True, max_length=max_length
    )
    new_text = tokenizer.decode(input_ids, skip_special_tokens=True)
    return new_text, len(input_ids)


PROMPT_FNS = {
    "LaMP-1": _classification_citation_prompt_fn,
    "LaMP-2": _classification_movie_prompt_fn,
    "LaMP-3": _regression_rating_prompt_fn,
    "LaMP-4": _generation_news_prompt_fn,
    "LaMP-5": _generation_scholar_prompt_fn,
    "LaMP-7": _generation_tweet_prompt_fn,
    "LongLaMP-2": _generation_abstract_prompt_fn,
    "LongLaMP-3": _generation_review_prompt_fn,
    "LongLaMP-4": _generation_topic_prompt_fn
}
