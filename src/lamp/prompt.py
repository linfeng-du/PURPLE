# https://github.com/LaMP-Benchmark/LaMP/blob/main/LaMP/prompts/prompts.py
# https://github.com/LongLaMP-benchmark/LongLaMP-Benchmark/blob/main/longLaMP/prompts/prompts.py
import logging
from collections.abc import Callable

from transformers import AutoTokenizer, PreTrainedTokenizerBase
from transformers.pipelines.text_generation import ChatType

from .metric import LABELS
from .retrieval import create_retrieval_fn


logger = logging.getLogger(__name__)
PromptFn = Callable[[str, list[dict[str, str]], str, list[str]], str]
ChatPromptFn = Callable[[str, str | None], ChatType]


def create_prompt_fn(
    task: str,
    retriever: str,
    num_retrieve: int,
    llm_model: str,
    max_prompt_length: int,
    factor: float = 0.6
) -> PromptFn:
    prompt_fn = PROMPT_FNS[task]
    retrieval_fn = create_retrieval_fn(retriever)
    tokenizer = AutoTokenizer.from_pretrained(llm_model)

    def retrieval_augmented_prompt_fn(
        source: str,
        profile: list[dict[str, str]],
        query: str,
        corpus: list[str]
    ) -> str:
        local_factor = factor
        retrieved_profile = retrieval_fn(query, corpus, profile, num_retrieve)
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
        # Extract first 750 words of abstract before truncation
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
        content, _ = _truncate_text(record["content"], max_length, tokenizer)
        prompts.append(f'"{record["summary"]}" is a summary for "{content}" ')

    return f"{', and '.join(prompts)}. Following the given patterns, {source}"


def _truncate_text(
    text: str,
    max_length: int,
    tokenizer: PreTrainedTokenizerBase
) -> tuple[str, int]:
    text_ids = tokenizer.encode(
        text, add_special_tokens=False, truncation=True, max_length=max_length
    )
    new_text = tokenizer.decode(text_ids, skip_special_tokens=True)
    return new_text, len(text_ids)


PROMPT_FNS = {
    "lamp1": _classification_citation_prompt_fn,
    "lamp2": _classification_movie_prompt_fn,
    "lamp3": _regression_rating_prompt_fn,
    "lamp4": _generation_news_prompt_fn,
    "lamp5": _generation_scholar_prompt_fn,
    "lamp7": _generation_tweet_prompt_fn,
    "longlamp2": _generation_abstract_prompt_fn,
    "longlamp3": _generation_review_prompt_fn,
    "longlamp4": _generation_topic_prompt_fn
}


SYSTEM_PROMPTS = {
    "lamp1": (
        "You are a personalized citation identification chatbot "
        f"who responds with one of the following: {LABELS['lamp1']} "
        "based on the given examples without any additional text."
    ),
    "lamp2": (
        "You are a personalized movie tagging chatbot "
        f"who responds with one of the following: {LABELS['lamp2']} "
        "based on the given examples without any additional text."
    ),
    "lamp3": (
        "You are a personalized product rating chatbot "
        f"who responds with one of the following: {LABELS['lamp3']} "
        "based on the given examples without any additional text."
    ),
    "lamp4": (
        "You are a personalized news headline generation chatbot "
        "who generates a news headline "
        "in a style similar to the given examples without any additional text."
    ),
    "lamp5": (
        "You are a personalized scholarly title generation chatbot "
        "who generates a scholarly title "
        "in a style similar to the given examples without any additional text."
    ),
    "lamp7": (
        "You are a personalized tweet paraphrasing chatbot "
        "who paraphrases a tweet "
        "in a style similar to the given examples without any additional text."
    ),
    "longlamp2": (
        "You are a personalized abstract generation chatbot "
        "who generates an abstract "
        "in a style similar to the given examples without any additional text."
    ),
    "longlamp3": (
        "You are a personalized product review generation chatbot "
        "who generates a product review "
        "in a style similar to the given examples without any additional text."
    ),
    "longlamp4": (
        "You are a personalized topic generation chatbot "
        "who generates a topic "
        "in a style similar to the given examples without any additional text."
    )
}


def create_chat_prompt_fn(task: str) -> ChatPromptFn:
    def chat_prompt_fn(
        prompt: str,
        completion_prefix: str | None = None
    ) -> ChatType:
        chat_prompt = [
            {"role": "system", "content": SYSTEM_PROMPTS[task]},
            {"role": "user", "content": prompt},
        ]

        if completion_prefix is not None:
            chat_prompt.append(
                {"role": "assistant", "content": completion_prefix}
            )

        return chat_prompt

    return chat_prompt_fn
