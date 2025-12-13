# Adapted from: https://github.com/sunnweiwei/RankGPT/blob/main/rank_gpt.py
import logging
import time

from openai import OpenAI, OpenAIError
from transformers import pipeline


logger = logging.getLogger(__name__)


class RankGPT:

    def __init__(
        self,
        model: str,
        backend: str,
        window_size: int = 20,
        window_stride: int = 10,
        max_passage_length: int = 300
    ) -> None:
        self.model = model
        self.backend = backend

        self.window_size = window_size
        self.window_stride = window_stride
        self.max_passage_length = max_passage_length

        if self.backend == "hf":
            self.pipeline = pipeline(
                task="text-generation",
                model=self.model,
                device_map="auto",
                torch_dtype="bfloat16"
            )
        elif self.backend == "openai":
            self.client = OpenAI()
        else:
            raise ValueError(f"Invalid backend: {self.backend}")

    def __call__(
        self,
        query: str,
        corpus: list[str],
        profile: list[dict[str, str]],
        num_retrieve: int
    ) -> list[dict[str, str]]:
        assert len(corpus) == len(profile) != 0
        num_retrieve = min(num_retrieve, len(profile))

        corpus = corpus.copy()
        ranking = list(range(len(profile)))
        end = len(profile)

        while end > 0:
            start = max(0, end - self.window_size)
            messages = _build_messages(
                query, corpus[start:end], self.max_passage_length
            )

            if self.backend == "hf":
                outputs = self.pipeline(
                    messages,
                    max_new_tokens=256,
                    do_sample=False,
                    temperature=None,
                    top_p=None,
                    pad_token_id=self.pipeline.tokenizer.eos_token_id
                )
                response = outputs[0]["generated_text"][-1]["content"]

            elif self.backend == "openai":
                response = None
                num_retries = 0

                while response is None:
                    try:
                        outputs = self.client.chat.completions.create(
                            messages=messages, model=self.model
                        )
                        response = outputs.choices[0].message.content
                    except OpenAIError as err:
                        num_retries += 1

                        if num_retries > 10:
                            raise err

                        logger.error(f"OpenAI API error: {err}", exc_info=True)
                        time.sleep(num_retries ** 2)

            ordering = _parse_response(response, end - start)
            corpus[start:end] = [corpus[start:end][idx] for idx in ordering]
            ranking[start:end] = [ranking[start:end][idx] for idx in ordering]

            end -= self.window_stride

        return [profile[idx] for idx in ranking[:num_retrieve]]


def _build_messages(
    query: str,
    corpus: list[str],
    max_passage_length: int
) -> list[dict[str, str]]:
    messages = _build_prefix(query, len(corpus))

    for index, passage in enumerate(corpus, start=1):
        passage = " ".join(passage.strip().split()[:max_passage_length])
        messages.extend([
            {"role": "user", "content": f"[{index}] {passage}"},
            {"role": "assistant", "content": f"Received passage [{index}]."}
        ])

    messages.append(_build_suffix(query, len(corpus)))
    return messages


def _build_prefix(query: str, num_passages: int) -> list[dict[str, str]]:
    return [
        {
            "role": "system",
            "content": (
                "You are RankGPT, an intelligent assistant "
                "that can rank passages based on their relevancy to the query."
            )
        },
        {
            "role": "user",
            "content": (
                f"I will provide you with {num_passages} passages, "
                "each indicated by number identifier []. \n"
                f"Rank the passages based on their relevance to query: {query}."
            )
        },
        {"role": "assistant", "content": "Okay, please provide the passages."}
    ]


def _build_suffix(query: str, num_passages: int) -> dict[str, str]:
    return {
        "role": "user",
        "content": (
            f"Search Query: {query}. \n"
            f"Rank the {num_passages} passages above "
            "based on their relevance to the search query. "
            "The passages should be listed "
            "in descending order using identifiers. "
            "The most relevant passages should be listed first. "
            "The output format should be [] > [], e.g., [1] > [2]. "
            "Only response the ranking results, do not say any word or explain."
        )
    }


def _parse_response(response: str, num_passages: int) -> list[int]:
    digits_str = "".join(char if char.isdigit() else " " for char in response)
    digits = [int(char) - 1 for char in digits_str.strip().split()]
    unique_digits = list(dict.fromkeys(digits))

    ordering = [idx for idx in unique_digits if idx in range(num_passages)]
    ordering += [idx for idx in range(num_passages) if idx not in unique_digits]
    return ordering
