# Adapted from: https://github.com/sunnweiwei/RankGPT/blob/main/rank_gpt.py
import logging
import time

from openai import OpenAI, OpenAIError
from transformers import pipeline
from transformers.pipelines.text_generation import ChatType


logger = logging.getLogger(__name__)


class RankGPT:

    def __init__(
        self,
        model: str,
        backend: str,
        sliding_window_size: int = 20,
        sliding_window_stride: int = 10,
        max_passage_length: int = 300
    ) -> None:
        self.model = model
        self.backend = backend

        self.sliding_window_size = sliding_window_size
        self.sliding_window_stride = sliding_window_stride
        self.max_passage_length = max_passage_length

        if self.backend == "hf":
            self.pipeline = pipeline(
                task="text-generation",
                model=self.model,
                device_map="auto",
                torch_dtype="bfloat16"
            )

            if self.pipeline.model.generation_config.pad_token_id is None:
                self.pipeline.model.generation_config.pad_token_id = (
                    self.pipeline.tokenizer.eos_token_id
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
            start = max(0, end - self.sliding_window_size)
            messages = _build_messages(
                query, corpus[start:end], self.max_passage_length
            )

            if self.backend == "hf":
                outputs = self.pipeline(
                    messages,
                    return_full_text=False,
                    max_new_tokens=256,
                    do_sample=False,
                    temperature=None,
                    top_p=None
                )
                completion = outputs[0]["generated_text"]

            elif self.backend == "openai":
                completion = None
                num_retries = 0

                while completion is None:
                    try:
                        outputs = self.client.chat.completions.create(
                            messages=messages, model=self.model
                        )
                        completion = outputs.choices[0].message.content
                    except OpenAIError as err:
                        num_retries += 1

                        if num_retries > 10:
                            raise

                        logger.warning(f"Retrying ({num_retries}/10). {err}")
                        time.sleep(min(2 ** num_retries, 60))

            ordering = _parse_completion(completion, end - start)
            corpus[start:end] = [corpus[start:end][idx] for idx in ordering]
            ranking[start:end] = [ranking[start:end][idx] for idx in ordering]

            end -= self.sliding_window_stride

        return [profile[idx] for idx in ranking[:num_retrieve]]


def _build_messages(
    query: str,
    corpus: list[str],
    max_passage_length: int
) -> ChatType:
    messages = _build_prefix_messages(query, len(corpus))

    for index, passage in enumerate(corpus, start=1):
        passage = " ".join(passage.strip().split()[:max_passage_length])
        messages.extend([
            {"role": "user", "content": f"[{index}] {passage}"},
            {"role": "assistant", "content": f"Received passage [{index}]."}
        ])

    messages.extend(_build_suffix_messages(query, len(corpus)))
    return messages


def _build_prefix_messages(query: str, num_passages: int) -> ChatType:
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


def _build_suffix_messages(query: str, num_passages: int) -> ChatType:
    return [
        {
            "role": "user",
            "content": (
                f"Search Query: {query}. \n"
                f"Rank the {num_passages} passages above "
                "based on their relevance to the search query. "
                "The passages should be listed "
                "in descending order using identifiers. "
                "The most relevant passages should be listed first. "
                "The output format should be [] > [], e.g., [1] > [2]. "
                "Only response the ranking results, "
                "do not say any word or explain."
            )
        }
    ]


def _parse_completion(completion: str, num_passages: int) -> list[int]:
    digits = "".join(char if char.isdigit() else " " for char in completion)
    indices = list(
        dict.fromkeys([int(char) - 1 for char in digits.strip().split()])
    )

    ordering = [i for i in indices if i in range(num_passages)]
    ordering += [i for i in range(num_passages) if i not in indices]
    return ordering
