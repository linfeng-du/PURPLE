# Adapted from https://github.com/sunnweiwei/RankGPT/blob/main/rank_gpt.py
from typing import TypeAlias

import torch
from transformers import pipeline

from ..data_types import Profile


Message: TypeAlias = list[dict[str, str]]


class RankGPT:

    def __init__(self, device: torch.device) -> None:
        self.pipeline = pipeline(
            task='text-generation',
            model='meta-llama/Meta-Llama-3-8B-Instruct',
            device=device,
            torch_dtype=torch.bfloat16
        )
        self.pipeline.tokenizer.padding_side = 'left'

        if self.pipeline.tokenizer.pad_token is None:
            self.pipeline.tokenizer.pad_token = self.pipeline.tokenizer.eos_token
            self.pipeline.model.generation_config.pad_token_id = self.pipeline.tokenizer.eos_token_id

    def __call__(
        self,
        query: str,
        corpus: list[str],
        profiles: list[Profile],
        num_rerank: int,
        window_size: int = 20,
        step: int = 10
    ) -> list[Profile]:
        rank_start = len(profiles) - window_size
        rank_end = len(profiles)
        indices = list(range(len(profiles)))

        while rank_start >= 0:
            message = _create_ranking_instruction(query, corpus, rank_start, rank_end)
            output = self.pipeline(
                message, max_new_tokens=256,
                do_sample=False, temperature=None, top_p=None
            )
            response = output[0]['generated_text'][-1]['content']

            indices = _receive_ranking(indices, response, rank_start, rank_end)
            corpus = [corpus[index] for index in indices]
            profiles = [profiles[index] for index in indices]

            rank_start -= step
            rank_end -= step

        return profiles[:num_rerank]


def _create_ranking_instruction(
    query: str,
    corpus: list[str],
    rank_start: int,
    rank_end: int,
    max_length: int = 300
) -> Message:
    num_passages = len(corpus[rank_start:rank_end])
    message = _create_prefix_prompt(query, num_passages)

    for index, document in enumerate(corpus[rank_start:rank_end], start=1):
        content = ' '.join(document.strip().split()[:max_length])
        message.append({'role': 'user', 'content': f'[{index}] {content}'})
        message.append({'role': 'assistant', 'content': f'Received passage [{index}].'})

    message.append({'role': 'user', 'content': _create_postfix_prompt(query, num_passages)})
    return message


def _create_prefix_prompt(query: str, num_passages: int) -> Message:
    return [
        {'role': 'system', 'content': (
            'You are RankGPT, an intelligent assistant that can '
            'rank passages based on their relevancy to the query.'
        )},
        {'role': 'user', 'content': (
            f'I will provide you with {num_passages} passages, each indicated by number identifier []. \n'
            f'Rank the passages based on their relevance to query: {query}.'
        )},
        {'role': 'assistant', 'content': 'Okay, please provide the passages.'}
    ]


def _create_postfix_prompt(query: str, num_passages: int) -> str:
    return (
        f'Search Query: {query}. \n'
        f'Rank the {num_passages} passages above based on their relevance to the search query. '
        'The passages should be listed in descending order using identifiers. '
        'The most relevant passages should be listed first. '
        'The output format should be [] > [], e.g., [1] > [2]. '
        'Only response the ranking results, do not say any word or explain.'
    )


def _receive_ranking(indices: list[int], response: str, rank_start: int, rank_end: int) -> list[int]:
    ranking = ''.join((char if char.isdigit() else ' ') for char in response)
    ranking = [int(char) - 1 for char in ranking.strip().split()]
    ranking = list(dict.fromkeys(ranking))

    window_indices = indices[rank_start:rank_end].copy()
    original_ranking = list(range(len(window_indices)))
    ranking = [rank for rank in ranking if rank in original_ranking]
    ranking += [rank for rank in original_ranking if rank not in ranking]

    for index, rank in enumerate(ranking):
        indices[index + rank_start] = window_indices[rank]

    return indices
