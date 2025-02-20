import os
import logging
from typing import Callable

from openai import OpenAI, InternalServerError


logger = logging.getLogger(__name__)


def initialize_openai_client(model: str, temperature: float) -> (
    Callable[[list[str]], list[str]]
):
    if model == 'gpt-4o':
        api_key = os.getenv('OPENAI_API_KEY')
        base_url = 'https://api.openai.com/v1'
    elif model == 'deepseek-chat':
        api_key = os.getenv('DEEPSEEK_API_KEY')
        base_url = 'https://api.deepseek.com'
    else:
        raise ValueError(f'Unsupported generation model: {model}')

    client = OpenAI(api_key=api_key, base_url=base_url)

    def request_completions(prompts: list[str]) -> list[str]:
        responses = [None for _ in range(len(prompts))]
        remaining_prompts = set(range(len(prompts)))

        while remaining_prompts:
            for prompt_index in list(remaining_prompts):
                try:
                    completion = client.chat.completions.create(
                        model=model,
                        messages=[{'role': 'user', 'content': prompts[prompt_index]}],
                        temperature=temperature
                    )
                    response = completion.choices[0].message.content

                    responses[prompt_index] = response
                    remaining_prompts.remove(prompt_index)
                except InternalServerError:
                    logger.warning(f'Retrying due to server error...')

        return responses

    return request_completions
