import os
from typing import Callable

from openai import OpenAI


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
        responses = []

        for prompt in prompts:
            completion = client.chat.completions.create(
                model=model,
                messages=[{'role': 'user', 'content': prompt}],
                temperature=temperature
            )
            response = completion.choices[0].message.content
            responses.append(response)

        return responses

    return request_completions
