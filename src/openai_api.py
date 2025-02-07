from typing import Callable

from openai import OpenAI


def initialize_openai_client(api_key: str, base_url: str, model: str, temperature: float) -> (
    Callable[[list[str]], list[str]]
):
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
