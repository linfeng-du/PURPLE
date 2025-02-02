from openai import OpenAI


client = OpenAI()


def request_completions(prompts: list[str], model: str, temperature: float) -> list[str]:
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
