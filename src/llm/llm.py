from typing import Any

import torch
from transformers.pipelines.text_generation import ChatType

from lamp import LABELS

from .huggingface import HuggingFaceLLM
from .vllm import VLLMClient


class LLM:
    def __init__(
        self,
        task: str,
        model: str,
        backend: str,
        authority: str | None,
        generation_kwargs: dict[str, Any]
    ) -> None:
        self.task = task

        if backend == "hf":
            self.llm = HuggingFaceLLM(model, generation_kwargs)
        elif backend == "vllm":
            self.llm = VLLMClient(
                model,
                authority,
                generation_kwargs,
                max_concurrency=5
            )
        else:
            raise ValueError(f"Invalid backend: {backend}")

    def generate(
        self,
        user_prompts: list[str],
        assistant_prompts: list[str] | None = None,
        verbose: bool = False
    ) -> list[str] | list[list[str]]:
        if assistant_prompts is None:
            assistant_prompts = [None] * len(user_prompts)

        prompts = [
            self._build_prompt(up, ap)
            for up, ap in zip(user_prompts, assistant_prompts, strict=True)
        ]
        responses = self.llm.generate(prompts, verbose)

        if all(len(r) == 1 for r in responses):
            responses = [r[0] for r in responses]

        return responses

    def compute_completion_logps(
        self,
        user_prompts: list[str],
        completions: list[str],
        assistant_prompts: list[str] | None = None
    ) -> torch.Tensor:
        if assistant_prompts is None:
            assistant_prompts = [None] * len(user_prompts)

        prompts = [
            self._build_prompt(up, ap)
            for up, ap in zip(user_prompts, assistant_prompts, strict=True)
        ]
        completion_logps = self.llm.compute_completion_logps(
            prompts, completions
        )
        return completion_logps

    def _build_prompt(
        self,
        user_prompt: str,
        assistant_prompt: str | None
    ) -> ChatType:
        prompt = [
            {"role": "system", "content": SYSTEM_PROMPTS[self.task]},
            {"role": "user", "content": user_prompt}
        ]

        if assistant_prompt is not None:
            prompt.append({"role": "assistant", "content": assistant_prompt})

        return prompt


SYSTEM_PROMPTS = {
    "LaMP-1": (
        "You are a personalized citation identification chatbot "
        f"who responds with one of the following: {LABELS['LaMP-1']} "
        "based on the given examples without any additional text."
    ),

    "LaMP-2": (
        "You are a personalized movie tagging chatbot "
        f"who responds with one of the following: {LABELS['LaMP-2']} "
        "based on the given examples without any additional text."
    ),

    "LaMP-3": (
        "You are a personalized product rating chatbot "
        f"who responds with one of the following: {LABELS['LaMP-3']} "
        "based on the given examples without any additional text."
    ),

    "LaMP-4": (
        "You are a personalized news headline generation chatbot "
        "who generates a news headline "
        "in a style similar to the given examples without any additional text."
    ),

    "LaMP-5": (
        "You are a personalized scholarly title generation chatbot "
        "who generates a scholarly title "
        "in a style similar to the given examples without any additional text."
    ),

    "LaMP-7": (
        "You are a personalized tweet paraphrasing chatbot "
        "who paraphrases a tweet "
        "in a style similar to the given examples without any additional text."
    ),
 
    "LongLaMP-2": (
        "You are a personalized abstract generation chatbot "
        "who generates an abstract "
        "in a style similar to the given examples without any additional text."
    ),

    "LongLaMP-3": (
        "You are a personalized topic generation chatbot "
        "who generates a topic "
        "in a style similar to the given examples without any additional text."
    ),

    "LongLaMP-4": (
        "You are a personalized product review generation chatbot "
        "who generates a product review "
        "in a style similar to the given examples without any additional text."
    )
}
