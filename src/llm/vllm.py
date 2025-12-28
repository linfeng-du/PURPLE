import asyncio
import logging
from typing import Any

from openai import AsyncOpenAI, OpenAIError
from tqdm import tqdm

import torch
from transformers import AutoTokenizer
from transformers.pipelines.text_generation import ChatType

from .utils import (
    compute_chat_template_length,
    encode_prompt_and_completion,
    truncate_user_prompt
)


logger = logging.getLogger(__name__)


class VLLMClient:

    def __init__(
        self,
        model: str,
        authority: str,
        generation_kwargs: dict[str, Any],
        max_concurrency: int
    ) -> None:
        self.model = model
        self.generation_kwargs = generation_kwargs
        self.max_concurrency = max_concurrency

        self.client = AsyncOpenAI(base_url=f"http://{authority}/v1")

        self.tokenizer = AutoTokenizer.from_pretrained(self.model)
        # The user instruction is often on the right;
        # keep it via left truncation
        self.tokenizer.truncation_side = "left"

        # Set by vLLM
        if self.model == "meta-llama/Meta-Llama-3-70B-Instruct":
            self.tokenizer.model_max_length = 8192
        elif self.model == "Qwen/Qwen3-Next-80B-A3B-Instruct":
            self.tokenizer.model_max_length = 262144

        self.chat_template_length = compute_chat_template_length(
            self.tokenizer
        )

    def generate(
        self,
        prompts: list[ChatType],
        verbose: bool
    ) -> list[list[str]]:
        new_prompts = [
            truncate_user_prompt(
                p,
                self.generation_kwargs["max_completion_tokens"],
                self.chat_template_length,
                self.tokenizer
            )
            for p in prompts
        ]

        async def generate_async() -> list[list[str]]:
            semaphore = asyncio.Semaphore(value=self.max_concurrency)

            with tqdm(
                total=len(new_prompts),
                desc="Requesting completions",
                disable=(not verbose)
            ) as pbar:
                tasks = [
                    self._request_completions(p, i, semaphore, pbar)
                    for i, p in enumerate(new_prompts)
                ]
                return await asyncio.gather(*tasks)

        return asyncio.run(generate_async())

    async def _request_completions(
        self,
        prompt: ChatType,
        index: int,
        semaphore: asyncio.Semaphore,
        pbar: tqdm
    ) -> list[str]:
        completions = None
        num_retries = 0

        continue_final_message = (prompt[-1]["role"] == "assistant")
        extra_body = {
            "add_generation_prompt": (not continue_final_message),
            "continue_final_message": continue_final_message,
        }

        while completions is None:
            try:
                async with semaphore:
                    output = await self.client.chat.completions.create(
                        messages=prompt,
                        model=self.model,
                        extra_body=extra_body,
                        **self.generation_kwargs
                    )
                    completions = [c.message.content for c in output.choices]

            except OpenAIError as err:
                num_retries += 1

                if num_retries > 10:
                    raise

                logger.warning(
                    f"[{index}] {err}\nRetrying {num_retries}/10..."
                )
                await asyncio.sleep(min(2 ** num_retries, 60))

        pbar.update()
        return completions

    def compute_completion_logps(
        self,
        prompts: list[ChatType],
        completions: list[str]
    ) -> torch.Tensor:
        input_ids = []
        prompt_lengths = []

        for prompt, completion in zip(prompts, completions, strict=True):
            prompt_completion_ids, prompt_length = (
                encode_prompt_and_completion(
                    prompt,
                    completion,
                    self.generation_kwargs["max_completion_tokens"],
                    self.chat_template_length,
                    self.tokenizer
                )
            )
            input_ids.append(prompt_completion_ids.squeeze(dim=0).tolist())
            prompt_lengths.append(prompt_length)

        async def compute_completion_logps_async() -> list[float]:
            semaphore = asyncio.Semaphore(value=self.max_concurrency)
            tasks = [
                self._request_completion_logp(ii, pl, i, semaphore)
                for i, (ii, pl) in enumerate(
                    zip(input_ids, prompt_lengths, strict=True)
                )
            ]
            return await asyncio.gather(*tasks)

        completion_logps = asyncio.run(compute_completion_logps_async())
        return torch.tensor(completion_logps, dtype=torch.float32)

    async def _request_completion_logp(
        self,
        prompt_completion_ids: list[int],
        prompt_length: int,
        index: int,
        semaphore: asyncio.Semaphore
    ) -> float:
        completion_logp = None
        num_retries = 0

        while completion_logp is None:
            try:
                async with semaphore:
                    output = await self.client.completions.create(
                        model=self.model,
                        prompt=prompt_completion_ids,
                        echo=True,
                        logprobs=0,
                        max_tokens=0
                    )
                    token_logps = output.choices[0].logprobs.token_logprobs
                    completion_logp = sum(token_logps[prompt_length:])

            except OpenAIError as err:
                num_retries += 1

                if num_retries > 10:
                    raise

                logger.warning(
                    f"[{index}] {err}\nRetrying {num_retries}/10..."
                )
                await asyncio.sleep(min(2 ** num_retries, 60))

        return completion_logp
