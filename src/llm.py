import asyncio
import copy
import logging
from typing import Any

from openai import AsyncOpenAI, OpenAIError
from tqdm import tqdm

import torch
from torch.utils.data import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedTokenizerBase,
    pipeline
)
from transformers.pipelines.text_generation import Chat, ChatType


logger = logging.getLogger(__name__)


def create_llm(backend: str, **kwargs: Any) -> "HFLLM | VLLMClient":
    if backend == "hf":
        return HFLLM(**kwargs)
    elif backend == "vllm":
        return VLLMClient(**kwargs)
    else:
        raise ValueError(f"Invalid backend: {backend}")


class HFLLM:
    def __init__(self, model: str, generation_kwargs: dict[str, Any]) -> None:
        self.generation_kwargs = generation_kwargs

        self.model = AutoModelForCausalLM.from_pretrained(
            model,
            dtype="bfloat16",
            device_map="auto",
            attn_implementation="flash_attention_2"
        )

        self.tokenizer = AutoTokenizer.from_pretrained(model)
        self.tokenizer.padding_side = "left"

        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        self.pipeline = pipeline(
            task="text-generation",
            model=self.model,
            tokenizer=self.tokenizer
        )
        self.chat_template_length = _compute_chat_template_length(
            self.tokenizer
        )

    def generate(
        self,
        prompts: list[ChatType],
        verbose: bool = False
    ) -> list[list[str]]:
        new_prompts = [
            _truncate_user_prompt(
                p,
                self.generation_kwargs["max_new_tokens"],
                self.chat_template_length,
                self.tokenizer
            )
            for p in prompts
        ]

        # Wrap in a Dataset object to show progress streamingly
        dataset = _ChatDataset(new_prompts)
        responses = []

        for outputs in tqdm(
            self.pipeline(
                dataset, return_full_text=False, **self.generation_kwargs
            ),
            desc="Generating completions",
            total=len(dataset),
            disable=not verbose
        ):
            responses.append([o["generated_text"] for o in outputs])

        return responses

    @torch.no_grad()
    def compute_completion_logprobs(
        self,
        prompts: list[ChatType],
        completions: list[str]
    ) -> torch.Tensor:
        completion_logprobs = []

        for prompt, completion in zip(prompts, completions, strict=True):
            input_ids, prompt_length = _encode_prompt_and_completion(
                prompt,
                completion,
                self.generation_kwargs["max_new_tokens"],
                self.chat_template_length,
                self.tokenizer
            )
            input_ids = input_ids.to(self.pipeline.device)

            labels = input_ids[:, 1:]
            outputs = self.pipeline.model(input_ids=input_ids)

            logprobs = outputs.logits[:, :-1, :].log_softmax(dim=-1)
            token_logprobs = (
                logprobs.gather(dim=-1, index=labels.unsqueeze(dim=-1))
                .squeeze(dim=-1)
            )
            token_mask = torch.ones_like(token_logprobs, dtype=torch.long)
            token_mask[0, :prompt_length - 1] = 0

            completion_logp = (token_logprobs * token_mask).sum(dim=-1)
            completion_logprobs.append(completion_logp)

        return torch.cat(completion_logprobs)


class _ChatDataset(Dataset):
    def __init__(self, prompts: list[ChatType]) -> None:
        # Wrap in Chat objects for pipeline compatibility
        self.chats = [Chat(p) for p in prompts]

    def __len__(self) -> int:
        return len(self.chats)

    def __getitem__(self, index: int) -> Chat:
        return self.chats[index]


class VLLMClient:
    def __init__(
        self,
        model: str,
        vllm_server_host: str,
        vllm_server_port: int,
        generation_kwargs: dict[str, Any],
        max_concurrency: int = 5
    ) -> None:
        self.model = model
        self.generation_kwargs = generation_kwargs
        self.max_concurrency = max_concurrency

        self.client = AsyncOpenAI(
            base_url=f"http://{vllm_server_host}:{vllm_server_port}/v1"
        )

        self.tokenizer = AutoTokenizer.from_pretrained(self.model)
        # The user instruction is often on the right;
        # keep it via left truncation
        self.tokenizer.truncation_side = "left"

        # Set by vLLM
        if self.model == "meta-llama/Meta-Llama-3-70B-Instruct":
            self.tokenizer.model_max_length = 8192
        elif self.model == "Qwen/Qwen3-Next-80B-A3B-Instruct":
            self.tokenizer.model_max_length = 262144

        self.chat_template_length = _compute_chat_template_length(
            self.tokenizer
        )

    def generate(
        self,
        prompts: list[ChatType],
        verbose: bool = False
    ) -> list[list[str]]:
        new_prompts = [
            _truncate_user_prompt(
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
                disable=not verbose
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

        continue_final_message = prompt[-1]["role"] == "assistant"
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

    def compute_completion_logprobs(
        self,
        prompts: list[ChatType],
        completions: list[str]
    ) -> torch.Tensor:
        input_ids = []
        prompt_lengths = []

        for prompt, completion in zip(prompts, completions, strict=True):
            prompt_completion_ids, prompt_length = (
                _encode_prompt_and_completion(
                    prompt,
                    completion,
                    self.generation_kwargs["max_completion_tokens"],
                    self.chat_template_length,
                    self.tokenizer
                )
            )
            input_ids.append(prompt_completion_ids.squeeze(dim=0).tolist())
            prompt_lengths.append(prompt_length)

        async def compute_completion_logprobs_async() -> list[float]:
            semaphore = asyncio.Semaphore(value=self.max_concurrency)
            tasks = [
                self._request_completion_logp(ii, pl, i, semaphore)
                for i, (ii, pl) in enumerate(
                    zip(input_ids, prompt_lengths, strict=True)
                )
            ]
            return await asyncio.gather(*tasks)

        completion_logprobs = asyncio.run(compute_completion_logprobs_async())
        return torch.tensor(completion_logprobs)

    async def _request_completion_logp(
        self,
        prompt_completion_ids: list[int],
        prompt_length: int,
        index: int,
        semaphore: asyncio.Semaphore
    ) -> float:
        completion_logprob = None
        num_retries = 0

        while completion_logprob is None:
            try:
                async with semaphore:
                    output = await self.client.completions.create(
                        model=self.model,
                        prompt=prompt_completion_ids,
                        echo=True,
                        logprobs=0,
                        max_tokens=0
                    )
                    token_logprobs = output.choices[0].logprobs.token_logprobs
                    completion_logprob = sum(token_logprobs[prompt_length:])

            except OpenAIError as err:
                num_retries += 1

                if num_retries > 10:
                    raise

                logger.warning(
                    f"[{index}] {err}\nRetrying {num_retries}/10..."
                )
                await asyncio.sleep(min(2 ** num_retries, 60))

        return completion_logprob


def _compute_chat_template_length(tokenizer: PreTrainedTokenizerBase) -> int:
    messages = [
        {"role": "system", "content": ""}, {"role": "user", "content": ""}
    ]
    chat_template_length = len(
        tokenizer.apply_chat_template(messages, add_generation_prompt=True)
    )
    return chat_template_length


def _truncate_user_prompt(
    prompt: ChatType,
    max_completion_length: int,
    chat_template_length: int,
    tokenizer: PreTrainedTokenizerBase
) -> ChatType:
    prompt = copy.deepcopy(prompt)

    if tokenizer.model_max_length >= 10 ** 30:
        # Llama-3 uses a large integer to indicate unbounded context length
        return prompt

    # If there is an assistant message,
    # assume it already fits within the context budget
    assert prompt[0]["role"] == "system" and prompt[1]["role"] == "user"
    system_prompt = prompt[0]["content"]
    user_prompt = prompt[1]["content"]

    # Truncate only the user prompt; keep the system prompt intact
    system_length = len(
        tokenizer.encode(system_prompt, add_special_tokens=False)
    )
    max_user_length = (
        tokenizer.model_max_length
        - max_completion_length
        - chat_template_length
        - system_length
        # Make space for edge cases
        - 10
    )
    assert max_user_length > 0

    # The user instruction is often on the right; keep it via left truncation
    truncation_side = tokenizer.truncation_side
    tokenizer.truncation_side = "left"

    user_prompt_ids = tokenizer.encode(
        user_prompt,
        add_special_tokens=False,
        truncation=True,
        max_length=max_user_length
    )

    # Restore truncation side
    tokenizer.truncation_side = truncation_side

    new_user_prompt = tokenizer.decode(
        user_prompt_ids, skip_special_tokens=True
    )

    prompt[1]["content"] = new_user_prompt
    return prompt


def _encode_prompt_and_completion(
    prompt: ChatType,
    completion: str,
    max_completion_length: int,
    chat_template_length: int,
    tokenizer: PreTrainedTokenizerBase
) -> tuple[torch.Tensor, int]:
    new_prompt = _truncate_user_prompt(
        prompt, max_completion_length, chat_template_length, tokenizer
    )

    continue_final_message = new_prompt[-1]["role"] == "assistant"
    prompt_ids = tokenizer.apply_chat_template(
        new_prompt,
        add_generation_prompt=not continue_final_message,
        continue_final_message=continue_final_message,
        return_tensors="pt"
    )
    prompt_length = prompt_ids.shape[1]

    prompt_completion = copy.deepcopy(new_prompt)

    if continue_final_message:
        prompt_completion[-1]["content"] += completion
    else:
        prompt_completion.append({"role": "assistant", "content": completion})

    model_max_length = tokenizer.model_max_length

    if model_max_length >= 10 ** 30:
        model_max_length = None

    # Truncate the completion on the right to keep the prompt intact
    truncation_side = tokenizer.truncation_side
    tokenizer.truncation_side = "right"

    # Set `continue_final_message=True` to avoid the EOS token,
    # since some models assign it a very low probability
    prompt_completion_ids = tokenizer.apply_chat_template(
        prompt_completion,
        continue_final_message=True,
        truncation=model_max_length is not None,
        max_length=model_max_length,
        return_tensors="pt"
    )

    # Restore truncation side
    tokenizer.truncation_side = truncation_side
    return prompt_completion_ids, prompt_length
