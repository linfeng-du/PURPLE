import asyncio
import logging
from typing import TypeAlias

import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer, pipeline
from transformers.pipelines.text_generation import Chat

from openai import AsyncOpenAI, OpenAIError
from tqdm import tqdm

from .system_prompts import SYSTEM_PROMPTS


logger = logging.getLogger(__name__)
Message: TypeAlias = list[dict[str, str]]


class LLM:

    def __init__(
        self, task: str, model: str,
        provider: str, endpoint: str | None,
        generate_config: dict
    ) -> None:
        self.task = task
        self.model = model
        self.provider = provider
        self.endpoint = endpoint
        self.generate_config = generate_config

        if self.provider == 'local':
            self.pipeline = pipeline(
                task='text-generation',
                model=self.model,
                device_map='cuda',
                torch_dtype='bfloat16'
            )
            self.tokenizer = self.pipeline.tokenizer
            self._setup_tokenizer()
        elif self.provider == 'vllm':
            self.client = AsyncOpenAI(api_key='EMPTY', base_url=f'http://{self.endpoint}/v1')
            self.tokenizer = AutoTokenizer.from_pretrained(self.model)
            self._setup_tokenizer()

            self.loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self.loop)
        else:
            raise ValueError(f'Invalid provider: {self.provider}')

    def _setup_tokenizer(self) -> None:
        if self.model == 'microsoft/Phi-4-mini-instruct':
            self.end_tokens = ['<|end|>', '<|endoftext|>']
            self.end_token_ids = self.tokenizer.encode(self.end_tokens)
        else:
            self.end_tokens = [self.tokenizer.eos_token]
            self.end_token_ids = [self.tokenizer.eos_token_id]

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

            if self.provider == 'local':
                self.pipeline.model.generation_config.pad_token_id = self.tokenizer.pad_token_id

        if self.model == 'meta-llama/Meta-Llama-3-70B-Instruct':
            self.tokenizer.model_max_length = 8192

        self.tokenizer.padding_side = 'left'

    def generate(self, prompts: list[str], apply_template: bool = True, verbose: bool = False) -> (
        list[str] | list[list[str]]
    ):
        if self.provider == 'local':
            return self._generate_local(prompts, apply_template, verbose)
        elif self.provider == 'vllm':
            return self.loop.run_until_complete(self._generate_api(prompts, apply_template, verbose))

    def _generate_local(self, prompts: list[str], apply_template: bool, verbose: bool) -> (
        list[str] | list[list[str]]
    ):
        responses = []

        if apply_template:
            dataset = _ChatDataset([self._create_message(prompt) for prompt in prompts])

            for outputs in tqdm(
                self.pipeline(dataset, **self.generate_config),
                desc='Generating responses',
                total=len(dataset),
                disable=(not verbose)
            ):
                if len(outputs) == 1:
                    response = outputs[0]['generated_text'][-1]['content']
                    responses.append(response)
                elif len(outputs) > 1:
                    all_responses = [output['generated_text'][-1]['content'] for output in outputs]
                    responses.append(all_responses)
        else:
            prompt_lengths = [len(prompt) for prompt in prompts]

            for outputs, prompt_length in tqdm(
                zip(self.pipeline(prompts, **self.generate_config), prompt_lengths),
                desc='Generating responses',
                total=len(prompts),
                disable=(not verbose)
            ):
                if len(outputs) == 1:
                    response = outputs[0]['generated_text'][prompt_length:]
                    responses.append(response)
                elif len(outputs) > 1:
                    all_responses = [output['generated_text'][prompt_length:] for output in outputs]
                    responses.append(all_responses)

        return responses

    async def _generate_api(self, prompts: list[str], apply_template: bool, verbose: bool) -> (
        list[str] | list[list[str]]
    ):
        semaphore = asyncio.Semaphore(value=5)

        async def _request_response(prompt: str, pbar: tqdm) -> str | list[str]:
            response = None
            num_retries = 0

            while response is None:
                try:
                    if apply_template:
                        # Truncate messages longer than the model's max length
                        message_ids = self.apply_chat_template([prompt])[0]
                        total_length = len(message_ids) + self.generate_config['max_completion_tokens']

                        if total_length > self.tokenizer.model_max_length:
                            prompt_ids = self.tokenizer.encode(prompt, add_special_tokens=False)
                            prompt = self.tokenizer.decode(
                                prompt_ids[total_length - self.tokenizer.model_max_length:],
                                skip_special_tokens=True
                            )

                        message = self._create_message(prompt)

                        async with semaphore:
                            output = await self.client.chat.completions.create(
                                messages=message,
                                model=self.model,
                                **self.generate_config
                            )

                        if len(output.choices) == 1:
                            response = output.choices[0].message.content
                        elif len(output.choices) > 1:
                            response = [choice.message.content for choice in output.choices]
                    else:
                        async with semaphore:
                            output = await self.client.completions.create(
                                model=self.model,
                                prompt=prompt,
                                **self.generate_config
                            )

                        if len(output.choices) == 1:
                            response = output.choices[0].text
                        elif len(output.choices) > 1:
                            response = [choice.text for choice in output.choices]

                    pbar.update(1)
                except OpenAIError as err:
                    logger.error(f'OpenAI API error: {err}', exc_info=True)
                    num_retries += 1
                    await asyncio.sleep(min(2 ** num_retries, 60))

            return response

        with tqdm(total=len(prompts), desc='Generating responses', disable=(not verbose)) as pbar:
            tasks = [asyncio.create_task(_request_response(prompt, pbar)) for prompt in prompts]
            responses = await asyncio.gather(*tasks)

        return responses

    def compute_target_logps(self, prompts: list[str], targets: list[str], apply_template: bool = True) -> (
        torch.Tensor
    ):
        if self.provider == 'local':
            return self._compute_target_logps_local(prompts, targets, apply_template)
        elif self.provider == 'vllm':
            return self.loop.run_until_complete(
                self._compute_target_logps_api(prompts, targets, apply_template)
            )
        else:
            raise ValueError(f'Invalid provider for computing target logps: {self.provider}')

    def _compute_target_logps_local(
        self, prompts: list[str], targets: list[str], apply_template: bool = True
    ) -> torch.Tensor:
        if apply_template:
            inputs_ids = self.apply_chat_template(prompts)
            targets_ids = self.tokenizer(targets, add_special_tokens=False)['input_ids']
            targets_ids = [target_ids + self.end_token_ids for target_ids in targets_ids]
        else:
            inputs_ids = self.tokenizer(prompts, add_special_tokens=False)['input_ids']
            targets_ids = self.tokenizer(targets, add_special_tokens=False)['input_ids']

        target_logps = []
        model_max_length = self.tokenizer.model_max_length

        for input_ids, target_ids in zip(inputs_ids, targets_ids):
            if len(input_ids) + len(target_ids) > model_max_length:
                target_max_length = model_max_length - len(input_ids)
                assert target_max_length > 0
                target_ids = target_ids[:target_max_length]

            concat_ids = input_ids + target_ids
            concat_ids = torch.tensor([concat_ids], device=self.pipeline.device)
            labels = concat_ids[:, 1:]

            with torch.no_grad():
                outputs = self.pipeline.model(input_ids=concat_ids)

            logps = torch.log_softmax(outputs.logits[:, :-1, :], dim=2)
            token_logps = logps.gather(dim=2, index=labels.unsqueeze(dim=2)).squeeze(dim=2)
            token_mask = torch.zeros_like(token_logps)
            token_mask[0, -len(target_ids):] = 1.

            target_logp = torch.sum(token_logps * token_mask, dim=1)
            target_logps.append(target_logp)

        return torch.cat(target_logps, dim=0)

    async def _compute_target_logps_api(
        self, prompts: list[str], targets: list[str], apply_template: bool = True
    ) -> torch.Tensor:
        semaphore = asyncio.Semaphore(value=5)

        async def _request_logp(prompt: str, target: str) -> float:
            # Truncate messages longer than the model's max length
            message_ids = (
                self.apply_chat_template([prompt])[0]
                if apply_template
                else self.tokenizer.encode(prompt, add_special_tokens=False)
            )
            target += ''.join(self.end_tokens)
            target_length = len(self.tokenizer.encode(target, add_special_tokens=False))
            total_length = len(message_ids) + target_length

            if total_length > self.tokenizer.model_max_length:
                prompt_ids = self.tokenizer.encode(prompt, add_special_tokens=False)
                prompt = self.tokenizer.decode(
                    prompt_ids[total_length - self.tokenizer.model_max_length + 5:],
                    skip_special_tokens=True
                )

            if apply_template:
                prompt = self.apply_chat_template([prompt], tokenize=False)[0]

            logp = None
            num_retries = 0

            while logp is None:
                try:
                    async with semaphore:
                        output = await self.client.completions.create(
                            model=self.model,
                            prompt=prompt + target,
                            echo=True,
                            logprobs=0,
                            max_tokens=0
                        )

                    logps = output.choices[0].logprobs.token_logprobs
                    logp = sum(logps[-target_length:])
                except OpenAIError as err:
                    logger.error(f'VLLM API error: {err}', exc_info=True)
                    num_retries += 1
                    await asyncio.sleep(min(2 ** num_retries, 60))

            return logp

        tasks = [
            asyncio.create_task(_request_logp(prompt, target))
            for prompt, target in zip(prompts, targets)
        ]
        logps = await asyncio.gather(*tasks)
        return torch.tensor(logps)

    def apply_chat_template(self, prompts: list[str], tokenize: bool = True) -> list[str] | list[list[int]]:
        return self.tokenizer.apply_chat_template(
            [self._create_message(prompt) for prompt in prompts],
            add_generation_prompt=True,
            tokenize=tokenize
        )

    def _create_message(self, prompt: str) -> Message:
        return [
            {'role': 'system', 'content': SYSTEM_PROMPTS[self.task]},
            {'role': 'user', 'content': prompt}
        ]


class _ChatDataset(Dataset):

    def __init__(self, messages: list[Message]) -> None:
        self.chats = [Chat(message) for message in messages]

    def __len__(self) -> int:
        return len(self.chats)

    def __getitem__(self, index: int) -> Chat:
        return self.chats[index]
