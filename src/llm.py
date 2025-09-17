import asyncio
import logging
from typing import TypeAlias

import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer, pipeline
from transformers.pipelines.text_generation import Chat

from openai import AsyncOpenAI, OpenAIError
from tqdm import tqdm

from lamp import get_labels


logger = logging.getLogger(__name__)
Message: TypeAlias = list[dict[str, str]]


class LLM:

    def __init__(self, task: str, model: str, provider: str, generate_config: dict, endpoint: str = None) -> None:
        self.task = task
        self.model = model
        self.provider = provider
        self.generate_config = generate_config

        if self.provider == 'local':
            self.pipeline = pipeline('text-generation', model=self.model, device_map='auto', dtype='bfloat16')
            self.tokenizer = self.pipeline.tokenizer
            self._setup_tokenizer()
        elif self.provider == 'vllm':
            self.client = AsyncOpenAI(api_key='EMPTY', base_url=f'http://{endpoint}/v1')
            self.tokenizer = AutoTokenizer.from_pretrained(self.model)
            self._setup_tokenizer()
        elif self.provider == 'openai':
            self.client = AsyncOpenAI(base_url='https://api.openai.com/v1')
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

        self.tokenizer.padding_side = 'left'

    def generate(self, prompts: list[str], verbose: bool = False) -> list[str] | list[list[str]]:
        if self.provider == 'local':
            return self._generate_local(prompts, verbose)
        elif self.provider == 'openai':
            return asyncio.run(self._generate_openai(prompts, verbose))

    def _generate_local(self, prompts: list[str], verbose: bool) -> list[str] | list[list[str]]:
        responses = []
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

        return responses

    async def _generate_openai(self, prompts: list[str], verbose: bool) -> list[str]:
        semaphore = asyncio.Semaphore(value=10)

        async def _request_response(prompt: str) -> str:
            response = None
            num_retries = 0

            while response is None:
                try:
                    async with semaphore:
                        output = await self.client.chat.completions.create(
                            messages=self._create_message(prompt),
                            model=self.model,
                            **self.generate_config
                        )

                    response = output.choices[0].message.content
                except OpenAIError as err:
                    logger.error(f'OpenAI API error: {err}', exc_info=True)
                    num_retries += 1
                    await asyncio.sleep(min(2 ** num_retries, 60))

            return response

        tqdm_prompts = tqdm(prompts, desc='Generating responses', disable=(not verbose))
        tasks = [asyncio.create_task(_request_response(prompt)) for prompt in tqdm_prompts]
        return await asyncio.gather(*tasks)

    def compute_target_logps(self, prompts: list[str], targets: list[str]) -> torch.Tensor:
        if self.provider == 'local':
            return self._compute_target_logps_local(prompts, targets)
        elif self.provider == 'vllm':
            return asyncio.run(self._compute_target_logps_vllm(prompts, targets))
        else:
            raise ValueError(f'Invalid provider for computing target logps: {self.provider}')

    def _compute_target_logps_local(self, prompts: list[str], targets: list[str]) -> torch.Tensor:
        inputs_ids = self.apply_chat_template(prompts)
        targets_ids = self.tokenizer(targets, add_special_tokens=False)['input_ids']
        targets_ids = [target_ids + self.end_token_ids for target_ids in targets_ids]
        return self.compute_target_id_logps(inputs_ids, targets_ids)

    async def _compute_target_logps_vllm(self, prompts: list[str], targets: list[str]) -> torch.Tensor:
        semaphore = asyncio.Semaphore(value=10)

        async def _request_logp(prompt: str) -> float:
            logp = None
            num_retries = 0

            while logp is None:
                try:
                    async with semaphore:
                        output = await self.client.completions.create(
                            model=self.model,
                            prompt=prompt,
                            echo=True,
                            logprobs=0,
                            max_tokens=0
                        )

                    logps = output.choices[0].logprobs.token_logprobs
                    logp = sum(logps[1:])
                except OpenAIError as err:
                    logger.error(f'VLLM API error: {err}', exc_info=True)
                    num_retries += 1
                    await asyncio.sleep(min(2 ** num_retries, 60))

            return logp

        chat_prompts = self.apply_chat_template(prompts, tokenize=False)
        targets = [target + ''.join(self.end_tokens) for target in targets]
        chat_prompts = [chat_prompt + target for chat_prompt, target in zip(chat_prompts, targets)]

        tasks = [asyncio.create_task(_request_logp(chat_prompt)) for chat_prompt in chat_prompts]
        logps = await asyncio.gather(*tasks)
        return torch.tensor(logps)

    @torch.no_grad()
    def compute_target_id_logps(self, inputs_ids: list[list[int]], targets_ids: list[list[int]]) -> torch.Tensor:
        target_logps = []
        model_max_length = self.tokenizer.model_max_length

        for input_ids, target_ids in zip(inputs_ids, targets_ids):
            if len(input_ids) + len(target_ids) > model_max_length:
                target_max_length = model_max_length - len(input_ids)
                assert target_max_length > 0
                target_ids = target_ids[:target_max_length]

            target_length = len(target_ids)
            concat_ids = input_ids + target_ids

            concat_ids = torch.tensor([concat_ids], device=self.pipeline.device)
            labels = concat_ids[:, 1:]
            outputs = self.pipeline.model(input_ids=concat_ids)

            logps = torch.log_softmax(outputs.logits[:, :-1, :], dim=2)
            token_logps = logps.gather(dim=2, index=labels.unsqueeze(dim=2)).squeeze(dim=2)
            token_mask = torch.zeros_like(token_logps)
            token_mask[0, -target_length:] = 1.

            target_logp = torch.sum(token_logps * token_mask, dim=1)
            target_logps.append(target_logp)

        return torch.cat(target_logps, dim=0)

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


SYSTEM_PROMPTS = {
    'LaMP-1': (
        f'You are a personalized citation identification chatbot '
        f'who responds with one of the following: {get_labels("LaMP-1")} based on the given examples.'
    ),
    'LaMP-2': (
        f'You are a personalized movie tagging chatbot '
        f'who responds with one of the following: {get_labels("LaMP-2")} based on the given examples.'
    ),
    'LaMP-3': (
        f'You are a personalized product rating chatbot '
        f'who responds with one of the following: {get_labels("LaMP-3")} based on the given examples.'
    ),
    'LaMP-4': (
        f'You are a personalized news headline generation chatbot '
        f'who generates a news headline in a style similar to the given examples without any additional text.'
    ),
    'LaMP-5': (
        f'You are a personalized scholarly title generation chatbot '
        f'who generates a scholarly title in a style similar to the given examples without any additional text.'
    ),
    'LaMP-6': (
        f'You are a personalized email subject generation chatbot '
        f'who generates an email subject in a style similar to the given examples without any additional text.'
    ),
    'LaMP-7': (
        f'You are a personalized tweet paraphrasing chatbot '
        f'who paraphrases a tweet in a style similar to the given examples without any additional text.'
    ),
    'LongLaMP-1': (
        f'You are a personalized email completion chatbot '
        f'who completes an email in a style similar to the given examples without any additional text.'
    ),
    'LongLaMP-2': (
        f'You are a personalized abstract generation chatbot '
        f'who generates an abstract in a style similar to the given examples without any additional text.'
    ),
    'LongLaMP-3': (
        f'You are a personalized topic generation chatbot '
        f'who generates a topic in a style similar to the given examples without any additional text.'
    ),
    'LongLaMP-4': (
        f'You are a personalized product review generation chatbot '
        f'who generates a product review in a style similar to the given examples without any additional text.'
    )
}
