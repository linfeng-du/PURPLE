import logging
import os
from typing import TypeAlias

import torch
from torch.utils.data import Dataset
from transformers import pipeline
from transformers.pipelines.text_generation import Chat

from openai import OpenAI, OpenAIError
from tqdm import tqdm

from lamp import get_labels


logger = logging.getLogger(__name__)
Message: TypeAlias = list[dict[str, str]]


class LLM:

    def __init__(self, task: str, model: str, provider: str, generate_config: dict) -> None:
        self.task = task
        self.model = model
        self.provider = provider
        self.generate_config = generate_config

        if self.provider == 'local':
            self.device = torch.device(f'cuda:{torch.cuda.device_count() - 1}')
            self.pipeline = pipeline(
                task='text-generation',
                model=self.model,
                device=self.device,
                torch_dtype=torch.bfloat16
            )
            self.pipeline.tokenizer.padding_side = 'left'

            if self.model == 'microsoft/Phi-4-mini-instruct':
                eos_token = '<|end|>'
                eos_token_id = self.pipeline.tokenizer.encode(eos_token)[0]
                self.pipeline.tokenizer.eos_token = eos_token
                self.pipeline.tokenizer.eos_token_id = eos_token_id

            if self.pipeline.tokenizer.pad_token is None:
                self.pipeline.tokenizer.pad_token = self.pipeline.tokenizer.eos_token
                self.pipeline.model.generation_config.pad_token_id = self.pipeline.tokenizer.eos_token_id
        elif self.provider == 'openai':
            self.client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'), base_url='https://api.openai.com/v1')
        else:
            raise ValueError(f'Invalid provider: {self.provider}')

    def generate(self, prompts: list[str], verbose: bool = False) -> list[str] | list[list[str]]:
        if self.provider == 'local':
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
        elif self.provider == 'openai':
            responses = [None] * len(prompts)
            remaining_prompts = set(range(len(prompts)))

            while remaining_prompts:
                for index in tqdm(
                    list(remaining_prompts),
                    desc='Requesting responses',
                    disable=(not verbose)
                ):
                    try:
                        completion = self.client.chat.completions.create(
                            model=self.model,
                            messages=self._create_message(prompts[index]),
                            **self.generate_config
                        )
                        responses[index] = completion.choices[0].message.content
                        remaining_prompts.remove(index)
                    except OpenAIError as err:
                        logger.error(f'OpenAI API error: {err}', exc_info=True)

            return responses

    def compute_target_logps(self, prompts: list[str], targets: list[str]) -> torch.Tensor:
        inputs_ids = self.apply_chat_template(prompts)
        targets_ids = self.pipeline.tokenizer(targets, add_special_tokens=False)['input_ids']
        targets_ids = [target_ids + [self.pipeline.tokenizer.eos_token_id] for target_ids in targets_ids]
        return self.compute_target_id_logps(inputs_ids, targets_ids)

    @torch.no_grad()
    def compute_target_id_logps(self, inputs_ids: list[list[int]], targets_ids: list[list[int]]) -> torch.Tensor:
        target_logps = []
        model_max_length = self.pipeline.tokenizer.model_max_length

        for input_ids, target_ids in zip(inputs_ids, targets_ids):
            if len(input_ids) + len(target_ids) > model_max_length:
                target_max_length = model_max_length - len(input_ids)
                assert target_max_length > 0
                target_ids = target_ids[:target_max_length]

            target_length = len(target_ids)
            concat_ids = input_ids + target_ids

            concat_ids = torch.tensor([concat_ids], device=self.device)
            labels = concat_ids[:, 1:]
            outputs = self.pipeline.model(input_ids=concat_ids)

            logps = torch.log_softmax(outputs.logits[:, :-1, :], dim=2)
            token_logps = logps.gather(dim=2, index=labels.unsqueeze(dim=2)).squeeze(dim=2)
            token_mask = torch.zeros_like(token_logps)
            token_mask[0, -target_length:] = 1.

            target_logp = torch.sum(token_logps * token_mask, dim=1)
            target_logps.append(target_logp)

        return torch.cat(target_logps, dim=0)

    def apply_chat_template(self, prompts: list[str]) -> list[list[int]]:
        return self.pipeline.tokenizer.apply_chat_template(
            [self._create_message(prompt) for prompt in prompts],
            add_generation_prompt=True,
            tokenize=True
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
