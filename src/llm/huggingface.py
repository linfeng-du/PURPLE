from typing import Any

from tqdm import tqdm

import torch
from torch.utils.data import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from transformers.pipelines.text_generation import Chat, ChatType

from .utils import (
    compute_chat_template_length,
    encode_prompt_and_completion,
    truncate_user_prompt
)


class HuggingFaceLLM:
    def __init__(self, model: str, generation_kwargs: dict[str, Any]) -> None:
        self.generation_kwargs = generation_kwargs

        self.model = AutoModelForCausalLM.from_pretrained(
            model,
            torch_dtype="bfloat16",
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
    def compute_completion_logps(
        self,
        prompts: list[ChatType],
        completions: list[str]
    ) -> torch.Tensor:
        completion_logps = []

        for prompt, completion in zip(prompts, completions, strict=True):
            input_ids, prompt_length = encode_prompt_and_completion(
                prompt,
                completion,
                self.generation_kwargs["max_new_tokens"],
                self.chat_template_length,
                self.tokenizer
            )
            input_ids = input_ids.to(self.pipeline.device)

            labels = input_ids[:, 1:]
            outputs = self.pipeline.model(input_ids=input_ids)

            logps = outputs.logits[:, :-1, :].log_softmax(dim=-1)
            token_logps = (
                logps.gather(dim=-1, index=labels.unsqueeze(dim=-1))
                .squeeze(dim=-1)
            )
            token_mask = torch.ones_like(token_logps, dtype=torch.long)
            token_mask[0, :prompt_length - 1] = 0

            completion_logp = (token_logps * token_mask).sum(dim=-1)
            completion_logps.append(completion_logp)

        return torch.cat(completion_logps)


class _ChatDataset(Dataset):
    def __init__(self, prompts: list[ChatType]) -> None:
        # Wrap in Chat objects for pipeline compatibility
        self.chats = [Chat(p) for p in prompts]

    def __len__(self) -> int:
        return len(self.chats)

    def __getitem__(self, index: int) -> Chat:
        return self.chats[index]
