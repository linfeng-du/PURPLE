from copy import deepcopy

import torch
from transformers import PreTrainedTokenizerBase
from transformers.pipelines.text_generation import ChatType


def compute_chat_template_length(tokenizer: PreTrainedTokenizerBase) -> int:
    messages = [
        {"role": "system", "content": ""}, {"role": "user", "content": ""}
    ]
    chat_template_length = len(
        tokenizer.apply_chat_template(messages, add_generation_prompt=True)
    )
    return chat_template_length


def truncate_user_prompt(
    prompt: ChatType,
    max_completion_length: int,
    chat_template_length: int,
    tokenizer: PreTrainedTokenizerBase
) -> ChatType:
    prompt = deepcopy(prompt)

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


def encode_prompt_and_completion(
    prompt: ChatType,
    completion: str,
    max_completion_length: int,
    chat_template_length: int,
    tokenizer: PreTrainedTokenizerBase
) -> tuple[torch.Tensor, int]:
    new_prompt = truncate_user_prompt(
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

    prompt_completion = deepcopy(new_prompt)

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

    assert torch.equal(prompt_completion_ids[0, :prompt_length], prompt_ids[0])
    return prompt_completion_ids, prompt_length
