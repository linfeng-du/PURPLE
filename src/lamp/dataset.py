import json
from typing import TypedDict

from torch.utils.data import Dataset

from .data_types import PromptGenerator


class LaMPExample(TypedDict):

    id: str
    source: str
    target: str


class LaMPDataset(Dataset):

    def __init__(self, task: str, split: str, prompt_generator: PromptGenerator | None = None) -> None:
        with open(f'./dataset/{task}/{split}_questions.json', 'r') as file:
            self.examples= json.load(file)

        with open(f'./dataset/{task}/{split}_outputs.json', 'r') as file:
            outputs = json.load(file)
            self.targets = {gold['id']: gold['output'] for gold in outputs['golds']}

        self.prompt_generator = prompt_generator

    def __getitem__(self, index: int) -> LaMPExample:
        example = self.examples[index]

        id_ = example['id']
        source = example['input']
        target = self.targets[id_]

        if self.prompt_generator is not None:
            source = self.prompt_generator(source, example['profile'])

        return LaMPExample(
            id=id_,
            source=source,
            target=target
        )

    def __len__(self) -> int:
        return len(self.examples)
