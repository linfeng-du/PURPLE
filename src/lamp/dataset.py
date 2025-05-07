import json

from .data_types import LaMPExample


def load_lamp_dataset(task: str, split: str) -> list[LaMPExample]:
    with open(f'./dataset/{task}/{split}_questions.json', 'r') as file:
        questions = json.load(file)

    with open(f'./dataset/{task}/{split}_outputs.json', 'r') as file:
        outputs = {gold['id']: gold['output'] for gold in json.load(file)['golds']}

    examples = []

    for question in questions:
        example = LaMPExample(
            source=question['input'],
            profile=question['profile'],
            target=outputs[question['id']]
        )
        examples.append(example)

    return examples
