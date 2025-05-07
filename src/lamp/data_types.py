from typing import Callable, TypeAlias, TypedDict


Profile: TypeAlias = dict[str, str]
Metric: TypeAlias = Callable[[list[str], list[str]], dict[str, float]]

PromptGenerator: TypeAlias = Callable[[str, list[Profile], float], str]
QueryCorpusGenerator: TypeAlias = Callable[[str, list[Profile]], tuple[str, list[str]]]
Retriever: TypeAlias = Callable[[str, list[Profile], int, QueryCorpusGenerator], list[Profile]]


class LaMPExample(TypedDict):

    source: str
    profile: list[Profile]
    target: str
