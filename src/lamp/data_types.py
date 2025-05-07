from typing import Callable, TypeAlias, TypedDict


Profile: TypeAlias = dict[str, str]
PromptGenerator: TypeAlias = Callable[[str, list[Profile], float], str]
QueryCorpusGenerator: TypeAlias = Callable[[str, list[Profile]], tuple[str, list[str]]]


class LaMPExample(TypedDict):

    id: str
    source: str
    target: str
