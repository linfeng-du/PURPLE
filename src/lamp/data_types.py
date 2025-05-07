from typing import Callable, TypeAlias


Profile: TypeAlias = dict[str, str]
PromptGenerator: TypeAlias = Callable[[str, list[Profile], float], str]
QueryCorpusGenerator: TypeAlias = Callable[[str, list[Profile]], tuple[str, list[str]]]
