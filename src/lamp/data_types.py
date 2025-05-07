from typing import Callable, TypeAlias


Profile: TypeAlias = dict[str, str]
Metric: TypeAlias = Callable[[list[str], list[str]], dict[str, float]]
PromptGenerator: TypeAlias = Callable[[str, list[Profile], str, list[str], float], str]
