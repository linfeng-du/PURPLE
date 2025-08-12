from typing import Callable, TypeAlias


Profile: TypeAlias = dict[str, str]
Message: TypeAlias = list[dict[str, str]]
Metric: TypeAlias = Callable[[list[str], list[str]], dict[str, float]]
PromptGenerator: TypeAlias = Callable[[str, list[Profile], str | None, list[str] | None, float], str]
