from typing import Callable, TypeAlias

import torch


Reward: TypeAlias = Callable[[list[str], list[str]], torch.Tensor]
