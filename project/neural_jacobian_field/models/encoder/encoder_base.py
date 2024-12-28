from abc import ABC, abstractmethod
from typing import Generic, TypeVar

from jaxtyping import Float
from omegaconf import DictConfig
from torch import Tensor, nn

T = TypeVar("T")


class Encoder(nn.Module, ABC, Generic[T]):
    cfg: DictConfig

    def __init__(self, cfg: T):
        super().__init__()
        self.cfg = cfg

    @abstractmethod
    def forward(
        self, rgb: Float[Tensor, "batch 3 height width"]
    ) -> Float[Tensor, "batch channel new_height new_width"]:
        pass

    @abstractmethod
    def get_output_dim(self) -> int:
        pass
