from abc import ABC, abstractmethod
from typing import Generic, TypeVar

from attr import dataclass
from jaxtyping import Float
from torch import Tensor, nn

T = TypeVar("T")


@dataclass
class PixelEncoding:
    features: Float[Tensor, "B C H W"]
    extrinsics: Float[Tensor, "B 4 4"]
    intrinsics: Float[Tensor, "B 3 3"]
    action: Float[Tensor, "B C"]


@dataclass
class DecoderOutput:
    density: Float[Tensor, "batch ray sample 1"]
    color: Float[Tensor, "batch ray sample 3"]
    flow: Float[Tensor, "batch ray sample 3"]
    action_features: Float[Tensor, "batch ray sample dim"]


@dataclass
class DecoderFeatureOnlyOutput:
    density: Float[Tensor, "batch ray sample 1"]
    action_features: Float[Tensor, "batch ray sample dim"]


class ActionDecoder(nn.Module, ABC, Generic[T]):
    cfg: T

    def __init__(self, cfg: T) -> None:
        super().__init__()
        self.cfg = cfg

    @abstractmethod
    def forward(
        self,
        world_space_xyz: Float[Tensor, "batch ray sample 3"],
        world_space_dir: Float[Tensor, "batch ray sample 3"],
        pixel_encoding: PixelEncoding,
    ) -> DecoderOutput:
        pass

    @abstractmethod
    def encode_image(
        self,
        world_space_xyz: Float[Tensor, "batch ray sample 3"],
        pixel_encoding: PixelEncoding,
    ) -> DecoderFeatureOnlyOutput:
        pass

    @abstractmethod
    def freeze_non_action_parameters(self) -> int:
        """
        Freeze all parameters except for the action decoder.
        Return number of frozen parameters.
        """

        pass
