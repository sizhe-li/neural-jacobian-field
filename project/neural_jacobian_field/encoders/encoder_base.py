from abc import ABC, abstractmethod
from typing import List

from einops import pack, rearrange
from jaxtyping import Float
from lightning_fabric.utilities.apply_func import apply_to_collection
from omegaconf import DictConfig
from torch import Tensor, nn


class EncoderBase(nn.Module, ABC):
    cfg: DictConfig

    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.cfg = cfg

    def encode_model_input(
        self,
        model_input: dict,
    ) -> List[Float[Tensor, "batch view _ _ _"]]:
        # Combine the context views into a single tensor.
        rgb = [view["rgb"] for view in model_input["context"]]
        rgb = pack(rgb, "batch * channel width height")[0]

        return self.encode_multiple_views(rgb, model_input)

    def encode_multiple_views(
        self,
        rgb: Float[Tensor, "batch view 3 height width"],
        model_input: dict,
    ) -> List[Float[Tensor, "batch view _ _ _"]]:
        """Encoders that jointly encode multiple context views should override this
        function. If not overridden, this function uses encode_single_view to encode
        the context views independently. Output features are "batch view channel height
        width," but channels, heights, and widths can vary between images, hence the
        triple underscore.
        """

        # Encode each view independently.
        _, num_views, *_ = rgb.shape
        flat_rgb = rearrange(rgb, "B V C H W -> (B V) C H W")
        features = self.encode_single_view(flat_rgb, model_input)

        # Reshape the outputs to "batch view channel height width."
        return apply_to_collection(
            features,
            Tensor,
            lambda f: rearrange(f, "(B V) C H W -> B V C H W", V=num_views),
        )

    def encode_single_view(
        self,
        rgb: Float[Tensor, "batch 3 height width"],
        model_input: dict,
    ) -> List[Float[Tensor, "batch _ _ _"]]:
        """Encoders that encode context views independently should override this
        function."""
        pass

    @property
    @abstractmethod
    def d_out(self) -> int:
        pass
