from dataclasses import dataclass
from typing import Literal

import torch.nn as nn
from einops import rearrange
from jaxtyping import Float
from nerfstudio.field_components.encodings import NeRFEncoding
from torch import Tensor

from .action_decoder import PixelEncoding
from ...model_components.activations import init_density_activation
from ...model_components.pixel_aligned_features import get_pixel_aligned_features
from ...model_components.resnet_fc import MlpCfg, ResnetFC


@dataclass
class DensityDecoderMlpCfg:
    name: Literal["density_mlp"]
    mlp: MlpCfg
    num_frequencies: int = 10


class DensityDecoderMlp(nn.Module):
    def __init__(
        self,
        cfg: DensityDecoderMlpCfg,
        encoder_dim: int,
    ):
        super().__init__()

        self.positional_encoding = NeRFEncoding(
            in_dim=3,
            num_frequencies=cfg.num_frequencies,
            min_freq_exp=0,
            max_freq_exp=cfg.num_frequencies - 1,
            implementation="torch",
            include_input=True,
        )
        positional_encoding_dim = self.positional_encoding.get_out_dim()
        self.density_head = ResnetFC(
            cfg.mlp, d_in=positional_encoding_dim, d_latent=encoder_dim, d_out=1
        )
        self.density_activation = init_density_activation("trunc_exp")

    def get_density(
        self,
        world_space_xyz: Float[Tensor, "batch ray sample 3"],
        pixel_encoding: PixelEncoding,
    ) -> Float[Tensor, "batch ray sample 1"]:
        batch, ray, sample = world_space_xyz.shape[:3]
        world_space_xyz = rearrange(
            world_space_xyz, "batch ray sample dim -> batch (ray sample) dim"
        )

        aligned_features, world_space_xyz, _ = get_pixel_aligned_features(
            world_space_xyz,
            pixel_encoding.extrinsics,
            pixel_encoding.intrinsics,
            pixel_encoding.features,
        )

        features = self.positional_encoding(world_space_xyz.contiguous())
        density = self.density_activation(
            self.density_head.forward(aligned_features, features).output
        )

        density = rearrange(
            density, "batch (ray sample) dim -> batch ray sample dim", sample=sample
        )

        return density
