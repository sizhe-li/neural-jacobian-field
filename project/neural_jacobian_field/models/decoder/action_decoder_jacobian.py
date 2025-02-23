from abc import abstractmethod
from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn as nn
from einops import rearrange, repeat, einsum
from jaxtyping import Float
from nerfstudio.field_components.encodings import NeRFEncoding, SHEncoding
from torch import Tensor

from .action_decoder import (
    ActionDecoder,
    PixelEncoding,
    DecoderOutput,
    DecoderFeatureOnlyOutput,
)
from ...model_components.activations import init_density_activation
from ...model_components.pixel_aligned_features import get_pixel_aligned_features
from ...model_components.resnet_fc import MlpCfg, ResnetFC
from ...model_components.transformer import Transformer


def get_normalized_directions(directions):
    """SH encoding must be in the range [0, 1]

    Args:
        directions: batch of directions
    """
    return (directions + 1.0) / 2.0


@dataclass
class TransformerCfg:
    attn_feat_dim: int
    attn_head_dim: int
    num_attn_heads: int
    attn_depth: int
    attn_mlp_dim: int


@dataclass
class ActionDecoderJacobianMlpCfg:
    name: Literal["jacobian_mlp"]
    mlp: MlpCfg
    num_frequencies: int = 10
    geometry_feature_dim: int = 15
    use_arm_model: bool = False
    arm_action_dim: int | None = None


@dataclass
class ActionDecoderJacobianTransformerCfg:
    name: Literal["jacobian_transformer"]
    mlp: MlpCfg
    transformer: TransformerCfg
    num_frequencies: int = 10
    geometry_feature_dim: int = 15
    use_arm_model: bool = False
    arm_action_dim: int | None = None


@dataclass
class DensityHeadOutput:
    density: Float[Tensor, "batch ray_sample 1"]
    density_features: Float[Tensor, "batch ray_sample dim"]
    xyz_features: Float[Tensor, "batch ray_sample dim"]
    pixel_aligned_features: Float[Tensor, "batch ray_sample dim"]


@dataclass
class FlowHeadOutput:
    flow: Float[Tensor, "batch ray_sample 3"]
    action_features: Float[Tensor, "batch ray_sample dim"]


# initialize with a small value
def initialize_jacobian_weights(m):
    if type(m) == nn.Linear:
        if m.weight is not None:
            torch.nn.init.normal_(m.weight, mean=0.0, std=1e-4)
        if m.bias is not None:
            torch.nn.init.normal_(m.bias, mean=0.0, std=1e-4)


class ActionDecoderJacobian(ActionDecoder):
    mode: Literal["regular", "arm"]

    def switch_mode(self, mode: Literal["regular", "arm"]):
        self.mode = mode

    def compute_density(
        self,
        world_space_xyz: Float[Tensor, "batch ray_sample 3"],
        pixel_encoding: PixelEncoding,
    ) -> DensityHeadOutput:
        pixel_aligned_features, world_space_xyz, _ = get_pixel_aligned_features(
            world_space_xyz,
            pixel_encoding.extrinsics,
            pixel_encoding.intrinsics,
            pixel_encoding.features,
        )

        xyz_features = self.positional_encoding(world_space_xyz.contiguous())

        density_features = self.density_head.forward(
            pixel_aligned_features, xyz_features
        ).output
        density_features, density_before_activation = torch.split(
            density_features, [self.cfg.geometry_feature_dim, 1], dim=-1
        )
        density = self.density_activation(density_before_activation)

        return DensityHeadOutput(
            density=density,
            density_features=density_features,
            xyz_features=xyz_features,
            pixel_aligned_features=pixel_aligned_features,
        )

    @abstractmethod
    def compute_jacobian(
        self,
        density_head_output: DensityHeadOutput,
    ) -> Float[Tensor, "batch ray_sample action_spatial_dim"]:
        pass

    def compute_flow(
        self,
        action: Float[Tensor, "batch ray_sample action_dim"],
        density_head_output: DensityHeadOutput,
    ) -> FlowHeadOutput:
        jacobian = self.compute_jacobian(density_head_output)

        flow = einsum(
            rearrange(
                jacobian,
                "batch ray_sample (action_dim spatial_dim) -> batch ray_sample action_dim spatial_dim",
                action_dim=self.action_dim,
            ),
            action,
            "batch ray_sample action_dim spatial_dim, batch ray_sample action_dim -> batch ray_sample spatial_dim",
        )

        return FlowHeadOutput(flow=flow, action_features=jacobian)

    def forward(
        self,
        world_space_xyz: Float[Tensor, "batch ray sample 3"],
        world_space_dir: Float[Tensor, "batch ray sample 3"],
        pixel_encoding: PixelEncoding,
    ) -> DecoderOutput:
        batch, ray, sample = world_space_xyz.shape[:3]

        world_space_xyz = rearrange(
            world_space_xyz, "batch ray sample dim -> batch (ray sample) dim"
        )

        density_head_output = self.compute_density(world_space_xyz, pixel_encoding)

        action = repeat(
            pixel_encoding.action,
            "batch dim -> batch ray_sample dim",
            ray_sample=ray * sample,
        )

        flow_head_output = self.compute_flow(action, density_head_output)
        flow, action_features = flow_head_output.flow, flow_head_output.action_features

        # reshape features
        (
            density,
            density_features,
            xyz_features,
            pixel_aligned_features,
            flow,
            action_features,
        ) = map(
            lambda x: rearrange(
                x,
                "batch (ray sample) dim -> batch ray sample dim",
                sample=sample,
            ),
            (
                density_head_output.density,
                density_head_output.density_features,
                density_head_output.xyz_features,
                density_head_output.pixel_aligned_features,
                flow,
                action_features,
            ),
        )

        world_space_dir = get_normalized_directions(world_space_dir)
        # tcnn implementation only accepts two dimensions, so we need to reshape
        world_space_dir = rearrange(
            world_space_dir, "batch ray sample dim -> (batch ray sample) dim"
        )
        dir_features = self.directional_encoding(world_space_dir.contiguous())
        # reshape back to four dimensions
        dir_features = rearrange(
            dir_features,
            "(batch ray sample) dim -> batch ray sample dim",
            batch=batch,
            ray=ray,
            sample=sample,
        )
        color = self.color_head(torch.cat((density_features, dir_features), dim=-1))

        return DecoderOutput(
            density=density,
            color=color,
            flow=flow,
            action_features=action_features,
        )

    def encode_image(
        self,
        world_space_xyz: Float[Tensor, "batch ray sample 3"],
        pixel_encoding: PixelEncoding,
    ) -> DecoderFeatureOnlyOutput:
        """To solve the inverse problem"""

        batch, ray, sample = world_space_xyz.shape[:3]
        world_space_xyz = rearrange(
            world_space_xyz, "batch ray sample dim -> batch (ray sample) dim"
        )

        density_head_output = self.compute_density(world_space_xyz, pixel_encoding)

        jacobian = self.compute_jacobian(density_head_output)

        # reshape features
        density, jacobian = map(
            lambda x: rearrange(
                x,
                "batch (ray sample) dim -> batch ray sample dim",
                sample=sample,
            ),
            (
                density_head_output.density,
                jacobian,
            ),
        )

        return DecoderFeatureOnlyOutput(
            density=density,
            action_features=jacobian,
        )

    def freeze_non_action_parameters(self) -> int:
        counts = 0
        for name, param in self.named_parameters():
            if self.action_param_glob_pattern not in name:
                param.requires_grad = False
                counts += 1

        return counts


class ActionDecoderJacobianMLP(ActionDecoderJacobian):
    cfg: ActionDecoderJacobianMlpCfg
    spatial_dim: int = 3  # modeling a 3d world
    action_param_glob_pattern = "jacobian_head"

    def __init__(
        self,
        cfg: ActionDecoderJacobianMlpCfg,
        action_dim: int,
        encoder_dim: int,
    ) -> None:
        super().__init__(cfg)

        self.action_dim = action_dim
        self.positional_encoding = NeRFEncoding(
            in_dim=self.spatial_dim,
            num_frequencies=cfg.num_frequencies,
            min_freq_exp=0,
            max_freq_exp=cfg.num_frequencies - 1,
            implementation="torch",
            include_input=True,
        )

        self.directional_encoding = SHEncoding(levels=4, implementation="tcnn")

        positional_encoding_dim = self.positional_encoding.get_out_dim()
        directional_encoding_dim = self.directional_encoding.get_out_dim()

        self.density_head = ResnetFC(
            cfg.mlp,
            d_in=positional_encoding_dim,
            d_latent=encoder_dim,
            d_out=cfg.geometry_feature_dim + 1,
        )
        self.density_activation = init_density_activation("trunc_exp")

        self.jacobian_head = ResnetFC(
            cfg.mlp,
            d_in=positional_encoding_dim,
            d_latent=encoder_dim,
            d_out=self.spatial_dim * action_dim,
        )
        self.jacobian_head.apply(initialize_jacobian_weights)

        self.mode: Literal["regular", "arm"] = "regular"
        if cfg.use_arm_model:
            self.jacobian_head_arm = ResnetFC(
                cfg.mlp,
                d_in=positional_encoding_dim,
                d_latent=encoder_dim,
                d_out=self.spatial_dim * cfg.arm_action_dim,
            )
            self.jacobian_head_arm.apply(initialize_jacobian_weights)

        self.color_head = nn.Sequential(
            nn.Linear(cfg.geometry_feature_dim + directional_encoding_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 3),
            nn.Sigmoid(),
        )

    def compute_jacobian(
        self,
        density_head_output: DensityHeadOutput,
    ) -> Float[Tensor, "batch ray_sample action_spatial_dim"]:
        if self.mode == "regular":
            jacobian_head = self.jacobian_head
        else:
            jacobian_head = self.jacobian_head_arm

        jacobian = jacobian_head.forward(
            density_head_output.pixel_aligned_features, density_head_output.xyz_features
        ).output

        return jacobian


class ActionDecoderJacobianTransformer(ActionDecoderJacobian):
    cfg: ActionDecoderJacobianTransformerCfg
    spatial_dim: int = 3  # modeling a 3d world
    action_param_glob_pattern = "jacobian"

    def __init__(
        self,
        cfg: ActionDecoderJacobianTransformerCfg,
        action_dim: int,
        encoder_dim: int,
    ) -> None:
        super().__init__(cfg)

        self.action_dim = action_dim
        self.positional_encoding = NeRFEncoding(
            in_dim=self.spatial_dim,
            num_frequencies=cfg.num_frequencies,
            min_freq_exp=0,
            max_freq_exp=cfg.num_frequencies - 1,
            implementation="torch",
            include_input=True,
        )

        self.directional_encoding = SHEncoding(levels=4, implementation="tcnn")

        positional_encoding_dim = self.positional_encoding.get_out_dim()
        directional_encoding_dim = self.directional_encoding.get_out_dim()

        self.density_head = ResnetFC(
            cfg.mlp,
            d_in=positional_encoding_dim,
            d_latent=encoder_dim,
            d_out=cfg.geometry_feature_dim + 1,
        )
        self.density_activation = init_density_activation("trunc_exp")

        self.jacobian_index_embedding = nn.Parameter(
            torch.randn(1, action_dim, self.cfg.transformer.attn_feat_dim),
            requires_grad=True,
        )
        self.jacobian_query_mlp = nn.Linear(
            encoder_dim + positional_encoding_dim, self.cfg.transformer.attn_feat_dim
        )

        self.jacobian_attn_decoder = Transformer(
            dim=self.cfg.transformer.attn_feat_dim,
            depth=self.cfg.transformer.attn_depth,
            heads=self.cfg.transformer.num_attn_heads,
            dim_head=self.cfg.transformer.attn_head_dim,
            mlp_dim=self.cfg.transformer.attn_mlp_dim,
            selfatt=False,
            kv_dim=self.cfg.transformer.attn_feat_dim,
        )
        self.jacobian_head = nn.Linear(
            self.cfg.transformer.attn_feat_dim, self.spatial_dim * action_dim
        )

        self.jacobian_head.apply(initialize_jacobian_weights)

        self.mode: Literal["regular", "arm"] = "regular"
        if cfg.use_arm_model:
            self.jacobian_head_arm = ResnetFC(
                cfg.mlp,
                d_in=positional_encoding_dim,
                d_latent=encoder_dim,
                d_out=self.spatial_dim * cfg.arm_action_dim,
            )
            self.jacobian_head_arm.apply(initialize_jacobian_weights)

        self.color_head = nn.Sequential(
            nn.Linear(cfg.geometry_feature_dim + directional_encoding_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 3),
            nn.Sigmoid(),
        )

    def compute_jacobian(
        self,
        density_head_output: DensityHeadOutput,
    ) -> Float[Tensor, "batch ray_sample action_spatial_dim"]:
        if self.mode == "regular":
            query = torch.cat(
                [
                    density_head_output.xyz_features,
                    density_head_output.pixel_aligned_features,
                ],
                dim=-1,
            )
            query = self.jacobian_query_mlp.forward(query)
            tokens, _ = self.jacobian_attn_decoder.forward(
                x=query,
                z=self.jacobian_index_embedding[:],
                get_layer_act=False,
            )
            jacobian = self.jacobian_head.forward(tokens)

        else:
            jacobian_head = self.jacobian_head_arm

            jacobian = jacobian_head.forward(
                density_head_output.pixel_aligned_features,
                density_head_output.xyz_features,
            ).output

        return jacobian
