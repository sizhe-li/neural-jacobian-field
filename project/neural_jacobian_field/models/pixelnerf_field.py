from typing import Literal, Optional, Tuple, TypedDict

import tinycudann as tcnn
import torch
import torch.nn as nn
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from jaxtyping import Float
from nerfstudio.field_components.encodings import NeRFEncoding, SHEncoding
from neural_jacobian_field.model_components.pixel_aligned_features import (
    pixel_aligned_features,
)
from neural_jacobian_field.model_components.resnet_fc import ResnetFC
from neural_jacobian_field.model_components.transformer import Transformer
from neural_jacobian_field.model_components.activations import init_density_activation
from omegaconf import DictConfig
from torch import Tensor


def get_normalized_directions(directions):
    """SH encoding must be in the range [0, 1]

    Args:
        directions: batch of directions
    """
    return (directions + 1.0) / 2.0


class PixelAlignedFeatures(TypedDict):
    features: Float[Tensor, "B C H W"]
    c2w: Float[Tensor, "B 4 4"]
    k: Float[Tensor, "B 3 3"]


class PixelAlignedActionFeatures(TypedDict):
    features: Float[Tensor, "B C H W"]
    c2w: Float[Tensor, "B 4 4"]
    k: Float[Tensor, "B 3 3"]
    action: Float[Tensor, "B C"]


class PixelNeRFDensityField(nn.Module):
    def __init__(
        self,
        mlp_cfg: DictConfig,
        d_latent: int,
        implementation: Literal["tcnn", "torch"] = "torch",
    ):
        super().__init__()

        num_frequencies = 10
        self.pos_enc = NeRFEncoding(
            in_dim=3,
            num_frequencies=num_frequencies,
            min_freq_exp=0,
            max_freq_exp=num_frequencies - 1,
            implementation=implementation,
            include_input=(False if implementation == "tcnn" else True),
        )
        d_in = self.pos_enc.get_out_dim()

        self.sigma_net = ResnetFC(mlp_cfg, d_in=d_in, d_latent=d_latent, d_out=1)
        self.density_activation = init_density_activation("trunc_exp")

    def get_density(
        self,
        world_space_xyz: Float[Tensor, "camera ray sample 3"],
        context: PixelAlignedFeatures,
    ) -> Float[Tensor, "camera ray sample 1"]:
        camera, ray, sample = world_space_xyz.shape[:3]
        world_space_xyz = rearrange(
            world_space_xyz, "camera ray sample C -> camera (ray sample) C"
        )

        aligned_features, world_space_xyz, _ = pixel_aligned_features(
            world_space_xyz,
            context["c2w"],
            context["k"],
            context["features"],
        )

        features = self.pos_enc(world_space_xyz.contiguous())
        density = self.density_activation(self.sigma_net(aligned_features, features))

        density = rearrange(
            density, "camera (ray sample) C -> camera ray sample C", sample=sample
        )

        return density


class PixelJacobianField(nn.Module):
    flow_scale: float = 1.0
    # flow_scale: float = 0.1

    def __init__(
        self,
        mlp_cfg: DictConfig,
        d_latent: int,
        n_joints: int = 4,
    ):
        super().__init__()

        self.n_joints = n_joints

        num_frequencies = 10
        self.pos_enc = NeRFEncoding(
            in_dim=3,
            num_frequencies=num_frequencies,
            min_freq_exp=0,
            max_freq_exp=num_frequencies - 1,
            implementation="torch",
            include_input=True,
        )
        self.dir_enc = SHEncoding(
            levels=4,
            implementation="tcnn",
        )

        pos_enc_dim = self.pos_enc.get_out_dim()
        dir_enc_dim = self.dir_enc.get_out_dim() if self.dir_enc is not None else 0

        self.geo_feat_dim = 15
        self.sigma_net = ResnetFC(
            mlp_cfg, d_in=pos_enc_dim, d_latent=d_latent, d_out=self.geo_feat_dim + 1
        )

        self.jacobian_pred_head = ResnetFC(
            mlp_cfg, d_in=pos_enc_dim, d_latent=d_latent, d_out=3 * self.n_joints
        )

        # TODO: tmp hack
        self.model_mode = 1

        act_dim2 = 6
        self.jacobian_pred_head2 = ResnetFC(
            mlp_cfg,
            d_in=pos_enc_dim,
            d_latent=d_latent,
            d_out=3 * act_dim2,
        )

        # initialize with a small value
        def init_weights(m):
            if type(m) == nn.Linear:
                if m.weight is not None:
                    torch.nn.init.normal_(m.weight, mean=0.0, std=1e-4)
                if m.bias is not None:
                    torch.nn.init.normal_(m.bias, mean=0.0, std=1e-4)

        self.jacobian_pred_head.apply(init_weights)

        self.color_net = nn.Sequential(
            nn.Linear(self.geo_feat_dim + dir_enc_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 3),
            nn.Sigmoid(),
        )

        self.density_activation = init_density_activation("trunc_exp")

    def get_density(
        self,
        context: PixelAlignedActionFeatures,
        world_space_xyz: Float[Tensor, "camera ray sample 3"],
    ):
        world_space_xyz = rearrange(
            world_space_xyz, "camera ray sample C -> camera (ray sample) C"
        )

        aligned_features, world_space_xyz, _ = pixel_aligned_features(
            world_space_xyz,
            context["c2w"],
            context["k"],
            context["features"],
        )

        xyz_features = self.pos_enc(world_space_xyz.contiguous())

        density_features = self.sigma_net(aligned_features, xyz_features)
        density_features, density_before_activation = torch.split(
            density_features, [self.geo_feat_dim, 1], dim=-1
        )

        density = self.density_activation(density_before_activation)

        return density, density_features, xyz_features, aligned_features

    def pred_flow_and_action_features(
        self,
        action,
        xyz_features,
        aligned_features,
        compute_action_features: bool = False,
    ):
        if self.model_mode == 1:
            jac = self.jacobian_pred_head.forward(aligned_features, xyz_features)
        else:
            jac = self.jacobian_pred_head2.forward(aligned_features, xyz_features)
            self.n_joints = 6

        action_features = jac if compute_action_features else None

        # multiply by action, which is batch x ray_sample x n_joints to get batch x ray_sample x 3
        jac = rearrange(
            jac,
            "batch ray_sample (n_joints d) -> batch ray_sample n_joints d",
            n_joints=self.n_joints,
        )

        flow = self.flow_scale * torch.einsum("brnd, brn -> brd", jac, action)

        return flow, action_features

    def get_density_and_flow(
        self,
        world_space_xyz: Float[Tensor, "camera ray sample 3"],
        context: PixelAlignedActionFeatures,
        compute_action_features: bool = False,
    ) -> Tuple[
        Float[Tensor, "camera ray sample 1"],
        Float[Tensor, "camera ray sample channel"],
        Float[Tensor, "camera ray sample channel"],
        Float[Tensor, "camera ray sample channel"],
    ]:
        camera, ray, sample = world_space_xyz.shape[:3]
        world_space_xyz = rearrange(
            world_space_xyz, "camera ray sample C -> camera (ray sample) C"
        )

        aligned_features, world_space_xyz, _ = pixel_aligned_features(
            world_space_xyz,
            context["c2w"],
            context["k"],
            context["features"],
        )

        xyz_features = self.pos_enc(world_space_xyz.contiguous())

        density_features = self.sigma_net(aligned_features, xyz_features)

        density_features, density_before_activation = torch.split(
            density_features, [self.geo_feat_dim, 1], dim=-1
        )

        density = self.density_activation(density_before_activation)

        # flow prediction
        action = context["action"]  # batch x n_joints
        action = repeat(
            action,
            pattern="camera C -> camera ray_sample C",
            ray_sample=ray * sample,
        )

        flow, action_features = self.pred_flow_and_action_features(
            action, xyz_features, aligned_features, compute_action_features
        )

        density = rearrange(
            density, "camera (ray sample) C -> camera ray sample C", sample=sample
        )

        density_features = rearrange(
            density_features,
            "camera (ray sample) C -> camera ray sample C",
            sample=sample,
        )

        if compute_action_features:
            action_features = rearrange(
                action_features,
                "camera (ray sample) C -> camera ray sample C",
                sample=sample,
            )
        flow = rearrange(
            flow, "camera (ray sample) C -> camera ray sample C", sample=sample
        )

        return density, density_features, flow, action_features

    def forward(
        self,
        world_space_xyz: Float[Tensor, "camera ray sample 3"],
        world_space_dir: Float[Tensor, "camera ray sample 3"],
        context: PixelAlignedActionFeatures,
        compute_action_features=False,
    ) -> Tuple[
        Float[Tensor, "camera ray sample 1"],
        Float[Tensor, "camera ray sample 3"],
        Float[Tensor, "camera ray sample 3"],
        Optional[Float[Tensor, "camera ray sample channel"]],
    ]:
        camera, ray, sample = world_space_xyz.shape[:3]
        density, density_features, flow, action_features = self.get_density_and_flow(
            world_space_xyz, context, compute_action_features=compute_action_features
        )

        world_space_dir = rearrange(
            world_space_dir, "camera ray sample C -> (camera ray sample) C"
        )
        world_space_dir = get_normalized_directions(world_space_dir).contiguous()

        dir_features = (
            self.dir_enc(world_space_dir) if self.dir_enc is not None else None
        )

        dir_features = rearrange(
            dir_features,
            "(camera ray sample) C -> camera ray sample C",
            camera=camera,
            ray=ray,
            sample=sample,
        )

        density_features = (
            torch.cat((density_features, dir_features), dim=-1)
            if self.dir_enc is not None
            else density_features
        )
        rgb = self.color_net(density_features)

        return density, rgb, flow, action_features

    def encode_image(
        self,
        world_space_xyz: Float[Tensor, "camera ray sample 3"],
        context: PixelAlignedActionFeatures,
    ) -> Tuple[
        Float[Tensor, "camera ray sample 1"],
        Float[Tensor, "camera ray sample n_joints d"],
    ]:
        camera, ray, sample = world_space_xyz.shape[:3]
        density, _, xyz_features, aligned_features = self.get_density(
            context, world_space_xyz
        )

        jacobian_predictor = (
            self.jacobian_pred_head
            if self.model_mode == 1
            else self.jacobian_pred_head2
        )
        n_joints = (
            self.n_joints if self.model_mode == 1 else 6
        )  # TODO: remove hardcoding

        jacobian = jacobian_predictor.forward(aligned_features, xyz_features)

        # unpacking tensors
        density = rearrange(
            density,
            "camera (ray sample) channel -> camera ray sample channel",
            sample=world_space_xyz.shape[2],
        )

        jacobian = rearrange(
            jacobian,
            "camera (ray sample) (n_joints channel) -> camera ray sample n_joints channel",
            ray=ray,
            sample=sample,
            n_joints=n_joints,
        )

        return density, jacobian


class PixelJacobianFieldOld(nn.Module):
    flow_scale: float = 1.0
    # flow_scale: float = 0.1

    def __init__(
        self,
        mlp_cfg: DictConfig,
        d_latent: int,
        n_joints: int = 4,
        attn_feat_dim: int = 64,
        attn_head_dim: int = 64,
        num_attn_heads: int = 8,
        attn_depth: int = 3,
        attn_mlp_dim: int = 64,
    ):
        super().__init__()

        self.n_joints = n_joints

        num_frequencies = 10
        self.pos_enc = NeRFEncoding(
            in_dim=3,
            num_frequencies=num_frequencies,
            min_freq_exp=0,
            max_freq_exp=num_frequencies - 1,
            implementation="torch",
            include_input=True,
        )
        self.dir_enc = SHEncoding(
            levels=4,
            implementation="tcnn",
        )

        pos_enc_dim = self.pos_enc.get_out_dim()
        dir_enc_dim = self.dir_enc.get_out_dim() if self.dir_enc is not None else 0

        self.geo_feat_dim = 15
        self.sigma_net = ResnetFC(
            mlp_cfg, d_in=pos_enc_dim, d_latent=d_latent, d_out=self.geo_feat_dim + 1
        )

        self.joint_idx_emb = nn.Parameter(
            torch.randn(1, self.n_joints, attn_feat_dim), requires_grad=True
        )
        self.query_mlp = nn.Linear(d_latent + pos_enc_dim, attn_feat_dim)
        self.jacobian_attn_decoder = Transformer(
            dim=attn_feat_dim,
            depth=attn_depth,
            heads=num_attn_heads,
            dim_head=attn_head_dim,
            mlp_dim=attn_mlp_dim,
            selfatt=False,
            kv_dim=attn_feat_dim,
        )
        self.jacobian_pred_head = nn.Linear(attn_feat_dim, 3 * self.n_joints)

        # initialize with a small value
        def init_weights(m):
            if type(m) == nn.Linear:
                if m.weight is not None:
                    torch.nn.init.normal_(m.weight, mean=0.0, std=1e-4)
                if m.bias is not None:
                    torch.nn.init.normal_(m.bias, mean=0.0, std=1e-4)

        self.jacobian_pred_head.apply(init_weights)

        self.color_net = nn.Sequential(
            nn.Linear(self.geo_feat_dim + dir_enc_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 3),
            nn.Sigmoid(),
        )

        self.density_activation = init_density_activation("trunc_exp")

    def pred_flow_and_action_features(
        self,
        action,
        xyz_features,
        aligned_features,
        compute_action_features: bool = False,
    ):

        query = torch.cat([xyz_features, aligned_features], dim=-1)
        query = self.query_mlp(query)

        jac, action_features = self.jacobian_attn_decoder.forward(
            query, z=self.joint_idx_emb[:], get_layer_act=False
        )
        jac = self.jacobian_pred_head(jac)

        action_features = jac if compute_action_features else None

        # multiply by action, which is batch x ray_sample x n_joints to get batch x ray_sample x 3
        jac = rearrange(
            jac,
            "batch ray_sample (n_joints d) -> batch ray_sample n_joints d",
            n_joints=self.n_joints,
        )

        flow = self.flow_scale * torch.einsum("brnd, brn -> brd", jac, action)

        misc = {"jacobian": jac}
        return flow, action_features, misc

    def get_density_and_flow(
        self,
        world_space_xyz: Float[Tensor, "camera ray sample 3"],
        context: PixelAlignedActionFeatures,
        compute_action_features: bool = False,
    ) -> Tuple[
        Float[Tensor, "camera ray sample 1"],
        Float[Tensor, "camera ray sample channel"],
        Float[Tensor, "camera ray sample channel"],
        Float[Tensor, "camera ray sample channel"],
    ]:
        camera, ray, sample = world_space_xyz.shape[:3]
        world_space_xyz = rearrange(
            world_space_xyz, "camera ray sample C -> camera (ray sample) C"
        )

        aligned_features, world_space_xyz, _ = pixel_aligned_features(
            world_space_xyz,
            context["c2w"],
            context["k"],
            context["features"],
        )

        xyz_features = self.pos_enc(world_space_xyz.contiguous())

        density_features = self.sigma_net(aligned_features, xyz_features)

        density_features, density_before_activation = torch.split(
            density_features, [self.geo_feat_dim, 1], dim=-1
        )

        density = self.density_activation(density_before_activation)

        # flow prediction
        action = context["action"]  # batch x n_joints
        action = repeat(
            action,
            pattern="camera C -> camera ray_sample C",
            ray_sample=ray * sample,
        )

        flow, action_features, _ = self.pred_flow_and_action_features(
            action, xyz_features, aligned_features, compute_action_features
        )

        density = rearrange(
            density, "camera (ray sample) C -> camera ray sample C", sample=sample
        )

        density_features = rearrange(
            density_features,
            "camera (ray sample) C -> camera ray sample C",
            sample=sample,
        )

        if compute_action_features:
            action_features = rearrange(
                action_features,
                "camera (ray sample) C -> camera ray sample C",
                sample=sample,
            )

        flow = rearrange(
            flow, "camera (ray sample) C -> camera ray sample C", sample=sample
        )

        return density, density_features, flow, action_features

    def forward(
        self,
        world_space_xyz: Float[Tensor, "camera ray sample 3"],
        world_space_dir: Float[Tensor, "camera ray sample 3"],
        context: PixelAlignedActionFeatures,
        compute_action_features=False,
    ) -> Tuple[
        Float[Tensor, "camera ray sample 1"],
        Float[Tensor, "camera ray sample 3"],
        Float[Tensor, "camera ray sample 3"],
        Optional[Float[Tensor, "camera ray sample channel"]],
    ]:
        camera, ray, sample = world_space_xyz.shape[:3]
        density, density_features, flow, action_features = self.get_density_and_flow(
            world_space_xyz, context, compute_action_features=compute_action_features
        )

        world_space_dir = rearrange(
            world_space_dir, "camera ray sample C -> (camera ray sample) C"
        )
        world_space_dir = get_normalized_directions(world_space_dir).contiguous()

        dir_features = (
            self.dir_enc(world_space_dir) if self.dir_enc is not None else None
        )

        dir_features = rearrange(
            dir_features,
            "(camera ray sample) C -> camera ray sample C",
            camera=camera,
            ray=ray,
            sample=sample,
        )

        density_features = (
            torch.cat([density_features, dir_features], dim=-1)
            if self.dir_enc is not None
            else density_features
        )
        rgb = self.color_net(density_features)

        return density, rgb, flow, action_features

    def get_density(
        self,
        context: PixelAlignedActionFeatures,
        world_space_xyz: Float[Tensor, "camera ray sample 3"],
    ):
        world_space_xyz = rearrange(
            world_space_xyz, "camera ray sample C -> camera (ray sample) C"
        )

        aligned_features, world_space_xyz, _ = pixel_aligned_features(
            world_space_xyz,
            context["c2w"],
            context["k"],
            context["features"],
        )

        xyz_features = self.pos_enc(world_space_xyz.contiguous())

        density_features = self.sigma_net(aligned_features, xyz_features)
        density_features, density_before_activation = torch.split(
            density_features, [self.geo_feat_dim, 1], dim=-1
        )

        density = self.density_activation(density_before_activation)

        return density, density_features, xyz_features, aligned_features

    def encode_image(
        self,
        world_space_xyz: Float[Tensor, "camera ray sample 3"],
        context: PixelAlignedActionFeatures,
    ):
        density, _, xyz_features, aligned_features = self.get_density(
            context, world_space_xyz
        )

        jacobian = self.jacobian_pred_head.forward(aligned_features, xyz_features)
        jacobian = rearrange(
            jacobian,
            "camera ray_sample (n_joints d) -> camera ray_sample n_joints d",
            n_joints=self.n_joints,
        )

        return density, jacobian
