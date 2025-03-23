from dataclasses import dataclass, field

import torch
import torch.nn as nn
from einops import einsum, rearrange, repeat
from jaxtyping import Float
from torch import Tensor
from typing import Literal

from ...model_components.vit_new import Rearrange, Transformer, pair, posemb_sincos_2d
from ...model_components.unet import UNet

from .base_jacobian import JacobianField, JacobianFieldCfg, JacobianNetOutput


@dataclass
class TransformerCfg:
    dim: int = 64
    depth: int = 4
    heads: int = 4
    dim_head: int = 64
    mlp_dim: int = 64
    self_attention: bool = True
    kv_dim: int = 64


@dataclass
class TransformerJacobianFieldCfg(JacobianFieldCfg):
    name: Literal["transformer"] = "transformer"
    patch_size: int = 16
    image_size: int = 256

    encoder: TransformerCfg = field(default_factory=TransformerCfg)
    decoder: TransformerCfg = field(default_factory=TransformerCfg)


# initialize with a small value
def initialize_jacobian_weights(m):
    if type(m) == nn.Linear:
        if m.weight is not None:
            torch.nn.init.normal_(m.weight, mean=0.0, std=1e-4)
        if m.bias is not None:
            torch.nn.init.normal_(m.bias, mean=0.0, std=1e-4)


class TransformerJacobianField(JacobianField):

    def __init__(self, model_cfg: TransformerJacobianFieldCfg):
        super(TransformerJacobianField, self).__init__(cfg=model_cfg)

        self.command_dim = model_cfg.command_dim
        self.spatial_dim = model_cfg.spatial_dim

        self.dino = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14")
        # freeze the DINO model
        for param in self.dino.parameters():
            param.requires_grad = False
        self.patch_size = self.dino.patch_size

        self.decoder = UNet(
            in_channels=3 + 384,
            out_channels=self.command_dim * self.spatial_dim,
            depth=3,
        )

        # self.decoder = nn.Conv2d(
        #     in_channels=384,
        #     out_channels=model_cfg.command_dim * model_cfg.spatial_dim,
        #     kernel_size=3,
        #     stride=1,
        #     padding=1,
        # )
        # self.jacobian_field = UNet(
        #     out_channels=self.command_dim * self.spatial_dim, in_channels=3
        # )

        # patch_height, patch_width = pair(model_cfg.patch_size)
        # image_height, image_width = pair(model_cfg.image_size)

        # patch_dim = 3 * patch_height * patch_width

        # self.to_patch_embedding = nn.Sequential(
        #     Rearrange(
        #         "b c (h p1) (w p2) -> b (h w) (p1 p2 c)",
        #         p1=patch_height,
        #         p2=patch_width,
        #     ),
        #     nn.LayerNorm(patch_dim),
        #     nn.Linear(patch_dim, model_cfg.encoder.dim),
        #     nn.LayerNorm(model_cfg.encoder.dim),
        # )

        # pos_embedding = posemb_sincos_2d(
        #     h=image_height // patch_height,
        #     w=image_width // patch_width,
        #     dim=model_cfg.encoder.dim,
        # )
        # self.register_buffer("pos_embedding", pos_embedding)

        # self.encoder = Transformer(
        #     dim=model_cfg.encoder.dim,
        #     depth=model_cfg.encoder.depth,
        #     heads=model_cfg.encoder.heads,
        #     dim_head=model_cfg.encoder.dim_head,
        #     mlp_dim=model_cfg.encoder.mlp_dim,
        #     selfatt=True,
        # )

        # self.decoder = Transformer(
        #     dim=model_cfg.decoder.dim,
        #     depth=model_cfg.decoder.depth,
        #     heads=model_cfg.decoder.heads,
        #     dim_head=model_cfg.decoder.dim_head,
        #     mlp_dim=model_cfg.decoder.mlp_dim,
        #     selfatt=False,
        #     kv_dim=model_cfg.decoder.kv_dim,
        # )

        # self.linear_head = nn.Linear(
        #     model_cfg.decoder.dim, model_cfg.command_dim * model_cfg.spatial_dim
        # )

        # self.linear_head.apply(initialize_jacobian_weights)

        # query_grid = posemb_sincos_2d(
        #     h=image_height,
        #     w=image_width,
        #     dim=model_cfg.encoder.dim,
        # )
        # query_grid = query_grid.unsqueeze(0)  # shape: (1, h*w, dim)
        # # query_grid = rearrange(query_grid, "(h w) dim -> () (h w) dim")
        # self.register_buffer("query_grid", query_grid)

    def get_features(self, input_img):
        tokens = self.dino.get_intermediate_layers(input_img)[0]
        tokens = repeat(
            tokens,
            "b (h w) c -> b c (h hps) (w wps)",
            h=input_img.shape[-2] // self.patch_size,
            w=input_img.shape[-1] // self.patch_size,
            hps=self.patch_size,
            wps=self.patch_size,
        )

        return tokens

    def compute_jacobian(
        self,
        input_img: Float[Tensor, "batch channels height width"],
    ):
        image_height, image_width = input_img.shape[-2:]

        tokens = self.dino.get_intermediate_layers(input_img)[0]
        tokens = repeat(
            tokens,
            "b (h w) c -> b c (h hps) (w wps)",
            h=image_height // self.patch_size,
            w=image_width // self.patch_size,
            hps=self.patch_size,
            wps=self.patch_size,
        )

        tokens = torch.cat([input_img, tokens], dim=1)
        jacobian = self.decoder(tokens)
        jacobian = rearrange(
            jacobian,
            "b (c_dim s_dim) h w -> b c_dim s_dim h w",
            h=image_height,
            w=image_width,
            c_dim=self.command_dim,
            s_dim=self.spatial_dim,
        )

        # # image_height, image_width = pair(self.cfg.image_size)

        # # import pdb

        # # pdb.set_trace()
        # tokens = self.to_patch_embedding(input_img)
        # tokens += self.pos_embedding[:]

        # tokens = self.encoder(tokens)

        # query_grid = self.query_grid.expand(input_img.shape[0], -1, -1)
        # tokens = self.decoder(query_grid, tokens)
        # jacobian = self.linear_head(tokens)

        # jacobian = self.jacobian_field(input_img)  # b (command_dim * spatial_dim) h w
        # jacobian = rearrange(
        #     jacobian,
        #     "b (h w) (c_dim s_dim)  -> b c_dim s_dim h w",
        #     h=image_height,
        #     w=image_width,
        #     c_dim=self.command_dim,
        #     s_dim=self.spatial_dim,
        # )

        return jacobian

    def forward(
        self,
        input_img: Float[Tensor, "batch channels height width"],
        input_cmd: Float[Tensor, "batch cmd_dim"],
    ) -> JacobianNetOutput:

        jacobian = self.compute_jacobian(input_img)

        flow = einsum(jacobian, input_cmd, "b c_dim s_dim h w, b c_dim -> b s_dim h w")

        return JacobianNetOutput(jacobian=jacobian, flow=flow)
