from typing import Literal

from einops import einsum, rearrange
from jaxtyping import Float
from torch import Tensor, nn

from ...model_components.unet import UNet
from .base_jacobian import JacobianField, JacobianFieldCfg, JacobianNetOutput

from dataclasses import dataclass


@dataclass
class UnetJacobianFieldCfg(JacobianFieldCfg):
    name: Literal["unet"] = "unet"


class UnetJacobianField(JacobianField):

    def __init__(self, model_cfg: UnetJacobianFieldCfg):
        super(UnetJacobianField, self).__init__(cfg=model_cfg)

        self.command_dim = model_cfg.command_dim
        self.spatial_dim = model_cfg.spatial_dim

        # image -> UNet -> Jacobian field
        self.jacobian_field = UNet(
            out_channels=self.command_dim * self.spatial_dim,
            in_channels=3,
            depth=3,
            start_filts=32,
        )

    def get_features(self, input_img: Float[Tensor, "batch channels height width"]):
        feats, _ = self.jacobian_field.encoder(input_img)
        return feats

    def compute_jacobian(
        self,
        input_img: Float[Tensor, "batch channels height width"],
    ):
        jacobian = self.jacobian_field(input_img)  # b (command_dim * spatial_dim) h w
        jacobian = rearrange(
            jacobian,
            "b (c_dim s_dim) h w -> b c_dim s_dim h w",
            c_dim=self.command_dim,
            s_dim=self.spatial_dim,
        )

        return jacobian

    def forward(
        self,
        input_img: Float[Tensor, "batch channels height width"],
        input_cmd: Float[Tensor, "batch cmd_dim"],
    ) -> JacobianNetOutput:

        jacobian = self.compute_jacobian(input_img)

        # import pdb

        # pdb.set_trace()

        flow = einsum(jacobian, input_cmd, "b c_dim s_dim h w, b c_dim -> b s_dim h w")

        return JacobianNetOutput(jacobian=jacobian, flow=flow)
