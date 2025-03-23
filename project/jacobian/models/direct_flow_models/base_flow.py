from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Generic, TypeVar

import torch.nn as nn
from jaxtyping import Float
from omegaconf import DictConfig
from torch import Tensor, nn

# from . import ModelCfg


@dataclass
class JacobianFieldCfg:
    command_dim: int
    spatial_dim: int


@dataclass
class JacobianNetOutput:
    jacobian: Float[Tensor, "batch height width jacobian_dim"]
    flow: Float[Tensor, "batch height width spatial_dim"]


T = TypeVar("T")


class JacobianField(nn.Module, ABC, Generic[T]):
    cfg: DictConfig

    def __init__(self, cfg: T):
        super().__init__()
        self.cfg = cfg

    @abstractmethod
    def compute_jacobian(
        self,
        input_img: Float[Tensor, "batch channels height width"],
    ) -> Float[Tensor, "batch channel height width"]:
        pass

    @abstractmethod
    def forward(
        self,
        input_img: Float[Tensor, "batch channels height width"],
        input_cmd: Float[Tensor, "batch cmd_dim"],
    ) -> JacobianNetOutput:
        pass


# class Model(nn.Module):

#     def __init__(self, model_cfg: ModelCfg):
#         super(Model, self).__init__()

#         self.command_dim = model_cfg.command_dim
#         self.spatial_dim = model_cfg.spatial_dim

#         # image -> UNet -> Jacobian field
#         self.jacobian_field = UNet(
#             out_channels=self.command_dim * self.spatial_dim, in_channels=3
#         )

#     def compute_jacobian(
#         self,
#         input_img: Float[Tensor, "batch channels height width"],
#     ):
#         jacobian = self.jacobian_field(input_img)  # b (command_dim * spatial_dim) h w
#         jacobian = rearrange(
#             jacobian,
#             "b (c_dim s_dim) h w -> b c_dim s_dim h w",
#             c_dim=self.command_dim,
#             s_dim=self.spatial_dim,
#         )

#         return jacobian

#     def forward(
#         self,
#         input_img: Float[Tensor, "batch channels height width"],
#         input_cmd: Float[Tensor, "batch cmd_dim"],
#     ) -> JacobianNetOutput:

#         jacobian = self.compute_jacobian(input_img)

#         flow = einsum(jacobian, input_cmd, "b c_dim s_dim h w, b c_dim -> b s_dim h w")

#         return JacobianNetOutput(jacobian=jacobian, flow=flow)
