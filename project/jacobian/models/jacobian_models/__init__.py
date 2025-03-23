from dataclasses import dataclass

import torch.nn as nn
from jaxtyping import Float
from torch import Tensor

from .base_jacobian import JacobianField, JacobianNetOutput
from .transformer_jacobian import TransformerJacobianField, TransformerJacobianFieldCfg
from .unet_jacobian import UnetJacobianField, UnetJacobianFieldCfg

ModelCfg = TransformerJacobianFieldCfg | UnetJacobianFieldCfg


def get_model(cfg: ModelCfg) -> JacobianField:
    if cfg.name == "transformer":
        return TransformerJacobianField(cfg)
    elif cfg.name == "unet":
        return UnetJacobianField(cfg)
    else:
        raise ValueError(f"Unknown model name {cfg.name}")
