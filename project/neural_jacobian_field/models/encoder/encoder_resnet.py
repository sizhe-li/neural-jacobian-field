from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from jaxtyping import Float
from torch import Tensor

from neural_jacobian_field.model_components.get_norm_layer import get_norm_layer
from neural_jacobian_field.models.encoder.encoder_base import Encoder


@dataclass
class EncoderResnetCfg:
    name: Literal["resnet"]
    upsample_interp: Literal["bilinear"]
    num_layers: int
    use_first_pool: bool
    norm_type: Literal["batch", "instance", "group", "none"]


class EncoderResnet(Encoder):
    def __init__(
        self,
        cfg: EncoderResnetCfg,
    ):
        super().__init__(cfg)

        self.pretrained = False

        self.use_first_pool = cfg.use_first_pool
        norm_layer = get_norm_layer(cfg.norm_type)
        self.model = getattr(torchvision.models, "resnet34")(
            pretrained=self.pretrained,
            norm_layer=norm_layer,
        )

        self.num_layers = cfg.num_layers
        self.upsample_interp = cfg.upsample_interp

        if not self.pretrained:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(
                        m.weight, mode="fan_out", nonlinearity="relu"
                    )
                elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)

    def forward(
        self,
        rgb: Float[Tensor, "batch 3 height width"],
    ) -> Float[Tensor, "batch channel new_height new_width"]:
        x = rgb
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)

        latents = [x]
        if self.num_layers > 1:
            if self.use_first_pool:
                x = self.model.maxpool(x)
            x = self.model.layer1(x)
            latents.append(x)
        if self.num_layers > 2:
            x = self.model.layer2(x)
            latents.append(x)
        if self.num_layers > 3:
            x = self.model.layer3(x)
            latents.append(x)
        if self.num_layers > 4:
            x = self.model.layer4(x)
            latents.append(x)

        latent_sz = latents[0].shape[-2:]
        for i in range(len(latents)):
            latents[i] = F.interpolate(
                latents[i],
                latent_sz,
                mode=self.upsample_interp,
                align_corners=False,
            )
        return torch.cat(latents, dim=1)

    def get_output_dim(self) -> int:
        return 512
