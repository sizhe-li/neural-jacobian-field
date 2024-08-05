from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

from jaxtyping import Float
from omegaconf import DictConfig
from torch import Tensor

from neural_jacobian_field.encoders.encoder_base import EncoderBase
from neural_jacobian_field.model_components.get_norm_layer import get_norm_layer
from neural_jacobian_field.utils.misc import cyan


class ResnetEncoder(EncoderBase):
    def __init__(
        self,
        cfg: DictConfig,
    ):
        super().__init__(cfg)
        encoder_cfg = cfg.model.encoder

        self.pretrained = False
        self.normalize = encoder_cfg.get("use_proprioception", False)
        if self.normalize:
            print(
                cyan("Normalizing images from (0, 1) to (-1, 1) inside resnet encoder!")
            )

        self.use_first_pool = encoder_cfg.use_first_pool
        norm_layer = get_norm_layer(encoder_cfg.norm_type)
        self.model = getattr(torchvision.models, "resnet34")(
            pretrained=self.pretrained,
            norm_layer=norm_layer,
        )
        # change the first layer to accept 16 more dimension

        if encoder_cfg.get("use_proprioception", False):
            self.model.conv1 = nn.Conv2d(
                3 + 16,
                64,
                kernel_size=(7, 7),
                stride=(2, 2),
                padding=(3, 3),
                bias=False,
            )

        self.num_layers = encoder_cfg.num_layers
        self.upsample_interp = encoder_cfg.upsample_interp

        if not self.pretrained:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(
                        m.weight, mode="fan_out", nonlinearity="relu"
                    )
                elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)

    def encode_single_view(
        self,
        rgb: Float[Tensor, "batch 3 height width"],
        model_input: dict,
    ) -> List[Float[Tensor, "batch _ _ _"]]:
        if self.normalize:
            x = rgb * 2 - 1
        else:
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
        return [torch.cat(latents, dim=1)]

    @property
    def d_out(self) -> int:
        return 512
