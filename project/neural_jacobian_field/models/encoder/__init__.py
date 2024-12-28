from dataclasses import dataclass
from omegaconf import DictConfig

from .encoder_base import Encoder
from .encoder_resnet import EncoderResnet, EncoderResnetCfg

ENCODERS = {
    "resnet": EncoderResnet,
}


EncoderCfg = EncoderResnetCfg


def get_encoder(cfg: EncoderCfg) -> Encoder:
    return ENCODERS[cfg.name](cfg)
