from omegaconf import DictConfig

from .encoder_base import EncoderBase
from .resnet_encoder import ResnetEncoder

ENCODERS = {
    "pixelnerf": ResnetEncoder,
}


def get_encoder(cfg: DictConfig) -> EncoderBase:
    return ENCODERS[cfg.model.encoder.name](cfg)