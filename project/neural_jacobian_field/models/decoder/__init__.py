from .action_decoder import ActionDecoder
from .action_decoder_flow import ActionDecoderFlowMlpCfg, ActionDecoderFlowMlp
from .action_decoder_jacobian import (
    ActionDecoderJacobianMlpCfg,
    ActionDecoderJacobianMLP,
    ActionDecoderJacobianTransformer,
    ActionDecoderJacobianTransformerCfg,
)
from .density_decoder import DensityDecoderMlpCfg, DensityDecoderMlp

DENSITY_DECODERS = {
    "density_mlp": DensityDecoderMlp,
}

ACTION_DECODERS = {
    "jacobian_mlp": ActionDecoderJacobianMLP,
    "jacobian_transformer": ActionDecoderJacobianTransformer,
    "flow_mlp": ActionDecoderFlowMlp,
}


DensityDecoderCfg = DensityDecoderMlpCfg
ActionDecoderCfg = (
    ActionDecoderJacobianMlpCfg
    | ActionDecoderFlowMlpCfg
    | ActionDecoderJacobianTransformerCfg
)


def get_density_decoder(cfg: DensityDecoderCfg, encoder_dim: int) -> DensityDecoderMlp:
    return DENSITY_DECODERS[cfg.name](
        cfg=cfg,
        encoder_dim=encoder_dim,
    )


def get_action_decoder(
    cfg: ActionDecoderCfg, action_dim: int, encoder_dim: int
) -> ActionDecoder:
    return ACTION_DECODERS[cfg.name](
        cfg=cfg,
        action_dim=action_dim,
        encoder_dim=encoder_dim,
    )
