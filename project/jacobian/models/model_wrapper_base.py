from .model_wrapper_planar_hand import PlanarHandModelWrapper, PlanarHandModelWrapperCfg
from .model_wrapper_pusher import PusherModelWrapper, PusherModelWrapperCfg

ModelWrapperCfg = PusherModelWrapperCfg | PlanarHandModelWrapperCfg


def get_wrapper(cfg: ModelWrapperCfg):
    if cfg.name == "planar_hand":
        return PlanarHandModelWrapper(cfg)
    # elif cfg.name in ["two_fingers", "shadow_finger"]:
    #     return PlanarHandWrapper(cfg)
    elif cfg.name == "pusher":
        return PusherModelWrapper(cfg)
    else:
        raise ValueError(f"Unknown model type: {cfg.name}")
