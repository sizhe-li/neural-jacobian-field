from .pusher_dataset import DatasetPusherCfg
from .planar_hand_dataset import DatasetPlanarHandCfg

DatasetCfg = DatasetPusherCfg | DatasetPlanarHandCfg
