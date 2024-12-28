from dataclasses import dataclass
from typing import Literal

from PIL import ImageFile

from .dataset import DatasetCfgCommon, DatasetCommon, Stage
from .dataset import QposItems
from ...utils import convention

ImageFile.LOAD_TRUNCATED_IMAGES = True


@dataclass
class DatasetToyArmCfg(DatasetCfgCommon):
    name: Literal["toy_arm"]


class DatasetToyArm(DatasetCommon):
    cfg: DatasetToyArmCfg
    near: float = 0.5
    far: float = 10.0
    repeat = 1000
    scale_factor = 1.0

    def __init__(
        self,
        cfg: DatasetToyArmCfg,
        stage: Stage = "train",
    ):
        super().__init__(cfg, stage)

    # def get_max_num_frames_per_traj(self, traj_idx: int) -> int:
    #     return 7 if traj_idx == 511 else self.cfg.max_num_frames_per_traj
