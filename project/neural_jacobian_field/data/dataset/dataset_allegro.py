from dataclasses import dataclass
from typing import Literal

from .dataset import DatasetCfgCommon, DatasetCommon, Stage
from PIL import ImageFile, Image

ImageFile.LOAD_TRUNCATED_IMAGES = True


@dataclass
class DatasetAllegroCfg(DatasetCfgCommon):
    name: Literal["allegro"]


class DatasetAllegro(DatasetCommon):
    cfg: DatasetAllegroCfg
    near: float = 0.5
    far: float = 10.0
    repeat = 1000
    scale_factor = 1.0

    def __init__(
        self,
        cfg: DatasetAllegroCfg,
        stage: Stage = "train",
        downscale_factor: int | None = None,
    ):
        super().__init__(cfg, stage)
