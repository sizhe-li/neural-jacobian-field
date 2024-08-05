from dataclasses import dataclass
from typing import Optional


@dataclass
class DatasetCfgCommon:
    overfit_to_scene: Optional[str]
