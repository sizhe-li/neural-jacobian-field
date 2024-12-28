from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader
from typing import Dict

# from .dataset.dataset_toy_arm import DatasetToyArmPointTrack
from .dataset.dataset import DatasetCommon
from .dataset import DatasetAllegro
from .dataset.dataset_hsa import DatasetHsa
from .dataset.dataset_toy_arm import DatasetToyArm
from .dataset.dataset_pneumatic import DatasetPneumaticHandOnly
from .validation_wrapper import ValidationWrapper
from ..config.common import PipelineCfg

DATASETS = {
    "allegro": DatasetAllegro,
    "hsa": DatasetHsa,
    "toy_arm": DatasetToyArm,
    "pneumatic_hand_only": DatasetPneumaticHandOnly,
}

# def get_dataset(
#         cfg: DatasetCfg,
#
# )


class DataModule(LightningDataModule):
    cfg: PipelineCfg

    def __init__(self, cfg: PipelineCfg):
        super().__init__()
        self.cfg = cfg

    def train_dataloader(self):
        return DataLoader(
            DATASETS[self.cfg.dataset.name](cfg=self.cfg.dataset, stage="train"),
            shuffle=True,
            batch_size=self.cfg.training.data.batch_size,
            num_workers=self.cfg.training.data.num_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            ValidationWrapper(
                DATASETS[self.cfg.dataset.name](cfg=self.cfg.dataset, stage="val"),
                1,
            ),
            batch_size=self.cfg.validation.data.batch_size,
            num_workers=self.cfg.validation.data.num_workers,
        )

    def test_dataloader(self):
        return DataLoader(
            ValidationWrapper(
                DATASETS[self.cfg.dataset.name](cfg=self.cfg.dataset, stage="test"),
                1,
            ),
            batch_size=self.cfg.validation.data.batch_size,
            num_workers=self.cfg.validation.data.num_workers,
        )
