from dataclasses import dataclass
from pathlib import Path

from jacobian.config.common import PipelineCfg
from jacobian.dataset.planar_hand_dataset import DatasetPlanarHand
from jacobian.dataset.pusher_dataset import DatasetPusher
from omegaconf import DictConfig
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader

DATASETS = {
    "pusher": DatasetPusher,
    "planar_hand": DatasetPlanarHand,
}


class DataModule(LightningDataModule):
    cfg: PipelineCfg

    def __init__(self, cfg: PipelineCfg):
        super().__init__()
        self.cfg = cfg
        print("dataset name", self.cfg.dataset.name)

    def train_dataloader(self):
        return DataLoader(
            DATASETS[self.cfg.dataset.name](cfg=self.cfg.dataset, stage="train"),
            shuffle=True,
            batch_size=self.cfg.training.data.batch_size,
            num_workers=self.cfg.training.data.num_workers,
        )

    def val_dataloader(self):
        if self.cfg.dataset.name == "two_fingers":
            # set batch size to the number of validation samples
            batch_size = len(list(Path((self.cfg.dataset.root), "val").glob("*.pkl")))
            return DataLoader(
                DATASETS[self.cfg.dataset.name](cfg=self.cfg.dataset, stage="val"),
                shuffle=False,
                batch_size=batch_size,
                num_workers=4,
            )
        else:
            return DataLoader(
                DATASETS[self.cfg.dataset.name](cfg=self.cfg.dataset, stage="val"),
                shuffle=False,
                batch_size=1,
                num_workers=4,
            )

    def test_dataloader(self):
        return DataLoader(
            DATASETS[self.cfg.dataset.name](cfg=self.cfg.dataset, stage="test"),
            shuffle=False,
            batch_size=1,
            num_workers=4,
        )
