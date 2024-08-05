from omegaconf import DictConfig
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader

from .dataset.dataset_toy_arm import DatasetToyArmPointTrack
from .validation_wrapper import ValidationWrapper

DATASETS = {
    "toy_arm": DatasetToyArmPointTrack,
}


class DataModule(LightningDataModule):
    cfg: DictConfig

    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.cfg = cfg

    def train_dataloader(self):
        # TODO: temp hack
        self.cfg.dataset.train_flow = self.cfg.model.get("train_flow", False)

        return DataLoader(
            DATASETS[self.cfg.dataset.name](cfg=self.cfg.dataset, stage="train"),
            shuffle=True,
            batch_size=self.cfg.training.data.batch_size,
            num_workers=self.cfg.training.data.num_workers,
        )

    def val_dataloader(self):
        # TODO: temp hack
        self.cfg.dataset.train_flow = self.cfg.model.get("train_flow", False)

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
            DATASETS[self.cfg.dataset.name](cfg=self.cfg.dataset, stage="test"),
            batch_size=self.cfg.testing.data.batch_size,
            num_workers=self.cfg.testing.data.num_workers,
            shuffle=False,
        )
