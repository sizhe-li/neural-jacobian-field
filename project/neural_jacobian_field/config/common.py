from dataclasses import dataclass
from typing import Literal, Type, TypeVar

from omegaconf import DictConfig

from ..models.model import ModelCfg
from ..data.dataset import DatasetCfg
from .tools import get_typed_config, separate_multiple_defaults


@dataclass
class WandbCfg:
    project: str
    entity: str
    mode: Literal["online", "offline", "disabled"]
    name: str | None
    group: str | None
    tags: list[str] | None


@dataclass
class CheckpointCfg:
    every_n_train_steps: int
    load: str | None  # str instead of Path, since it could be wandb://...


@dataclass
class DataloaderCfg:
    rays_per_batch: int | None
    batch_size: int = 32
    num_workers: int = 4


@dataclass
class OptimizerCfg:
    lr: float = 1e-4
    warm_up_steps: int = 10000


@dataclass
class TrainingCfg:
    data: DataloaderCfg
    optim: OptimizerCfg
    val_check_interval: int
    max_steps: int


@dataclass
class ValidationVideoCfg:
    frequency: float = 0.25
    num_frames: int = 30
    frame_rate: int = 30


@dataclass
class ValidationCfg:
    data: DataloaderCfg
    video: ValidationVideoCfg


@dataclass
class PipelineCfg:
    wandb: WandbCfg
    model: ModelCfg
    dataset: DatasetCfg
    training: TrainingCfg
    validation: ValidationCfg
    checkpoint: CheckpointCfg


T = TypeVar("T")


def get_typed_root_config(cfg_dict: DictConfig, cfg_type: Type[T]) -> T:
    return get_typed_config(
        cfg_type,
        cfg_dict,
        {
            list[DatasetCfg]: separate_multiple_defaults(DatasetCfg),
        },
    )
