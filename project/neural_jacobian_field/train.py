import os
import sys
from pathlib import Path

import hydra
import torch
import wandb
from jaxtyping import install_import_hook

from colorama import Fore

from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.loggers.wandb import WandbLogger
from neural_jacobian_field.data.data_module import DataModule
from neural_jacobian_field.models import ActionModel, ActionWrapper
from neural_jacobian_field.utils.wandb_logger import SpaceEfficientWandbLogger
from neural_jacobian_field.utils.misc import cyan


# TODO (@lester): support type checking
# with install_import_hook()


@hydra.main(
    version_base=None,
    config_path="../configurations",
    config_name="config",
)
def train(cfg: DictConfig):
    # Set up the output directory.
    output_dir = Path(
        hydra.core.hydra_config.HydraConfig.get()["runtime"]["output_dir"]
    )
    print(cyan(f"Saving outputs to {output_dir}."))
    latest_run = output_dir.parents[1] / "latest-run"
    os.system(f"rm {latest_run}")
    os.system(f"ln -s {output_dir} {latest_run}")
    # print(f"Saving outputs to {output_dir}")
    # os.system(f"ln -sf {output_dir} {output_dir.parents[1]}/latest-run")

    # Set up logging with wandb.
    callbacks = []
    if cfg.wandb.mode != "disabled":
        if not hasattr(cfg.wandb, "name"):
            from omegaconf import open_dict

            with open_dict(cfg):
                cfg.wandb.name = cfg.model.name

        logger = SpaceEfficientWandbLogger(
            project=cfg.wandb.project,
            entity=cfg.wandb.entity,
            mode=cfg.wandb.mode,
            name=f"{cfg.wandb.name} ({output_dir.parent.name}/{output_dir.name})",
            log_model="all",
            save_dir=str(output_dir),
            config=OmegaConf.to_container(cfg),
        )
        callbacks.append(LearningRateMonitor("step", True))

        # On rank != 0, wandb.run is None.
        logger.__getstate__()  # hack to initialize wandb.run
        if wandb.run is not None:
            print(f"wandb mode: {wandb.run.settings.mode}")
            wandb.run.log_code(".")
    else:
        logger = None

    # If resuming a run, download the checkpoint.
    resume_id = cfg.wandb.get("resume", None)
    checkpoint_path = None
    if resume_id is not None:
        if Path(resume_id).exists():
            checkpoint_path = Path(resume_id)
            print(cyan(f"Loading checkpoint from {str(checkpoint_path)}"))
        else:
            raise ValueError(f"Checkpoint {resume_id} does not exist")

    # (Optional) set matmul precision (for newer GPUs, e.g., A6000) for faster training.
    # if hasattr(torch, "set_float32_matmul_precision"):
    #     torch.set_float32_matmul_precision("high")

    # Get the model and loaders.
    data_module = DataModule(cfg)
    model = ActionModel(cfg)
    if checkpoint_path is None:
        model_wrapper = ActionWrapper(cfg, model)
    else:
        model_wrapper = ActionWrapper.load_from_checkpoint(
            checkpoint_path=checkpoint_path,
            cfg=cfg,
            model=model,
            strict=False,
        )

    trainer = Trainer(
        max_epochs=-1,
        accelerator="gpu",
        logger=logger,
        # Instead of changing this, run: pip install --upgrade pytorch_lightning
        devices="auto",
        strategy=(
            "ddp_find_unused_parameters_true"
            if torch.cuda.device_count() > 1
            else "auto"
        ),
        callbacks=callbacks,
        val_check_interval=cfg.validation.interval,
        accumulate_grad_batches=cfg.training.optim.accumulate_grad_batches,
    )

    trainer.fit(
        model_wrapper,
        datamodule=data_module,
    )


if __name__ == "__main__":
    train()
