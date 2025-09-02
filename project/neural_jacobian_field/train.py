from pathlib import Path

import hydra
import torch
import wandb
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import Trainer

from .config.common import PipelineCfg, get_typed_root_config

# with install_import_hook(
#     ("neural_jacobian_field",),
#     ("beartype", "beartype"),
# ):
from .data.data_module import DataModule
from .models.model import Model
from .models.model_wrapper import ModelWrapper
from .utils.misc import cyan
from .utils.wandb_ckpt import update_checkpoint_path
from .utils.wandb_logger import SpaceEfficientWandbLogger


@hydra.main(
    version_base=None,
    config_path="configurations",
    config_name="config",
)
def train(cfg: DictConfig):
    # Set up the output directory.
    output_dir = Path(
        hydra.core.hydra_config.HydraConfig.get()["runtime"]["output_dir"]
    )
    print(cyan(f"Saving outputs to {output_dir}."))

    omega_conf = OmegaConf.create(cfg)

    cfg: PipelineCfg = get_typed_root_config(cfg, PipelineCfg)

    data_module = DataModule(cfg)
    model = Model(cfg.model)
    model_wrapper = ModelWrapper(cfg=cfg, model=model)
    print(model)

    logger = SpaceEfficientWandbLogger(
        project=cfg.wandb.project,
        entity=cfg.wandb.entity,
        mode=cfg.wandb.mode,
        name=f"{cfg.wandb.name} ({output_dir.parent.name}/{output_dir.name})",
        log_model="all",
        save_dir=str(output_dir),
        config=OmegaConf.to_container(omega_conf),
    )

    # Only load the model's saved state (so that optimization restarts).
    if cfg.checkpoint.load is not None:
        checkpoint_path = update_checkpoint_path(cfg.checkpoint.load, cfg.wandb)
        checkpoint = torch.load(checkpoint_path)
        model_wrapper.load_state_dict(checkpoint["state_dict"], strict=False)

    # Log code to wandb if rank is 0. On rank != 0, wandb.run is None.
    if wandb.run is not None:
        wandb.run.log_code("project")

    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")

    trainer = Trainer(
        max_steps=cfg.training.max_steps,
        accelerator="gpu",
        logger=logger,
        devices="auto",
        strategy=(
            "ddp_find_unused_parameters_true"
            if torch.cuda.device_count() > 1
            else "auto"
        ),
        # callbacks=callbacks,
        val_check_interval=cfg.training.val_check_interval,
    )

    trainer.fit(
        model_wrapper,
        datamodule=data_module,
    )


if __name__ == "__main__":
    train()
