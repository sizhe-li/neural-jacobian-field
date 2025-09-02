from pathlib import Path

import hydra
import torch
from colorama import Fore
from jacobian.config.common import PipelineCfg, get_typed_root_config
from jacobian.models.model_wrapper_base import get_wrapper
from jaxtyping import install_import_hook
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import Trainer


def cyan(text: str) -> str:
    return f"{Fore.CYAN}{text}{Fore.RESET}"


with install_import_hook(
    ("jacobian",),
    ("beartype", "beartype"),
):
    from jacobian.dataset.data_module import DataModule
    from jacobian.utils.wandb_logger import SpaceEfficientWandbLogger


@hydra.main(
    version_base=None,
    config_path="configurations",
    config_name="config",
)
def train(cfg: DictConfig) -> None:

    # Set up the output directory.
    output_dir = Path(
        hydra.core.hydra_config.HydraConfig.get()["runtime"]["output_dir"]
    )

    # change output dir to have a different parent
    # parent = Path("/data/scene-rep/u/iyu/scene-jacobian-discovery/")

    output_dir = Path("outputs") / output_dir.name
    print(cyan(f"Saving outputs to {output_dir}."))

    omega_conf = OmegaConf.create(cfg)

    cfg: PipelineCfg = get_typed_root_config(cfg, PipelineCfg)

    data_module = DataModule(cfg)
    model_wrapper = get_wrapper(
        cfg=cfg.wrapper,
    )

    logger = SpaceEfficientWandbLogger(
        project=cfg.wandb.project,
        entity=cfg.wandb.entity,
        mode=cfg.wandb.mode,
        name=f"{cfg.wandb.name} ({output_dir.parent.name}/{output_dir.name})",
        log_model="all",
        save_dir=str(output_dir),
        config=OmegaConf.to_container(omega_conf),
    )

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
