from dataclasses import asdict, dataclass
from typing import Literal

import numpy as np
import torch
import tqdm
import wandb
from einops import einsum, rearrange, reduce
from jacobian.utils import logging_utils
from lightning.pytorch.utilities.types import OptimizerLRScheduler
from pytorch_lightning import LightningModule
from torch import optim
from torchvision.utils import flow_to_image

from .jacobian_models import ModelCfg, get_model

from jaxtyping import Float
from torch import Tensor


def compute_sensitivity(
    input_jacobian: Float[Tensor, "... C_cmd C_spatial H W"],
):
    # norm over spatial dimension
    sensitivity = input_jacobian.norm(dim=-3)

    minima = reduce(sensitivity, "... C_cmd H W -> ... C_cmd () ()", "min")
    maxima = reduce(sensitivity, "... C_cmd H W -> ... C_cmd () ()", "max")

    sensitivity = (sensitivity - minima) / (maxima - minima + 1e-10)
    sensitivity = sensitivity.clip(0, 1)

    return sensitivity


def visualize_sensitivity(input_sensitivity, color_map):
    input_sensitivity = einsum(
        input_sensitivity, color_map, "... feature H W, ... feature rgb -> ... rgb H W"
    )

    minima = reduce(input_sensitivity, "... H W -> ... () ()", "min")
    maxima = reduce(input_sensitivity, "... H W -> ... () ()", "max")

    input_sensitivity = (input_sensitivity - minima) / (maxima - minima + 1e-10)
    input_sensitivity = input_sensitivity.clip(0, 1)

    return input_sensitivity


@dataclass
class OptimzierCfg:
    lr: float = 1e-4
    warm_up_steps: int = 10000


@dataclass
class PusherModelWrapperCfg:
    name: Literal["pusher"]
    optim: OptimzierCfg
    model: ModelCfg


class PusherModelWrapper(LightningModule):
    def __init__(self, cfg: PusherModelWrapperCfg) -> None:
        super().__init__()

        self.cfg = cfg
        self.model = get_model(cfg.model)
        self.is_jacobian = True

    def training_step(self, batch, batch_idx):
        input_frame_curr = batch["input_frame_curr"]
        input_frame_next = batch["input_frame_next"]
        input_command = batch["input_command"]
        trgt_flow_curr = batch["trgt_flow_curr"]

        net_output = self.model.forward(input_frame_curr, input_command)

        loss = torch.nn.functional.mse_loss(net_output.flow, trgt_flow_curr)

        for k, v in logging_utils.get_sanity_metrics(batch).items():
            self.log(f"sanity/input_{k}", v)

        # convert datalcass output to dict, TODO: @lester hack because gradient problems
        net_output_dict = {
            "flow": net_output.flow,
        }
        if self.is_jacobian:
            net_output_dict["jacobian"] = net_output.jacobian

        for k, v in logging_utils.get_sanity_metrics(net_output_dict).items():
            self.log(f"sanity/output_{k}", v)

        # log loss
        self.log("train/loss", loss)

        return loss

    def validation_step(self, batch):

        input_video_sequence = batch["input_video_sequence"][0]  # 1 x T x C X H X W
        input_command_sequence = batch["input_command_sequence"][0]  # 1 x T x C_command
        trgt_flow_sequence = batch["trgt_flow_sequence"][0]  # 1 x T x 2 x H x W

        input_frame_curr = input_video_sequence[:-1]
        input_frame_next = input_video_sequence[1:]

        # TODO: batch process this
        batch_size = 12
        num_frames = input_frame_curr.shape[0]
        num_batches = num_frames // batch_size

        pred_flow, pred_jacobian = [], []
        with torch.no_grad():
            for i in range(0, num_frames, batch_size):
                start_frame_index = i
                end_frame_index = min(i + batch_size, num_frames)

                net_output = self.model.forward(
                    input_frame_curr[start_frame_index:end_frame_index],
                    input_command_sequence[start_frame_index:end_frame_index],
                )

                pred_flow.append(net_output.flow)
                if self.is_jacobian:
                    pred_jacobian.append(net_output.jacobian)

            pred_flow = torch.cat(pred_flow, dim=0)
            if self.is_jacobian:
                pred_jacobian = torch.cat(pred_jacobian, dim=0)

        video_visualizations = dict()

        # 1. draw predicted optical flow
        video_visualizations[f"video/pred_optical_flow"] = wandb.Video(
            # B X 2 x H X W -> B X C X H X W (min,max=0, 255, uint8)
            flow_to_image(pred_flow).cpu().numpy(),
            fps=30,
            format="mp4",
        )
        # 2. draw ground truth optical flow
        video_visualizations["video/ground_truth_optical_flow"] = wandb.Video(
            flow_to_image(trgt_flow_sequence).cpu().numpy(), fps=30, format="mp4"
        )

        # 3. draw input video
        video_visualizations["video/input_video"] = wandb.Video(
            (255 * input_video_sequence.cpu().numpy()).astype(np.uint8),
            fps=30,
            format="mp4",
        )

        # 4. draw jacobian video if model type is jacobian
        if self.is_jacobian:
            # convert sensitivity to color code
            color_codes = torch.tensor(
                [
                    [0, 1, 0],
                    [0, 0, 1],
                ],
                dtype=pred_jacobian.dtype,
                device=pred_jacobian.device,
            )  # shape C_cmd x 3

            pred_jacobian_q0 = pred_jacobian.clone()
            pred_jacobian_q0[..., 1, :, :, :] = 0

            pred_jacobian_q1 = pred_jacobian.clone()
            pred_jacobian_q1[..., 0, :, :, :] = 0

            sensitivity_q0 = compute_sensitivity(pred_jacobian_q0)
            vis_sensitivity_q0 = visualize_sensitivity(sensitivity_q0, color_codes)

            sensitivity_q1 = compute_sensitivity(pred_jacobian_q1)
            vis_sensitivity_q1 = visualize_sensitivity(sensitivity_q1, color_codes)

            # 4.1 draw x dimension
            video_visualizations["video/jacobian_q0"] = wandb.Video(
                (255 * vis_sensitivity_q0.cpu().numpy()).astype(np.uint8),
                fps=30,
                format="mp4",
            )
            # # 4.2 draw y dimension
            video_visualizations["video/jacobian_q1"] = wandb.Video(
                (255 * vis_sensitivity_q1.cpu().numpy()).astype(np.uint8),
                fps=30,
                format="mp4",
            )

        wandb.log(video_visualizations)

        return

    def configure_optimizers(self) -> OptimizerLRScheduler:
        return optim.Adam(self.parameters(), lr=self.cfg.optim.lr, weight_decay=1e-5)

    @torch.no_grad()
    def visualize_model(
        self,
        input_video_sequence: Float[Tensor, "T C H W"],
        input_command_sequence: Float[Tensor, "T C_command"],
        trgt_flow_sequence: Float[Tensor, "T 2 H W"],
    ):

        input_frame_curr = input_video_sequence[:-1]
        input_frame_next = input_video_sequence[1:]

        # TODO: batch process this
        batch_size = 12
        num_frames = input_frame_curr.shape[0]
        num_batches = num_frames // batch_size

        pred_flow, pred_jacobian = [], []
        with torch.no_grad():
            for i in range(0, num_frames, batch_size):
                start_frame_index = i
                end_frame_index = min(i + batch_size, num_frames)

                net_output = self.model.forward(
                    input_frame_curr[start_frame_index:end_frame_index],
                    input_command_sequence[start_frame_index:end_frame_index],
                )

                pred_flow.append(net_output.flow)
                if self.is_jacobian:
                    pred_jacobian.append(net_output.jacobian)

            pred_flow = torch.cat(pred_flow, dim=0)
            print(pred_flow.shape, trgt_flow_sequence.shape)
            total_flow_error = torch.nn.functional.mse_loss(
                pred_flow, trgt_flow_sequence
            )
            if self.is_jacobian:
                pred_jacobian = torch.cat(pred_jacobian, dim=0)

        video_visualizations = dict()

        # 1. draw predicted optical flow
        video_visualizations[f"video/pred_optical_flow"] = (
            flow_to_image(pred_flow)
            .cpu()
            .numpy()
            # torch.stack([flow_to_image(x) for x in pred_flow], dim=0)
            # .cpu()
            # .numpy()
        )

        # 3. draw input video
        video_visualizations["video/input_video"] = (
            255 * input_video_sequence.cpu().numpy()
        ).astype(np.uint8)

        # 4. draw jacobian video if model type is jacobian
        if self.is_jacobian:
            # convert sensitivity to color code
            color_codes = torch.tensor(
                [
                    [0, 1, 0],
                    [0, 0, 1],
                ],
                dtype=pred_jacobian.dtype,
                device=pred_jacobian.device,
            )  # shape C_cmd x 3

            pred_jacobian_q0 = pred_jacobian.clone()
            pred_jacobian_q0[..., 1, :, :, :] = 0

            pred_jacobian_q1 = pred_jacobian.clone()
            pred_jacobian_q1[..., 0, :, :, :] = 0

            sensitivity_q0 = compute_sensitivity(pred_jacobian_q0)
            vis_sensitivity_q0 = visualize_sensitivity(sensitivity_q0, color_codes)

            sensitivity_q1 = compute_sensitivity(pred_jacobian_q1)
            vis_sensitivity_q1 = visualize_sensitivity(sensitivity_q1, color_codes)

            # 4.1 draw x dimension
            video_visualizations["video/jacobian_q0"] = (
                255 * vis_sensitivity_q0.cpu().numpy()
            ).astype(np.uint8)
            # # 4.2 draw y dimension
            video_visualizations["video/jacobian_q1"] = (
                255 * vis_sensitivity_q1.cpu().numpy()
            ).astype(np.uint8)

        video_visualizations["video/ground_truth_optical_flow"] = (
            flow_to_image(trgt_flow_sequence).cpu().numpy()
        )
        # video_visualizations["video/ground_truth_optical_flow"] = (
        #     # flow_to_image(trgt_flow_sequence).cpu().numpy()
        #     torch.stack([flow_to_image(x) for x in trgt_flow_sequence], dim=0)
        #     .cpu()
        #     .numpy()
        # )
        # print(video_visualizations["video/ground_truth_optical_flow"].shape)

        return video_visualizations, total_flow_error
