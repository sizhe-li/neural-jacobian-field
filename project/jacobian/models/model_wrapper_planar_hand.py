from dataclasses import asdict, dataclass
from typing import Literal

import numpy as np
import torch
from math import sqrt
import tqdm
import wandb
from einops import einsum, rearrange, reduce
from jacobian.utils import logging_utils
from lightning.pytorch.utilities.types import OptimizerLRScheduler
from pytorch_lightning import LightningModule
from torch import optim
from torchvision.utils import flow_to_image

from .jacobian_models import JacobianNetOutput, ModelCfg, get_model

from jaxtyping import Float
from torch import Tensor

from torch.utils.data.dataloader import default_collate
from lightning_fabric.utilities.apply_func import apply_to_collection

# from lightning.pytorch.utilities.apply_func import apply_to_collection


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


def create_video_grid(images: torch.Tensor):

    B, T, C, H, W = images.shape
    grid_size = int(sqrt(B))

    if grid_size**2 != B:
        raise ValueError("B must be a perfect square.")

    # Reshape to (grid_size, grid_size, T, 3, H, W)
    images = images.view(grid_size, grid_size, T, C, H, W)

    # Permute to (T, 3, grid_size, H, grid_size, W)
    images = images.permute(2, 3, 0, 4, 1, 5)

    # Reshape to (T, 3, grid_size * H, grid_size * W)
    grid = images.reshape(T, C, grid_size * H, grid_size * W)

    return grid


@dataclass
class OptimzierCfg:
    lr: float = 1e-4
    warm_up_steps: int = 10000


@dataclass
class PlanarHandModelWrapperCfg:
    name: Literal["planar_hand"]
    optim: OptimzierCfg
    model: ModelCfg


class PlanarHandModelWrapper(LightningModule):
    jac_color_codes = torch.tensor(
        [
            [1, 0, 0],
            [0, 0, 1],
            [0, 1, 0],
            [1, 0, 1],
            [1, 0.5, 1],
            [0.5, 0.5, 1],
        ],
        dtype=torch.float32,
    )  # shape: cmd_dim x 3

    def __init__(self, cfg: PlanarHandModelWrapperCfg) -> None:
        super().__init__()

        self.cfg = cfg
        self.model = get_model(cfg.model)

        self.register_buffer("jac_color_codes_bufffer", self.jac_color_codes)

    def training_step(self, batch, batch_idx):
        input_frame_curr = batch["input_frame_curr"]
        input_command = batch["input_command"]
        trgt_flow_curr = batch["trgt_flow_curr"]

        net_output: JacobianNetOutput = self.model.forward(
            input_frame_curr, input_command
        )

        mse_loss = torch.nn.functional.mse_loss(net_output.flow, trgt_flow_curr)

        consistency_loss = 0.0

        self.log("train/consistency_loss", consistency_loss)
        self.log("train/mse_loss", mse_loss)

        mse_weight = 1.0 - self.cfg.model.consistency_loss_weight
        consistency_weight = self.cfg.model.consistency_loss_weight
        loss = mse_loss + consistency_weight * consistency_loss

        for k, v in logging_utils.get_sanity_metrics(batch).items():
            self.log(f"sanity/input_{k}", v)

        # convert datalcass output to dict, TODO: @lester hack because gradient problems
        net_output_dict = {
            "flow": net_output.flow,
            "jacobian": net_output.jacobian,
        }

        for k, v in logging_utils.get_sanity_metrics(net_output_dict).items():
            self.log(f"sanity/output_{k}", v)

        # log loss
        self.log("train/loss", loss)

        return loss

    def validation_step(self, batch):
        # here batch is all examples from validation set (81 of them)
        input_video_sequence = batch["input_video_sequence"]  # B x T x C X H X W
        input_command_sequence = batch["input_command_sequence"]  # B x T x C_command
        trgt_flow_sequence = batch["trgt_flow_sequence"]  # B x T x 2 x H x W

        input_frame_curr = input_video_sequence[:, :]
        # input_frame_next = input_video_sequence[:, 1:]

        # TODO: batch process this
        batch_size = int(sqrt(input_video_sequence.shape[0]))
        num_examples = input_frame_curr.shape[0]
        num_batches = num_examples // batch_size

        pred_flows, pred_jacobians, trgt_flow_sequences, input_video_sequences = (
            [],
            [],
            [],
            [],
        )
        loss = 0.0
        with torch.no_grad():
            for i in range(0, num_examples, batch_size):
                start_frame_index = i
                end_frame_index = min(i + batch_size, num_examples)

                # reshape to B*T x C X H X W
                input_frame_curr_reshaped = input_frame_curr[
                    start_frame_index:end_frame_index
                ].reshape(-1, *input_frame_curr.shape[2:])
                input_command_sequence_reshaped = input_command_sequence[
                    start_frame_index:end_frame_index
                ].reshape(-1, *input_command_sequence.shape[2:])

                net_output: JacobianNetOutput = self.model.forward(
                    input_frame_curr_reshaped,
                    input_command_sequence_reshaped,
                )  # B*T x 2 x H x W

                flow = net_output.flow.reshape(
                    batch_size, -1, *net_output.flow.shape[1:]
                )
                jacobian = net_output.jacobian.reshape(
                    batch_size, -1, *net_output.jacobian.shape[1:]
                )

                pred_flows.append(flow)
                pred_jacobians.append(jacobian)
                trgt_flow_sequences.append(
                    trgt_flow_sequence[start_frame_index:end_frame_index]
                )
                input_video_sequences.append(
                    input_video_sequence[start_frame_index:end_frame_index]
                )

                # loss += torch.nn.functional.mse_loss(
                #     flow, trgt_flow_sequence[start_frame_index:end_frame_index]
                # )

            # pred_flow = torch.cat(pred_flow, dim=0)
            # pred_jacobian = torch.cat(pred_jacobian, dim=0)

            loss /= num_batches

        video_visualizations = dict()
        self.log("val/average_temporal_MSE_loss", loss)

        for i, (pred_flow, pred_jacobian, trgt_flow, input_video_sequence) in enumerate(
            zip(pred_flows, pred_jacobians, trgt_flow_sequences, input_video_sequences)
        ):
            # each item in the tuple has shape B x T x C x H x W, where B is batch_size (9 by default)
            # we want to organize the data in 3 x 3 grid, or an array of shape T x C x 3*H x 3*W

            # since flow_to_image expects B x 2 x H x W, we need to convert pred_flow and trgt_flow to this shape
            pred_flow_reshaped = pred_flow.reshape(-1, *pred_flow.shape[2:])
            trgt_flow_reshaped = trgt_flow.reshape(-1, *trgt_flow.shape[2:])

            pred_flow_image = flow_to_image(pred_flow_reshaped)  # B x 3 x H x W
            pred_flow_image_grid = (
                create_video_grid(
                    pred_flow_image.reshape(
                        *pred_flow.shape[:2], *pred_flow_image.shape[1:]
                    )
                )
                .cpu()
                .numpy()
            )

            trgt_flow_image = flow_to_image(trgt_flow_reshaped)  # B x 3 x H x W
            trgt_flow_image_grid = (
                create_video_grid(
                    trgt_flow_image.reshape(
                        *trgt_flow.shape[:2], *trgt_flow_image.shape[1:]
                    )
                )
                .cpu()
                .numpy()
            )

            input_video_sequence_grid = (
                create_video_grid(input_video_sequence).cpu().numpy()
            )

            # 1. draw predicted optical flow
            video_visualizations[f"video/pred_optical_flow_{i}"] = wandb.Video(
                # B X 2 x H X W -> B X C X H X W (min,max=0, 255, uint8)
                pred_flow_image_grid,
                fps=12,
                format="mp4",
            )
            # 2. draw ground truth optical flow
            video_visualizations[f"video/ground_truth_optical_flow_{i}"] = wandb.Video(
                trgt_flow_image_grid, fps=12, format="mp4"
            )

            # 3. draw input video
            video_visualizations[f"video/input_video_{i}"] = wandb.Video(
                (255 * input_video_sequence_grid).astype(np.uint8),
                fps=12,
                format="mp4",
            )

            # 4. draw jacobian video

            # convert sensitivity to color code
            jacobian_color_codes = torch.tensor(
                [
                    [1, 0, 0],
                    [0, 0, 1],
                    [0, 1, 0],
                    [1, 0, 1],
                    [1, 0.5, 1],
                    [0.3, 0.5, 1],
                ],
                dtype=pred_jacobian.dtype,
                device=pred_jacobian.device,
            )  # shape C_cmd x 3

            num_command_channels = pred_jacobian.shape[-4]

            for cmd_idx in range(num_command_channels):
                pred_jacobian_qi = pred_jacobian.clone()

                for i in range(num_command_channels):
                    if i != cmd_idx:
                        pred_jacobian_qi[..., i, :, :, :] = 0

                sensitivity_qi = compute_sensitivity(pred_jacobian_qi)
                vis_sensitivity_qi = visualize_sensitivity(
                    sensitivity_qi, jacobian_color_codes[:num_command_channels]
                )

                vis_sensitivity_qi_grid = (
                    255 * create_video_grid(vis_sensitivity_qi).cpu().numpy()
                ).astype(np.uint8)

                video_visualizations[f"video/jacobian_q{cmd_idx}_{i}"] = wandb.Video(
                    vis_sensitivity_qi_grid,
                    fps=12,
                    format="mp4",
                )

        wandb.log(video_visualizations)

        return {
            "pred_flows": pred_flows,
            "pred_jacobians": pred_jacobians,
            "trgt_flow_sequences": trgt_flow_sequences,
            "input_video_sequences": input_video_sequences,
        }

    @torch.no_grad()
    def visualize_model(
        self,
        input_video_sequence: Float[Tensor, "batch time channels height width"],
        input_command_sequence: Float[Tensor, "batch time command_channels"],
        trgt_flow_sequence: Float[Tensor, "batch time 2 height width"],
    ):

        orig_batch_size, orig_timesteps = input_video_sequence.shape[:2]

        # TODO: batch process this
        total_num_frames = input_video_sequence.shape[0] * input_video_sequence.shape[1]
        process_batch_size = int(sqrt(total_num_frames))
        # trgt_flow_sequence = rearrange(

        input_video_sequence, input_command_sequence, trgt_flow_sequence = (
            apply_to_collection(
                (input_video_sequence, input_command_sequence, trgt_flow_sequence),
                torch.Tensor,
                rearrange,
                "b t ... -> (b t) ...",
            )
        )

        # print(
        #     f"""
        #     input_video_sequence.shape: {input_video_sequence.shape},
        #     input_command_sequence.shape: {input_command_sequence.shape},
        #     trgt_flow_sequence.shape: {trgt_flow_sequence.shape}
        #     """
        # )

        pred_outputs = {
            "total_pred_flow_raw": [],
            "total_pred_flow_rgb": [],
            "total_pred_jacobian_raw": [],
            "total_pred_jacobian_rgb": [],
        }

        # total_pred_flow_raw, total_pred_jacobian, total_pred_flow_rgb = [], [], []

        validation_loss = 0.0
        for i in tqdm.trange(
            0, total_num_frames, process_batch_size, desc="Processing validation"
        ):
            batch_idx_start = i
            batch_idx_end = min(i + process_batch_size, total_num_frames)

            batch_input_video = input_video_sequence[batch_idx_start:batch_idx_end]
            batch_input_command = input_command_sequence[batch_idx_start:batch_idx_end]
            batch_trgt_flow = trgt_flow_sequence[batch_idx_start:batch_idx_end]

            # print(
            #     f"""batch_input_video.shape: {batch_input_video.shape},
            #     batch_input_command.shape: {batch_input_command.shape},
            #     batch_trgt_flow.shape: {batch_trgt_flow.shape}"""
            # )

            batch_net_output: JacobianNetOutput = self.model.forward(
                batch_input_video,
                batch_input_command,
            )
            batch_pred_flow = batch_net_output.flow
            batch_pred_jacobian_raw = batch_net_output.jacobian

            batch_pred_jacobian_rgb = self.visualize_jacobian(batch_pred_jacobian_raw)

            pred_outputs["total_pred_flow_raw"].append(batch_pred_flow)
            pred_outputs["total_pred_flow_rgb"].append(flow_to_image(batch_pred_flow))
            pred_outputs["total_pred_jacobian_raw"].append(batch_pred_jacobian_raw)
            pred_outputs["total_pred_jacobian_rgb"].append(batch_pred_jacobian_rgb)

            validation_loss += torch.nn.functional.mse_loss(
                batch_pred_flow,
                batch_trgt_flow,
                reduction="sum",
            )

        validation_loss /= total_num_frames

        # restore original batch and time dimensions
        for k, v in pred_outputs.items():
            pred_outputs[k] = torch.cat(v, dim=0)

        pred_outputs = apply_to_collection(
            pred_outputs,
            torch.Tensor,
            lambda x: rearrange(
                x,
                "(b t) ... -> b t ...",
                b=orig_batch_size,
                t=orig_timesteps,
            ),
        )

        input_video_sequence, input_command_sequence, trgt_flow_sequence = (
            apply_to_collection(
                (input_video_sequence, input_command_sequence, trgt_flow_sequence),
                torch.Tensor,
                rearrange,
                "(b t) ... -> b t ...",
                b=orig_batch_size,
                t=orig_timesteps,
            )
        )

        validation_outputs = {
            "trgt_flow_sequences": trgt_flow_sequence,
            "input_video_sequences": input_video_sequence,
            "validation_loss": validation_loss,
        }

        validation_outputs.update(pred_outputs)

        return validation_outputs

    def visualize_jacobian(
        self, batch_pred_jacobian: Float[Tensor, "... cmd_dim height width"]
    ) -> Float[Tensor, " ... cmd_dim rgb height width"]:

        num_command_channels = batch_pred_jacobian.shape[-4]

        vis_sensitivity_q_list = []
        for cmd_idx in range(num_command_channels):
            pred_jacobian_qi = batch_pred_jacobian.clone()

            for i in range(num_command_channels):
                if i != cmd_idx:
                    pred_jacobian_qi[..., i, :, :, :] = 0

            sensitivity_qi = compute_sensitivity(pred_jacobian_qi)

            # print(
            #     sensitivity_qi.shape,
            #     self.get_buffer("jac_color_codes_bufffer")[:num_command_channels],
            # )

            vis_sensitivity_qi = visualize_sensitivity(
                sensitivity_qi,
                self.get_buffer("jac_color_codes_bufffer")[:num_command_channels],
            )

            vis_sensitivity_q_list.append(vis_sensitivity_qi)

        vis_sensitivity_q_list = torch.stack(vis_sensitivity_q_list, dim=-4)

        return vis_sensitivity_q_list

    def configure_optimizers(self) -> OptimizerLRScheduler:
        return optim.Adam(self.parameters(), lr=self.cfg.optim.lr, weight_decay=1e-5)
