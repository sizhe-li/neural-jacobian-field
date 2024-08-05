import math
from typing import Optional, Dict, Any

import cv2
import numpy as np
import torch
import tqdm
import wandb
from einops import pack, rearrange
from lightning_fabric.utilities.apply_func import apply_to_collection
from nerfstudio.model_components.losses import distortion_loss, interlevel_loss
from nerfstudio.utils.colormaps import apply_depth_colormap
from omegaconf import DictConfig
from pytorch_lightning import LightningModule
from pytorch_lightning.loggers.wandb import WandbLogger
from pytorch_lightning.utilities.rank_zero import rank_zero_only
from torch import Tensor, optim
from torch.utils.data.dataloader import default_collate
from torchvision.utils import flow_to_image

from neural_jacobian_field.models.action_model import ActionModel, ActionModelInput
from neural_jacobian_field.rendering.geometry import get_world_rays_with_z
from neural_jacobian_field.utils import convention, logging_utils, loss_utils
from neural_jacobian_field.visualization.view_interpolation import (
    interpolate_intrinsics,
    interpolate_pose,
)


class ActionWrapper(LightningModule):
    logger: Optional[WandbLogger]
    model: ActionModel
    cfg: DictConfig

    def __init__(
            self,
            cfg: DictConfig,
            model: ActionModel,
    ):
        super().__init__()
        self.model = model
        self.cfg = cfg
        self.should_decay_sigma: bool = False
        self.depth_sigma: float = 0.001
        self.starting_depth_sigma: float = 0.01
        self.sigma_decay_rate: float = 0.99985

        if self.should_decay_sigma:
            self.depth_sigma = torch.tensor([self.starting_depth_sigma])
        else:
            self.depth_sigma = torch.tensor([self.depth_sigma])

    def on_load_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        """This function loads the model and informs the user about skipped parameters"""
        state_dict = checkpoint["state_dict"]
        model_state_dict = self.state_dict()
        is_changed = False
        for k in state_dict:
            if k in model_state_dict:
                if state_dict[k].shape != model_state_dict[k].shape:
                    print(
                        f"Skip loading parameter: {k}, "
                        f"required shape: {model_state_dict[k].shape}, "
                        f"loaded shape: {state_dict[k].shape}"
                    )
                    state_dict[k] = model_state_dict[k]
                    is_changed = True
            else:
                print(f"Dropping parameter {k}")
                is_changed = True

        if is_changed:
            checkpoint.pop("optimizer_states", None)

    def on_train_batch_start(self, batch, batch_idx):
        super().on_train_batch_start(batch, batch_idx)
        self.model.step_before_iter(self.global_step)

    def on_train_batch_end(self, outputs, batch, batch_idx):
        super().on_train_batch_end(outputs, batch, batch_idx)
        self.model.step_after_iter(self.global_step)

    def training_step(self, batch, batch_idx):
        coordinates = batch["scene"]["coordinates"]
        image_height, image_width = coordinates.shape[1:3]

        ### unpack target signals
        trgt_rgb = batch["target"]["rgb"]
        trgt_depth = batch["target"]["depth"]

        ### (A) random subsample
        if self.cfg.model.train_encoder:
            indices = torch.floor(
                torch.rand((self.cfg.rays_per_batch, 2))
                * torch.tensor([image_height, image_width])
            ).long()
            y, x = (i.flatten() for i in torch.split(indices, 1, dim=-1))
            # subsample rays randomly
            trgt_rgb = trgt_rgb[:, :, y, x].transpose(1, 2)
            trgt_depth = trgt_depth[:, :, y, x].transpose(1, 2)
            coordinates = coordinates[:, y, x, :]

        elif self.cfg.model.train_flow:
            if "pixel_motion" not in batch["target"]:
                indices = torch.floor(
                    torch.rand((self.cfg.rays_per_batch, 2))
                    * torch.tensor([image_height, image_width])
                ).long()
                y, x = (i.flatten() for i in torch.split(indices, 1, dim=-1))
                # subsample rays randomly
                trgt_rgb = trgt_rgb[:, :, y, x].transpose(1, 2)
                trgt_depth = trgt_depth[:, :, y, x].transpose(1, 2)
                coordinates = coordinates[:, y, x, :]

                # add to batch for sanity logging
                batch["trgt_flow"] = batch["target"]["flow"]

                trgt_flow = batch["target"]["flow"]
                # subsample rays
                trgt_flow = trgt_flow[:, :, y, x].transpose(1, 2)

            else:
                ### (B) subsample according to pixel selector and pixel motion
                pixel_selector = batch["target"]["pixel_selector"]  # (B, N)
                trgt_flow = batch["target"]["pixel_motion"]  # (B, N, 2)
                trgt_visible_mask = batch["target"]["pixel_visible_mask"]  # (B, N)

                trgt_rgb = rearrange(trgt_rgb, "b c h w -> b (h w) c")
                trgt_depth = rearrange(trgt_depth, "b c h w -> b (h w) c")
                coordinates = rearrange(coordinates, "b h w c -> b (h w) c")

                trgt_rgb = torch.gather(
                    trgt_rgb,
                    dim=1,
                    index=pixel_selector.unsqueeze(-1).expand(
                        -1, -1, trgt_rgb.shape[-1]
                    ),
                )
                trgt_depth = torch.gather(
                    trgt_depth,
                    dim=1,
                    index=pixel_selector.unsqueeze(-1).expand(
                        -1, -1, trgt_depth.shape[-1]
                    ),
                )
                coordinates = torch.gather(
                    coordinates,
                    dim=1,
                    index=pixel_selector.unsqueeze(-1).expand(
                        -1, -1, coordinates.shape[-1]
                    ),
                )

        origins, directions, z_values = get_world_rays_with_z(
            coordinates,
            batch["target"]["intrinsics"],
            batch["target"]["extrinsics"],
        )

        # normalize trgt_depth by z_values
        trgt_depth = trgt_depth / z_values
        # add to batch for sanity logging
        batch["trgt_depth"] = trgt_depth

        inputs: ActionModelInput
        inputs = {
            "ctxt_rgb": batch["context"]["rgb"],
            "ctxt_c2w": batch["context"]["extrinsics"],
            "ctxt_intr": batch["context"]["intrinsics"],
            "robot_action": batch["context"]["robot_action"],
            "trgt_c2w": batch["target"]["extrinsics"],
            "trgt_intr": convention.denormalize_intrinsics(
                batch["target"]["intrinsics"], image_width, image_height
            ),
        }

        output = self.model.forward(
            origins,
            directions,
            batch["scene"]["near"],
            batch["scene"]["far"],
            inputs,
        )

        loss = 0.0

        if self.cfg.model.train_encoder:
            # rgb loss
            rgb_loss = torch.nn.functional.mse_loss(output["rgb"], trgt_rgb)

            # ds depth loss
            weights_list = output["weights_list"]
            ray_samples_list = output["ray_samples_list"]
            depth_sigma = self._get_sigma().to(trgt_depth.device)
            depth_loss = 0.0
            for i in range(len(weights_list)):
                weights = weights_list[i]
                steps = (ray_samples_list[i].starts + ray_samples_list[i].ends) / 2
                lengths = ray_samples_list[i].ends - ray_samples_list[i].starts
                depth_loss += loss_utils.ds_nerf_depth_loss(
                    weights, trgt_depth, steps, lengths, depth_sigma
                ) / len(weights_list)
            depth_loss = 0.08 * depth_loss

            interlevel_loss_val = 1.0 * interlevel_loss(weights_list, ray_samples_list)
            distortion_loss_val = 0.01 * distortion_loss(weights_list, ray_samples_list)

            loss += rgb_loss + interlevel_loss_val + distortion_loss_val + depth_loss
            # log the loss.
            self.log("loss/rgb", rgb_loss)
            self.log("loss/depth", depth_loss)
            self.log("loss/distortion", distortion_loss_val)
            self.log("loss/interlevel", interlevel_loss_val)

        if self.cfg.model.train_flow:
            # flow-related loss
            flow_loss = 0.01 * torch.nn.functional.mse_loss(
                output["optical_flow"], trgt_flow, reduction="none"
            )

            if "pixel_motion" in batch["target"]:
                # apply masking
                flow_loss = flow_loss * trgt_visible_mask.unsqueeze(-1)
                flow_loss = flow_loss.sum() / trgt_visible_mask.sum()
            else:
                flow_loss = flow_loss.mean()

            loss += flow_loss

            self.log("loss/flow_loss", flow_loss)

        # log sanity metrics.
        for k, v in logging_utils.get_sanity_metrics(batch).items():
            self.log(f"sanity/input_{k}", v)
        for k, v in logging_utils.get_sanity_metrics(output).items():
            self.log(f"sanity/output_{k}", v)

        # log the first 10 percentile of target depth values.
        valid_depth = trgt_depth[trgt_depth > 0]
        self.log("depth/trgt_10_percentile", torch.quantile(valid_depth, 0.1))
        # log the last 10 percentile of target depth values.
        self.log("depth/trgt_90_percentile", torch.quantile(valid_depth, 0.9))
        # log the mean of target depth values.
        self.log("depth/trgt_mean", torch.mean(valid_depth))
        # log the first 10 percentile of predicted depth values.
        self.log("depth/pred_10_percentile", torch.quantile(output["depth"], 0.1))
        # log the last 10 percentile of predicted depth values.
        self.log("depth/pred_90_percentile", torch.quantile(output["depth"], 0.9))
        # log the mean of predicted depth values.
        self.log("depth/pred_mean", torch.mean(output["depth"]))

        return loss

    @rank_zero_only
    def video_validation_step(self, batch):
        if self.cfg.wandb.mode == "disabled":
            return

        # unpack batch
        coordinates = batch["scene"]["coordinates"]
        image_height, image_width = coordinates.shape[1:3]
        coordinates = rearrange(coordinates, "b h w c -> b (h w) c")[0:1]

        self.render_interpolated_view(
            batch,
            coordinates=coordinates,
            image_height=image_height,
            image_width=image_width,
        )

    def validation_step(self, batch, batch_idx):
        with torch.no_grad():
            self.video_validation_step(batch)

        return

    def test_step(self, batch, batch_idx):
        coordinates = batch["scene"]["coordinates"]
        image_height, image_width = coordinates.shape[1:3]

        ### unpack target signals
        trgt_rgb = batch["target"]["rgb"]
        trgt_depth = batch["target"]["depth"]

        coordinates = rearrange(coordinates, "b h w c -> b (h w) c")
        origins, directions, z_values = get_world_rays_with_z(
            coordinates,
            batch["target"]["intrinsics"],
            batch["target"]["extrinsics"],
        )

        scale_factor = 15
        output = self.model.patch_render(
            ctxt_image=batch["context"]["rgb"],
            robot_action=batch["context"]["robot_action"],
            origins=origins,
            directions=directions,
            ctxt_c2w=batch["context"]["extrinsics"],
            trgt_c2w=batch["target"]["extrinsics"],
            ctxt_intr=batch["context"]["intrinsics"],
            trgt_intr=batch["target"]["intrinsics"],
            z_near=batch["scene"]["near"],
            z_far=batch["scene"]["far"],
            patch_size=(image_height * image_width) // scale_factor,
            render_height=image_height,
            render_width=image_width,
            compute_action_features=False,
            verbose=True,
        )

        pred_depth = rearrange(output["pred_depth"], "b h w c -> b (h w) c")
        pred_depth = pred_depth * z_values  # denormalize depth to meter unit

        pred_flow = rearrange(output["pred_flow"], "b c h w -> b (h w) c")
        # for visualization purposes
        pred_depth_rgb = output["pred_depth_rgb"]
        pred_flow_rgb = output["pred_flow_rgb"]

        if "pixel_motion" not in batch["target"]:
            trgt_flow = batch["target"]["flow"]
            trgt_flow = rearrange(trgt_flow, "b c h w -> b (h w) c")
        else:
            ### (B) subsample according to pixel selector and pixel motion
            pixel_selector = batch["target"]["pixel_selector"]  # (B, N)
            trgt_flow = batch["target"]["pixel_motion"]  # (B, N, 2)
            trgt_visible_mask = batch["target"]["pixel_visible_mask"]  # (B, N)

            pred_flow = torch.gather(
                pred_flow,
                dim=1,
                index=pixel_selector.unsqueeze(-1).expand(-1, -1, pred_flow.shape[-1]),
            )

        trgt_depth = rearrange(trgt_depth, "b c h w -> b (h w) c")

        # compute meter-unit depth loss
        # TODO: configure this hard-code
        depth_valid_mask = (trgt_depth > 0) & (trgt_depth < 5)

        depth_loss = torch.nn.functional.mse_loss(
            pred_depth[depth_valid_mask], trgt_depth[depth_valid_mask], reduction="mean"
        )
        flow_loss = torch.nn.functional.mse_loss(pred_flow, trgt_flow, reduction="none")

        if "pixel_motion" in batch["target"]:
            # apply masking
            flow_loss = flow_loss * trgt_visible_mask.unsqueeze(-1)
            flow_loss = flow_loss.sum() / trgt_visible_mask.sum()
        else:
            flow_loss = flow_loss.mean()

        # log the loss.
        self.log("loss/depth", depth_loss)
        self.log("loss/flow", flow_loss)

        # import pdb
        # pdb.set_trace()
        # log context image
        visualizations = {
            "input/rgb": wandb.Image(batch["context"]["rgb"][0].squeeze(0)),
            "output/flow": wandb.Image(pred_flow_rgb.cpu().numpy()[0]),
            "output/depth": wandb.Image(pred_depth_rgb.cpu().numpy()[0]),
        }

        wandb.log(visualizations)

        self.test_epoch_outputs.append(
            {
                "depth_loss": depth_loss.item(),
                "flow_loss": flow_loss.item(),
            }
        )

        # return depth_loss, flow_loss

    def on_test_epoch_start(self) -> None:
        self.test_epoch_outputs = []

    def on_test_epoch_end(self) -> None:
        # compute mean, std, and median
        depth_loss_th = torch.tensor([x["depth_loss"] for x in self.test_epoch_outputs])
        flow_loss_th = torch.tensor([x["flow_loss"] for x in self.test_epoch_outputs])

        depth_loss_mean = depth_loss_th.mean()
        flow_loss_mean = flow_loss_th.mean()

        depth_loss_std = depth_loss_th.std()
        flow_loss_std = flow_loss_th.std()

        depth_loss_median = depth_loss_th.median()
        flow_loss_median = flow_loss_th.median()

        self.log("test/depth_loss_mean", depth_loss_mean)
        self.log("test/flow_loss_mean", flow_loss_mean)
        self.log("test/depth_loss_std", depth_loss_std)
        self.log("test/flow_loss_std", flow_loss_std)
        self.log("test/depth_loss_median", depth_loss_median)
        self.log("test/flow_loss_median", flow_loss_median)

    def configure_optimizers(self):
        optimizer = optim.Adam(
            self.parameters(), lr=self.cfg.training.optim.lr, weight_decay=1e-5
        )
        warm_up_steps = self.cfg.training.optim.warm_up_steps
        warm_up = torch.optim.lr_scheduler.LinearLR(
            optimizer,
            1 / warm_up_steps,
            1,
            total_iters=warm_up_steps,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": warm_up,
                "interval": "step",
                "frequency": 1,
            },
        }

    def _get_sigma(self):
        if not self.should_decay_sigma:
            return self.depth_sigma

        self.depth_sigma = torch.maximum(
            self.sigma_decay_rate * self.depth_sigma,
            torch.tensor([self.depth_sigma]),
        )
        return self.depth_sigma

    def render_other_views(self, batch, coordinates, image_width, image_height):
        other_flows = batch["other_views"]["flows"].squeeze(0)
        other_extrinsics = batch["other_views"]["extrinsics"].squeeze(0)
        other_intrinsics = batch["other_views"]["intrinsics"].squeeze(0)

        render_other_view_logs = {}
        ### render other views
        for view_idx in range(other_flows.shape[0]):
            interp_trgt_c2w = other_extrinsics[view_idx: view_idx + 1]
            trgt_intr = other_intrinsics[view_idx: view_idx + 1]

            origins, directions, z_values = get_world_rays_with_z(
                coordinates, trgt_intr, interp_trgt_c2w
            )

            inputs: ActionModelInput
            inputs = {
                "ctxt_rgb": batch["context"]["rgb"],
                "ctxt_c2w": batch["context"]["extrinsics"],
                "ctxt_intr": batch["context"]["intrinsics"],
                "robot_action": batch["context"]["robot_action"],
                "trgt_c2w": interp_trgt_c2w,
                "trgt_intr": convention.denormalize_intrinsics(
                    trgt_intr, image_width, image_height
                ),
            }
            output = self.model.forward(
                origins,
                directions,
                batch["scene"]["near"],
                batch["scene"]["far"],
                inputs,
            )
            output = {k: v for k, v in output.items() if k in ["optical_flow"]}

            pred_flow = flow_to_image(
                rearrange(
                    output["optical_flow"],
                    "() (h w) c -> c h w",
                    h=image_height,
                    w=image_width,
                )
            )
            trgt_flow = flow_to_image(other_flows[view_idx])
            pred_flow = rearrange(pred_flow, "c h w -> h w c")
            trgt_flow = rearrange(trgt_flow, "c h w -> h w c")

            render_other_view_logs[f"flow/{view_idx}/pred"] = wandb.Image(
                pred_flow.to(torch.uint8).cpu().numpy()
            )
            render_other_view_logs[f"flow/{view_idx}/trgt"] = wandb.Image(
                trgt_flow.to(torch.uint8).cpu().numpy()
            )

        wandb.log(render_other_view_logs)

    def render_interpolated_view(
            self,
            batch,
            coordinates,
            image_height,
            image_width,
    ):
        trgt_depth_t0 = batch["target"]["depth"]

        ### render video output
        outputs = []
        pred_depth_t0 = None
        pred_flow_t0 = None

        for t in tqdm.tqdm(
                torch.linspace(0, 1, self.cfg.validation.video.num_frames),
                desc="Rendering video frames",
        ):
            # Apply smoothing.
            t = (math.cos(math.pi * (t.item() + 1)) + 1) / 2

            # Video validation always uses a batch size of 1.
            interp_trgt_c2w = interpolate_pose(
                batch["target"]["extrinsics"][0],
                batch["context"]["extrinsics"][0],
                t,
            )[None]

            interp_trgt_intr = interpolate_intrinsics(
                batch["target"]["intrinsics"][0],
                batch["context"]["intrinsics"][0],
                t,
            )[None]

            origins, directions, z_values = get_world_rays_with_z(
                coordinates, interp_trgt_intr, interp_trgt_c2w
            )

            inputs: ActionModelInput
            inputs = {
                "ctxt_rgb": batch["context"]["rgb"],
                "ctxt_c2w": batch["context"]["extrinsics"],
                "ctxt_intr": batch["context"]["intrinsics"],
                "robot_action": batch["context"]["robot_action"],
                "trgt_c2w": interp_trgt_c2w,
                "trgt_intr": convention.denormalize_intrinsics(
                    interp_trgt_intr, image_width, image_height
                ),
            }

            output = self.model.forward(
                origins,
                directions,
                batch["scene"]["near"],
                batch["scene"]["far"],
                inputs,
            )
            output = {
                k: v for k, v in output.items() if k in ["rgb", "depth", "optical_flow"]
            }

            if t == 0:
                # normalize trgt_depth by z_values
                z_values = rearrange(
                    z_values,
                    "b (h w) 1 -> b 1 h w",
                    h=image_height,
                    w=image_width,
                )
                trgt_depth_t0 = trgt_depth_t0 / z_values

                pred_depth_t0 = rearrange(
                    output["depth"],
                    "b (h w) c -> b c h w",
                    h=image_height,
                    w=image_width,
                ).clone()

                pred_flow_t0 = rearrange(
                    output["optical_flow"],
                    "b (h w) c -> b c h w",
                    h=image_height,
                    w=image_width,
                ).clone()

            output["rgb"] = rearrange(
                output["rgb"],
                "b (h w) c -> b c h w",
                h=image_height,
                w=image_width,
            )
            output["depth"] = rearrange(
                apply_depth_colormap(
                    rearrange(
                        output["depth"],
                        "b (h w) c -> b h w c",
                        h=image_height,
                        w=image_width,
                    ),
                ),
                "b h w c -> b c h w",
            )

            output["optical_flow"] = flow_to_image(
                rearrange(
                    output["optical_flow"],
                    "b (h w) c -> b c h w",
                    h=image_height,
                    w=image_width,
                )
            )

            outputs.append(output)

        # Collate the video and convert from TBCHW/TBHW to BTCHW/BTHW.
        outputs = default_collate(outputs)
        outputs = apply_to_collection(
            outputs, Tensor, lambda value: rearrange(value, "t b ... -> b t ...")
        )

        # Run the visualizers on the batch.
        # Since there's no corresponding ground truth, don't provide model_input.'
        visualizations = {}
        visualizations["rgb"] = (
            (outputs["rgb"].clip(min=0, max=1) * 255).type(torch.uint8).cpu().numpy()
        )
        visualizations["depth"] = (
            (outputs["depth"].clip(min=0, max=1) * 255).type(torch.uint8).cpu().numpy()
        )
        visualizations["optical_flow"] = (
            outputs["optical_flow"].type(torch.uint8).cpu().numpy()
        )

        # Make the animation seamless by reversing it.
        visualizations = {
            k: pack([v, v[:, ::-1][:, 1:-1]], "b * c h w")[0]
            for k, v in visualizations.items()
        }

        # Convert to wandb video format.
        visualizations = {
            f"video/{k}": wandb.Video(
                v, fps=self.cfg.validation.video.frame_rate, format="mp4"
            )
            for k, v in visualizations.items()
        }

        # apply the same transform to depth
        trgt_depth_t0 = torch.nn.functional.interpolate(
            trgt_depth_t0,
            size=pred_depth_t0.shape[-2:],
            mode="nearest",
        )

        # compare the mse between the predicted depth and the ground truth depth
        depth_loss = torch.nn.functional.mse_loss(
            pred_depth_t0,
            trgt_depth_t0,
        )

        wandb.log({"loss/validation_depth_loss": depth_loss.item()})

        if self.cfg.model.train_flow:
            visualizations["optical_flow"] = (
                outputs["optical_flow"].type(torch.uint8).cpu().numpy()
            )

            # render pixel motion
            if "pixel_motion" in batch["target"]:
                self.render_pixel_motion(batch, coordinates, image_height, image_width)

        # log context image
        visualizations["rgb/ctxt"] = wandb.Image(batch["context"]["rgb"].squeeze(0))

        wandb.log(visualizations)

    def render_pixel_motion(self, batch, coordinates, image_height, image_width):
        pixel_selector = batch["target"]["pixel_selector"].long()
        pixel_motion = batch["target"]["pixel_motion"]

        orig_image_height, orig_image_width = batch["target"]["rgb"].shape[-2:]

        # draw motion arrows.
        # start and end points should have circles. if flow is non zero then draw the arrow.
        pixel_selector_y = pixel_selector // orig_image_width
        pixel_selector_x = pixel_selector % orig_image_width

        render_logs = {}
        # draw the motion arrows
        for view_idx in range(pixel_selector.shape[0]):
            # draw the motion arrows
            canvas = batch["target"]["rgb"][view_idx].permute(1, 2, 0).cpu().numpy()
            canvas = (canvas * 255).clip(0, 255).astype(np.uint8)
            canvas = np.ascontiguousarray(canvas)

            num_pts_to_draw = 200
            for ray_idx in tqdm.trange(num_pts_to_draw, desc="drawing motion arrows"):
                y_coord, x_coord = (
                    pixel_selector_y[view_idx, ray_idx],
                    pixel_selector_x[view_idx, ray_idx],
                )
                motion_x, motion_y = pixel_motion[view_idx, ray_idx].split(1, dim=-1)

                start_pt = (int(x_coord), int(y_coord))
                end_pt = (int(x_coord + motion_x), int(y_coord + motion_y))

                # draw green circle at start point
                canvas = cv2.circle(canvas, start_pt, 1, (0, 255, 0), -1)
                # draw red circle at end point
                canvas = cv2.circle(canvas, end_pt, 1, (255, 0, 0), -1)

                if (torch.abs(motion_x) + torch.abs(motion_y)).sum() > 0:
                    canvas = cv2.arrowedLine(
                        canvas,
                        start_pt,
                        end_pt,
                        (255, 0, 0),
                        1,
                    )

                render_logs[f"motion_gt/{view_idx}"] = wandb.Image(canvas)

        wandb.log(render_logs)
