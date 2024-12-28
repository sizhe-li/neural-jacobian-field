from typing import Dict, Any, Tuple

import cv2
import math
import numpy as np
import torch
import tqdm
import wandb
from einops import pack, rearrange
from jaxtyping import Float
from lightning_fabric.utilities.apply_func import apply_to_collection
from nerfstudio.model_components.losses import distortion_loss, interlevel_loss
from nerfstudio.utils.colormaps import apply_depth_colormap
from pytorch_lightning import LightningModule
from pytorch_lightning.utilities.rank_zero import rank_zero_only
from torch import Tensor, optim
from torch.utils.data.dataloader import default_collate
from torchvision.utils import flow_to_image

from neural_jacobian_field.models.model import (
    Model,
    ModelInput,
    ModelTarget,
    CameraInput,
    RenderingInput,
    RobotInput,
)
from ..rendering.geometry import get_world_rays_with_z
from ..utils import convention, logging_utils, loss_utils
from ..utils.logging_utils import safe_asdict
from ..utils.misc import cyan
from ..visualization.view_interpolation import (
    interpolate_intrinsics,
    interpolate_pose,
)
from ..config.common import PipelineCfg


class ModelWrapper(LightningModule):
    should_decay_sigma: bool = False
    sigma_decay_rate: float = 0.99985
    starting_depth_sigma: float = 0.01
    depth_sigma: Float[Tensor, "1"]

    def __init__(
        self,
        cfg: PipelineCfg,
        model: Model,
    ):
        super().__init__()
        self.model = model
        self.cfg = cfg

        self.initialize_depth_loss()
        self.freeze_parameters()

    def initialize_depth_loss(self):
        # depth loss related
        if self.should_decay_sigma:
            depth_sigma = torch.tensor([self.starting_depth_sigma])
        else:
            depth_sigma = torch.tensor([0.001])
        self.register_buffer("depth_sigma", depth_sigma)

    def _get_sigma(self):
        if not self.should_decay_sigma:
            return self.depth_sigma

        self.depth_sigma = torch.maximum(
            self.sigma_decay_rate * self.depth_sigma,
            self.depth_sigma,
        )
        return self.depth_sigma

    def freeze_parameters(self):
        if self.cfg.dataset.mode == "action":
            # action parameters only exist in the decoder
            num_frozen = self.model.decoder.freeze_non_action_parameters()

            # freeze all other parameters:
            for name, param in self.model.named_parameters():
                if "decoder" not in name:
                    param.requires_grad = False

            print(cyan(f"Froze {num_frozen} layers"))

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

    def training_step(self, batch, batch_idx):
        model_input, model_target = self.prepare_training_input_output(batch)
        model_output = self.model.forward(
            model_input.camera_input,
            model_input.rendering_input,
            model_input.robot_input,
        )

        loss = 0.0

        if self.cfg.dataset.mode == "perception":
            # rgb loss
            rgb_loss = torch.nn.functional.mse_loss(
                model_output.standard_output.rgb, model_target.rgb
            )

            # ds depth loss
            weights_list = model_output.training_output.weights_list
            ray_samples_list = model_output.training_output.ray_samples_list

            depth_sigma = self._get_sigma().to(model_target.depth.device)
            depth_loss = 0.0
            for i in range(len(weights_list)):
                weights = weights_list[i]
                steps = (ray_samples_list[i].starts + ray_samples_list[i].ends) / 2
                lengths = ray_samples_list[i].ends - ray_samples_list[i].starts
                depth_loss += loss_utils.ds_nerf_depth_loss(
                    weights, model_target.depth, steps, lengths, depth_sigma
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

        else:  # mode is "action"
            # flow-related loss
            flow_loss = 0.01 * torch.nn.functional.mse_loss(
                model_output.standard_output.optical_flow,
                model_target.optical_flow,
                reduction="none",
            )
            if model_target.visible_mask is not None:  # apply masking
                flow_loss = flow_loss * model_target.visible_mask.unsqueeze(-1)
                flow_loss = flow_loss.sum() / model_target.visible_mask.sum()
            else:
                flow_loss = flow_loss.mean()

            loss += flow_loss

            self.log("loss/flow_loss", flow_loss)
        # log sanity metrics.
        for k, v in logging_utils.get_sanity_metrics(
            logging_utils.safe_asdict(model_input)
        ).items():
            self.log(f"sanity/input_{k}", v)
        for k, v in logging_utils.get_sanity_metrics(safe_asdict(model_target)).items():
            self.log(f"sanity/target_{k}", v)
        for k, v in logging_utils.get_sanity_metrics(safe_asdict(model_output)).items():
            self.log(f"sanity/output_{k}", v)

        # log the first 10 percentile of target depth values.
        valid_depth = model_target.depth[model_target.depth > 0]
        self.log("depth/trgt_10_percentile", torch.quantile(valid_depth, 0.1))
        # log the last 10 percentile of target depth values.
        self.log("depth/trgt_90_percentile", torch.quantile(valid_depth, 0.9))
        # log the mean of target depth values.
        self.log("depth/trgt_mean", torch.mean(valid_depth))
        # log the first 10 percentile of predicted depth values.
        self.log("depth/pred_10_percentile", torch.quantile(model_target.depth, 0.1))
        # log the last 10 percentile of predicted depth values.
        self.log("depth/pred_90_percentile", torch.quantile(model_target.depth, 0.9))
        # log the mean of predicted depth values.
        self.log("depth/pred_mean", torch.mean(model_target.depth))

        return loss

    def validation_step(self, batch, batch_idx):
        with torch.no_grad():
            self.video_validation_step(batch)

        return

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

            model_output = self.model.forward(
                camera_input=CameraInput(
                    input_image=batch["context"]["rgb"],
                    ctxt_extrinsics=batch["context"]["extrinsics"],
                    ctxt_intrinsics=batch["context"]["intrinsics"],
                    trgt_extrinsics=interp_trgt_c2w,
                    trgt_intrinsics=convention.denormalize_intrinsics(
                        interp_trgt_intr, width=image_width, height=image_height
                    ),
                ),
                rendering_input=RenderingInput(
                    origins=origins,
                    directions=directions,
                    z_near=batch["scene"]["near"],
                    z_far=batch["scene"]["far"],
                ),
                robot_input=RobotInput(
                    robot_action=batch["context"]["robot_action"],
                ),
                compute_vis_features=False,
            )

            output = {}

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
                    model_output.standard_output.depth,
                    "b (h w) c -> b c h w",
                    h=image_height,
                    w=image_width,
                ).clone()

                pred_flow_t0 = rearrange(
                    model_output.standard_output.optical_flow,
                    "b (h w) c -> b c h w",
                    h=image_height,
                    w=image_width,
                ).clone()

            output["rgb"] = rearrange(
                model_output.standard_output.rgb,
                "b (h w) c -> b c h w",
                h=image_height,
                w=image_width,
            )
            output["depth"] = rearrange(
                apply_depth_colormap(
                    rearrange(
                        model_output.standard_output.depth,
                        "b (h w) c -> b h w c",
                        h=image_height,
                        w=image_width,
                    ),
                ),
                "b h w c -> b c h w",
            )

            output["optical_flow"] = flow_to_image(
                rearrange(
                    model_output.standard_output.optical_flow,
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

        if self.cfg.dataset.mode == "action":
            visualizations["optical_flow"] = (
                outputs["optical_flow"].type(torch.uint8).cpu().numpy()
            )

            # render pixel motion
            if "pixel_motion" in batch["target"]:
                self.render_pixel_motion(batch, coordinates, image_height, image_width)

        # log context image
        visualizations["rgb/ctxt"] = wandb.Image(batch["context"]["rgb"].squeeze(0))

        wandb.log({"loss/validation_depth_loss": depth_loss.item()})
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

    @staticmethod
    def random_sample_ray_yx_indices(image_height, image_width, num_samples):
        indices = torch.floor(
            torch.rand((num_samples, 2)) * torch.tensor([image_height, image_width])
        ).long()
        y, x = (i.flatten() for i in torch.split(indices, 1, dim=-1))

        return y, x

    def prepare_training_input_output(
        self, batch: dict
    ) -> Tuple[ModelInput, ModelTarget]:
        coordinates = batch["scene"]["coordinates"]
        trgt_rgb = batch["target"]["rgb"]
        trgt_depth = batch["target"]["depth"]

        image_height, image_width = coordinates.shape[1:3]

        trgt_flow = None
        trgt_visible_mask = None
        if self.cfg.dataset.mode == "perception":
            y, x = self.random_sample_ray_yx_indices(
                image_height, image_width, self.cfg.training.data.rays_per_batch
            )
            trgt_rgb = trgt_rgb[:, :, y, x].transpose(1, 2)
            trgt_depth = trgt_depth[:, :, y, x].transpose(1, 2)
            coordinates = coordinates[:, y, x, :]

        else:
            if "pixel_motion" not in batch["target"]:
                y, x = self.random_sample_ray_yx_indices(
                    image_height, image_width, self.cfg.training.data.rays_per_batch
                )
                trgt_rgb = trgt_rgb[:, :, y, x].transpose(1, 2)
                trgt_depth = trgt_depth[:, :, y, x].transpose(1, 2)
                coordinates = coordinates[:, y, x, :]

                trgt_flow = batch["target"]["flow"]
                trgt_flow = trgt_flow[:, :, y, x].transpose(1, 2)

            else:
                ### subsample according to pixel selector and pixel motion
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
        # update batch for sanity logging
        batch["target"]["depth"] = trgt_depth

        ### pack into model input and target
        model_input = ModelInput(
            camera_input=CameraInput(
                input_image=batch["context"]["rgb"],
                ctxt_extrinsics=batch["context"]["extrinsics"],
                ctxt_intrinsics=batch["context"]["intrinsics"],
                trgt_extrinsics=batch["target"]["extrinsics"],
                trgt_intrinsics=convention.denormalize_intrinsics(
                    batch["target"]["intrinsics"],
                    width=image_width,
                    height=image_height,
                ),
            ),
            rendering_input=RenderingInput(
                origins=origins,
                directions=directions,
                z_near=batch["scene"]["near"],
                z_far=batch["scene"]["far"],
            ),
            robot_input=RobotInput(
                robot_action=batch["context"]["robot_action"],
            ),
        )

        model_target = ModelTarget(
            rgb=trgt_rgb,
            depth=trgt_depth,
            optical_flow=trgt_flow,
            visible_mask=trgt_visible_mask,
        )

        return model_input, model_target

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
