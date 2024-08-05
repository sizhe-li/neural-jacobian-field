import functools
from typing import Optional, Tuple, TypedDict

import numpy as np
import torch
import torch.nn as nn
import tqdm
from einops import einsum, rearrange
from jaxtyping import Float
from nerfstudio.utils.colormaps import apply_depth_colormap
from omegaconf import DictConfig
from torch import Tensor
from torchvision.utils import flow_to_image

from neural_jacobian_field.encoders import get_encoder
from neural_jacobian_field.models.pixelnerf_field import (  # PixelFlowField,
    PixelAlignedActionFeatures,
    PixelJacobianField,
    PixelJacobianFieldOld,
    PixelNeRFDensityField,
)
from neural_jacobian_field.rendering.geometry import project_world_coords_to_camera
from neural_jacobian_field.rendering.ray_samplers import (
    ProposalNetworkSampler,
    RayBundle,
    RaySamples,
    UniformSampler,
)


class ActionModelInput(TypedDict):
    # context rgb
    ctxt_rgb: Float[Tensor, "B C H W"]
    # context batch params
    ctxt_c2w: Float[Tensor, "B 4 4"]
    ctxt_intr: Float[Tensor, "B 3 3"]
    # target batch params
    trgt_c2w: Float[Tensor, "B 4 4"]
    trgt_intr: Float[Tensor, "B 3 3"]
    # robot action
    robot_action: Float[Tensor, "B C"]


class SceneRepresentation(TypedDict):
    density: Float[Tensor, "batch ray sample 1"]
    jacobian: Float[Tensor, "batch ray sample n_joints 3"]
    weights: Float[Tensor, "batch ray sample 1"]
    ray_positions: Float[Tensor, "batch ray sample 3"]
    trgt_c2w: Float[Tensor, "batch 4 4"]
    trgt_intr: Float[Tensor, "batch 3 3"]


class ActionModel(nn.Module):
    num_proposal_samples: Tuple[int] = (256,)
    num_nerf_samples: int = 256
    single_jitter: bool = False
    proposal_warmup: int = 5000
    proposal_update_every: int = 5
    use_proposal_weight_anneal: bool = True
    proposal_weights_anneal_max_num_iters: int = 1000
    proposal_weights_anneal_slope: float = 10.0

    def __init__(
        self,
        cfg: DictConfig,
    ):
        super().__init__()

        model_cfg = cfg.model

        self.model_cfg = model_cfg
        self.action_dim = self.model_cfg.action_dim

        # rendering-related
        self.num_proposal_samples = tuple(self.model_cfg.rendering.num_proposal_samples)
        self.num_nerf_samples = self.model_cfg.rendering.num_nerf_samples
        self.single_jitter = self.model_cfg.rendering.single_jitter
        self.proposal_warmup = self.model_cfg.rendering.proposal_warmup
        self.proposal_update_every = self.model_cfg.rendering.proposal_update_every
        self.use_proposal_weight_anneal = (
            self.model_cfg.rendering.use_proposal_weight_anneal
        )
        self.proposal_weights_anneal_max_num_iters = (
            self.model_cfg.rendering.proposal_weights_anneal_max_num_iters
        )
        self.proposal_weights_anneal_slope = (
            self.model_cfg.rendering.proposal_weights_anneal_slope
        )

        # initialize encoder
        self.encoder = get_encoder(cfg)

        PixelFieldCls = None
        # if model_cfg.action_model_type == "flow":
        #     PixelFieldCls = PixelFlowField
        # elif model_cfg.action_model_type == "jacobian":
        #     PixelFieldCls = PixelJacobianField

        if model_cfg.action_model_type == "jacobian":
            PixelFieldCls = PixelJacobianField
        elif model_cfg.action_model_type == "jacobian_old":
            PixelFieldCls = PixelJacobianFieldOld

        else:
            raise NotImplementedError

        self.field = PixelFieldCls(
            mlp_cfg=model_cfg.mlp,
            d_latent=self.encoder.d_out,
            n_joints=self.action_dim,
        )

        # proposal network definition
        self.density_fns = []
        self.num_proposal_iterations = len(self.num_proposal_samples)

        self.proposal_networks = nn.ModuleList()

        self.proposal_networks.extend(
            [
                PixelNeRFDensityField(
                    mlp_cfg=model_cfg.mlp,
                    d_latent=self.encoder.d_out,
                )
                for _ in range(self.num_proposal_iterations)
            ]
        )
        self.density_fns.extend(
            [network.get_density for network in self.proposal_networks]
        )

        update_schedule = lambda step: np.clip(
            np.interp(
                step,
                [0, self.proposal_warmup],
                [0, self.proposal_update_every],
            ),
            1,
            self.proposal_update_every,
        )

        initial_sampler = UniformSampler(single_jitter=self.single_jitter)
        self.proposal_sampler = ProposalNetworkSampler(
            num_nerf_samples_per_ray=self.num_nerf_samples,
            num_proposal_samples_per_ray=self.num_proposal_samples,
            num_proposal_network_iterations=self.num_proposal_iterations,
            single_jitter=self.single_jitter,
            update_sched=update_schedule,
            initial_sampler=initial_sampler,
        )

        flow_module_keywords = [
            "query_mlp",
            "joint_idx_emb",
            # for flow model
            "flow_attn_decoder",
            "flow_pred_head",
            # for jacobian model
            "jacobian_attn_decoder",
            "jacobian_pred_head",
            "jacobian_decoder",
        ]

        if not model_cfg.train_encoder:
            for name, param in self.field.named_parameters():
                # TODO: temporary hack, wrap everything next time
                if all([keyword not in name for keyword in flow_module_keywords]):
                    param.requires_grad = False

            # freeze all other networks
            for name, param in self.named_parameters():
                if "field" not in name:
                    param.requires_grad = False

        # if not model_cfg.train_flow:
        #     for name, param in self.field.named_parameters():
        #         if any([keyword in name for keyword in flow_module_keywords]):
        #             param.requires_grad = False

    def step_before_iter(self, step: int):
        if self.use_proposal_weight_anneal:
            # anneal the weights of the proposal network before doing PDF sampling
            N = self.proposal_weights_anneal_max_num_iters
            # https://arxiv.org/pdf/2111.12077.pdf eq. 18
            train_frac = np.clip(step / N, 0, 1)
            bias = lambda x, b: (b * x) / ((b - 1) * x + 1)
            anneal = bias(train_frac, self.proposal_weights_anneal_slope)
            self.proposal_sampler.set_anneal(step)

    def step_after_iter(self, step: int):
        if self.use_proposal_weight_anneal:
            self.proposal_sampler.step_cb(step)

    @staticmethod
    def render_rgb(
        rgb: torch.Tensor,
        weights: torch.Tensor,
        bg_color: Optional[torch.Tensor],
    ):
        comp_rgb = torch.sum(weights * rgb, dim=-2)
        accumulation = torch.sum(weights, dim=-2)
        if bg_color is None:
            pass
        else:
            comp_rgb = comp_rgb + (1.0 - accumulation) * bg_color
        return comp_rgb

    @staticmethod
    def render_depth(weights: torch.Tensor, ray_samples: RaySamples):
        steps = (ray_samples.starts + ray_samples.ends) / 2
        eps = 1e-10

        depth = torch.sum(weights * steps, dim=-2) / (torch.sum(weights, -2) + eps)
        depth = torch.clip(depth, steps.min(), steps.max())

        return depth, steps

    @staticmethod
    def render_optical_flow(
        weights: Float[Tensor, "batch ray sample 1"],
        ray_positions: Float[Tensor, "batch ray sample 3"],
        scene_flow: Float[Tensor, "batch ray sample 3"],
        trgt_c2w: Float[Tensor, "batch 4 4"],
        trgt_intr: Float[Tensor, "batch 3 3"],
        # )-> Float[Tensor, "batch ray 2"] # TODO: update return type annotations
    ):
        ray_positions_warped = ray_positions + scene_flow

        ray_positions = torch.sum(weights * ray_positions, dim=-2)
        ray_positions_warped = torch.sum(weights * ray_positions_warped, dim=-2)

        uv = project_world_coords_to_camera(ray_positions, trgt_c2w, trgt_intr)
        uv_warped = project_world_coords_to_camera(
            ray_positions_warped, trgt_c2w, trgt_intr
        )

        optical_flow = uv_warped - uv

        return optical_flow, ray_positions, ray_positions_warped

    def compute_proposal(
        self, ray_bundle: RayBundle, context: PixelAlignedActionFeatures
    ):
        density_fns = self.density_fns
        density_fns = [functools.partial(fn, context=context) for fn in density_fns]

        ray_samples: RaySamples
        (
            ray_samples,
            weights_list,
            ray_samples_list,
        ) = self.proposal_sampler.generate_ray_samples(
            ray_bundle, density_fns=density_fns
        )

        ray_samples_positions = ray_samples.get_positions()

        directions = ray_bundle.directions[..., None, :].expand(
            ray_samples_positions.shape
        )

        return (
            ray_samples,
            ray_samples_positions,
            directions,
            weights_list,
            ray_samples_list,
        )

    def pack_decoder_input(
        self,
        features,
        context: ActionModelInput,
    ) -> PixelAlignedActionFeatures:
        return {
            "features": features,
            "c2w": context["ctxt_c2w"],
            "k": context["ctxt_intr"],
            "action": context["robot_action"],
        }

    def forward_feature_extraction(
        self,
        rgb_images: Float[Tensor, "B C H W"],
        c2d_cam: Float[Tensor, "B N 2"],
        interp="bilinear",
    ) -> Float[Tensor, "B C H_ W_"]:
        img_features = self.encoder.encode_single_view(rgb_images, dict())[0]

        # now between 0 and 1. Map to -1 and 1
        c2d_norm = (c2d_cam - 0.5) * 2
        c2d_norm = rearrange(c2d_norm, "b n ch -> b n () ch")
        c2d_norm = c2d_norm[..., :2]

        # grid_sample
        feats = torch.nn.functional.grid_sample(
            img_features,
            c2d_norm,
            align_corners=True,
            padding_mode="border",
            mode=interp,
        )
        feats = feats.squeeze(-1)  # b ch n
        feats = rearrange(feats, "b ch n -> b n ch")

        return feats

    def forward(
        self,
        origins: Float[Tensor, "batch ray 3"],
        directions: Float[Tensor, "batch ray 3"],
        z_near: Float[Tensor, "batch"],
        z_far: Float[Tensor, "batch"],
        inputs: ActionModelInput,
        compute_action_features=False,
    ):
        trgt_c2w, trgt_intr = inputs["trgt_c2w"], inputs["trgt_intr"]
        ones = torch.ones_like(origins[..., 0:1])

        z_near = ones * z_near[:, None, None]
        z_far = ones * z_far[:, None, None]

        ray_bundle = RayBundle(
            origins=origins, directions=directions, nears=z_near, fars=z_far
        )

        features = self.encoder.encode_single_view(inputs["ctxt_rgb"], dict())[0]

        inputs: PixelAlignedActionFeatures
        inputs = self.pack_decoder_input(features, inputs)

        (
            ray_samples,
            ray_samples_positions,
            directions,
            weights_list,
            ray_samples_list,
        ) = self.compute_proposal(ray_bundle, inputs)

        density, rgb, scene_flow, action_features = self.field.forward(
            ray_samples_positions,
            directions,
            inputs,
            compute_action_features=compute_action_features,
        )

        weights = ray_samples.get_weights(density)
        weights_list.append(weights)
        ray_samples_list.append(ray_samples)

        rgb = self.render_rgb(rgb=rgb, weights=weights, bg_color=None)
        depth, steps = self.render_depth(
            weights=weights,
            ray_samples=ray_samples,
        )
        optical_flow, ray_positions, ray_position_warped = self.render_optical_flow(
            weights=weights,
            ray_positions=ray_samples_positions,
            scene_flow=scene_flow[..., :3],
            trgt_c2w=trgt_c2w,
            trgt_intr=trgt_intr,
        )

        outputs = {
            "rgb": rgb,
            "depth": depth,
            "optical_flow": optical_flow,
        }

        if self.training:
            outputs["weights_list"] = weights_list
            outputs["ray_samples_list"] = ray_samples_list
            outputs["scene_flow"] = scene_flow

        if compute_action_features:
            outputs["action_features"] = torch.sum(weights * action_features, dim=-2)
            outputs["steps"] = steps.squeeze(-1)
            outputs["weights"] = weights.squeeze(-1)
            # B x N x 3
            outputs["ray_positions"] = ray_positions
            outputs["ray_positions_warped"] = ray_position_warped

        return outputs

    def encode_image(
        self,
        origins: Float[Tensor, "batch ray 3"],
        directions: Float[Tensor, "batch ray 3"],
        z_near: Float[Tensor, "batch"],
        z_far: Float[Tensor, "batch"],
        inputs: ActionModelInput,
    ) -> SceneRepresentation:
        """Function used by inference algorithms for encoding an input image"""

        ones = torch.ones_like(origins[..., 0:1])

        z_near = ones * z_near[:, None, None]
        z_far = ones * z_far[:, None, None]

        ray_bundle = RayBundle(
            origins=origins, directions=directions, nears=z_near, fars=z_far
        )

        features = self.encoder.encode_single_view(inputs["ctxt_rgb"], dict())[0]

        pixelnerf_inputs: PixelAlignedActionFeatures
        pixelnerf_inputs = self.pack_decoder_input(features, inputs)

        (
            ray_samples,
            ray_samples_positions,
            directions,
            _,
            _,
        ) = self.compute_proposal(ray_bundle, pixelnerf_inputs)

        density, jacobian = self.field.encode_image(
            ray_samples_positions,
            pixelnerf_inputs,
        )

        weights = ray_samples.get_weights(density)

        return {
            "density": density,
            "jacobian": jacobian,
            "weights": weights,
            "ray_positions": ray_samples_positions,
            "trgt_c2w": inputs["trgt_c2w"],
            "trgt_intr": inputs["trgt_intr"],
        }

    def predict_optical_flow(
        self,
        jacobian_info: SceneRepresentation,
        action: Float[Tensor, "batch n_joints"],
    ) -> Float[Tensor, "batch ray 2"]:
        """Function used by inference algorithms for optimizing acitons"""

        batch, ray, sample = jacobian_info["density"].shape[:3]

        trgt_c2w = jacobian_info["trgt_c2w"]
        trgt_intr = jacobian_info["trgt_intr"]

        weights = jacobian_info["weights"]
        ray_samples_positions = jacobian_info["ray_positions"]
        jacobian = jacobian_info["jacobian"]

        scene_flow = einsum(
            jacobian,
            action,
            "batch ray sample n_joints d, batch n_joints -> batch ray sample d",
        )

        optical_flow, _, _ = self.render_optical_flow(
            weights=weights,
            ray_positions=ray_samples_positions,
            scene_flow=scene_flow[..., :3],
            trgt_c2w=trgt_c2w,
            trgt_intr=trgt_intr,
        )

        return optical_flow

    def patch_render(
        self,
        ctxt_image: Float[Tensor, "B C H W"],
        robot_action: Float[Tensor, "B action_dim"],
        origins: Float[Tensor, "B N 3"],
        directions: Float[Tensor, "B N 3"],
        ctxt_c2w: Float[Tensor, "B 4 4"],
        ctxt_intr: Float[Tensor, "B 3 3"],
        trgt_c2w: Float[Tensor, "B 4 4"],
        trgt_intr: Float[Tensor, "B 3 3"],
        z_near: Float[Tensor, "B"],
        z_far: Float[Tensor, "B"],
        patch_size: int = 2048,
        render_height: int = 480,
        render_width: int = 640,
        compute_action_features: bool = True,
        verbose: bool = False,
    ):
        """Function used by inference algorithms for making visualizations"""

        pixelnerf_packed_inputs: ActionModelInput = {
            "robot_action": robot_action,
            "ctxt_rgb": ctxt_image,
            "ctxt_c2w": ctxt_c2w,
            "trgt_c2w": trgt_c2w,
            "ctxt_intr": ctxt_intr,
            "trgt_intr": trgt_intr,
        }

        pred_rgb_full = []
        pred_depth_full = []
        pred_flow_full = []
        pred_feat_full = []
        pred_steps_full = []
        pred_weights_full = []
        pred_ray_positions_full = []
        pred_ray_positions_warped_full = []

        ran = tqdm.trange if verbose else range
        patchified_range = ran(0, origins.shape[1], patch_size)

        for start_idx in patchified_range:
            end_idx = start_idx + patch_size

            with torch.no_grad():
                pred_output_patch = self.forward(
                    origins[:, start_idx:end_idx, :],
                    directions[:, start_idx:end_idx, :],
                    z_near,
                    z_far,
                    pixelnerf_packed_inputs,
                    compute_action_features=compute_action_features,
                )

            pred_rgb_patch = pred_output_patch["rgb"]
            pred_depth_patch = pred_output_patch["depth"]
            pred_flow_patch = pred_output_patch["optical_flow"]
            if compute_action_features:
                pred_feat_full.append(pred_output_patch["action_features"])
                pred_steps_full.append(pred_output_patch["steps"])
                pred_weights_full.append(pred_output_patch["weights"])
                pred_ray_positions_full.append(pred_output_patch["ray_positions"])
                pred_ray_positions_warped_full.append(
                    pred_output_patch["ray_positions_warped"]
                )

            pred_rgb_full.append(pred_rgb_patch)
            pred_depth_full.append(pred_depth_patch)
            pred_flow_full.append(pred_flow_patch)

        pred_rgb_full = torch.cat(pred_rgb_full, dim=1)
        pred_depth_full = torch.cat(pred_depth_full, dim=1)
        pred_flow_full = torch.cat(pred_flow_full, dim=1)

        if compute_action_features:
            pred_feat_full = torch.cat(pred_feat_full, dim=1)
            pred_steps_full = torch.cat(pred_steps_full, dim=1)
            pred_weights_full = torch.cat(pred_weights_full, dim=1)
            pred_ray_positions_full = torch.cat(pred_ray_positions_full, dim=1)
            pred_ray_positions_warped_full = torch.cat(
                pred_ray_positions_warped_full, dim=1
            )

            pred_feat_full = rearrange(
                pred_feat_full,
                "b (h w) c -> b c h w",
                h=render_height,
                w=render_width,
            )

            pred_weights_full = rearrange(
                pred_weights_full,
                "b (h w) c -> b c h w",
                h=render_height,
                w=render_width,
            )

            pred_steps_full = rearrange(
                pred_steps_full,
                "b (h w) c -> b c h w",
                h=render_height,
                w=render_width,
            )

            pred_ray_positions_full = rearrange(
                pred_ray_positions_full,
                "b (h w) c -> b c h w",
                h=render_height,
                w=render_width,
            )

            pred_ray_positions_warped_full = rearrange(
                pred_ray_positions_warped_full,
                "b (h w) c -> b c h w",
                h=render_height,
                w=render_width,
            )

        pred_rgb_full = rearrange(
            pred_rgb_full,
            "b (h w) c -> b h w c",
            h=render_height,
            w=render_width,
        )

        pred_depth_original_full = rearrange(
            pred_depth_full.clone(),
            "b (h w) c -> b h w c",
            h=render_height,
            w=render_width,
        )

        pred_depth_rgb_full = apply_depth_colormap(
            rearrange(
                pred_depth_full,
                "b (h w) c -> b h w c",
                h=render_height,
                w=render_width,
            )
        )

        pred_flow_full = rearrange(
            pred_flow_full,
            "b (h w) c -> b c h w",
            h=render_height,
            w=render_width,
        )

        pred_flow_full_rgb = flow_to_image(pred_flow_full)
        pred_flow_full_rgb = rearrange(
            pred_flow_full_rgb,
            "b c h w -> b h w c",
        )

        return {
            "pred_rgb": pred_rgb_full,
            "pred_depth": pred_depth_original_full,
            "pred_depth_rgb": pred_depth_rgb_full,
            "pred_flow": pred_flow_full,
            "pred_flow_rgb": pred_flow_full_rgb,
            "pred_ray_positions": pred_ray_positions_full,
            "pred_ray_positions_warped": pred_ray_positions_warped_full,
            "pred_feat": pred_feat_full,
            "pred_steps": pred_steps_full,
            "pred_weights": pred_weights_full,
        }
