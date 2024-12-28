import functools
from dataclasses import dataclass
from typing import Optional, List, Tuple, Dict

import numpy as np
import torch
import torch.nn as nn
import tqdm
from einops import rearrange, einsum
from jaxtyping import Float
from nerfstudio.utils.colormaps import apply_depth_colormap
from torch import Tensor
from torchvision.utils import flow_to_image

from .decoder import (
    DensityDecoderCfg,
    ActionDecoderCfg,
    get_density_decoder,
    get_action_decoder,
)
from .decoder.action_decoder import PixelEncoding
from .encoder import EncoderCfg, get_encoder
from ..rendering.geometry import project_world_coords_to_camera
from ..rendering.ray_samplers import (
    ProposalNetworkSampler,
    RayBundle,
    RaySamples,
    UniformSampler,
)


@dataclass
class RenderingCfg:
    num_proposal_samples: Tuple[int]
    num_nerf_samples: int
    single_jitter: bool
    proposal_warmup: int
    proposal_update_every: int
    use_proposal_weight_anneal: bool
    proposal_weights_anneal_max_num_iters: int
    proposal_weights_anneal_slope: float


@dataclass
class ModelCfg:
    action_dim: int
    rendering: RenderingCfg
    encoder: EncoderCfg
    density_decoder: DensityDecoderCfg
    action_decoder: ActionDecoderCfg


@dataclass
class CameraInput:
    input_image: Float[Tensor, "batch C H W"]
    # context batch params
    ctxt_extrinsics: Float[Tensor, "batch 4 4"]
    ctxt_intrinsics: Float[Tensor, "batch 3 3"]
    # target batch params
    trgt_extrinsics: Float[Tensor, "batch 4 4"]
    trgt_intrinsics: Float[Tensor, "batch 3 3"]


@dataclass
class RenderingInput:
    origins: Float[Tensor, "batch ray 3"]
    directions: Float[Tensor, "batch ray 3"]
    z_near: Float[Tensor, "batch"]
    z_far: Float[Tensor, "batch"]


@dataclass
class RobotInput:
    robot_action: Float[Tensor, "batch dim"]


@dataclass
class ModelInput:
    camera_input: CameraInput
    rendering_input: RenderingInput
    robot_input: RobotInput


@dataclass
class ModelTarget:
    rgb: Float[Tensor, "batch num_ray dim"]
    depth: Float[Tensor, "batch num_ray dim"]
    optical_flow: Float[Tensor, "batch num_ray dim"] | None
    visible_mask: Float[Tensor, "batch num_ray"] | None


@dataclass
class ModelStandardOutput:
    rgb: Float[Tensor, "batch H W C"]
    depth: Float[Tensor, "batch H W"]
    optical_flow: Float[Tensor, "batch H W 2"]


@dataclass
class ModelTrainingOutput:
    weights_list: List[Float[Tensor, "batch ray sample 1"]]
    ray_samples_list: List[RaySamples]
    # scene_flow: Float[Tensor, "batch ray sample 3"]


@dataclass
class ModelVisOutput:
    action_features: Float[Tensor, "batch ray sample dim"]
    ray_positions: Float[Tensor, "batch ray sample 3"]
    ray_positions_warped: Float[Tensor, "batch ray sample 3"]
    weights: Float[Tensor, "batch ray sample 1"]
    steps: Float[Tensor, "batch ray sample 1"]


@dataclass
class ModelOutput:
    standard_output: ModelStandardOutput
    training_output: ModelTrainingOutput | None
    vis_output: ModelVisOutput | None


@dataclass
class ModelInferenceEncoding:
    density: Float[Tensor, "batch ray sample 1"]
    action_features: Float[Tensor, "batch ray sample dim"]
    weights: Float[Tensor, "batch ray sample 1"]
    ray_samples_positions: Float[Tensor, "batch ray sample 3"]


@dataclass
class RenderingOutput:
    rgb: Float[Tensor, "batch height width channel"]
    depth_raw: Float[Tensor, "batch height width channel"]
    depth_rgb: Float[Tensor, "batch height width channel"]
    flow_raw: Float[Tensor, "batch height width channel"]
    flow_rgb: Float[Tensor, "batch height width channel"]
    ray_positions: Float[Tensor, "batch height width channel"]
    ray_positions_warped: Float[Tensor, "batch height width channel"]
    action_features: Float[Tensor, "batch height width channel"]
    steps: Float[Tensor, "batch height width channel"]
    weights: Float[Tensor, "batch height width channel"]


class Model(nn.Module):
    def __init__(
        self,
        cfg: ModelCfg,
    ):
        super().__init__()

        self.cfg = cfg

        # initialize encoder
        self.encoder = get_encoder(cfg.encoder)
        self.decoder = get_action_decoder(
            cfg.action_decoder,
            action_dim=cfg.action_dim,
            encoder_dim=self.encoder.get_output_dim(),
        )

        # proposal network definition
        num_proposal_iterations = len(self.cfg.rendering.num_proposal_samples)
        # has to make it module list to track the parameters
        self.proposal_networks = nn.ModuleList()
        self.proposal_networks.extend(
            [
                get_density_decoder(
                    cfg.density_decoder, encoder_dim=self.encoder.get_output_dim()
                )
                for _ in range(num_proposal_iterations)
            ]
        )
        self.density_fns = []
        self.density_fns.extend(
            [network.get_density for network in self.proposal_networks]
        )

        update_schedule = lambda step: np.clip(
            np.interp(
                step,
                [0, self.cfg.rendering.proposal_warmup],
                [0, self.cfg.rendering.proposal_update_every],
            ),
            1,
            self.cfg.rendering.proposal_update_every,
        )

        initial_sampler = UniformSampler(single_jitter=self.cfg.rendering.single_jitter)
        self.proposal_sampler = ProposalNetworkSampler(
            num_nerf_samples_per_ray=self.cfg.rendering.num_nerf_samples,
            num_proposal_samples_per_ray=self.cfg.rendering.num_proposal_samples,
            num_proposal_network_iterations=num_proposal_iterations,
            single_jitter=self.cfg.rendering.single_jitter,
            update_sched=update_schedule,
            initial_sampler=initial_sampler,
        )

    def step_before_iter(self, step):
        if self.cfg.rendering.use_proposal_weight_anneal:
            # anneal the weights of the proposal network before doing PDF sampling
            N = self.cfg.rendering.proposal_weights_anneal_max_num_iters
            # https://arxiv.org/pdf/2111.12077.pdf eq. 18
            train_frac = np.clip(step / N, 0, 1)
            bias = lambda x, b: (b * x) / ((b - 1) * x + 1)
            anneal = bias(train_frac, self.cfg.rendering.proposal_weights_anneal_slope)
            self.proposal_sampler.set_anneal(anneal)

    def step_after_iter(self, step):
        if self.cfg.rendering.use_proposal_weight_anneal:
            self.proposal_sampler.step_cb(step)

    def compute_ray_bundle(self, rendering_input: RenderingInput) -> RayBundle:
        ones = torch.ones_like(rendering_input.origins[..., 0:1])
        z_near = ones * rendering_input.z_near[:, None, None]
        z_far = ones * rendering_input.z_far[:, None, None]

        ray_bundle = RayBundle(
            origins=rendering_input.origins,
            directions=rendering_input.directions,
            nears=z_near,
            fars=z_far,
        )
        return ray_bundle

    def compute_proposal(self, ray_bundle: RayBundle, pixel_encoding: PixelEncoding):
        density_fns = self.density_fns
        density_fns = [
            functools.partial(fn, pixel_encoding=pixel_encoding) for fn in density_fns
        ]

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
    def render_action_features(
        action_features: Float[Tensor, "batch ray sample dim"],
        weights: Float[Tensor, "batch ray sample 1"],
    ):
        return torch.sum(weights * action_features, dim=-2)

    @staticmethod
    def render_optical_flow(
        weights: Float[Tensor, "batch ray sample 1"],
        ray_positions: Float[Tensor, "batch ray sample 3"],
        scene_flow: Float[Tensor, "batch ray sample 3"],
        trgt_extrinsics: Float[Tensor, "batch 4 4"],
        trgt_intrinsics: Float[Tensor, "batch 3 3"],
    ) -> Tuple[
        Float[Tensor, "batch ray 2"],
        Float[Tensor, "batch ray 3"],
        Float[Tensor, "batch ray 3"],
    ]:
        ray_positions_warped = ray_positions + scene_flow

        ray_positions = torch.sum(weights * ray_positions, dim=-2)
        ray_positions_warped = torch.sum(weights * ray_positions_warped, dim=-2)

        uv = project_world_coords_to_camera(
            ray_positions, trgt_extrinsics, trgt_intrinsics
        )
        uv_warped = project_world_coords_to_camera(
            ray_positions_warped, trgt_extrinsics, trgt_intrinsics
        )

        optical_flow = uv_warped - uv

        return optical_flow, ray_positions, ray_positions_warped

    def forward(
        self,
        camera_input: CameraInput,
        rendering_input: RenderingInput,
        robot_input: RobotInput,
        compute_vis_features: bool = False,
    ):
        ray_bundle = self.compute_ray_bundle(rendering_input)

        features = self.encoder.forward(camera_input.input_image)
        pixel_encoding = PixelEncoding(
            features=features,
            extrinsics=camera_input.ctxt_extrinsics,
            intrinsics=camera_input.ctxt_intrinsics,
            action=robot_input.robot_action,
        )

        # import pdb
        #
        # pdb.set_trace()

        (
            ray_samples,
            ray_samples_positions,
            directions,
            weights_list,
            ray_samples_list,
        ) = self.compute_proposal(ray_bundle, pixel_encoding)

        decoder_output = self.decoder.forward(
            world_space_xyz=ray_samples_positions,
            world_space_dir=directions,
            pixel_encoding=pixel_encoding,
        )

        weights = ray_samples.get_weights(decoder_output.density)
        weights_list.append(weights)
        ray_samples_list.append(ray_samples)

        rgb = self.render_rgb(rgb=decoder_output.color, weights=weights, bg_color=None)
        depth, steps = self.render_depth(
            weights=weights,
            ray_samples=ray_samples,
        )

        optical_flow, ray_positions, ray_position_warped = self.render_optical_flow(
            weights=weights,
            ray_positions=ray_samples_positions,
            scene_flow=decoder_output.flow[..., :3],
            trgt_extrinsics=camera_input.trgt_extrinsics,
            trgt_intrinsics=camera_input.trgt_intrinsics,
        )

        model_output = ModelOutput(
            standard_output=ModelStandardOutput(
                rgb=rgb, depth=depth, optical_flow=optical_flow
            ),
            training_output=None,
            vis_output=None,
        )

        if self.training:
            model_output.training_output = ModelTrainingOutput(
                weights_list=weights_list,
                ray_samples_list=ray_samples_list,
                # scene_flow=decoder_output.flow,
            )

        if compute_vis_features:
            action_features = self.render_action_features(
                decoder_output.action_features, weights
            )
            model_output.vis_output = ModelVisOutput(
                action_features=action_features,
                steps=steps.squeeze(-1),
                weights=weights.squeeze(-1),
                ray_positions=ray_positions,
                ray_positions_warped=ray_position_warped,
            )

        return model_output

    def encode_image(
        self,
        camera_input: CameraInput,
        rendering_input: RenderingInput,
        robot_input: RobotInput,
    ) -> ModelInferenceEncoding:
        """Function used by inference algorithms for encoding an input image"""

        ray_bundle = self.compute_ray_bundle(rendering_input)

        features = self.encoder.forward(camera_input.input_image)
        pixel_encoding = PixelEncoding(
            features=features,
            extrinsics=camera_input.ctxt_extrinsics,
            intrinsics=camera_input.ctxt_intrinsics,
            action=robot_input.robot_action,
        )

        (
            ray_samples,
            ray_samples_positions,
            directions,
            _,
            _,
        ) = self.compute_proposal(ray_bundle, pixel_encoding)

        decoder_feature_only_output = self.decoder.encode_image(
            ray_samples_positions, pixel_encoding
        )

        weights = ray_samples.get_weights(decoder_feature_only_output.density)

        return ModelInferenceEncoding(
            density=decoder_feature_only_output.density,
            action_features=decoder_feature_only_output.action_features,
            weights=weights,
            ray_samples_positions=ray_samples_positions,
        )

    def infer_optical_flow(
        self,
        model_inference_encoding: ModelInferenceEncoding,
        camera_input: CameraInput,
        robot_input: RobotInput,
    ) -> Float[Tensor, "batch ray 2"]:
        """Function used by inference algorithms for optimizing actions"""

        assert self.cfg.action_decoder.name == "jacobian_mlp"

        scene_flow = einsum(
            rearrange(
                model_inference_encoding.action_features,
                "batch ray sample (action_dim spatial_dim) -> batch ray sample action_dim spatial_dim",
                action_dim=robot_input.robot_action.shape[-1],
            ),
            robot_input.robot_action,
            "batch ray sample action_dim spatial_dim, batch action_dim -> batch ray sample spatial_dim",
        )

        optical_flow, _, _ = self.render_optical_flow(
            weights=model_inference_encoding.weights,
            ray_positions=model_inference_encoding.ray_samples_positions,
            scene_flow=scene_flow[..., :3],
            trgt_extrinsics=camera_input.trgt_extrinsics,
            trgt_intrinsics=camera_input.trgt_intrinsics,
        )

        return optical_flow

    def patch_render(
        self,
        camera_input: CameraInput,
        rendering_input: RenderingInput,
        robot_input: RobotInput,
        patch_size: int = 2048,
        render_height: int = 480,
        render_width: int = 640,
        verbose: bool = False,
    ) -> RenderingOutput:
        """Function used by inference algorithms for making visualizations"""

        rendering_output: Dict[
            str, Float[Tensor, "batch channel height width"] | List[torch.Tensor]
        ] = {
            k: []
            for k in [
                "rgb",
                "depth_raw",
                "depth_rgb",
                "flow_raw",
                "flow_rgb",
                "ray_positions",
                "ray_positions_warped",
                "action_features",
                "steps",
                "weights",
            ]
        }

        ran = tqdm.trange if verbose else range
        num_rays = rendering_input.origins.shape[1]

        for ray_start_idx in ran(0, num_rays, patch_size):
            ray_end_idx = ray_start_idx + patch_size

            patch_rendering_input = RenderingInput(
                origins=rendering_input.origins[:, ray_start_idx:ray_end_idx, :],
                directions=rendering_input.directions[:, ray_start_idx:ray_end_idx, :],
                z_near=rendering_input.z_near,
                z_far=rendering_input.z_far,
            )

            with torch.no_grad():
                patch_model_output = self.forward(
                    camera_input=camera_input,
                    rendering_input=patch_rendering_input,
                    robot_input=robot_input,
                    compute_vis_features=True,
                )

            rendering_output["rgb"].append(patch_model_output.standard_output.rgb)
            rendering_output["depth_raw"].append(
                patch_model_output.standard_output.depth
            )
            rendering_output["flow_raw"].append(
                patch_model_output.standard_output.optical_flow
            )
            rendering_output["action_features"].append(
                patch_model_output.vis_output.action_features
            )
            rendering_output["steps"].append(patch_model_output.vis_output.steps)
            rendering_output["weights"].append(patch_model_output.vis_output.weights)
            rendering_output["ray_positions"].append(
                patch_model_output.vis_output.ray_positions
            )
            rendering_output["ray_positions_warped"].append(
                patch_model_output.vis_output.ray_positions_warped
            )

        for key in rendering_output:
            if len(rendering_output[key]) > 0:
                rendering_output[key] = rearrange(
                    torch.cat(rendering_output[key], dim=1),
                    "b (h w) c -> b h w c",
                    h=render_height,
                    w=render_width,
                )

        rendering_output["depth_rgb"] = apply_depth_colormap(
            rearrange(
                rendering_output["depth_raw"],
                "b h w c -> b h w c",
                h=render_height,
                w=render_width,
            )
        )

        rendering_output["flow_rgb"] = rearrange(
            flow_to_image(
                rearrange(
                    rendering_output["flow_raw"],
                    "b h w c -> b c h w",
                    h=render_height,
                    w=render_width,
                )
            ),
            "b c h w -> b h w c",
        )

        return RenderingOutput(**rendering_output)
