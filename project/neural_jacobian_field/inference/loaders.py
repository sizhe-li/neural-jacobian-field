import torch
import nerfstudio.utils.poses as pose_utils

from hydra import compose, initialize

from pathlib import Path
from typing import Tuple, TypedDict, Dict
from omegaconf import DictConfig
from neural_jacobian_field.rendering.geometry import (
    get_pixel_coordinates,
    homogenize_points,
    transform_world2cam,
)

from neural_jacobian_field.models import Model, ModelWrapper
from neural_jacobian_field.data.dataset.config_parser import (
    DNeRFDataParser,
    DNeRFDataParserConfig,
    DNeRFDataParserOutputs,
)
from nerfstudio.cameras.cameras import Cameras
from copy import deepcopy


class CameraContext(TypedDict):
    # extrinsics
    ctxt_c2w: torch.Tensor
    trgt_c2w: torch.Tensor
    # intrinsics
    ctxt_intr: torch.Tensor
    trgt_intr: torch.Tensor
    ctxt_intr_raw: torch.Tensor
    trgt_intr_raw: torch.Tensor
    # camera coordinates
    coordinates: torch.Tensor
    selector: torch.Tensor
    # misc
    inverse_ctxt_c2w: torch.Tensor
    # all views
    all_trgt_c2w: torch.Tensor
    all_trgt_intr: torch.Tensor


def c2w_convention_transform(c2w):
    conversion = torch.eye(4, dtype=torch.float32)
    conversion[1:3, 1:3] *= -1
    c2w = c2w @ conversion
    return c2w


def parse_nerfstudio_camera(
    cameras: Cameras,
    ctxt_camera_idx: int = 0,
    trgt_camera_idx: int = 1,
    device: torch.device = torch.device("cuda:0"),
) -> CameraContext:
    multiview_cam2worlds = pose_utils.to4x4(cameras.camera_to_worlds)
    multiview_intrinsics = cameras.get_intrinsics_matrices()

    ctxt_width = cameras.width[ctxt_camera_idx].clone()
    ctxt_height = cameras.height[ctxt_camera_idx].clone()
    trgt_width = cameras.width[trgt_camera_idx].clone().item()
    trgt_height = cameras.height[trgt_camera_idx].clone().item()

    ctxt_intr = multiview_intrinsics[ctxt_camera_idx].clone().to(device)  # [3, 3]
    ctxt_intr_raw = ctxt_intr.clone().to(device)
    trgt_intr = multiview_intrinsics[trgt_camera_idx].clone().to(device)  # [3, 3]
    trgt_intr_raw = trgt_intr.clone().to(device)  # [3, 3]

    # normalize intrinsics
    ctxt_intr[:2] /= torch.tensor([ctxt_width, ctxt_height])[:, None].float().to(device)
    trgt_intr[:2] /= torch.tensor([trgt_width, trgt_height])[:, None].float().to(device)

    # get pixel coordinates
    coordinates, selector = get_pixel_coordinates(
        int(trgt_height), int(trgt_width), device
    )

    ctxt_c2w = multiview_cam2worlds[ctxt_camera_idx].clone()
    trgt_c2w = multiview_cam2worlds[trgt_camera_idx].clone()
    ctxt_c2w = c2w_convention_transform(ctxt_c2w)
    trgt_c2w = c2w_convention_transform(trgt_c2w)

    inverse_ctxt_c2w = torch.inverse(ctxt_c2w)

    ctxt_c2w = torch.einsum("ij, jk -> ik", inverse_ctxt_c2w, ctxt_c2w).to(device)
    trgt_c2w = torch.einsum("ij, jk -> ik", inverse_ctxt_c2w, trgt_c2w).to(device)

    all_trgt_c2w = multiview_cam2worlds.clone()
    all_trgt_c2w = c2w_convention_transform(all_trgt_c2w)
    all_trgt_c2w = torch.einsum("ij, bjk -> bik", inverse_ctxt_c2w, all_trgt_c2w).to(
        device
    )

    all_trgt_intr = multiview_intrinsics.clone().to(device)
    all_trgt_intr[..., :2] /= torch.tensor([trgt_width, trgt_height], device=device)[
        None, :
    ].float()

    return {
        "selector": selector,
        "coordinates": coordinates,
        "ctxt_c2w": ctxt_c2w,
        "trgt_c2w": trgt_c2w,
        "ctxt_intr": ctxt_intr,
        "trgt_intr": trgt_intr,
        "ctxt_intr_raw": ctxt_intr_raw,
        "trgt_intr_raw": trgt_intr_raw,
        "inverse_ctxt_c2w": inverse_ctxt_c2w,
        "all_trgt_c2w": all_trgt_c2w,
        "all_trgt_intr": all_trgt_intr,
    }


def load_model_cfg(
    config_path: Path,
    overrides: list = [],
):
    with initialize(
        version_base=None,
        config_path=str("../../" / config_path.parent),
    ):
        model_cfg = compose(config_name=str(config_path.name), overrides=overrides)

    return model_cfg


def load_model(
    model_cfg: DictConfig,
    model_ckpt: Path,
    device: torch.device = torch.device("cuda:0"),
) -> Model:
    model_wrapper = ModelWrapper.load_from_checkpoint(
        str(model_ckpt),
        cfg=model_cfg,
        model=Model(model_cfg),
        strict=False,
    )

    model = model_wrapper.model
    model.to(device)
    model.eval()

    return model


def load_nerfstudio_data(
    data_path: Path,
    scale_factor: float = 1.0,
    downscale_factor: int = 1,
) -> Tuple[Cameras, Dict]:
    dataparser: DNeRFDataParser
    dataparser = DNeRFDataParserConfig(
        data=Path(data_path),
        center_method="focus",
        scale_factor=scale_factor,
        downscale_factor=downscale_factor,
        depth_unit_scale_factor=1e-3,
    ).setup()

    dataparser_outputs: DNeRFDataParserOutputs
    dataparser_outputs = dataparser.get_dataparser_outputs(split="train")

    cameras = deepcopy(dataparser_outputs.cameras)
    metadata = deepcopy(dataparser_outputs.metadata)
    metadata["render_height"] = cameras.image_height[0].item()
    metadata["render_width"] = cameras.image_width[0].item()

    return cameras, metadata


# def prepare_model(
#     model_cfg_path: Path,
#     model_ckpt_path: Path,
#     action_dim: int = 10,
# ) -> Tuple[ActionModel, model_utils.ModelInfo]:
#     # TODO: make the following adaptable according to cmd line args
#     model_cfg = model_utils.load_model_cfg(
#         model_cfg_path,
#         overrides=[
#             "model=allegro",
#             "dataset=allegro",
#             "model.action_model_type=jacobian",
#             "model.rendering.num_proposal_samples=[256]",
#             "model.rendering.num_nerf_samples=256",
#             f"model.action_dim={action_dim}",
#             "+model.train_encoder=False",
#         ],
#     )

#     model, model_info = model_utils.load_model(
#         model_cfg=model_cfg,
#         model_ckpt=model_ckpt_path,
#     )
#     model_info["model_cfg"] = model_cfg

#     return model, model_info


# def prepare_camera(
#     downscale_factor: int = 1,
#     ctxt_camera_idx: int = 0,
#     trgt_camera_idx: int = 1,
#     recording_fps: int = 10,
#     recording_pix_fmt: str = "bgr24",
#     video_crf: int = 21,
#     thread_per_video: int = 2,
#     capture_folder: Path = Path(""),
#     device: str = "cuda:0",
# ):
#     print(f"Loading cameras to device {device}...")

#     cameras, metadata = model_utils.load_cameras(
#         data_path=capture_folder,
#         downscale_factor=downscale_factor,  # because of memory limit we cannot support full resolution
#     )

#     camera_context = model_utils.parse_camera_info(
#         cameras=cameras,
#         ctxt_camera_idx=ctxt_camera_idx,
#         trgt_camera_idx=trgt_camera_idx,
#         device=device,
#     )

#     ctxt_view_str = CAMERA_NAMES[ctxt_camera_idx]

#     return {
#         "cameras": cameras,
#         "metadata": metadata,
#         "camera_context": camera_context,
#     }
