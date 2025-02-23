import glob
from typing import Tuple, Literal

import torch
from jaxtyping import Float
from torch import Tensor


def post_process_camera_to_world(c2w):
    conversion = torch.eye(4, dtype=torch.float32)
    conversion[1:3, 1:3] *= -1
    c2w = c2w @ conversion
    return c2w


def normalize(curr_vals, old_min, old_max, new_min=0.0, new_max=1.0):
    values = (curr_vals - old_min) / (old_max - old_min)
    values = (new_max - new_min) * values + new_min
    return values


def normalize_optical_flow(
    flow: Float[Tensor, "*B 2 H W"],
):
    # normalize optical flow by image size
    H, W = flow.shape[-2:]

    flow[..., 0, :, :] /= W
    flow[..., 1, :, :] /= H

    return flow


def resize_optical_flow(
    flow: Float[Tensor, "B 2 H1 W1"],
    trgt_size: Tuple[int, int],
    inter_mode: Literal["bilinear", "nearest"] = "bilinear",
) -> Float[Tensor, "B 2 H2 W2"]:
    # normalize optical flow by image size
    curr_height, curr_width = flow.shape[-2:]
    trgt_height, trgt_width = trgt_size

    flow[:, 0, :, :] *= trgt_width / curr_width
    flow[:, 1, :, :] *= trgt_height / curr_height

    flow = torch.nn.functional.interpolate(
        flow,
        (trgt_height, trgt_width),
        mode=inter_mode,
    )

    return flow


def get_traj_and_frame_idx(image_filename: str):
    keynames = image_filename.split("/")
    traj_idx, frame_idx = keynames[-1].split("_")

    traj_idx = int(traj_idx)
    frame_idx = int(frame_idx.split(".")[0])

    return traj_idx, frame_idx


def get_optical_flow_filename(
    image_filename: str,
    traj_idx: int,
    frame_idx: int,
    flow_type: Literal["fwd", "bwd"],
):
    optical_flow_filename = image_filename.replace("rgb", "flow").replace(
        f"{traj_idx:05d}_{frame_idx:05d}.png",
        f"{traj_idx:05d}_{frame_idx:05d}_{flow_type}.npz",
    )
    return optical_flow_filename


def get_optical_flow_filename_new(
    image_filename: str,
    flow_type: Literal["fwd", "bwd"],
):
    optical_flow_filename = image_filename.replace("rgb", "flow").replace(
        ".png",
        f"_{flow_type}.npz",
    )
    return optical_flow_filename


def get_tracking_data_filenames(
    image_filename: str,
    traj_idx: int,
    frame_idx: int,
):
    glob_pattern = image_filename.replace("rgb", "tapir").replace(
        f"{traj_idx:05d}_{frame_idx:05d}.png", f"{traj_idx:05d}_*.npz"
    )

    return glob.glob(glob_pattern)


def get_trgt_view_filename(
    ctxt_img_filename: str, ctxt_cam_idx: int, trgt_cam_idx: int
):
    trgt_img_filename = str(ctxt_img_filename).replace(
        f"view_{ctxt_cam_idx}", f"view_{trgt_cam_idx}"
    )
    return trgt_img_filename


def denormalize_intrinsics(
    intrinsics: Float[Tensor, "B 3 3"],
    width: int,
    height: int,
) -> Float[Tensor, "*B 3 3"]:
    """Denormalize intrinsics to the original image size.

    Args:
        intrinsics: The intrinsics to denormalize.
        image_size: The image size to denormalize to.
    """
    intrinsics = intrinsics.clone()
    intrinsics[..., 0, :] *= width
    intrinsics[..., 1, :] *= height

    return intrinsics


def sample_around_pixels(pixel_indices_yx, image_height, image_width, radius=5):
    """Samples pixels around the given pixel indices.

    Args:
        pixel_indices_yx: The pixel indices to sample around.
        image_height: The image height.
        image_width: The image width.
        radius: The radius around the pixel indices to sample.
    Returns:
        The sampled pixel indices (num_pixels, num_samples, 2).
    """

    offsets_y, offsets_x = torch.meshgrid(
        torch.arange(-radius, radius + 1), torch.arange(-radius, radius + 1)
    )
    offsets_yx = torch.stack([offsets_y, offsets_x], dim=-1).reshape(-1, 2)

    pixel_indices_yx = pixel_indices_yx[:, None, :] + offsets_yx[None, :, :]
    pixel_indices_yx[..., 0] = pixel_indices_yx[..., 0].clip(0, image_height - 1)
    pixel_indices_yx[..., 1] = pixel_indices_yx[..., 1].clip(0, image_width - 1)

    return pixel_indices_yx


def sample_pixel_locations_from_mask(mask, num_points: int = 500):
    """
    Args:
        mask: The mask to sample from.
        num_points: The number of points to sample.
    Returns:
        The sampled pixel locations (num_points, 2).
    """

    yx_indices = torch.nonzero(mask)
    yx_indices = yx_indices[torch.randperm(yx_indices.shape[0])[:num_points]]

    return yx_indices
