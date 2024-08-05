import torch
from einops import rearrange, repeat
from jaxtyping import Float
from scipy.spatial.transform import Rotation as R
from torch import Tensor
from typing import Optional


@torch.no_grad()
def interpolate_pose(
    initial: Float[Tensor, "4 4"],
    final: Float[Tensor, "4 4"],
    t: float,
) -> Float[Tensor, "4 4"]:
    # Get the relative rotation.
    r_initial = initial[:3, :3]
    r_final = final[:3, :3]
    r_relative = r_final @ r_initial.T

    # Convert it to axis-angle to interpolate it.
    r_relative = R.from_matrix(r_relative.cpu().numpy()).as_rotvec()
    r_relative = R.from_rotvec(r_relative * t).as_matrix()
    r_relative = torch.tensor(r_relative, dtype=final.dtype, device=final.device)
    r_interpolated = r_relative @ r_initial

    # Interpolate the position.
    t_initial = initial[:3, 3]
    t_final = final[:3, 3]
    t_interpolated = t_initial + (t_final - t_initial) * t

    # Assemble the result.
    result = torch.zeros_like(initial)
    result[3, 3] = 1
    result[:3, :3] = r_interpolated
    result[:3, 3] = t_interpolated
    return result


@torch.no_grad()
def interpolate_intrinsics(
    initial: Float[Tensor, "3 3"],
    final: Float[Tensor, "3 3"],
    t: float,
) -> Float[Tensor, "3 3"]:
    return initial + (final - initial) * t


@torch.no_grad()
def generate_wobble(
    tf: Float[Tensor, "batch 4 4"],
    t: Float[Tensor, " view"],
) -> Float[Tensor, "batch view 4 4"]:
    tf_translation = torch.eye(4, dtype=torch.float32, device=t.device)
    tf_translation = repeat(tf_translation, "i j -> () v i j", v=len(t)).clone()
    tf_translation[0, :, 0, 3] = torch.sin(2 * torch.pi * t) * (0.5 * t)
    tf_translation[0, :, 1, 3] = -torch.cos(2 * torch.pi * t) * (0.5 * t)
    return rearrange(tf, "b i j -> b () i j") @ tf_translation


def reproj_best_torch(
    src_steps: Float[Tensor, "H W N"],
    src_weights: Float[Tensor, "H W N"],
    src_c2w: Float[Tensor, "4 4"],
    tgt_c2w: Float[Tensor, "4 4"],
    src_intrinsics: Float[Tensor, "3 3"],
    tgt_intrinsics: Optional[Float[Tensor, "3 3"]] = None,
):
    device = src_steps.device
    # grab best indices
    best_inds = src_weights.argmax(-1, keepdims=True)  # H, W
    z_A = torch.gather(src_steps, -1, best_inds)  # H, W, 1

    H, W = src_steps.shape[:2]
    xy_pix_map = torch.stack(
        torch.meshgrid(torch.arange(W), torch.arange(H)), dim=-1
    ).to(device)
    xy_pix_map = xy_pix_map.transpose(0, 1)  # H, W, 2

    padding = torch.ones((H, W, 1)).to(device)

    xyz_A_camera = torch.cat([xy_pix_map, padding], dim=-1) * z_A
    xyz_A_camera = xyz_A_camera @ torch.inverse(src_intrinsics).T  # H, W, 3

    xyz_A_world = torch.cat([xyz_A_camera, padding], dim=-1) @ src_c2w.T  # H, W, 4

    xy_B = (xyz_A_world @ torch.inverse(tgt_c2w).T)[
        ..., :3
    ] @ tgt_intrinsics.T  # H, W, 3
    xy_B = xy_B[..., :2] / xy_B[..., 2:]  # H, W, 2
    xy_B = xy_B.round().long()  # H, W, 2

    # clip to image bounds
    xy_B[..., 0] = xy_B[..., 0].clamp(0, W - 1)
    xy_B[..., 1] = xy_B[..., 1].clamp(0, H - 1)

    return xy_B
