import torch.nn.functional as F
from einops import rearrange

from neural_jacobian_field.rendering.geometry import (
    deprecated_project,
    homogenize_points,
    transform_world2cam,
)


def get_pixel_aligned_features(
    coords_3d_world, cam2world, intrinsics, img_features, interp="bilinear"
):
    # Args:
    #     coords_3d_world: shape (b, n, 3)
    #     cam2world: camera pose of shape (..., 4, 4)

    # project 3d points to 2D
    c3d_world_hom = homogenize_points(coords_3d_world)
    c3d_cam_hom = transform_world2cam(c3d_world_hom, cam2world[:, None])
    c2d_cam, depth = deprecated_project(c3d_cam_hom, intrinsics.unsqueeze(1))

    # now between 0 and 1. Map to -1 and 1
    c2d_norm = (c2d_cam - 0.5) * 2
    c2d_norm = rearrange(c2d_norm, "b n ch -> b n () ch")
    c2d_norm = c2d_norm[..., :2]

    # grid_sample
    feats = F.grid_sample(
        img_features, c2d_norm, align_corners=True, padding_mode="border", mode=interp
    )
    feats = feats.squeeze(-1)  # b ch n

    feats = rearrange(feats, "b ch n -> b n ch")
    return feats, c3d_cam_hom[..., :3], c2d_cam
