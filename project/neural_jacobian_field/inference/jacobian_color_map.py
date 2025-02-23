import torch
import numpy as np
from torch import Tensor

from typing import Dict, List
from neural_jacobian_field.rendering import geometry
from einops import einsum, rearrange, reduce
from jaxtyping import Float, Int


JACOBIAN_COLORMAP: Dict[str, List[List[float]]] = {
    "model_allegro": [
        [0.0, 0.5, 0.5],
        [0, 1, 0],
        [0.8, 0.1, 0.1],
        [0.8, 0.0, 0.8],
        [0.0, 0.8, 0],
        [1.0, 0.8, 0],
        [1, 1, 0],
        [1, 0.0, 0.0],
    ],
    "model_allegro_transformer": [
        [0.0, 0.5, 0.5],
        [0, 1, 0],
        [0.8, 0.1, 0.1],
        [0.8, 0.0, 0.8],
        [0.0, 0.8, 0],
        [1.0, 0.8, 0],
        [1, 1, 0],
        [1, 0.0, 0.0],
    ],
}


def compute_joint_sensitivity(
    jacobians: Float[Tensor, "*batch height width action_spatial_dim"],
    extrinsics: Float[Tensor, "*batch 4 4"],
) -> Float[Tensor, "*batch height width action_spatial_dim"]:
    # convert to the same coordinate system
    jacobians = geometry.homogenize_vecs(
        rearrange(
            jacobians,
            "... height width (action_dim spatial_dim) -> ... height width action_dim spatial_dim",
            spatial_dim=3,  # world is 3D
        )
    )

    jacobians = einsum(
        extrinsics, jacobians, "... row_dim col_dim , ... col_dim -> ... row_dim"
    )[..., :3]
    jacobians = torch.norm(jacobians, dim=-1)

    sensitivity = rearrange(
        jacobians, "... height width action_dim -> ... action_dim height width"
    )

    # rescale to 0 and 1
    minima = reduce(sensitivity, "... C H W -> ... C () ()", "min")
    maxima = reduce(sensitivity, "... C H W -> ... C () ()", "max")

    sensitivity = (sensitivity - minima) / (maxima - minima + 1e-10)
    sensitivity = sensitivity.clip(0, 1)

    return sensitivity


def visualize_joint_sensitivity(
    sensitivity: Float[Tensor, "*batch action_dim height width"],
    color_map: Float[Tensor, "*batch rgb action_dim"],
) -> Int[np.ndarray, "*batch height width rgb"]:
    sensitivity = einsum(
        sensitivity, color_map, "... feature H W, ... rgb feature -> ... rgb H W"
    )
    # normalize to 0, 1
    minima = reduce(sensitivity, "... H W -> ... () ()", "min")
    maxima = reduce(sensitivity, "... H W -> ... () ()", "max")
    sensitivity = (sensitivity - minima) / (maxima - minima + 1e-10)
    sensitivity = sensitivity.clip(0, 1)

    # convert to a numpy image
    sensitivity = rearrange(sensitivity, "... C H W -> ... H W C").cpu().numpy()
    sensitivity = ((1 - sensitivity) * 255).astype(np.uint8)

    return sensitivity
