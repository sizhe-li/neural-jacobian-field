from typing import Dict, List, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from einops import einsum, rearrange, reduce
from jaxtyping import Float, Int
from neural_jacobian_field.rendering import geometry
from PIL import Image
from torch import Tensor

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
    "model_toy_arm": [
        [0.5, 0.8, 0.2],
        [0.9, 0.2, 0.0],
        [0, 0.8, 0],
        [1.0, 0.0, 1.0],
        [0, 0, 1],
        [0.1, 0.9, 0.7],
    ],
    "model_pneumatic_hand_only": [
        [0, 0, 1],
        [0.9, 0.2, 0.0],
        [0, 0.9, 0],
        [1.0, 0.0, 1.0],
        [0.1, 0.9, 0.7],
        [0.5, 0.8, 0.2],
    ],
}


def compute_joint_sensitivity(
    jacobians: Float[Tensor, "*batch height width action_spatial_dim"],
    extrinsics: None | Float[Tensor, "*batch 4 4"] = None,
    mode: int = 0,
) -> Float[Tensor, "*batch height width action_spatial_dim"]:
    # convert to the same coordinate system
    jacobians = geometry.homogenize_vecs(
        rearrange(
            jacobians,
            "... height width (action_dim spatial_dim) -> ... height width action_dim spatial_dim",
            spatial_dim=3,  # world is 3D
        )
    )

    if extrinsics is not None:
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

    if mode == 1:
        sensitivity = 1.1 - sensitivity

    sensitivity = sensitivity.clip(0, 1)

    return sensitivity


def visualize_joint_sensitivity(
    sensitivity: Float[Tensor, "*batch action_dim height width"],
    color_map: Float[Tensor, "rgb action_dim"],
) -> Int[np.ndarray, "*batch height width rgb"]:
    sensitivity = einsum(
        sensitivity, color_map, "... action_dim H W, rgb action_dim -> ... rgb H W"
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


def compute_joint_sensitivity_point_cloud(
    jacobians: Float[Tensor, "num_points action_dim spatial_dim"],
    extrinsics: None | Float[Tensor, "*batch 4 4"] = None,
    mode: int = 0,
) -> Float[Tensor, "*batch action_dim spatial_dim"]:

    jacobians = torch.norm(jacobians, dim=-1)
    sensitivity = jacobians

    # rescale to 0 and 1
    minima = reduce(sensitivity, "N C -> () C", "min")
    maxima = reduce(sensitivity, "N C -> () C", "max")

    sensitivity = (sensitivity - minima) / (maxima - minima + 1e-10)

    sensitivity = sensitivity.clip(0, 1)

    return sensitivity


def visualize_joint_sensitivity_point_cloud(
    sensitivity: Float[Tensor, "num_points action_dim"],
    color_map: Float[Tensor, "rgb action_dim"],
    mode: int = 0,
) -> Float[Tensor, "num_points rgb"]:
    colors = einsum(
        sensitivity,
        color_map,
        "num_points action_dim, rgb action_dim -> num_points rgb",
    )

    # normalize to 0, 1
    minima = reduce(colors, "num_points rgb -> () rgb", "min")
    maxima = reduce(colors, "num_points rgb -> () rgb", "max")

    colors = (colors - minima) / (maxima - minima + 1e-10)

    if mode == 0:
        colors = colors.clip(0, 1)
        colors = 1 - colors

    elif mode == 1:
        colors = 1.1 - colors
        colors = colors.clip(0, 1)

    return colors


def normalize_image(image):
    return (image - image.min()) / (image.max() - image.min())


def denormalize_torch_image(img_tensor: torch.Tensor) -> np.ndarray:
    """
    Denormalize a [0, 1] torch image of shape [1, 3, H, W] to a numpy array of shape [H, W, 3]
    with values in the range [0, 255].

    Args:
        img_tensor (torch.Tensor): Input tensor image of shape [1, 3, H, W].

    Returns:
        np.ndarray: Denormalized image of shape [H, W, 3] with values in the range [0, 255].
    """
    # Denormalize
    img_numpy = img_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255.0
    # Convert to uint8
    img_numpy = img_numpy.astype(np.uint8)
    return img_numpy


def visualize_jacobian_chain_structure(
    input_image_np: Int[np.ndarray, "H W rgb"],
    input_joint_sensitivity_np: Float[Tensor, "action_dim H W"],
    joint_colors_np: Float[np.ndarray, "action_dim rgb"],
    debug: bool = False,
) -> Tuple[Int[np.ndarray, "H W rgb"], Int[np.ndarray, "H W rgb"]]:
    image_height, image_width, _ = input_image_np.shape

    canvas_overlay = input_image_np.copy()

    list_of_diff_masks = []
    list_of_norm_projected = []

    for i in range(1, 5):
        # changin the threshold will make the mask sharper
        prev_s = ((input_joint_sensitivity_np[i])).clip(0.10, 1.5)
        next_s = ((input_joint_sensitivity_np[i + 1])).clip(0.10, 1.5)
        diff = (prev_s - next_s).clip(0.01, 1)
        diff = normalize_image(diff)
        diff = cv2.resize(diff, (image_width, image_height))

        if debug:
            # create a 1x3 grid of plots
            fig, ax = plt.subplots(1, 3, figsize=(15, 5))
            # draw
            ax[0].imshow(prev_s)
            ax[0].set_title("prev_s")
            ax[1].imshow(next_s)
            ax[1].set_title("next_s")
            ax[2].imshow(diff)
            ax[2].set_title("diff")
            fig.colorbar(ax[0].imshow(prev_s), ax=ax[0])
            fig.colorbar(ax[1].imshow(next_s), ax=ax[1])
            fig.colorbar(ax[2].imshow(diff), ax=ax[2])
            plt.show()

        projected = diff[..., None] * np.array(joint_colors_np[i])
        minima = reduce(projected, "C H W -> C () ()", "min")
        maxima = reduce(projected, "C H W -> C () ()", "max")

        norm_projected = (projected - minima) / (maxima - minima + 1e-10)
        norm_projected = (norm_projected * 255).clip(0, 255).astype(np.uint8)

        if debug:
            overlay = cv2.addWeighted(input_image_np, 0.5, norm_projected, 0.8, 0)
            plt.imshow(overlay)
            plt.show()

        list_of_diff_masks.append(diff)
        list_of_norm_projected.append(norm_projected)

    next_s = cv2.resize(next_s, (image_width, image_height))

    list_of_diff_masks.append(next_s.copy())
    next_s = normalize_image(next_s.clip(0.25, 1.0))

    projected = next_s[..., None] * np.array(joint_colors_np[i + 1])
    minima = reduce(projected, "C H W -> C () ()", "min")
    maxima = reduce(projected, "C H W -> C () ()", "max")

    norm_projected = (projected - minima) / (maxima - minima + 1e-10)
    norm_projected = (norm_projected * 255).clip(0, 255).astype(np.uint8)
    list_of_norm_projected.append(norm_projected)

    canvas_overlay = input_image_np.copy()
    canvas_overlay = cv2.cvtColor(canvas_overlay, cv2.COLOR_RGB2RGBA)
    canvas_overlay[..., 3] = 155
    canvas_overlay = Image.fromarray(canvas_overlay)

    canvas_white_bkgd = np.ones_like(input_image_np, dtype=np.uint8) * 255
    canvas_white_bkgd = cv2.cvtColor(canvas_white_bkgd, cv2.COLOR_RGB2RGBA)

    canvas_white_bkgd[..., 3] = 255
    canvas_white_bkgd = Image.fromarray(canvas_white_bkgd)

    for i in range(len(list_of_norm_projected)):
        norm_projected_rgba = cv2.cvtColor(
            list_of_norm_projected[i], cv2.COLOR_RGB2RGBA
        )
        diff = list_of_diff_masks[i]
        diff = normalize_image(diff)

        alpha_mask = diff * 1.5
        alpha_mask = (alpha_mask.clip(0, 1) * 255).astype(np.uint8)
        norm_projected_rgba[..., 3] = alpha_mask

        overlay = Image.fromarray(norm_projected_rgba)

        canvas_overlay.paste(overlay, (0, 0), overlay)
        canvas_white_bkgd.paste(overlay, (0, 0), overlay)

    return np.asarray(canvas_overlay), np.asarray(canvas_white_bkgd)
