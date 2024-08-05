import cv2
import numpy as np
import torch
from torch import Tensor
from matplotlib import cm

from jaxtyping import Float
from neural_jacobian_field.rendering import geometry
from einops import einsum, pack, rearrange, reduce


def compute_joint_sensitivity(original_feats, trgt_c2w):
    # convert to the same coordinate system
    homo_feats = geometry.homogenize_vecs(
        rearrange(original_feats, "(n_joints dim) h w -> h w n_joints dim", dim=3)
    )
    homo_feats = einsum(trgt_c2w, homo_feats, "i j, ... j -> ... i")[..., :3]
    homo_feats = torch.norm(homo_feats, dim=-1)

    sensitivity = rearrange(homo_feats, "h w n_joints-> n_joints h w")

    # rescale to 0 and 1
    minima = reduce(sensitivity, "C H W -> C () ()", "min")
    maxima = reduce(sensitivity, "C H W -> C () ()", "max")

    sensitivity = (sensitivity - minima) / (maxima - minima + 1e-10)
    sensitivity = sensitivity.clip(0, 1)

    # sensitivity[sensitivity < joint_sensitivity_thresh] = 0

    return sensitivity


def visualize_joint_sensitivity(joint_sensitivity_th, joint_colors_th):

    projected_sensitvity = einsum(
        joint_sensitivity_th, joint_colors_th, "feature H W, rgb feature -> rgb H W"
    )
    # normalize to 0, 1
    minima = reduce(projected_sensitvity, "C H W -> C () ()", "min")
    maxima = reduce(projected_sensitvity, "C H W -> C () ()", "max")
    projected_sensitvity = (projected_sensitvity - minima) / (maxima - minima + 1e-10)
    projected_sensitvity = projected_sensitvity.clip(0, 1)

    # convert to an image
    projected_sensitvity = projected_sensitvity.permute(1, 2, 0).cpu().numpy()
    # projected_feats = ((projected_feats) * 255).astype(np.uint8)
    # projected_sensitvity = ((projected_sensitvity) * 255).astype(np.uint8)
    projected_sensitvity = ((1 - projected_sensitvity) * 255).astype(np.uint8)

    # projected_feats = projected_feats[..., [1, 2, 0]]

    alpha_mask = ((2.0 - (projected_sensitvity.sum(-1) / 765)).clip(0, 1) * 255).astype(
        np.uint8
    )
    colored_sensitivity_image = cv2.merge((projected_sensitvity, alpha_mask))

    return colored_sensitivity_image


def draw_flow_on_image(
    curr_frame,
    pix_to_draw_y,
    pix_to_draw_x,
    flow_pred_combined: Float[Tensor, "1 N 2"],
    color_map_choice: str = "cool",
    length_multiplier: int = 50,
    line_thickness: int = 2,
    use_norm: bool = True,
):
    canvas_frame = curr_frame.copy()
    color_map = cm.get_cmap(color_map_choice)

    for i, (this_x, this_y) in enumerate(zip(pix_to_draw_x, pix_to_draw_y)):
        this_flow = flow_pred_combined[:, i, :].squeeze()
        if use_norm:
            this_flow = (this_flow / torch.norm(this_flow, p=2)) * length_multiplier
        else:
            this_flow = this_flow * length_multiplier

        y1, x1 = int(this_y), int(this_x)
        y2, x2 = int(this_y + this_flow[1]), int(this_x + this_flow[0])

        # cut the line into 5 segments
        num_segments = 10
        line_segments = np.linspace(0, 1, num_segments)
        line_segments = np.stack(
            [
                np.interp(line_segments, [0, 1], [y1, y2]),
                np.interp(line_segments, [0, 1], [x1, x2]),
            ],
            axis=1,
        )

        for segment_index in range(num_segments - 1):
            segment_y1, segment_x1 = line_segments[segment_index]
            segment_y2, segment_x2 = line_segments[segment_index + 1]

            segment_frac = segment_index / (num_segments - 1)
            segment_color = list(np.array(color_map(segment_frac)[:3]) * 255)

            # draw arrow if it's the last segment
            if segment_index == num_segments - 2:
                cv2.arrowedLine(
                    canvas_frame,
                    (int(segment_x1), int(segment_y1)),
                    (int(segment_x2), int(segment_y2)),
                    segment_color,
                    thickness=line_thickness,
                    tipLength=1,
                )
            else:
                cv2.line(
                    canvas_frame,
                    (int(segment_x1), int(segment_y1)),
                    (int(segment_x2), int(segment_y2)),
                    segment_color,
                    thickness=line_thickness,
                )

    return canvas_frame


def draw_point_matching_on_image(
    input_frame,
    curr_pix_locs,
    trgt_pix_locs,
    color_map_choice="cool",
    line_thickness: int = 1,
):
    canvas_frame = input_frame.copy()
    color_map = cm.get_cmap(color_map_choice)

    num_points = len(curr_pix_locs)

    for i in range(num_points):
        y1, x1 = map(int, np.array(curr_pix_locs[i]))
        y2, x2 = map(int, np.array(trgt_pix_locs[i]))

        # cut the line into 5 segments
        num_segments = 10
        line_segments = np.linspace(0, 1, num_segments)
        line_segments = np.stack(
            [
                np.interp(line_segments, [0, 1], [y1, y2]),
                np.interp(line_segments, [0, 1], [x1, x2]),
            ],
            axis=1,
        )

        for segment_index in range(num_segments - 1):
            segment_y1, segment_x1 = line_segments[segment_index]
            segment_y2, segment_x2 = line_segments[segment_index + 1]

            segment_frac = segment_index / (num_segments - 1)
            segment_color = list(np.array(color_map(segment_frac)[:3]) * 255)

            # draw arrow if it's the last segment
            if segment_index == num_segments - 2:
                cv2.arrowedLine(
                    canvas_frame,
                    (int(segment_y1), int(segment_x1)),
                    (int(segment_y2), int(segment_x2)),
                    segment_color,
                    thickness=line_thickness,
                    tipLength=1,
                )
            else:
                cv2.line(
                    canvas_frame,
                    (int(segment_y1), int(segment_x1)),
                    (int(segment_y2), int(segment_x2)),
                    segment_color,
                    thickness=line_thickness,
                )

    return canvas_frame
