import torch
import torch.nn as nn
import tqdm

from neural_jacobian_field.models.action_model import (
    ActionModel,
    ActionModelInput,
    SceneRepresentation,
)
from jaxtyping import Float
from torch import Tensor
from . import CameraContext


def optimize_actions(
    model: ActionModel,
    camera_context: CameraContext,
    ctxt_image: Float[Tensor, "C H W"],
    z_near: Float[Tensor, "()"],
    z_far: Float[Tensor, "()"],
    origins: Float[Tensor, "batch_size num_rays 3"],
    directions: Float[Tensor, "batch_size num_rays 3"],
    curr_pix_locs: Float[Tensor, "num_points 2"],
    trgt_pix_flow: Float[Tensor, "num_points 2"],
    num_optim_iter: int = 200,
    learning_rate: float = 1e-2,
    action_dim: int = 8,
    action_init_var: float = 0.001,
    reg_multiplier: float = 0.000,
    return_history: bool = False,
    device: str = "cuda:0",
):
    image_height, image_width = ctxt_image.shape[-2:]

    # create index selector
    pixel_locations_t0 = curr_pix_locs[:, :].to(torch.long)
    index_selector_t0 = pixel_locations_t0[..., 0]
    index_selector_t0 += image_width * pixel_locations_t0[..., 1]

    # create action param
    action_param = torch.randn((1, action_dim), device=device)
    action_param *= action_init_var
    action_param = nn.Parameter(action_param)

    # create optimizer
    optim = torch.optim.Adam([action_param], lr=learning_rate)
    ran = tqdm.trange(num_optim_iter)

    # create pixelnerf inputs
    pixelnerf_packed_inputs: ActionModelInput = {
        "robot_action": action_param,
        "ctxt_rgb": ctxt_image[None],
        "ctxt_c2w": camera_context["ctxt_c2w"][None],
        "ctxt_intr": camera_context["ctxt_intr"][None],
        "trgt_c2w": camera_context["trgt_c2w"][None],
        "trgt_intr": camera_context["trgt_intr_raw"][None],
    }

    with torch.no_grad():

        jacobian_info: SceneRepresentation = model.encode_image(
            origins[:, index_selector_t0, :],
            directions[:, index_selector_t0, :],
            z_near,
            z_far,
            pixelnerf_packed_inputs,
        )

    action_history = [] if return_history else None
    for i in ran:
        optim.zero_grad()
        pred_flow = model.predict_optical_flow(
            jacobian_info=jacobian_info,
            action=action_param,
        )

        loss = torch.nn.functional.mse_loss(
            pred_flow,
            trgt_pix_flow,
            reduction="mean",
        )
        # regularize dq
        dq_reg = reg_multiplier * action_param.pow(2).mean()
        loss += dq_reg

        loss.backward()
        optim.step()
        ran.set_description_str(f"loss: {loss.item()} | dq_reg: {dq_reg}")

        if action_history is not None:
            action_history.append(action_param.detach().clone())

    return action_param.data, action_history
