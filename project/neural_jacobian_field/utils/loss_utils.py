import torch
from jaxtyping import Float
from torch import Tensor


EPS = 1.0e-7


def ds_nerf_depth_loss(
    weights: Float[Tensor, "*batch num_samples 1"],
    termination_depth: Float[Tensor, "*batch 1"],
    steps: Float[Tensor, "*batch num_samples 1"],
    lengths: Float[Tensor, "*batch num_samples 1"],
    sigma: Float[Tensor, "0"],
) -> Float[Tensor, "*batch 1"]:
    """Depth loss from Depth-supervised NeRF (Deng et al., 2022).

    Args:
        weights: Weights predicted for each sample.
        termination_depth: Ground truth depth of rays.
        steps: Sampling distances along rays.
        lengths: Distances between steps.
        sigma: Uncertainty around depth values.
    Returns:
        Depth loss scalar.
    """
    depth_mask = termination_depth > 0

    loss = (
        -torch.log(weights + EPS)
        * torch.exp(-((steps - termination_depth[..., None, :]) ** 2) / (2 * sigma))
        * lengths
    )
    loss = loss.sum(-2) * depth_mask
    return torch.mean(loss)
