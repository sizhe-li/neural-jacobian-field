import torch
from einops import einsum


def compute_joint_sensitivity(
    original_feats, trgt_c2w, action_dim=6, joint_sensitivity_thresh=0.5
):
    # # convert to the same coordinate system
    # homo_feats = geometry.homogenize_vecs(
    #     rearrange(
    #         original_feats, "(n_joints d) h w -> h w n_joints d", n_joints=action_dim
    #     )
    # )
    homo_feats = einsum(trgt_c2w, homo_feats, "i j, ... j -> ... i")[..., :3]
    homo_feats = torch.norm(homo_feats, dim=-1)

    sensitivity = rearrange(homo_feats, "h w n_joints-> n_joints h w")

    # rescale to 0 and 1
    minima = reduce(sensitivity, "C H W -> C () ()", "min")
    maxima = reduce(sensitivity, "C H W -> C () ()", "max")

    sensitivity = (sensitivity - minima) / (maxima - minima + 1e-10)
    sensitivity = sensitivity.clip(0, 1)

    sensitivity[sensitivity < joint_sensitivity_thresh] = 0

    return sensitivity
