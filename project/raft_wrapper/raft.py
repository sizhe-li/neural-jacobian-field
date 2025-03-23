import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Tuple
from einops import rearrange
from torch import Tensor
from torchvision.models.optical_flow import Raft_Large_Weights, raft_large
from jaxtyping import Float
from torchvision.utils import flow_to_image


def resize_batched_frames(
    color_data: Float[Tensor, "N C H W"],
    long_dim: int = 768,
    inter_mode="bilinear",
) -> Tuple[Float[Tensor, "N C H W"], Tuple[int, int]]:
    orig_H, orig_W = color_data.shape[-2:]

    if orig_H > orig_W:
        new_H, new_W = long_dim, int(long_dim / orig_H * orig_W)
    else:
        new_H, new_W = int(long_dim / orig_W * orig_H), long_dim

    color_data = torch.nn.functional.interpolate(
        color_data, (new_H, new_W), mode=inter_mode
    )

    return color_data, (orig_H, orig_W)


def resize_batched_flow(
    flow: Float[Tensor, "N 2 H W"],
    new_H: int,
    new_W: int,
    inter_mode="bilinear",
) -> Float[Tensor, "N 2 H W"]:
    curr_H, curr_W = flow.shape[-2:]

    # for interpretability and avoiding dimension mismatch
    W_dim = 0
    H_dim = 1

    flow[:, :, W_dim] *= new_W / curr_W
    flow[:, :, H_dim] *= new_H / curr_H

    flow = torch.nn.functional.interpolate(flow, (new_H, new_W), mode=inter_mode)
    return flow


class InputPadder:
    """Pads images such that dimensions are divisible by 8"""

    def __init__(self, dims, mode="sintel"):
        self.ht, self.wd = dims[-2:]
        pad_ht = (((self.ht // 8) + 1) * 8 - self.ht) % 8
        pad_wd = (((self.wd // 8) + 1) * 8 - self.wd) % 8
        if mode == "sintel":
            self._pad = [
                pad_wd // 2,
                pad_wd - pad_wd // 2,
                pad_ht // 2,
                pad_ht - pad_ht // 2,
            ]
        else:
            self._pad = [pad_wd // 2, pad_wd - pad_wd // 2, 0, pad_ht]

    def pad(self, *inputs):
        return [F.pad(x, self._pad, mode="replicate") for x in inputs]

    def unpad(self, x):
        ht, wd = x.shape[-2:]
        c = [self._pad[2], ht - self._pad[3], self._pad[0], wd - self._pad[1]]
        return x[..., c[0] : c[1], c[2] : c[3]]


class RaftWrapper(nn.Module):
    def __init__(self, weights=Raft_Large_Weights.DEFAULT):
        super().__init__()

        self.model = raft_large(weights=weights, progress=False)
        self._transforms = weights.transforms()

        self.register_buffer("_motion_thresh", torch.tensor(0.5, dtype=torch.float32))
        self._freeze_params()

    def _freeze_params(self):
        # freezing
        for param in self.model.parameters():
            param.requires_grad = False

    def _forward_flow(self, img1_batch, img2_batch):

        list_of_flows = self.model(
            *self._transforms(img1_batch, img2_batch),
            num_flow_updates=20,
        )
        flow = list_of_flows[-1]

        return flow

    # @typechecked
    def forward_flow(
        self,
        images: Float[Tensor, "batch time channel height width"],
        chunk: int = 50,
    ):
        T = images.shape[1]
        img1_batch = images[:, :-1]
        img2_batch = images[:, 1:]

        img1_batch = rearrange(img1_batch, "b t c h w -> (b t) c h w")
        img2_batch = rearrange(img2_batch, "b t c h w -> (b t) c h w")

        # compute optical flow, process in chunks to avoid OOM
        pred_flows = []
        for i in range(0, len(img1_batch), chunk):
            pred_flows += [
                self._forward_flow(img1_batch[i : i + chunk], img2_batch[i : i + chunk])
            ]
        pred_flows = torch.cat(pred_flows, dim=0)

        return pred_flows

    def forward(self, img1_batch, img2_batch):
        return self._forward_flow(img1_batch, img2_batch)

    def convert_flow_to_video(
        self, flow: Float[Tensor, "N 2 H W"]
    ) -> Float[Tensor, "N C H W"]:
        H, W = flow.shape[-2:]
        flow_vid = flow_to_image(flow)
        flow_vid = torch.nn.functional.interpolate(flow_vid, (H, W))
        flow_vid = rearrange(flow_vid, "t c h w -> t h w c")
        flow_vid = flow_vid.cpu().numpy()

        return flow_vid
