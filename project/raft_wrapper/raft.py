import torch
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torchtyping import TensorType
from torchvision.models.optical_flow import Raft_Large_Weights, raft_large


# from typeguard import typechecked

# patch_typeguard()


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
        images: TensorType["b", "t", 3, "h", "w", torch.float32],
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
