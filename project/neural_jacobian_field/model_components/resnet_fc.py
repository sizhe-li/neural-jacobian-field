import torch
import torch.autograd.profiler as profiler
from jaxtyping import Float
from omegaconf import DictConfig
from torch import Tensor, nn


# Resnet Blocks
class ResnetBlockFC(nn.Module):
    """
    Fully connected ResNet Block class.
    Taken from DVR code.
    :param size_in (int): input dimension
    :param size_out (int): output dimension
    :param size_h (int): hidden dimension
    """

    def __init__(self, size_in, size_out=None, size_h=None, beta=0.0):
        super().__init__()
        # Attributes
        if size_out is None:
            size_out = size_in

        if size_h is None:
            size_h = min(size_in, size_out)

        self.size_in = size_in
        self.size_h = size_h
        self.size_out = size_out
        # Submodules
        self.fc_0 = nn.Linear(size_in, size_h)
        self.fc_1 = nn.Linear(size_h, size_out)

        # Init
        nn.init.constant_(self.fc_0.bias, 0.0)
        nn.init.kaiming_normal_(self.fc_0.weight, a=0, mode="fan_in")
        nn.init.constant_(self.fc_1.bias, 0.0)
        nn.init.zeros_(self.fc_1.weight)

        if beta > 0:
            self.activation = nn.Softplus(beta=beta)
        else:
            self.activation = nn.ReLU()

        if size_in == size_out:
            self.shortcut = None
        else:
            self.shortcut = nn.Linear(size_in, size_out, bias=False)
            nn.init.constant_(self.shortcut.bias, 0.0)
            nn.init.kaiming_normal_(self.shortcut.weight, a=0, mode="fan_in")

    def forward(self, x):
        with profiler.record_function("resblock"):
            net = self.fc_0(self.activation(x))
            dx = self.fc_1(self.activation(net))

            if self.shortcut is not None:
                x_s = self.shortcut(x)
            else:
                x_s = x
            return x_s + dx


class ResnetFC(nn.Module):
    resnet_cfg: DictConfig
    d_latent: int

    def __init__(self, resnet_cfg: DictConfig, d_in: int, d_latent: int, d_out: int):
        """
        :param d_in input size
        :param d_out output size
        :param n_blocks number of Resnet blocks
        :param d_latent latent size, added in each resnet block (0 = disable)
        :param d_hidden hiddent dimension throughout network
        :param beta softplus beta, 100 is reasonable; if <=0 uses ReLU activations
        instead
        """
        super().__init__()
        self.resnet_cfg = resnet_cfg
        self.d_latent = d_latent

        if d_in > 0:
            self.lin_in = nn.Linear(d_in, self.resnet_cfg.d_hidden)
            nn.init.constant_(self.lin_in.bias, 0.0)
            nn.init.kaiming_normal_(self.lin_in.weight, a=0, mode="fan_in")

        self.lin_out = nn.Linear(self.resnet_cfg.d_hidden, d_out)
        nn.init.constant_(self.lin_out.bias, 0.0)
        nn.init.kaiming_normal_(self.lin_out.weight, a=0, mode="fan_in")

        self.blocks = nn.ModuleList(
            [
                ResnetBlockFC(resnet_cfg.d_hidden, beta=resnet_cfg.beta)
                for i in range(resnet_cfg.n_blocks)
            ]
        )

        if d_latent != 0:
            n_lin_z = min(resnet_cfg.combine_layer, resnet_cfg.n_blocks)
            self.lin_z = nn.ModuleList(
                [nn.Linear(d_latent, resnet_cfg.d_hidden) for i in range(n_lin_z)]
            )
            for i in range(n_lin_z):
                nn.init.constant_(self.lin_z[i].bias, 0.0)
                nn.init.kaiming_normal_(self.lin_z[i].weight, a=0, mode="fan_in")

        if resnet_cfg.beta > 0:
            self.activation = nn.Softplus(beta=resnet_cfg.beta)
        else:
            self.activation = nn.ReLU()

    def forward(
        self,
        z: Float[Tensor, "camera view sample d_latent"],  # features
        x: Float[Tensor, "camera view sample d_in"],  # encoded rays
        compute_action_features: bool = False,
    ) -> Float[Tensor, "camera sample d_out"]:
        x = self.lin_in(x)

        action_features = [] if compute_action_features else None
        for blkid in range(self.resnet_cfg.n_blocks):
            if self.d_latent > 0 and blkid < self.resnet_cfg.combine_layer:
                tz = self.lin_z[blkid](z)
                x = x + tz

            x = self.blocks[blkid](x)

            if compute_action_features:
                action_features.append(x)

        output = self.lin_out(self.activation(x))
        if compute_action_features:
            return output, torch.cat(action_features, dim=-1)

        return output
