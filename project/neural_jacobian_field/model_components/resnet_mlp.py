from torch import nn
from dataclasses import dataclass
from typing import Literal


class ResMLP(nn.Module):
    def __init__(self, ch_in, ch_hid, out_ch, num_res_block=1):
        super().__init__()

        self.res_blocks = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(ch_hid, ch_hid),
                    nn.ReLU(),
                    nn.LayerNorm([ch_hid], elementwise_affine=True),
                    nn.Linear(ch_hid, ch_hid),
                    nn.ReLU(),
                )
                for _ in range(num_res_block)
            ]
        )

        self.proj_in = nn.Linear(ch_in, ch_hid)
        self.out = nn.Linear(ch_hid, out_ch)

    def forward(self, x):
        x = self.proj_in(x)

        for i, block in enumerate(self.res_blocks):
            x_in = x

            x = block(x)

            if i != len(self.res_blocks) - 1:
                x = x + x_in

        return self.out(x)