from typing import Optional

import numpy as np
import torch
from einops import rearrange, repeat


class PositionalEncoding(torch.nn.Module):
    """
    Implement NeRF's positional encoding
    """

    def __init__(
        self,
        num_freqs=6,
        d_in=3,
        freq_factor: Optional[float] = None,
        period: Optional[float] = None,
        include_input=True,
    ):
        super().__init__()

        # Either specify base frequency coefficient or base period.
        assert period is None or freq_factor is None
        if period is not None:
            freq_factor = 2 * np.pi / period

        self.num_freqs = num_freqs
        self.d_in = d_in
        self.freqs = freq_factor * 2.0 ** torch.arange(0, num_freqs)
        self.d_out = self.num_freqs * 2 * d_in
        self.include_input = include_input
        if include_input:
            self.d_out += d_in
        # f1 f1 f2 f2 ... to multiply x by
        self.register_buffer(
            "_freqs", torch.repeat_interleave(self.freqs, 2).view(1, -1, 1)
        )
        # 0 pi/2 0 pi/2 ... so that
        # (sin(x + _phases[0]), sin(x + _phases[1]) ...) = (sin(x), cos(x)...)
        _phases = torch.zeros(2 * self.num_freqs)
        _phases[1::2] = np.pi * 0.5
        self.register_buffer("_phases", _phases.view(1, -1, 1))

    def forward(self, x):
        """
        Apply positional encoding (new implementation)
        :param x (batch, self.d_in)
        :return (batch, self.d_out)
        """
        embed = x.unsqueeze(-2)
        embed = repeat(embed, "... j n -> ... (k j) n", k=2 * self.num_freqs)
        embed = torch.sin(torch.addcmul(self._phases, embed, self._freqs))
        embed = rearrange(embed, "... j n -> ... (j n)")

        if self.include_input:
            embed = torch.cat((x, embed), dim=-1)
        return embed