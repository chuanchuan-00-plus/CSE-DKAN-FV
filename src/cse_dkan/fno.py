"""Compact one-dimensional Fourier neural operator baseline."""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class SpectralConv1d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, modes: int) -> None:
        super().__init__()
        if modes < 1:
            raise ValueError("modes must be positive")
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.modes = int(modes)
        scale = 1.0 / max(1, in_channels * out_channels)
        weight = scale * torch.randn(in_channels, out_channels, modes, dtype=torch.cfloat)
        self.weight = nn.Parameter(weight)

    def forward(self, values: Tensor) -> Tensor:
        if values.ndim != 3:
            raise ValueError("spectral input must have shape [batch, channels, points]")
        spectrum = torch.fft.rfft(values, dim=-1)
        retained = min(self.modes, spectrum.shape[-1])
        output = torch.zeros(
            values.shape[0],
            self.out_channels,
            spectrum.shape[-1],
            dtype=spectrum.dtype,
            device=values.device,
        )
        output[..., :retained] = torch.einsum(
            "bix,iox->box", spectrum[..., :retained], self.weight[..., :retained]
        )
        return torch.fft.irfft(output, n=values.shape[-1], dim=-1)


class FNO1d(nn.Module):
    """Map grid functions plus coordinates/parameters to grid functions."""

    def __init__(
        self,
        in_channels: int = 5,
        out_channels: int = 3,
        width: int = 32,
        modes: int = 24,
        layers: int = 4,
    ) -> None:
        super().__init__()
        if layers < 1:
            raise ValueError("layers must be positive")
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.width = int(width)
        self.modes = int(modes)
        self.layers = int(layers)
        self.lift = nn.Linear(in_channels, width)
        self.spectral = nn.ModuleList(
            [SpectralConv1d(width, width, modes) for _ in range(layers)]
        )
        self.local = nn.ModuleList([nn.Conv1d(width, width, 1) for _ in range(layers)])
        self.project = nn.Sequential(
            nn.Linear(width, 2 * width), nn.GELU(), nn.Linear(2 * width, out_channels)
        )

    def forward(self, values: Tensor) -> Tensor:
        if values.ndim != 3 or values.shape[-1] != self.in_channels:
            raise ValueError("FNO input must have shape [batch, points, in_channels]")
        hidden = self.lift(values).transpose(1, 2)
        for index, (spectral, local) in enumerate(zip(self.spectral, self.local)):
            hidden = spectral(hidden) + local(hidden)
            if index + 1 < self.layers:
                hidden = F.gelu(hidden)
        return self.project(hidden.transpose(1, 2))
