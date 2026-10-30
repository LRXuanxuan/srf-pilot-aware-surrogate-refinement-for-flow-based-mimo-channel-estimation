"""U-Net backbone of the SRF flow prior.

The network predicts the conditional mixture ``p(u | H_t, t)`` of the additive
path displacement ``u``.  The convolutional topology (time embedding, residual
blocks, down/up sampling, GroupNorm/SiLU blocks) is unchanged from the legacy
implementation; only the final head is a Gaussian-mixture head:

* ``means``      -- ``(B, K, C, H, W)``
* ``logstds``    -- ``(B, 1, 1, 1, 1)``, a single sample-wise scale bounded to
  ``[gm_min_logstd, gm_max_logstd]``
* ``logweights`` -- ``(B, K, 1, H, W)``, log-softmax normalised over ``K``
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["GMUNet"]


class SinusoidalPosEmb(nn.Module):
    """Sinusoidal embedding of the time step ``t``."""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        return torch.cat((emb.sin(), emb.cos()), dim=-1)


class Downsample(nn.Module):
    """Strided 2x2 downsampling convolution."""

    def __init__(self, dim: int):
        super().__init__()
        self.conv = nn.Conv2d(dim, dim, 3, 2, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class Upsample(nn.Module):
    """Transposed-convolution 2x2 upsampling."""

    def __init__(self, dim: int):
        super().__init__()
        self.conv = nn.ConvTranspose2d(dim, dim, 4, 2, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class Block(nn.Module):
    """Convolution + GroupNorm + optional time conditioning + SiLU."""

    def __init__(self, dim: int, dim_out: int, groups: int = 8):
        super().__init__()
        self.proj = nn.Conv2d(dim, dim_out, 3, padding=1)
        self.norm = nn.GroupNorm(min(groups, dim_out), dim_out)
        self.act = nn.SiLU()

    def forward(
        self,
        x: torch.Tensor,
        scale_shift: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        x = self.proj(x)
        x = self.norm(x)
        if scale_shift is not None:
            scale, shift = scale_shift
            x = x * (scale + 1) + shift
        return self.act(x)


class ResnetBlock(nn.Module):
    """Pre-activation residual block with FiLM time conditioning."""

    def __init__(
        self,
        dim: int,
        dim_out: int,
        time_emb_dim: int | None = None,
        groups: int = 8,
    ):
        super().__init__()
        self.mlp = (
            nn.Sequential(nn.SiLU(), nn.Linear(time_emb_dim, dim_out * 2))
            if time_emb_dim is not None
            else None
        )
        self.block1 = Block(dim, dim_out, groups=groups)
        self.block2 = Block(dim_out, dim_out, groups=groups)
        self.res_conv = nn.Conv2d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(
        self, x: torch.Tensor, time_emb: torch.Tensor | None = None
    ) -> torch.Tensor:
        scale_shift = None
        if self.mlp is not None and time_emb is not None:
            scale_shift = self.mlp(time_emb).unsqueeze(-1).unsqueeze(-1).chunk(2, dim=1)
        h = self.block1(x, scale_shift=scale_shift)
        h = self.block2(h)
        return h + self.res_conv(x)


class GMUNet(nn.Module):
    """U-Net with a Gaussian-mixture output head (SRF flow prior).

    Args:
        in_channels: Input channels (real/imaginary parts, so 2).
        out_channels: Output channels of the mixture means (real/imaginary).
        model_channels: Base channel count (32 in the paper).
        channel_mult: Channel multipliers per resolution (1, 2, 4).
        num_res_blocks: Residual blocks per resolution.
        gm_num_components: Number of mixture components ``K`` (1 for DPR).
        gm_min_logstd: Lower bound of the predicted log standard deviation.
        gm_max_logstd: Upper bound of the predicted log standard deviation.
        gm_logstd_hidden: Hidden width of the log-std MLP (defaults to the time
            embedding width).
        gm_logstd_layers: Number of linear layers of the log-std MLP.
    """

    def __init__(
        self,
        in_channels: int = 2,
        out_channels: int = 2,
        model_channels: int = 32,
        channel_mult: tuple[int, ...] = (1, 2, 4),
        num_res_blocks: int = 2,
        gm_num_components: int = 1,
        gm_min_logstd: float = -3.5,
        gm_max_logstd: float = 0.5,
        gm_logstd_hidden: int | None = None,
        gm_logstd_layers: int = 2,
    ):
        super().__init__()
        self.out_channels = out_channels
        self.gm_num_components = gm_num_components
        self.gm_min_logstd = gm_min_logstd
        self.gm_max_logstd = gm_max_logstd

        time_dim = model_channels * 4
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(model_channels),
            nn.Linear(model_channels, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim),
        )

        self.init_conv = nn.Conv2d(in_channels, model_channels, 3, padding=1)
        self.downs = nn.ModuleList([])
        ch = model_channels
        ds_channels = [ch]
        for i, mult in enumerate(channel_mult):
            out_ch = model_channels * mult
            for _ in range(num_res_blocks):
                self.downs.append(ResnetBlock(ch, out_ch, time_emb_dim=time_dim))
                ch = out_ch
                ds_channels.append(ch)
            if i != len(channel_mult) - 1:
                self.downs.append(Downsample(ch))
                ds_channels.append(ch)

        self.mid_block1 = ResnetBlock(ch, ch, time_emb_dim=time_dim)
        self.mid_block2 = ResnetBlock(ch, ch, time_emb_dim=time_dim)

        self.ups = nn.ModuleList([])
        for i, mult in enumerate(reversed(channel_mult)):
            out_ch = model_channels * mult
            for _ in range(num_res_blocks + 1):
                self.ups.append(
                    ResnetBlock(ch + ds_channels.pop(), out_ch, time_emb_dim=time_dim)
                )
                ch = out_ch
            if i != len(channel_mult) - 1:
                self.ups.append(Upsample(ch))

        self.final_res_block = ResnetBlock(ch, ch, time_emb_dim=time_dim)

        gm_head_channels = gm_num_components * out_channels + gm_num_components
        self.final_conv = nn.Conv2d(ch, gm_head_channels, 3, padding=1)

        logstd_hidden = gm_logstd_hidden or time_dim
        logstd_layers: list[nn.Module] = []
        in_dim = time_dim
        for _ in range(max(gm_logstd_layers - 1, 0)):
            logstd_layers.extend([nn.SiLU(), nn.Linear(in_dim, logstd_hidden)])
            in_dim = logstd_hidden
        logstd_layers.extend([nn.SiLU(), nn.Linear(in_dim, 1)])
        self.logstd_mlp = nn.Sequential(*logstd_layers)

    def _bound_logstd(self, raw_logstd: torch.Tensor) -> torch.Tensor:
        bounded = torch.tanh(raw_logstd)
        logstd = self.gm_min_logstd + 0.5 * (bounded + 1.0) * (
            self.gm_max_logstd - self.gm_min_logstd
        )
        return logstd.view(raw_logstd.shape[0], 1, 1, 1, 1)

    def _format_gm_output(self, raw: torch.Tensor, t_emb: torch.Tensor) -> dict:
        b, _, h, w = raw.shape
        k = self.gm_num_components
        c = self.out_channels
        means_raw, logits_raw = torch.split(raw, [k * c, k], dim=1)
        means = means_raw.view(b, k, c, h, w)
        logweights = logits_raw.view(b, k, 1, h, w).log_softmax(dim=1)
        # The scale head is intentionally detached from the gradient graph,
        # as in the legacy implementation.
        raw_logstd = self.logstd_mlp(t_emb.detach())
        logstds = self._bound_logstd(raw_logstd)
        return {"means": means, "logstds": logstds, "logweights": logweights}

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> dict:
        """Predict the mixture of ``u`` for ``x = (B, C, H, W)`` and times ``t``."""
        if t.dim() > 1:
            t = t.flatten()
        t_emb = self.time_mlp(t * 1000)
        x = self.init_conv(x)
        h_list = [x]
        for module in self.downs:
            if isinstance(module, ResnetBlock):
                x = module(x, t_emb)
            else:
                x = module(x)
            h_list.append(x)
        x = self.mid_block1(x, t_emb)
        x = self.mid_block2(x, t_emb)
        for module in self.ups:
            if isinstance(module, ResnetBlock):
                skip = h_list.pop()
                if x.shape[2:] != skip.shape[2:]:
                    x = F.interpolate(x, size=skip.shape[2:], mode="nearest")
                x = torch.cat((x, skip), dim=1)
                x = module(x, t_emb)
            else:
                x = module(x)
        x = self.final_res_block(x, t_emb)
        raw = self.final_conv(x)
        return self._format_gm_output(raw, t_emb)
