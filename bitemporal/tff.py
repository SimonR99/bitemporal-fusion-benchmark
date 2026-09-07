"""Transformer Feature Fusion (TFF), the fusion stage of Text-ITSR.

Ported from the RSICCFormer reference implementation and kept architecturally
faithful so the comparison in the paper is against the published design rather
than a reinterpretation of it.

    Liu et al., "Remote Sensing Image Change Captioning With Dual-Branch
    Transformers: A New Method and a Large Scale Dataset", IEEE TGRS 2022.
    https://github.com/Chen-Yang-Liu/RSICC  (DOI 10.1109/TGRS.2022.3218921)

It differs from the other seven modules in using the difference features as
attention keys/values while the two acquisitions act as queries, and in mixing
the per-layer outputs through a convolutional residual block, which is why it
needs the patch grid shape.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.init import xavier_uniform_

__all__ = ["TransformerFeatureFusion"]


class _CrossTransformer(nn.Module):
    """One cross-attention layer using the temporal difference as key/value."""

    def __init__(self, d_model: int, n_head: int, dropout: float):
        super().__init__()
        self.attention = nn.MultiheadAttention(d_model, n_head, dropout=dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.linear1 = nn.Linear(d_model, d_model * 4)
        self.linear2 = nn.Linear(d_model * 4, d_model)

    def _cross(self, query: torch.Tensor, diff: torch.Tensor) -> torch.Tensor:
        attn, _ = self.attention(query.contiguous(), diff.contiguous(), diff.contiguous())
        out = self.norm1(query + self.dropout1(attn))
        ff = self.linear2(self.dropout2(F.relu(self.linear1(out))))
        return self.norm2(out + self.dropout3(ff))

    def forward(self, x1: torch.Tensor, x2: torch.Tensor):
        diff = x2 - x1
        return self._cross(x1, diff), self._cross(x2, diff)


class _ResBlock(nn.Module):
    """Bottleneck residual block used to mix the per-layer cross-attention outputs."""

    def __init__(self, channels: int):
        super().__init__()
        mid = channels // 2
        self.body = nn.Sequential(
            nn.Conv2d(channels, mid, kernel_size=1),
            nn.BatchNorm2d(mid),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid, mid, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid, channels, kernel_size=1),
            nn.BatchNorm2d(channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu(self.body(x) + x)


class TransformerFeatureFusion(nn.Module):
    """TFF. Same ``(B, L, F) x 2 -> (L, B, 2 * d_model)`` contract as the rest."""

    def __init__(
        self,
        feature_dim: int,
        d_model: int = 320,
        n_layers: int = 3,
        n_head: int = 16,
        dropout: float = 0.1,
        grid: int = 14,
        **_,
    ):
        super().__init__()
        self.d_model = d_model
        self.grid = grid
        self.projection = nn.Linear(feature_dim, d_model)
        self.pos_embedding = nn.Embedding(grid * grid, d_model)
        self.layers = nn.ModuleList(
            [_CrossTransformer(d_model, n_head, dropout) for _ in range(n_layers)]
        )
        self.res_blocks = nn.ModuleList([_ResBlock(d_model * 2) for _ in range(n_layers)])
        self.norms = nn.ModuleList([nn.LayerNorm(d_model * 2) for _ in range(n_layers)])
        for p in self.parameters():
            if p.dim() > 1:
                xavier_uniform_(p)

    @property
    def out_dim(self) -> int:
        return 2 * self.d_model

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        batch, length, _ = x1.shape
        if x1.size(-1) != self.d_model:
            x1, x2 = self.projection(x1), self.projection(x2)

        pos = self.pos_embedding(torch.arange(length, device=x1.device))
        # (L, B, D): MultiheadAttention here is sequence-first.
        out1 = (x1 + pos).permute(1, 0, 2)
        out2 = (x2 + pos).permute(1, 0, 2)

        fused = x1.new_zeros((length, batch, 2 * self.d_model))
        for layer, res, norm in zip(self.layers, self.res_blocks, self.norms):
            out1, out2 = layer(out1, out2)
            fused = fused + torch.cat([out1, out2], dim=-1)
            fused = fused.permute(1, 2, 0).view(batch, 2 * self.d_model, self.grid, self.grid)
            fused = res(fused).view(batch, 2 * self.d_model, -1).permute(2, 0, 1)
            fused = norm(fused)
        return fused
