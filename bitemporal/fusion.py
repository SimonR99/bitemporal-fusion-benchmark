"""The eight bi-temporal fusion modules compared in the paper.

Every module has the same contract: it takes the per-patch features of the two
acquisitions of one pair and returns a single fused sequence.

    forward(x1, x2) : (B, L, F) x 2  ->  (L, B, out_dim)

``out_dim`` differs by family -- compression-based modules emit ``bottleneck``
while the rest emit ``2 * d_model`` -- so the model head reads ``out_dim``
rather than assuming a width. This is what makes the modules interchangeable
under a single training recipe.

Note that only ``Subtraction`` consumes the backbone width directly; every other
module first projects both streams to ``d_model``.
"""

from __future__ import annotations

import torch
from torch import nn

__all__ = ["FUSION_MODULES", "build_fusion"]


def _bottleneck_mlp(in_dim: int, out_dim: int) -> nn.Sequential:
    """Concat-then-reduce block shared by TBF and Bottleneck Mamba."""
    return nn.Sequential(
        nn.Linear(in_dim, out_dim),
        nn.ReLU(),
        nn.Linear(out_dim, out_dim),
        nn.LayerNorm(out_dim),
    )


class _MambaStack(nn.Module):
    """Pre-norm-free residual stack of Mamba blocks, shared by all SSM variants."""

    def __init__(self, width: int, n_layers: int, dropout: float):
        super().__init__()
        from mamba_ssm import Mamba  # imported lazily: CUDA-only dependency

        self.layers = nn.ModuleList(
            [Mamba(d_model=width, d_state=16, d_conv=4, expand=2) for _ in range(n_layers)]
        )
        self.norms = nn.ModuleList([nn.LayerNorm(width) for _ in range(n_layers)])
        self.dropouts = nn.ModuleList([nn.Dropout(dropout) for _ in range(n_layers)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer, norm, drop in zip(self.layers, self.norms, self.dropouts):
            x = norm(x + drop(layer(x)))
        return x


def _transformer_stack(width: int, n_layers: int, n_head: int, dropout: float) -> nn.Module:
    layer = nn.TransformerEncoderLayer(
        d_model=width, nhead=n_head, dropout=dropout, batch_first=True
    )
    return nn.TransformerEncoder(layer, num_layers=n_layers)


class FusionModule(nn.Module):
    """Base class: projects both streams to ``d_model`` and fixes the I/O contract.

    Subclasses implement :meth:`fuse`, which receives the projected streams as
    ``(B, L, d_model)`` and returns ``(B, L, out_dim)``.
    """

    def __init__(self, feature_dim: int, d_model: int = 320, project: bool = True):
        super().__init__()
        self.d_model = d_model
        self.projection = nn.Linear(feature_dim, d_model) if project else None

    @property
    def out_dim(self) -> int:
        raise NotImplementedError

    def fuse(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        if self.projection is not None and x1.size(-1) != self.d_model:
            x1, x2 = self.projection(x1), self.projection(x2)
        return self.fuse(x1, x2).transpose(0, 1)


# --------------------------------------------------------------------------- #
# Simple baselines
# --------------------------------------------------------------------------- #


class Subtraction(FusionModule):
    """Element-wise difference followed by a linear projection.

    Operates on backbone features directly (no shared projection), which is why
    it is by far the cheapest module and usable as a cascade prefilter.
    """

    def __init__(self, feature_dim, d_model=320, dropout=0.2, bottleneck=None, **_):
        super().__init__(feature_dim, d_model, project=False)
        self.bottleneck = bottleneck or d_model
        self.mlp = nn.Sequential(
            nn.Linear(feature_dim, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, self.bottleneck),
        )

    @property
    def out_dim(self):
        return self.bottleneck

    def fuse(self, x1, x2):
        return self.mlp(x2 - x1)


class MLPFusion(FusionModule):
    """Channel mixing without token interaction."""

    def __init__(self, feature_dim, d_model=320, n_layers=3, dropout=0.2, **_):
        super().__init__(feature_dim, d_model)
        width = 2 * d_model
        self.blocks = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(width, width),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(width, width),
                    nn.LayerNorm(width),
                )
                for _ in range(n_layers)
            ]
        )

    @property
    def out_dim(self):
        return 2 * self.d_model

    def fuse(self, x1, x2):
        x = torch.cat([x1, x2], dim=-1)
        for block in self.blocks:
            x = block(x) + x
        return x


# --------------------------------------------------------------------------- #
# Attention-based
# --------------------------------------------------------------------------- #


class ConcatTransformer(FusionModule):
    """Channel-wise concatenation, then self-attention at the doubled width."""

    def __init__(self, feature_dim, d_model=320, n_layers=3, n_head=16, dropout=0.1, **_):
        super().__init__(feature_dim, d_model)
        self.transformer = _transformer_stack(2 * d_model, n_layers, n_head, dropout)

    @property
    def out_dim(self):
        return 2 * self.d_model

    def fuse(self, x1, x2):
        return self.transformer(torch.cat([x1, x2], dim=-1))


class TemporalBottleneckFusion(FusionModule):
    """TBF: concat-then-reduce, then self-attention in the compressed space.

    Because projections and feed-forward blocks scale quadratically with width,
    attending at ``bottleneck`` instead of ``2 * d_model`` removes most of the
    concatenation cost while keeping global attention over all patches.
    """

    def __init__(
        self, feature_dim, d_model=320, n_layers=3, n_head=16, dropout=0.1, bottleneck=None, **_
    ):
        super().__init__(feature_dim, d_model)
        self.bottleneck = bottleneck or d_model
        if self.bottleneck % n_head:
            raise ValueError(f"bottleneck {self.bottleneck} must be divisible by n_head {n_head}")
        self.compress = _bottleneck_mlp(2 * d_model, self.bottleneck)
        self.transformer = _transformer_stack(self.bottleneck, n_layers, n_head, dropout)

    @property
    def out_dim(self):
        return self.bottleneck

    def fuse(self, x1, x2):
        return self.transformer(self.compress(torch.cat([x1, x2], dim=-1)))


# --------------------------------------------------------------------------- #
# SSM-based
# --------------------------------------------------------------------------- #


class ConcatMamba(FusionModule):
    """Concatenation baseline with Mamba blocks in place of the Transformer."""

    def __init__(self, feature_dim, d_model=320, n_layers=3, dropout=0.2, **_):
        super().__init__(feature_dim, d_model)
        self.mamba = _MambaStack(2 * d_model, n_layers, dropout)

    @property
    def out_dim(self):
        return 2 * self.d_model

    def fuse(self, x1, x2):
        return self.mamba(torch.cat([x1, x2], dim=-1))


class BottleneckMamba(FusionModule):
    """TBF's compression MLP followed by Mamba blocks at the reduced width."""

    def __init__(self, feature_dim, d_model=320, n_layers=3, dropout=0.2, bottleneck=None, **_):
        super().__init__(feature_dim, d_model)
        self.bottleneck = bottleneck or d_model
        self.compress = _bottleneck_mlp(2 * d_model, self.bottleneck)
        self.mamba = _MambaStack(self.bottleneck, n_layers, dropout)

    @property
    def out_dim(self):
        return self.bottleneck

    def fuse(self, x1, x2):
        return self.mamba(self.compress(torch.cat([x1, x2], dim=-1)))


class InterleavedMamba(FusionModule):
    """Interleaves the two acquisitions into one sequence of length 2L.

    This is the only variant that lengthens the sequence rather than widening
    it, so it is the one that would benefit if the selective scan's linear-time
    advantage materialised at these sequence lengths. It does not (see paper).
    """

    def __init__(self, feature_dim, d_model=320, n_layers=3, dropout=0.2, **_):
        super().__init__(feature_dim, d_model)
        self.mamba = _MambaStack(d_model, n_layers, dropout)

    @property
    def out_dim(self):
        return 2 * self.d_model

    def fuse(self, x1, x2):
        b, length, dim = x1.shape
        x = torch.stack([x1, x2], dim=2).view(b, 2 * length, dim)
        x = self.mamba(x)
        return x.view(b, length, 2 * dim)


FUSION_MODULES = {
    "subtraction": Subtraction,
    "mlp": MLPFusion,
    "concat_transformer": ConcatTransformer,
    "tbf": TemporalBottleneckFusion,
    "concat_mamba": ConcatMamba,
    "bottleneck_mamba": BottleneckMamba,
    "interleaved_mamba": InterleavedMamba,
}


def build_fusion(name: str, feature_dim: int, **kwargs) -> FusionModule:
    """Instantiate a fusion module by name.

    ``tff`` lives in :mod:`bitemporal.tff` because it is a third-party
    architecture; it is registered here so all eight share one entry point.
    """
    if name == "tff":
        from .tff import TransformerFeatureFusion

        return TransformerFeatureFusion(feature_dim, **kwargs)
    if name not in FUSION_MODULES:
        raise KeyError(f"unknown fusion {name!r}; choose from {sorted(FUSION_MODULES) + ['tff']}")
    return FUSION_MODULES[name](feature_dim, **kwargs)
