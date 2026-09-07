"""The two-tower retrieval model.

Image tower:  frozen backbone -> patch tokens -> fusion -> pooling -> head
Text tower:   frozen text encoder -> head

The towers stay independent and meet only in a cosine score, which is what lets
the gallery be indexed offline and makes the fusion module the only component
that has to run per candidate pair at query time.

Only the fusion module changes across the eight configurations in the paper;
everything else here is held fixed.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from .fusion import build_fusion

__all__ = ["BiTemporalRetrieval", "contrastive_loss", "trainable_state"]

# state_dict prefixes belonging to the frozen backbone
_FROZEN_PREFIXES = ("image_encoder.", "text_encoder.")

EMBED_DIM = 128


class _Head(nn.Module):
    """Shared projection to the joint embedding space."""

    def __init__(self, in_dim: int, hidden: int, dropout: float = 0.0):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden, EMBED_DIM)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.dropout(torch.relu(self.fc1(x))))


class BiTemporalRetrieval(nn.Module):
    """Scores (before, after, text) triples.

    Args:
        fusion: key into :data:`bitemporal.fusion.FUSION_MODULES` (or ``"tff"``).
        image_encoder: frozen module mapping ``(B, 3, H, W) -> (B, L, feature_dim)``
            patch tokens, CLS already removed.
        text_encoder: frozen callable mapping a list of strings -> ``(B, text_dim)``.
    """

    def __init__(
        self,
        fusion: str,
        image_encoder: nn.Module,
        text_encoder: nn.Module,
        feature_dim: int = 768,
        text_dim: int = 512,
        d_model: int = 320,
        n_layers: int = 3,
        n_head: int = 16,
        dropout: float = 0.25,
        grid: int = 14,
        bottleneck: int | None = None,
    ):
        super().__init__()
        self.image_encoder = image_encoder.requires_grad_(False).eval()
        self.text_encoder = text_encoder.requires_grad_(False).eval()

        self.fusion = build_fusion(
            fusion,
            feature_dim=feature_dim,
            d_model=d_model,
            n_layers=n_layers,
            n_head=n_head,
            dropout=dropout,
            grid=grid,
            bottleneck=bottleneck,
        )
        self.image_head = _Head(self.fusion.out_dim, d_model, dropout)
        self.text_head = _Head(text_dim, 256)
        # Learned temperature, clamped as in CLIP.
        self.logit_scale = nn.Parameter(torch.tensor(2.6593))

    def encode_pair(self, before: torch.Tensor, after: torch.Tensor) -> torch.Tensor:
        """Fuse one bi-temporal pair into a single embedding. Query-independent."""
        with torch.no_grad():
            f1 = self.image_encoder(before).float()
            f2 = self.image_encoder(after).float()
        fused = self.fusion(f1, f2)          # (L, B, out_dim)
        pooled = fused.permute(1, 0, 2).mean(dim=1)
        return self.image_head(pooled)

    def encode_text(self, captions) -> torch.Tensor:
        with torch.no_grad():
            feats = self.text_encoder(captions).float()
        return self.text_head(feats)

    def forward(self, before, after, captions):
        return self.encode_pair(before, after), self.encode_text(captions)


def trainable_state(model: nn.Module) -> dict[str, torch.Tensor]:
    """The parts of ``model`` that actually train, as CPU tensors.

    The two towers hold a frozen CLIP each (the text tower keeps the whole
    model), so a full ``state_dict`` is about a gigabyte of weights that are
    identical in every run. Restore with ``load_state_dict(..., strict=False)``.
    """
    return {
        k: v.detach().cpu().clone()
        for k, v in model.state_dict().items()
        if not k.startswith(_FROZEN_PREFIXES)
    }


def contrastive_loss(
    image_emb: torch.Tensor, text_emb: torch.Tensor, logit_scale: torch.Tensor
) -> torch.Tensor:
    """Symmetric InfoNCE over the in-batch similarity matrix."""
    image_emb = F.normalize(image_emb, dim=-1)
    text_emb = F.normalize(text_emb, dim=-1)
    logits = logit_scale.exp().clamp(max=100.0) * image_emb @ text_emb.t()
    target = torch.arange(len(logits), device=logits.device)
    return (F.cross_entropy(logits, target) + F.cross_entropy(logits.t(), target)) / 2
