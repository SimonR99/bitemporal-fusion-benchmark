"""Frozen vision-language backbones.

Both towers are frozen: the image encoder is run once per frame (offline, in a
real deployment) and the text encoder once per query, so neither contributes to
the per-candidate cost the paper measures.

The image side returns *patch tokens* rather than a pooled embedding, because
the fusion modules operate over patches. The CLS token is dropped.
"""

from __future__ import annotations

import torch
from torch import nn

__all__ = ["load_backbone", "BACKBONES"]

# name -> (open_clip architecture, pretrained tag, patch width, patch grid, input px)
#
# The ``-quickgelu`` suffix is required, not cosmetic: OpenAI trained these
# weights with QuickGELU, and open_clip's plain ``ViT-B-16`` builds the same
# weights with nn.GELU. That mismatch loads without error (open_clip only warns)
# and silently degrades every frozen feature the fusion modules see.
BACKBONES = {
    "clip": ("ViT-B-16-quickgelu", "openai", 768, 14, 224),
    "clip32": ("ViT-B-32-quickgelu", "openai", 768, 7, 224),
}


def _vit_patch_tokens(visual: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """Run an open_clip VisionTransformer transformer stack, keeping every token.

    ``ln_post`` is deliberately NOT applied. In stock CLIP it normalises the CLS
    token just before the projection to the joint embedding space, but the
    fusion modules consume patch tokens instead and the reference implementation
    skips it for exactly that reason. Applying it flattens the per-token
    magnitude the fusion modules rely on, which costs a large amount of
    retrieval accuracy on LEVIR-CC.
    """
    x = x.to(visual.conv1.weight.dtype)
    x = visual.conv1(x).reshape(x.shape[0], visual.conv1.out_channels, -1).permute(0, 2, 1)
    cls = visual.class_embedding.to(x.dtype).expand(x.shape[0], 1, -1)
    x = torch.cat([cls, x], dim=1) + visual.positional_embedding.to(x.dtype)
    x = visual.ln_pre(x)

    # open_clip >= 3 runs its Transformer batch-first; older versions want
    # (L, N, D). Getting this wrong does not raise -- the shapes still line up --
    # it silently swaps the batch and sequence axes, so self-attention runs
    # across the images in a batch instead of across the patches of one image.
    batch_first = getattr(visual.transformer, "batch_first", False)
    if not batch_first:
        x = x.permute(1, 0, 2)
    x = visual.transformer(x)
    if not batch_first:
        x = x.permute(1, 0, 2)
    return x.float()


class ImageEncoder(nn.Module):
    """``(B, 3, H, W) -> (B, L, patch_dim)`` patch tokens, CLS removed."""

    def __init__(self, model: nn.Module, input_px: int):
        super().__init__()
        self.visual = model.visual
        self.input_px = input_px

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        if images.shape[-1] != self.input_px:
            images = nn.functional.interpolate(
                images, size=(self.input_px, self.input_px), mode="bilinear", align_corners=False
            )
        return _vit_patch_tokens(self.visual, images)[:, 1:, :]


class TextEncoder(nn.Module):
    """``list[str] -> (B, embed_dim)``."""

    def __init__(self, model: nn.Module, tokenizer):
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer

    def forward(self, captions) -> torch.Tensor:
        device = next(self.model.parameters()).device
        tokens = self.tokenizer(list(captions)).to(device)
        return self.model.encode_text(tokens).float()


def load_backbone(name: str = "clip", device: str = "cuda"):
    """Return ``(image_encoder, text_encoder, meta)``, both frozen and in eval mode.

    ``meta`` carries ``feature_dim``, ``grid`` and ``text_dim``, which the model
    needs to size the fusion module and the text head.
    """
    import open_clip

    if name not in BACKBONES:
        raise KeyError(f"unknown backbone {name!r}; choose from {sorted(BACKBONES)}")
    arch, tag, feature_dim, grid, input_px = BACKBONES[name]

    model, _, _ = open_clip.create_model_and_transforms(arch, pretrained=tag)
    tokenizer = open_clip.get_tokenizer(arch)
    model = model.to(device).eval().requires_grad_(False)

    image_encoder = ImageEncoder(model, input_px).to(device).eval()
    text_encoder = TextEncoder(model, tokenizer).to(device).eval()
    meta = {
        "feature_dim": feature_dim,
        "grid": grid,
        "seq_len": grid * grid,
        "text_dim": model.text_projection.shape[1],
    }
    return image_encoder, text_encoder, meta
