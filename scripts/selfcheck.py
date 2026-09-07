#!/usr/bin/env python3
"""Fast correctness checks. Run after installing, before trusting any number.

    python scripts/selfcheck.py

Needs no dataset. Each check guards a bug that this repository actually had and
that produced plausible-looking numbers rather than an error.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from bitemporal.backbones import load_backbone
from bitemporal.fusion import build_fusion
from bitemporal.profiling import count_parameters

# Table 1 of the paper.
PAPER_PARAMS_M = {
    "subtraction": 0.35, "mlp": 2.71, "concat_transformer": 13.05, "tff": 8.02,
    "tbf": 5.73, "concat_mamba": 8.14, "bottleneck_mamba": 2.58,
    "interleaved_mamba": 2.27,
}


def check_encoder_is_batch_invariant(device: str) -> bool:
    """One image must encode identically whoever it shares a batch with.

    Attention has to run over an image's own patches. If the batch and sequence
    axes are swapped it runs over the batch instead, which raises nothing --
    the shapes still line up -- but makes every embedding depend on its
    neighbours and quietly wrecks retrieval.
    """
    image_encoder, _, _ = load_backbone("clip", device)
    torch.manual_seed(0)
    images = torch.rand(4, 3, 224, 224, device=device)
    with torch.no_grad():
        together = image_encoder(images)
        alone = torch.cat([image_encoder(images[i : i + 1]) for i in range(len(images))])
    # Compared by direction, not absolute difference: batching changes which
    # attention kernel runs, which perturbs values at the 1e-3 level. The bug
    # this guards against is not subtle -- it drove cosine down to 0.36.
    cos = torch.nn.functional.cosine_similarity(together, alone, dim=-1).min().item()
    ok = cos > 0.999
    print(f"{'OK ' if ok else 'FAIL'} encoder batch-invariance: min cosine = {cos:.6f}")
    return ok


def check_fusion_contract(device: str) -> bool:
    """Every module must match Table 1 and return (L, B, out_dim)."""
    ok = True
    x = torch.randn(2, 196, 768, device=device)
    for name, expected in PAPER_PARAMS_M.items():
        module = build_fusion(
            name, feature_dim=768, d_model=320, n_layers=3, n_head=16, grid=14
        ).to(device).eval()
        with torch.no_grad():
            out = module(x, x)
        params = count_parameters(module) / 1e6
        good = tuple(out.shape) == (196, 2, module.out_dim) and abs(params - expected) < 0.005
        ok &= good
        print(f"{'OK ' if good else 'FAIL'} {name:<20} {params:>6.2f}M (paper {expected:.2f}M)"
              f"  out {tuple(out.shape)}")
    return ok


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("note: no CUDA, the Mamba variants cannot be checked\n")
    passed = check_encoder_is_batch_invariant(device)
    print()
    passed &= check_fusion_contract(device)
    print("\nall checks passed" if passed else "\nSOME CHECKS FAILED")
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
