#!/usr/bin/env python3
"""Latency vs sequence length, for the SSM crossover claim in the paper.

At short L the selective scan is memory-bound and its latency is nearly flat,
while attention grows with L. The two cross somewhere above the bi-temporal
patch count, which is why linear-time complexity does not pay off at L=196 but
does for long multi-temporal stacks.

    python scripts/latency_vs_length.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from bitemporal.fusion import build_fusion
from bitemporal.profiling import measure_latency

MODULES = [("Concat. Transformer", "concat_transformer"), ("TBF", "tbf"),
           ("Interleaved Mamba", "interleaved_mamba")]
LENGTHS = [196, 392, 784, 1568, 3136, 6272]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--lengths", type=int, nargs="+", default=LENGTHS)
    p.add_argument("--feature-dim", type=int, default=768)
    p.add_argument("--d-model", type=int, default=320)
    p.add_argument("--layers", type=int, default=3)
    p.add_argument("--heads", type=int, default=16)
    p.add_argument("--runs", type=int, default=200)
    args = p.parse_args()

    if not torch.cuda.is_available():
        sys.exit("needs CUDA: the Mamba kernel is GPU-only")

    modules = {
        label: build_fusion(
            name,
            feature_dim=args.feature_dim,
            d_model=args.d_model,
            n_layers=args.layers,
            n_head=args.heads,
        ).cuda().eval()
        for label, name in MODULES
    }

    header = f"{'L':>7}" + "".join(f"{label:>22}" for label in modules)
    print(header)
    print("-" * len(header))
    for length in args.lengths:
        x = torch.randn(1, length, args.feature_dim, device="cuda")
        times = [measure_latency(m, (x, x), runs=args.runs, warmup=20) for m in modules.values()]
        best = min(times)
        cells = "".join(f"{t:>20.2f}{'*' if t == best else ' '} " for t in times)
        print(f"{length:>7}{cells}")
    print("\n* fastest at that length")


if __name__ == "__main__":
    main()
