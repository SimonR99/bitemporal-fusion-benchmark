#!/usr/bin/env python3
"""Reproduce Table 1: parameters, FLOPs and latency of the eight fusion modules.

Needs no dataset -- it runs on random features of the right shape, so it is the
quickest way to check the environment and the efficiency claims.

    python scripts/benchmark_efficiency.py
    python scripts/benchmark_efficiency.py --seq-len 392 --runs 200
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from bitemporal.fusion import build_fusion
from bitemporal.profiling import count_flops, count_parameters, measure_latency

# Paper ordering: baselines, attention, SSM.
MODULES = [
    ("Subtraction", "subtraction"),
    ("MLP Fusion", "mlp"),
    ("Concat. Transformer", "concat_transformer"),
    ("TFF (Text-ITSR)", "tff"),
    ("TBF", "tbf"),
    ("Concat. Mamba", "concat_mamba"),
    ("Bottleneck Mamba", "bottleneck_mamba"),
    ("Interleaved Mamba", "interleaved_mamba"),
]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--feature-dim", type=int, default=768, help="backbone patch width")
    p.add_argument("--d-model", type=int, default=320, help="fusion width D")
    p.add_argument("--seq-len", type=int, default=196, help="patch count L")
    p.add_argument("--layers", type=int, default=3)
    p.add_argument("--heads", type=int, default=16)
    p.add_argument("--runs", type=int, default=1000)
    p.add_argument("--warmup", type=int, default=50)
    p.add_argument(
        "--stat",
        choices=("mean", "min"),
        default="mean",
        help="mean over the timed loop (the paper's protocol, assumes an idle "
        "GPU) or the fastest single pass (use this if the GPU is shared)",
    )
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument(
        "--flops",
        action="store_true",
        help="also estimate FLOPs (coarse: traced ops x2 plus analytic SSM cost; "
        "the paper's Table 1 used per-operator handlers, so values differ)",
    )
    args = p.parse_args()

    grid = int(round(args.seq_len**0.5))
    if grid * grid != args.seq_len:
        print(f"note: L={args.seq_len} is not square; TFF needs a square patch grid, skipping it")

    x = torch.randn(1, args.seq_len, args.feature_dim, device=args.device)
    print(f"device={args.device}  L={args.seq_len}  D={args.d_model}  "
          f"layers={args.layers}  heads={args.heads}  runs={args.runs}  "
          f"stat={args.stat}\n")
    cols = f"{'Model':<22}{'Params (M)':>12}{'Latency (ms)':>14}"
    print(cols + (f"{'FLOPs (G)':>12}" if args.flops else ""))
    print("-" * (len(cols) + (12 if args.flops else 0)))

    for label, name in MODULES:
        if name == "tff" and grid * grid != args.seq_len:
            continue
        if name.endswith("mamba") and args.device == "cpu":
            print(f"{label:<22}{'(needs CUDA)':>38}")
            continue

        module = build_fusion(
            name,
            feature_dim=args.feature_dim,
            d_model=args.d_model,
            n_layers=args.layers,
            n_head=args.heads,
            grid=grid,
        ).to(args.device).eval()

        params = count_parameters(module) / 1e6
        latency = measure_latency(
            module, (x, x), runs=args.runs, warmup=args.warmup, stat=args.stat
        )
        row = f"{label:<22}{params:>12.2f}{latency:>14.2f}"
        if args.flops:
            row += f"{count_flops(module, (x, x)) / 1e9:>12.2f}"
        print(row)


if __name__ == "__main__":
    main()
