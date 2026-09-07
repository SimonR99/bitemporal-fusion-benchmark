"""Parameter, FLOP and wall-clock measurement for the fusion modules.

FLOPs follow the convention 1 MAC = 2 FLOPs. ``fvcore`` cannot see through
Mamba's fused CUDA kernel, so SSM blocks are costed analytically with the
selective-scan formula from the Mamba paper and excluded from the traced count.

Latency is deliberately measured at batch 1 under CUDA synchronisation: that is
the per-candidate cost the deployment model in the paper assumes, and it is the
worst case for hardware utilisation. Batched throughput will differ.
"""

from __future__ import annotations

import time
from contextlib import suppress

import torch
from torch import nn

__all__ = ["count_parameters", "count_flops", "measure_latency"]


def count_parameters(module: nn.Module, trainable_only: bool = True) -> int:
    return sum(p.numel() for p in module.parameters() if p.requires_grad or not trainable_only)


def _mamba_block_flops(block: nn.Module, seq_len: int, batch: int = 1) -> int:
    """Analytic cost of one Mamba block, following the reference implementation.

    The selective scan itself is ``9 * B * L * d_inner * d_state``; the rest is
    the surrounding projections, depthwise conv and gating.
    """
    d_model, d_state, d_conv = block.d_model, block.d_state, block.d_conv
    d_inner = block.expand * d_model
    dt_rank = getattr(block, "dt_rank", "auto")
    if not isinstance(dt_rank, int):
        dt_rank = max(1, d_model // 16)

    bl = batch * seq_len
    return (
        2 * bl * d_model * 2 * d_inner          # in projection
        + 2 * batch * d_inner * seq_len * d_conv  # depthwise conv
        + 2 * bl * d_inner * (dt_rank + 2 * d_state)  # x_proj
        + 2 * bl * dt_rank * d_inner            # dt_proj
        + 9 * bl * d_inner * d_state            # selective scan
        + 2 * bl * d_inner * d_model            # out projection
        + 10 * bl * d_inner                     # SiLU x2, softplus, gate
    )


def count_flops(module: nn.Module, inputs: tuple[torch.Tensor, ...]) -> int:
    """Total FLOPs for one forward pass: traced ops plus analytic SSM blocks.

    The two sources must not overlap. fvcore does see the projections and the
    depthwise conv *inside* a Mamba block, and the analytic formula covers the
    whole block, so the traced subtree of every Mamba block is subtracted before
    the analytic cost is added back.
    """
    from fvcore.nn import FlopCountAnalysis

    named_mamba = [(n, m) for n, m in module.named_modules() if type(m).__name__ == "Mamba"]
    seq_len = inputs[0].shape[1]
    batch = inputs[0].shape[0]

    # Not every variant scans L tokens: Interleaved Mamba builds a 2L sequence,
    # so the length is observed rather than assumed.
    seen: dict[str, int] = {}

    def _record(name):
        def hook(_mod, args, _out):
            seen[name] = args[0].shape[1]

        return hook

    handles = [m.register_forward_hook(_record(n)) for n, m in named_mamba]
    try:
        with torch.no_grad():
            module(*inputs)
    finally:
        for h in handles:
            h.remove()

    analytic = sum(
        _mamba_block_flops(m, seen.get(n, seq_len), batch) for n, m in named_mamba
    )

    traced = FlopCountAnalysis(module, inputs)
    traced.unsupported_ops_warnings(False).uncalled_modules_warnings(False)
    with suppress(Exception):
        by_module = traced.by_module()
        # by_module values are subtree totals, so summing the block roots
        # counts each traced Mamba op exactly once.
        traced_mamba = sum(by_module.get(name, 0) for name, _ in named_mamba)
        return 2 * (traced.total() - traced_mamba) + analytic
    return analytic


@torch.no_grad()
def measure_latency(
    module: nn.Module,
    inputs: tuple[torch.Tensor, ...],
    runs: int = 1000,
    warmup: int = 50,
    stat: str = "mean",
) -> float:
    """Wall-clock milliseconds per forward pass.

    ``stat="mean"`` is the paper's protocol: one timed loop of ``runs`` passes
    after ``warmup``, divided by ``runs``. It assumes the GPU is otherwise idle
    -- any other process sharing the device inflates every module, and not
    evenly, which can reorder the table.

    ``stat="min"`` times each pass separately with CUDA events and returns the
    fastest. On a shared GPU that is much closer to the uncontended cost, at the
    price of no longer being the number the paper reports.
    """
    if stat not in ("mean", "min"):
        raise ValueError(f"stat must be 'mean' or 'min', got {stat!r}")

    cuda = inputs[0].is_cuda
    for _ in range(warmup):
        module(*inputs)
    if cuda:
        torch.cuda.synchronize()

    if stat == "min":
        if not cuda:
            times = []
            for _ in range(runs):
                t0 = time.perf_counter()
                module(*inputs)
                times.append((time.perf_counter() - t0) * 1000.0)
            return min(times)
        pairs = [
            (torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
            for _ in range(runs)
        ]
        for begin, end in pairs:
            begin.record()
            module(*inputs)
            end.record()
        torch.cuda.synchronize()
        return min(begin.elapsed_time(end) for begin, end in pairs)

    start = time.perf_counter()
    for _ in range(runs):
        module(*inputs)
    if cuda:
        torch.cuda.synchronize()
    return (time.perf_counter() - start) * 1000.0 / runs
