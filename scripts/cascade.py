#!/usr/bin/env python3
"""Reproduce Table 4: two-stage retrieval against full fusion.

Stage 1 ranks the whole gallery with a cheap module (Subtraction); stage 2
re-ranks only the top-N shortlist with an accurate one. Both checkpoints must
come from the same training seed -- the cascade reuses trained models as they
are, so there is nothing to fit here.

    python scripts/train.py --dataset levir_cc --fusion subtraction --seed 1 \\
        --checkpoint runs/subtraction_seed1.pt
    python scripts/train.py --dataset levir_cc --fusion tbf --seed 1 \\
        --checkpoint runs/tbf_seed1.pt
    python scripts/cascade.py --dataset levir_cc \\
        --prefilter runs/subtraction_seed1.pt --reranker runs/tbf_seed1.pt

Queries are change captions; the gallery is the full test split, which is the
protocol behind Table 3 and Table 4 alike.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from bitemporal.backbones import load_backbone
from bitemporal.data import build_dataset
from bitemporal.engine import encode_gallery, encode_queries
from bitemporal.model import BiTemporalRetrieval
from bitemporal.retrieval import cascade_rank, query_cost_ms

DATA_ROOTS = {
    "levir_cc": "data/levir-cc/output_015",
    "dubai_cc": "data/dubai-cc",
}

# Per-candidate fusion latency (ms) from Table 1, used for the cost column.
# Measure your own with scripts/benchmark_efficiency.py and pass --latency.
PAPER_LATENCY_MS = {
    "subtraction": 0.04,
    "mlp": 0.20,
    "concat_transformer": 0.76,
    "tff": 1.90,
    "tbf": 0.47,
    "concat_mamba": 0.58,
    "bottleneck_mamba": 0.62,
    "interleaved_mamba": 0.60,
}


def load_model(path: str, backbone, meta, args) -> tuple[torch.nn.Module, str]:
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    image_encoder, text_encoder = backbone
    model = BiTemporalRetrieval(
        fusion=ckpt["fusion"],
        image_encoder=image_encoder,
        text_encoder=text_encoder,
        feature_dim=meta["feature_dim"],
        text_dim=meta["text_dim"],
        grid=meta["grid"],
        d_model=args.d_model,
        n_layers=args.layers,
        n_head=args.heads,
    ).to(args.device)
    # strict=False: checkpoints hold only the trained parts, the frozen
    # backbone having just been rebuilt from its tag.
    model.load_state_dict(ckpt["state_dict"], strict=False)
    return model.eval(), ckpt["fusion"]


def recall(ranking: torch.Tensor, ground_truth: torch.Tensor, ks) -> dict[int, float]:
    hits = ranking == ground_truth.to(ranking.device).unsqueeze(1)
    return {k: hits[:, :k].any(dim=1).float().mean().item() * 100 for k in ks}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", choices=list(DATA_ROOTS), default="levir_cc")
    p.add_argument("--root", default=None)
    p.add_argument("--prefilter", required=True, help="cheap stage-1 checkpoint")
    p.add_argument("--reranker", required=True, help="accurate stage-2 checkpoint")
    p.add_argument("--shortlist", type=int, nargs="+",
                   default=[10, 25, 50, 100, 250, 500])
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--d-model", type=int, default=320)
    p.add_argument("--layers", type=int, default=3)
    p.add_argument("--heads", type=int, default=16)
    p.add_argument("--backbone", default="clip")
    p.add_argument("--latency", type=json.loads, default=None,
                   help='override per-pair latency, e.g. \'{"tbf":0.5}\' (ms)')
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--out", default=None, help="write the table to this JSON path")
    args = p.parse_args()

    latency = dict(PAPER_LATENCY_MS, **(args.latency or {}))
    root = args.root or DATA_ROOTS[args.dataset]
    image_encoder, text_encoder, meta = load_backbone(args.backbone, args.device)
    test_set = build_dataset(args.dataset, root, "test")

    ks = (1, 5, 10)
    stage1_model, stage1_name = load_model(args.prefilter, (image_encoder, text_encoder), meta, args)
    prefilter = encode_gallery(stage1_model, test_set, args.batch_size, args.device).to(args.device)
    query_emb, ground_truth = encode_queries(
        stage1_model, test_set, args.batch_size, args.device, change_only=True
    )
    del stage1_model

    stage2_model, stage2_name = load_model(args.reranker, (image_encoder, text_encoder), meta, args)
    reranker = encode_gallery(stage2_model, test_set, args.batch_size, args.device).to(args.device)
    # Stage 2 has its own text head, so the queries must be re-encoded with it.
    query_emb2, ground_truth2 = encode_queries(
        stage2_model, test_set, args.batch_size, args.device, change_only=True
    )
    del stage2_model
    if not torch.equal(ground_truth, ground_truth2):
        raise RuntimeError("query order differs between the two checkpoints")

    gallery_size = len(test_set.pairs)
    t_pre, t_fuse = latency[stage1_name], latency[stage2_name]
    print(
        f"{args.dataset}: G={gallery_size}  queries={len(ground_truth)}  "
        f"stage1={stage1_name} ({t_pre} ms/pair)  stage2={stage2_name} ({t_fuse} ms/pair)\n"
    )

    full_cost = query_cost_ms(gallery_size, t_pre, t_fuse, shortlist=None)
    rows = []

    def emit(label, ranking, cost):
        r = recall(ranking, ground_truth, ks)
        rows.append({"stage": label, **{f"R@{k}": round(v, 2) for k, v in r.items()},
                     "ms_per_query": round(cost, 1),
                     "speedup": round(full_cost / cost, 1)})
        print(f"{label:>8} | " + "  ".join(f"R@{k} {r[k]:5.2f}" for k in ks)
              + f" | {cost:8.1f} ms  {full_cost / cost:5.1f}x")

    q1, q2 = query_emb.to(args.device), query_emb2.to(args.device)

    def rank_alone(query, gallery):
        return cascade_rank(query, gallery, query, gallery, shortlist=gallery.shape[0])

    emit("stage1", rank_alone(q1, prefilter),
         query_cost_ms(gallery_size, t_pre, t_fuse, shortlist=0))

    for n in args.shortlist:
        if n >= gallery_size:
            continue
        emit(str(n), cascade_rank(q1, prefilter, q2, reranker, shortlist=n),
             query_cost_ms(gallery_size, t_pre, t_fuse, shortlist=n))

    emit("full", rank_alone(q2, reranker), full_cost)

    if args.out:
        Path(args.out).write_text(json.dumps(
            {"dataset": args.dataset, "gallery_size": gallery_size,
             "stage1": stage1_name, "stage2": stage2_name, "rows": rows}, indent=1))


if __name__ == "__main__":
    main()
