#!/usr/bin/env python3
"""Train one fusion module and report strict Recall@K.

    python scripts/train.py --dataset dubai_cc --fusion tbf --epochs 30
    python scripts/train.py --dataset levir_cc --fusion tbf --epochs 30

Defaults match the paper: frozen CLIP ViT-B/16, D=320, 3 layers, 16 heads,
AdamW at 8e-5 with cosine annealing, batch 32, dropout 0.25, 30 epochs.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from bitemporal.backbones import load_backbone
from bitemporal.data import UniqueImageSampler, build_dataset, one_caption_per_pair
from bitemporal.engine import evaluate, train_epoch, validation_loss
from bitemporal.model import BiTemporalRetrieval, trainable_state

DATA_ROOTS = {
    "levir_cc": "data/levir-cc/output_015",
    "dubai_cc": "data/dubai-cc",
}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", choices=list(DATA_ROOTS), default="dubai_cc")
    p.add_argument("--root", default=None, help="dataset root (defaults per dataset)")
    p.add_argument("--fusion", default="tbf")
    p.add_argument("--backbone", default="clip")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=8e-5)
    p.add_argument("--weight-decay", type=float, default=5e-4)
    p.add_argument("--d-model", type=int, default=320)
    p.add_argument("--layers", type=int, default=3)
    p.add_argument("--heads", type=int, default=16)
    p.add_argument("--dropout", type=float, default=0.25)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument(
        "--select",
        choices=("best", "final"),
        default="best",
        help="which weights to evaluate: the epoch with the lowest validation "
        "loss (paper protocol) or the last epoch",
    )
    p.add_argument(
        "--grad-clip",
        type=float,
        default=1.0,
        help="max gradient norm; 0 disables clipping (paper uses 1.0)",
    )
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--out", default=None, help="write final metrics to this JSON path")
    p.add_argument("--checkpoint", default=None, help="save trained weights here")
    args = p.parse_args()

    torch.manual_seed(args.seed)
    root = args.root or DATA_ROOTS[args.dataset]

    image_encoder, text_encoder, meta = load_backbone(args.backbone, args.device)
    train_set = build_dataset(args.dataset, root, "train")
    val_set = build_dataset(args.dataset, root, "val")
    test_set = build_dataset(args.dataset, root, "test")
    loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        sampler=UniqueImageSampler(train_set, seed=args.seed),
        num_workers=args.workers,
        drop_last=True,
    )
    val_loader = DataLoader(
        one_caption_per_pair(val_set),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
    )

    model = BiTemporalRetrieval(
        fusion=args.fusion,
        image_encoder=image_encoder,
        text_encoder=text_encoder,
        feature_dim=meta["feature_dim"],
        text_dim=meta["text_dim"],
        grid=meta["grid"],
        d_model=args.d_model,
        n_layers=args.layers,
        n_head=args.heads,
        dropout=args.dropout,
    ).to(args.device)

    trainable = [p_ for p_ in model.parameters() if p_.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-6
    )

    print(
        f"{args.dataset} | fusion={args.fusion} | {len(train_set.pairs)} train pairs "
        f"| {len(test_set.pairs)} gallery pairs | {sum(p_.numel() for p_ in trainable)/1e6:.2f}M trainable"
    )

    start = time.time()
    best = {"loss": float("inf"), "epoch": 0, "state": None}
    for epoch in range(args.epochs):
        loss = train_epoch(
            model, loader, optimizer, args.device, grad_clip=args.grad_clip or None
        )
        scheduler.step()
        val_loss = validation_loss(model, val_loader, args.device)
        marker = ""
        if val_loss < best["loss"]:
            best.update(loss=val_loss, epoch=epoch + 1, state=trainable_state(model))
            marker = "  *"
        print(
            f"  epoch {epoch + 1:>3}/{args.epochs}  loss {loss:.4f}  "
            f"val {val_loss:.4f}{marker}",
            flush=True,
        )

    if args.select == "best" and best["state"] is not None:
        model.load_state_dict(best["state"], strict=False)
        print(f"\nevaluating best-validation epoch {best['epoch']} (val {best['loss']:.4f})")
    else:
        print(f"\nevaluating final epoch {args.epochs}")

    mins = (time.time() - start) / 60
    results = {}
    n_change = sum(p_["change"] for p_ in test_set.pairs)
    for scope, size in (("full", len(test_set.pairs)), ("change", n_change)):
        recall = evaluate(
            model, test_set, args.batch_size, args.device,
            change_only=True, gallery_scope=scope,
        )
        results[scope] = recall
        print(
            f"\nchange-query Recall@K, {scope} gallery (G={size}): "
            + "  ".join(f"R@{k} {v:.2f}" for k, v in recall.items())
        )
    print(f"\ntrained in {mins:.1f} min")

    if args.checkpoint:
        # Only the trained parts: the frozen backbone is rebuilt from its tag.
        torch.save(
            {"fusion": args.fusion, "backbone": args.backbone,
             "state_dict": trainable_state(model)},
            args.checkpoint,
        )
        print(f"checkpoint -> {args.checkpoint}")

    if args.out:
        Path(args.out).write_text(
            json.dumps(
                {"dataset": args.dataset, "fusion": args.fusion, "seed": args.seed,
                 "epochs": args.epochs, "select": args.select,
                 "best_epoch": best["epoch"], "val_loss": best["loss"],
                 "recall": results}, indent=1
            )
        )


if __name__ == "__main__":
    main()
