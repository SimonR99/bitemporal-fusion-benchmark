"""Training and evaluation loops.

Evaluation mirrors the deployment model: every pair in the split is encoded once
to build the gallery, then each caption is used as a query against it. Only the
pair a caption annotates counts as correct, so Recall@K here is strict
instance-level retrieval.
"""

from __future__ import annotations

import torch
from torch.utils.data import DataLoader

from .model import contrastive_loss
from .retrieval import recall_at_k

__all__ = [
    "train_epoch",
    "validation_loss",
    "encode_gallery",
    "encode_queries",
    "evaluate",
]

GRAD_CLIP = 1.0


def train_epoch(
    model,
    loader: DataLoader,
    optimizer,
    device: str = "cuda",
    grad_clip: float | None = GRAD_CLIP,
) -> float:
    model.train()
    model.image_encoder.eval()
    model.text_encoder.eval()

    trainable = [p for p in model.parameters() if p.requires_grad]
    total, batches = 0.0, 0
    for before, after, captions, _, _ in loader:
        before, after = before.to(device), after.to(device)
        image_emb, text_emb = model(before, after, captions)
        loss = contrastive_loss(image_emb, text_emb, model.logit_scale)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if grad_clip:
            torch.nn.utils.clip_grad_norm_(trainable, grad_clip)
        optimizer.step()
        with torch.no_grad():
            model.logit_scale.clamp_(0, 4.6052)  # ln(100), as in CLIP

        total += loss.item()
        batches += 1
    return total / max(batches, 1)


@torch.no_grad()
def validation_loss(model, loader: DataLoader, device: str = "cuda") -> float:
    """Mean contrastive loss over a validation loader.

    This is the quantity the paper's implementation selects its checkpoint on,
    so it is computed the same way: deterministic order, no augmentation, and a
    single caption per pair, so every in-batch negative is a genuine negative.
    Build the loader with :func:`bitemporal.data.one_caption_per_pair`.
    """
    model.eval()
    total, count = 0.0, 0
    for before, after, captions, _, _ in loader:
        before, after = before.to(device), after.to(device)
        image_emb, text_emb = model(before, after, captions)
        loss = contrastive_loss(image_emb, text_emb, model.logit_scale)
        total += loss.item() * len(captions)
        count += len(captions)
    return total / max(count, 1)


@torch.no_grad()
def encode_gallery(model, dataset, batch_size: int = 64, device: str = "cuda") -> torch.Tensor:
    """Encode every pair once, in ``dataset.pairs`` order."""
    from .data import CAPTIONS_PER_PAIR

    model.eval()
    augment, dataset.augment = dataset.augment, False  # gallery must be deterministic
    try:
        embeddings = []
        for start in range(0, len(dataset.pairs), batch_size):
            rows = range(start, min(start + batch_size, len(dataset.pairs)))
            items = [dataset[r * CAPTIONS_PER_PAIR] for r in rows]
            before = torch.stack([i[0] for i in items]).to(device)
            after = torch.stack([i[1] for i in items]).to(device)
            embeddings.append(model.encode_pair(before, after).cpu())
    finally:
        dataset.augment = augment
    return torch.cat(embeddings)


@torch.no_grad()
def encode_queries(
    model,
    dataset,
    batch_size: int = 64,
    device: str = "cuda",
    change_only: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Encode captions as retrieval queries.

    Returns ``(Q, D)`` text embeddings and the ``(Q,)`` index of the pair each
    caption annotates -- the ground truth, as a position in ``dataset.pairs``.
    """
    model.eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    queries, targets = [], []
    for _, _, captions, pair_idx, change in loader:
        keep = change.bool() if change_only else torch.ones_like(change, dtype=torch.bool)
        if not keep.any():
            continue
        queries.append(model.encode_text([c for c, k in zip(captions, keep.tolist()) if k]).cpu())
        targets.append(pair_idx[keep])
    return torch.cat(queries), torch.cat(targets)


@torch.no_grad()
def evaluate(
    model,
    dataset,
    batch_size: int = 64,
    device: str = "cuda",
    change_only: bool = True,
    gallery_scope: str = "full",
    ks: tuple[int, ...] = (1, 5, 10),
) -> dict[int, float]:
    """Strict Recall@K for change-caption queries.

    Args:
        change_only: restrict *queries* to captions of pairs annotated as
            containing a change.
        gallery_scope: ``"full"`` ranks against every pair in the split
            (1929 on LEVIR-CC, 150 on Dubai-CC) and is the paper's protocol --
            use it to compare against Tables 3 and 4. ``"change"`` ranks only
            against change pairs (964 and 97), a smaller and easier gallery
            that inflates Recall@K by roughly the size ratio; it is reported
            only as a diagnostic and its random baseline differs accordingly.
    """
    model.eval()
    gallery = encode_gallery(model, dataset, batch_size, device).to(device)

    keep_pair = torch.tensor([p["change"] for p in dataset.pairs], dtype=torch.bool)
    if gallery_scope == "change":
        remap = torch.full((len(dataset.pairs),), -1, dtype=torch.long)
        remap[keep_pair] = torch.arange(int(keep_pair.sum()))
        gallery = gallery[keep_pair.to(gallery.device)]
    elif gallery_scope == "full":
        remap = torch.arange(len(dataset.pairs))
    else:
        raise ValueError(f"gallery_scope must be 'full' or 'change', got {gallery_scope!r}")

    query_emb, pair_idx = encode_queries(model, dataset, batch_size, device, change_only)
    ground_truth = remap[pair_idx]
    if (ground_truth < 0).any():
        raise ValueError("change query maps outside the change gallery")
    return recall_at_k(query_emb.to(device), gallery, ground_truth, ks)
