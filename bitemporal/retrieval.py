"""Retrieval scoring: strict Recall@K and the two-stage cascade.

The cascade is the paper's most reusable result. It needs no additional
training: a cheap difference model ranks the whole gallery, then an accurate
fusion module re-ranks only the top-N shortlist. On LEVIR-CC this matches or
improves full-fusion recall at 10-15x lower query cost, and the saving grows
with gallery size toward the per-pair cost ratio of the two stages.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

__all__ = ["recall_at_k", "cascade_rank", "query_cost_ms"]


def _similarity(query_emb: torch.Tensor, gallery_emb: torch.Tensor) -> torch.Tensor:
    return F.normalize(query_emb, dim=-1) @ F.normalize(gallery_emb, dim=-1).t()


def recall_at_k(
    query_emb: torch.Tensor,
    gallery_emb: torch.Tensor,
    ground_truth: torch.Tensor,
    ks: tuple[int, ...] = (1, 5, 10),
) -> dict[int, float]:
    """Strict instance-level Recall@K, in percent.

    Args:
        query_emb: ``(Q, D)`` text embeddings.
        gallery_emb: ``(G, D)`` fused pair embeddings.
        ground_truth: ``(Q,)`` index into the gallery of the single correct pair.

    Only the annotated pair counts as correct, so these numbers are a lower
    bound on practical utility when several pairs share a change description.
    """
    ranking = _similarity(query_emb, gallery_emb).argsort(dim=1, descending=True)
    hits = ranking == ground_truth.to(ranking.device).unsqueeze(1)
    return {k: hits[:, :k].any(dim=1).float().mean().item() * 100 for k in ks}


def cascade_rank(
    prefilter_query: torch.Tensor,
    prefilter_gallery: torch.Tensor,
    reranker_query: torch.Tensor,
    reranker_gallery: torch.Tensor,
    shortlist: int = 25,
) -> torch.Tensor:
    """Rank the gallery with a cheap stage 1, then re-rank its top-N with stage 2.

    The two stages are *separately trained models*, each with its own text head,
    so each stage brings its own query embedding as well as its own gallery.
    Scoring both stages with one text embedding silently compares vectors from
    two different spaces.

    Args:
        prefilter_query: ``(Q, D)`` queries encoded by the cheap module.
        prefilter_gallery: ``(G, D)`` gallery under the cheap module (e.g. Subtraction).
        reranker_query: ``(Q, D)`` the same queries encoded by the accurate module.
        reranker_gallery: ``(G, D)`` gallery under the accurate module.
        shortlist: N, the number of candidates handed to stage 2.

    Returns:
        ``(Q, G)`` gallery indices, best first. Shortlisted candidates are
        ordered by the re-ranker and placed above the stage-1 remainder, so the
        result is a full ranking and Recall@K stays comparable to full fusion.
    """
    stage1 = _similarity(prefilter_query, prefilter_gallery).argsort(dim=1, descending=True)
    top, rest = stage1[:, :shortlist], stage1[:, shortlist:]

    rerank_scores = _similarity(reranker_query, reranker_gallery)
    shortlist_scores = rerank_scores.gather(1, top)
    order = shortlist_scores.argsort(dim=1, descending=True)
    return torch.cat([top.gather(1, order), rest], dim=1)


def query_cost_ms(
    gallery_size: int, prefilter_ms: float, fusion_ms: float, shortlist: int | None = None
) -> float:
    """Per-query fusion cost under the single-stream model of the paper.

    ``shortlist=None`` gives the full-fusion cost ``G * fusion_ms``; otherwise
    ``G * prefilter_ms + N * fusion_ms``. Batching a real gallery scan lowers
    both terms, so treat these as an upper bound and the ratio as the claim.
    """
    if shortlist is None:
        return gallery_size * fusion_ms
    return gallery_size * prefilter_ms + shortlist * fusion_ms
