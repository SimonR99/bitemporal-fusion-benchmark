"""Bi-temporal fusion for text-based change retrieval.

See https://arxiv.org/abs/2607.28571
"""

from .data import UniqueImageSampler, build_dataset, one_caption_per_pair
from .engine import encode_gallery, encode_queries, evaluate, train_epoch, validation_loss
from .fusion import FUSION_MODULES, build_fusion
from .model import BiTemporalRetrieval, contrastive_loss, trainable_state
from .retrieval import cascade_rank, query_cost_ms, recall_at_k

__all__ = [
    "FUSION_MODULES",
    "build_fusion",
    "BiTemporalRetrieval",
    "contrastive_loss",
    "trainable_state",
    "build_dataset",
    "one_caption_per_pair",
    "UniqueImageSampler",
    "train_epoch",
    "validation_loss",
    "encode_gallery",
    "encode_queries",
    "evaluate",
    "cascade_rank",
    "query_cost_ms",
    "recall_at_k",
]
