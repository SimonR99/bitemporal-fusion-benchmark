# Bi-Temporal Fusion Benchmark

Code and split indices for **"Finding Change in Satellite Archives from Text:
How to Combine Before-and-After Images Efficiently"** (MACLEAN @ ECML/PKDD 2026).

Text-based change retrieval searches an archive of before-and-after satellite
image pairs and ranks each pair by how well it matches a natural-language
description of the change. The *fusion* module — the component that merges the
two acquisitions into one visual embedding — runs at query time over every
candidate pair, so its cost largely sets the cost of every search.

This repository holds a controlled comparison of eight ways to build that
module, under one frozen backbone and one training recipe.

## Findings

1. **A training-free cascade cuts query cost 10–15×.** Rank the whole gallery
   with a cheap difference model, then re-rank only the top-25 with attention.
   On LEVIR-CC this matches or improves full-fusion recall, and the saving grows
   with gallery size toward the per-pair cost ratio of the two stages.
2. **Mamba's linear-time scan does not pay off at L=196.** The selective scan is
   memory-bandwidth-bound while attention maps onto parallel hardware. The
   crossover sits near two frames' worth of patches; SSMs win for long
   multi-temporal stacks, not bi-temporal tiles.
3. **Bottleneck compression is cheap but not free.** TBF cuts parameters 2.3×
   and latency 1.6× for 0.007 change-only BLEU-1. Tighter compression degrades
   change-only quality significantly while pooled metrics stay flat — aggregate
   reporting hides the loss that matters.

Full numbers: [`results/paper_tables.md`](results/paper_tables.md).

## The split

The paper uses a retrieval-oriented LEVIR-CC split: all 5,038 change pairs are
kept and no-change pairs are subsampled to 2,142.

| Split | Pairs | Change | No-change |
|---|---:|---:|---:|
| train | 3,918 | 3,407 (87.0%) | 511 |
| val | 1,333 | 667 (50.0%) | 666 |
| test | 1,929 | 964 (50.0%) | 965 |
| **total** | **7,180** | **5,038 (70.2%)** | **2,142** |

The 70/30 ratio is pooled; validation and test stay balanced. Indices are in
[`splits/levir_cc/`](splits/levir_cc/) as `{"file", "change"}` records, so the
protocol is reproducible even though it is not comparable with published
change-captioning splits.

## Install

```bash
pip install -r requirements.txt
```

`mamba-ssm` needs a CUDA GPU; the attention and baseline modules run on CPU.

## Quick start

Reproduce the efficiency table without any dataset — it runs on random features:

```bash
python scripts/benchmark_efficiency.py           # params + latency
python scripts/latency_vs_length.py              # the SSM crossover
```

Parameter counts match the paper exactly. Latency depends on your GPU and its
load; the SSM rows move the most, since they are bandwidth-bound.

## Training

Check the install first — this needs no dataset and takes seconds:

```bash
python scripts/selfcheck.py
```

It verifies the frozen encoder is batch-composition invariant and that all eight
fusion modules match the paper's parameter counts. Both guard bugs that produced
plausible-looking numbers instead of errors.

Then get the data — see [`data/README.md`](data/README.md). Dubai-CC needs no
preprocessing and trains in minutes, so start there:

```bash
python scripts/train.py --dataset dubai_cc --fusion tbf --epochs 30
python scripts/train.py --dataset levir_cc --fusion concat_transformer --epochs 30
```

`--fusion` takes any of the eight names. Defaults match the paper: frozen CLIP
ViT-B/16, D=320, 3 layers, 16 heads, AdamW at 8e-5 with cosine annealing,
gradient clipping at 1.0, batch 32, dropout 0.25, 30 epochs.

Validation loss is computed every epoch and the weights evaluated are those of
the best epoch, not the last (`--select final` overrides). This matters: on
LEVIR-CC validation loss bottoms out well before epoch 30 while training loss
keeps falling, so the final epoch is not the best one.

A single Dubai-CC run takes about a minute and reproduces the paper:

| Change queries, TBF, 30 epochs | R@1 | R@5 | R@10 | time |
|---|---:|---:|---:|---:|
| Dubai-CC paper (10 seeds) | 4.82 ± 1.17 | 19.73 ± 1.63 | 33.63 ± 1.74 | — |
| Dubai-CC this repo (seed 1) | 4.95 | 20.41 | 32.99 | 1.1 min |
| LEVIR-CC paper (10 seeds) | 2.06 ± 0.42 | 8.29 ± 0.94 | 14.41 ± 1.39 | — |
| LEVIR-CC this repo (seed 1) | 0.77 | 3.86 | 7.51 | 16.7 min |

Dubai-CC lands within half a standard deviation of the paper. **LEVIR-CC does
not yet reproduce** — it learns (15× the random baseline) but sits ~2.5× low;
see [`results/paper_tables.md`](results/paper_tables.md) for the remaining known
differences. The paper reports mean ± std over seeds 1–10, so expect a single
run to land inside that spread rather than on the mean.

Build a model directly:

```python
from bitemporal.model import BiTemporalRetrieval, contrastive_loss

model = BiTemporalRetrieval(
    fusion="tbf",                # or subtraction | mlp | concat_transformer |
    image_encoder=clip_visual,   #    tff | concat_mamba | bottleneck_mamba |
    text_encoder=clip_text,      #    interleaved_mamba
    feature_dim=768, d_model=320,
)
img, txt = model(before, after, captions)
loss = contrastive_loss(img, txt, model.logit_scale)
```

Apply the cascade to any two trained modules. Each stage has its own text head,
so each brings its own query embedding as well as its own gallery:

```python
from bitemporal.retrieval import cascade_rank, recall_at_k, query_cost_ms

ranking = cascade_rank(sub_text, sub_gallery, tbf_text, tbf_gallery, shortlist=25)
query_cost_ms(gallery_size=1929, prefilter_ms=0.04, fusion_ms=0.47, shortlist=25)
```

`scripts/cascade.py` does this end to end from two checkpoints of the same seed:

```bash
python scripts/train.py --dataset levir_cc --fusion subtraction --seed 1 \
    --checkpoint runs/subtraction_seed1.pt
python scripts/train.py --dataset levir_cc --fusion tbf --seed 1 \
    --checkpoint runs/tbf_seed1.pt
python scripts/cascade.py --dataset levir_cc \
    --prefilter runs/subtraction_seed1.pt --reranker runs/tbf_seed1.pt
```

## Layout

```
bitemporal/
  fusion.py      seven fusion modules + registry
  tff.py         TFF (Text-ITSR), ported from RSICCFormer
  model.py       two-tower retrieval model + InfoNCE
  backbones.py   frozen CLIP image/text encoders (open_clip)
  data.py        LEVIR-CC and Dubai-CC datasets
  engine.py      train loop, gallery encoding, Recall@K evaluation
  retrieval.py   Recall@K, cascade, query-cost model
  profiling.py   parameters, FLOPs, latency
scripts/
  selfcheck.py              fast correctness checks, no dataset needed
  train.py                  train one fusion module and evaluate
  cascade.py                Table 4, from two checkpoints of one seed
  benchmark_efficiency.py   Table 1
  latency_vs_length.py      the SSM crossover
  prepare_levir.py          build the LEVIR-CC bundle from a download
splits/levir_cc/ the retrieval-oriented split
data/            datasets (gitignored except the README, see data/README.md)
results/
  paper_tables.md  reference numbers from the paper
  reproduction.md  what this repository regenerates, and how closely
  repro/           metrics from the runs behind reproduction.md
```

All eight modules share one contract — `(B, L, F) × 2 → (L, B, out_dim)` — which
is what lets them be swapped under a fixed recipe. Compression-based modules
emit `bottleneck`, the rest emit `2·d_model`; the model head reads `out_dim`
rather than assuming a width.

## Citation

```bibtex
@misc{roy2026findingchangesatellitearchives,
      title={Finding Change in Satellite Archives from Text: How to Combine Before-and-After Images Efficiently},
      author={Simon Roy and Mark Bong and Giovanni Beltrame},
      year={2026},
      eprint={2607.28571},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2607.28571},
}
```

## License

MIT — see [`LICENSE`](LICENSE).

The datasets are not redistributed here and keep their original terms; see
[`data/README.md`](data/README.md) for where to get them.

## Acknowledgements

TFF is ported from [RSICCFormer](https://github.com/Chen-Yang-Liu/RSICC)
(Liu et al., IEEE TGRS 2022). LEVIR-CC is from the same work; Dubai-CC is from
Hoxha et al., IEEE TGRS 2022.
