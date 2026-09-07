# Reference numbers

Everything below is from the paper (LEVIR-CC and Dubai-CC, mean ± std over 10
seeds, frozen CLIP ViT-B/16, D=320, 3 layers, 16 heads). Reproduced here so
results can be compared without re-deriving them from the PDF.

## Table 1 — Efficiency at L=196

Latency is a single fused forward pass at batch 1, averaged over 1,000 runs on
an RTX 4070. Bold marks the best among the six learned modules.

| Model | Params (M) | FLOPs (G) | Latency (ms) |
|---|---:|---:|---:|
| Subtraction | 0.35 | 0.14 | 0.04 |
| MLP Fusion | 2.71 | 1.06 | 0.20 |
| Concat. Transformer | 13.05 | 5.42 | 0.76 |
| TFF (Text-ITSR) | 8.02 | 2.65 | 1.90 |
| TBF | 5.73 | 2.41 | **0.47** |
| Concat. Mamba | 8.14 | 3.28 | 0.58 |
| Bottleneck Mamba | 2.58 | **1.06** | 0.62 |
| Interleaved Mamba | **2.27** | 1.77 | 0.60 |

`scripts/benchmark_efficiency.py` reproduces the parameter counts exactly.
Latency is hardware- and load-dependent; the SSM rows are the most sensitive,
since the selective scan is memory-bandwidth-bound. FLOPs used per-operator
handlers not reproduced here — the script's `--flops` estimate is coarser.

## Table 2 — Caption-similarity metrics

Averaged over both retrieval directions. The change-only regime is the
diagnostic one: the full set also contains no-change queries, whose captions are
near-identical across pairs, so their text-to-image score is near-degenerate and
inflates full-set values.

**This table has no reproduction path in this repository.** It needs the
caption-similarity evaluator (BLEU/METEOR/ROUGE-L over retrieved captions),
which was not ported; the numbers are reproduced here as reference only. The
retrieval and efficiency tables below do have scripts.

### Full set

| Model | LEVIR B-1 | LEVIR B-4 | LEVIR MET | LEVIR R-L | Dubai B-1 | Dubai B-4 | Dubai MET | Dubai R-L |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Subtraction | 0.631±0.007 | 0.326±0.004 | 0.294±0.004 | 0.573±0.008 | 0.575±0.011 | 0.270±0.014 | 0.274±0.008 | 0.514±0.011 |
| MLP Fusion | 0.658±0.020 | 0.352±0.023 | 0.307±0.012 | 0.610±0.026 | 0.576±0.010 | 0.268±0.010 | 0.266±0.009 | 0.500±0.015 |
| Concat. Transformer | **0.684±0.017** | **0.377±0.018** | **0.323±0.012** | **0.638±0.027** | 0.581±0.012 | 0.279±0.010 | 0.276±0.009 | 0.512±0.017 |
| TFF (Text-ITSR) | 0.662±0.025 | 0.353±0.029 | 0.310±0.024 | 0.617±0.040 | 0.581±0.024 | 0.276±0.022 | 0.271±0.015 | 0.508±0.030 |
| TBF | 0.662±0.030 | 0.357±0.030 | 0.308±0.018 | 0.610±0.043 | 0.573±0.011 | 0.268±0.010 | 0.270±0.009 | 0.507±0.015 |
| Concat. Mamba | 0.661±0.017 | 0.355±0.017 | 0.309±0.013 | 0.612±0.028 | **0.585±0.018** | **0.280±0.017** | **0.277±0.011** | **0.518±0.020** |
| Bottleneck Mamba | 0.656±0.014 | 0.354±0.012 | 0.308±0.006 | 0.607±0.018 | 0.573±0.012 | 0.274±0.011 | 0.270±0.009 | 0.505±0.014 |
| Interleaved Mamba | 0.648±0.014 | 0.344±0.013 | 0.305±0.008 | 0.602±0.019 | 0.579±0.011 | 0.273±0.016 | 0.275±0.010 | 0.514±0.014 |

### Change-only subset

| Model | LEVIR B-1 | LEVIR B-4 | LEVIR MET | LEVIR R-L | Dubai B-1 | Dubai B-4 | Dubai MET | Dubai R-L |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Subtraction | 0.640±0.003 | 0.249±0.002 | 0.251±0.002 | 0.452±0.003 | 0.574±0.013 | 0.254±0.014 | 0.259±0.009 | 0.471±0.017 |
| MLP Fusion | 0.640±0.009 | 0.250±0.012 | 0.249±0.006 | 0.449±0.009 | 0.584±0.011 | 0.261±0.013 | 0.254±0.007 | 0.472±0.014 |
| Concat. Transformer | **0.655±0.004** | **0.267±0.003** | **0.259±0.002** | **0.464±0.004** | 0.585±0.015 | 0.270±0.015 | 0.263±0.007 | 0.483±0.016 |
| TFF (Text-ITSR) | 0.640±0.007 | 0.255±0.006 | 0.254±0.003 | 0.452±0.004 | 0.580±0.022 | 0.263±0.020 | 0.258±0.010 | 0.469±0.023 |
| TBF | 0.648±0.009 | 0.259±0.010 | 0.255±0.005 | 0.459±0.010 | 0.576±0.010 | 0.257±0.009 | 0.256±0.006 | 0.469±0.009 |
| Concat. Mamba | 0.645±0.003 | 0.255±0.006 | 0.251±0.003 | 0.453±0.003 | **0.592±0.013** | **0.272±0.017** | **0.263±0.011** | **0.484±0.014** |
| Bottleneck Mamba | 0.643±0.003 | 0.255±0.004 | 0.252±0.003 | 0.452±0.004 | 0.577±0.014 | 0.263±0.014 | 0.256±0.008 | 0.467±0.012 |
| Interleaved Mamba | 0.639±0.005 | 0.253±0.006 | 0.252±0.003 | 0.453±0.005 | 0.586±0.013 | 0.262±0.022 | 0.260±0.011 | 0.475±0.016 |

## Table 3 — Strict Recall@K, change queries over the full gallery

LEVIR-CC gallery G=1929 (4,820 change-caption queries from 964 change pairs);
Dubai-CC gallery G=150. The random baseline reflects gallery size.

| Model | LEVIR R@1 | LEVIR R@5 | LEVIR R@10 | Dubai R@1 | Dubai R@5 | Dubai R@10 |
|---|---:|---:|---:|---:|---:|---:|
| Random | 0.05 | 0.26 | 0.52 | 0.67 | 3.33 | 6.67 |
| Subtraction | 1.88±0.17 | 7.93±0.25 | 13.66±0.39 | 3.11±0.82 | 14.70±2.38 | 26.33±2.79 |
| MLP Fusion | 1.93±0.26 | 7.85±0.69 | 13.66±1.23 | 4.82±1.05 | 19.51±1.97 | 33.38±2.21 |
| Concat. Transformer | **2.28±0.26** | **8.91±0.81** | **15.27±1.15** | 5.20±1.02 | 21.01±1.70 | 35.81±2.20 |
| TFF (Text-ITSR) | 2.08±0.22 | 8.00±0.40 | 13.77±0.58 | 4.21±0.70 | 18.97±2.56 | 33.44±3.38 |
| TBF | 2.06±0.42 | 8.29±0.94 | 14.41±1.39 | 4.82±1.17 | 19.73±1.63 | 33.63±1.74 |
| Concat. Mamba | 1.90±0.23 | 7.80±0.43 | 13.38±0.69 | **5.32±1.61** | **22.10±2.09** | **36.33±2.02** |
| Bottleneck Mamba | 2.05±0.15 | 8.32±0.27 | 14.33±0.40 | 4.97±0.55 | 19.20±2.05 | 33.07±2.27 |
| Interleaved Mamba | 2.05±0.27 | 8.07±0.55 | 13.75±0.74 | 4.58±1.16 | 19.63±1.64 | 32.54±1.33 |

## Table 4 — Cascaded retrieval on LEVIR-CC

Per-query cost is the single-stream fusion cost `G·t_sub + N·t_fuse` from Table 1.

| Re-ranker | N | R@1 | R@5 | R@10 | ms/query | speedup |
|---|---:|---:|---:|---:|---:|---:|
| TBF | 25 | **2.33±0.34** | **8.91±0.58** | **15.17±0.67** | 89 | 10.2× |
| TBF | full | 2.06±0.42 | 8.29±0.94 | 14.41±1.39 | 907 | 1.0× |
| Concat. Transf. | 25 | **2.43±0.31** | **9.33±0.53** | **15.53±0.53** | 96 | 15.2× |
| Concat. Transf. | full | 2.28±0.26 | 8.91±0.81 | 15.27±1.15 | 1466 | 1.0× |

## Latency vs sequence length

Batch 1, RTX 4070. Reproduce with `scripts/latency_vs_length.py`.

| L | Concat. Transformer | TBF | Interleaved Mamba |
|---:|---:|---:|---:|
| 196 | 0.81 | **0.48** | 0.69 |
| 392 | 1.48 | 0.94 | **0.67** |
| 784 | 2.99 | 2.32 | **1.07** |
| 1568 | 7.51 | 7.07 | **2.09** |
| 3136 | 22.77 | 23.84 | **4.06** |
| 6272 | 73.02 | 85.28 | **7.55** |

The SSM overtakes the best attention module at roughly two frames' worth of
patches, and is ~10× faster by L=6272 — long multi-temporal stacks, not the
bi-temporal tile sizes this paper targets.

## Reproduction with this repository

Single seed, `scripts/train.py` defaults (30 epochs), frozen CLIP ViT-B/16,
RTX 4070. The paper reports mean ± std over seeds 1–10.

| Change queries, TBF, 30 epochs | R@1 | R@5 | R@10 | time |
|---|---:|---:|---:|---:|
| **Dubai-CC** paper (10 seeds), G=150 | 4.82 ± 1.17 | 19.73 ± 1.63 | 33.63 ± 1.74 | — |
| **Dubai-CC** this repo (seed 1) | 4.95 | 20.41 | 32.99 | 1.1 min |
| **LEVIR-CC** paper (10 seeds), G=1929 | 2.06 ± 0.42 | 8.29 ± 0.94 | 14.41 ± 1.39 | — |
| **LEVIR-CC** this repo (seed 1) | 0.77 | 3.86 | 7.51 | 16.7 min |
| *random baseline*, G=1929 | *0.05* | *0.26* | *0.52* | — |

Dubai-CC reproduces: all three metrics land within half a standard deviation.

**LEVIR-CC does not yet reproduce.** The model clearly learns — R@1 is 15x the
random baseline and the loss was still falling at epoch 30 — but it lands about
2.5x below the paper. Two known differences remain, neither yet tested:

- The reference implementation evaluates the checkpoint with the best
  *validation* loss; `scripts/train.py` evaluates the final epoch. With 3,918
  training pairs and no early stopping, the final epoch need not be the best.
- LEVIR-CC augmentation is reimplemented here rather than ported line by line,
  and 256x256 tiles are more sensitive to rotation/crop choices than Dubai-CC's
  50x50 ones.

Treat the LEVIR-CC row as a working training pipeline, not a reproduction.

Two implementation details matter for this, and both are easy to get wrong:

- **`UniqueImageSampler` is required.** It draws one caption per pair per epoch,
  so an epoch is a pass over *pairs*, not captions. Training with a plain
  shuffle over all captions makes each epoch 5× longer and lets two captions of
  the same pair land in one batch, where InfoNCE treats them as negatives. With
  a plain shuffle at the same epoch count, R@5/R@10 come out ~13 and ~17 points
  high (33.4 / 50.3) purely from the extra optimisation.
- **The gallery is the full split**, not the change subset. Ranking change
  queries against only the 97 change pairs inflates every number; `evaluate`
  exposes this as `gallery_scope`.
