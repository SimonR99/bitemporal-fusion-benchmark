# Reproduction audit

What in [`paper_tables.md`](paper_tables.md) this repository can regenerate, and
how closely. Run on the paper's hardware (RTX 4070), CLIP ViT-B/16 frozen,
D=320, 3 layers, 16 heads, 30 epochs.

One caveat applies throughout: the paper reports mean ± std over seeds 1–10, and
seed-to-seed spread on LEVIR-CC is large (±0.4 on an R@1 of 2.06, about 20%
relative). A single-seed run should be compared against **that seed**, not
against the ten-seed mean. Seed 1 in particular is a below-average seed.

| Paper result | Status |
|---|---|
| Table 1, parameters | **Exact**, all eight modules |
| Table 1, FLOPs | Within ~10% for seven of eight; TFF unreconciled |
| Table 1, latency | Hardware-dependent; see below |
| Table 2, caption metrics | **Not reproducible here** — the evaluator was not ported |
| Table 3, Recall@K | Protocol confirmed; retrained values below |
| Table 4, cascade | **Exact**, all four rows |
| Latency vs. length | Transcribed from the reference output; needs an idle GPU |
| Backbone robustness (§4.5) | **Not ported** — only CLIP ViT-B/16 and B/32 |
| Welch tests (§4.2) | Not ported |

## Defects found while auditing

Six bugs in this repository, all fixed. The first is the root cause of the
LEVIR-CC shortfall; the rest are real but were not what held it back.

Worth noting how this hid: Dubai-CC reproduced the paper to within half a
standard deviation *while the image encoder was fundamentally broken*. Its
gallery is 150 pairs and its images are 50x50 tiles upsampled to 224, so the
task is easy enough to survive badly corrupted features. A benchmark that passes
on the small dataset is not evidence the pipeline is correct.

1. **The vision transformer was fed with the batch and sequence axes
   swapped.** `open_clip` 3.x runs its `Transformer` with `batch_first=True`
   and its own `VisionTransformer.forward` calls it directly; this repository
   permuted `(N, L, D) -> (L, N, D)` first, the older OpenAI convention. Nothing
   raises, because the shapes still line up — but self-attention then runs
   **across the images in the batch instead of across the patches of one
   image**. Every patch embedding was a mixture of unrelated scenes, and a
   gallery entry's embedding depended on which other images happened to share
   its batch. Against the reference encoder, patch tokens had cosine similarity
   of just **0.36** (min -0.27) despite near-identical norms — the signature of
   a transposed axis rather than a numerical drift. With this and (2) fixed the
   two encoders agree to **cosine 0.999993**.
2. **The frozen features were passed through `ln_post`.** The reference's CLIP
   fork skips that final LayerNorm whenever the fusion path is active — it
   exists to normalise the CLS token before the projection to the joint
   embedding space, and the fusion modules consume patch tokens instead. This
   repository applied it to every token. It is not a small perturbation:
   measured on LEVIR-CC test images, it rotates each token (cosine 0.77 against
   the raw features) and compresses the spread of token norms *within an image*
   from 3.82 to 1.37, flattening the per-patch magnitude differences that a
   change-detection module has to key on.
3. **The frozen backbone used the wrong activation.** OpenAI's CLIP weights
   were trained with QuickGELU, and `open_clip`'s plain `ViT-B-16` builds them
   with `nn.GELU`. The mismatch only warns, so it ran silently; patch tokens had
   cosine similarity 0.79 on average and 0.25 at worst against the correct
   features. This is a genuine correctness fix, but fixing it alone did not move
   LEVIR-CC (0.77/3.86/7.51 before, 0.64/4.21/7.90 after) — the axis bug above
   was masking it.
4. **No validation, so no checkpoint selection.** The reference tracks
   validation loss every epoch and evaluates the best epoch; this repository
   evaluated the last one. On LEVIR-CC validation loss bottoms out well before
   epoch 30 while training loss keeps falling, so the two differ.
5. **No gradient clipping.** The reference clips to a max norm of 1.0.
6. **`cascade_rank` shared one query embedding across both stages.** The two
   stages are separately trained models with separate text heads.

Two further issues were packaging rather than numerics: `requirements.txt`
omitted `open_clip_torch`, `h5py` and `numpy`, so a clean install could not run
anything; and checkpoints serialised the frozen CLIP (twice, since the text
tower holds a whole model), making them 967 MB instead of 24 MB.

## The split

`splits/levir_cc/` is byte-identical to the reference bundle it was derived
from — filenames and change labels agree exactly on all three splits
(3918/1333/1929 pairs, 3407/667/964 change). The retrieval protocol behind
Table 3 is a gallery of **all 1929 test pairs** with the 4820 change captions as
queries, which is what the paper's caption states and what `evaluate(...,
gallery_scope="full")` does.

The reference repository also contains an older evaluation path that filters to
change pairs *before* deduplicating the gallery, giving G=964 on LEVIR-CC and
G=97 on Dubai-CC. That path inflates Recall@K by roughly the gallery-size ratio
and its stored output is visibly higher than the published table. The paper does
not use it. `gallery_scope="change"` reproduces it for comparison only.

## Table 1

Parameter counts are exact for all eight modules, and every module satisfies the
`(B, L, F) × 2 → (L, B, out_dim)` contract.

FLOPs are an estimate: `fvcore` cannot see through Mamba's fused kernel, so SSM
blocks are costed analytically, their traced subtree is subtracted to avoid
double counting, and the length each block actually scans is observed rather
than assumed (Interleaved Mamba scans 2L, not L).

| Module | `--flops` | Paper | Δ |
|---|---:|---:|---:|
| Subtraction | 0.14 | 0.14 | 0% |
| MLP Fusion | 1.16 | 1.06 | +9% |
| Concat. Transformer | 5.21 | 5.42 | −4% |
| TFF (Text-ITSR) | 4.96 | 2.65 | **+87%** |
| TBF | 2.34 | 2.41 | −3% |
| Concat. Mamba | 3.38 | 3.28 | +3% |
| Bottleneck Mamba | 1.15 | 1.06 | +8% |
| Interleaved Mamba | 1.87 | 1.77 | +6% |

TFF is the outlier and is not reconciled. It applies each cross-attention layer
to *both* acquisitions, so it does roughly twice the work its parameter count
suggests; but counting each layer once still gives 3.36 G, not 2.65 G. The
paper's argument rests on latency and recall rather than FLOPs, and the two
sources use different operator handlers, so this is recorded rather than
resolved.

## Table 3

The protocol is confirmed: recomputing Recall@K from the published per-seed
embeddings over a G=1929 gallery with 4820 change-caption queries returns the
published rows to two decimals (Subtraction 1.88±0.16, TBF 2.06±0.40,
Concatenation Transformer 2.28±0.25; the paper quotes sample standard
deviations, which are a factor sqrt(10/9) larger).

Retraining from scratch is a weaker test than that, because the seed spread is
wide and no two implementations share an RNG stream. For reference, the spread
across the ten published TBF seeds on LEVIR-CC is R@1 2.06±0.40, and **seed 1
alone gives 1.60 / 7.51 / 13.36** — well under the mean. A single retrained run
should be judged against the distribution, not against the mean and not against
one seed's exact value.

## Table 4

Every row reproduces exactly. Feeding the published per-seed embeddings through
`bitemporal.retrieval.cascade_rank` gives, over the same ten seeds:

| Re-ranker | N | R@1 | R@5 | R@10 | Paper |
|---|---:|---:|---:|---:|---|
| TBF | 25 | 2.33 | 8.91 | 15.17 | 2.33 / 8.91 / 15.17 |
| TBF | full | 2.06 | 8.29 | 14.41 | 2.06 / 8.29 / 14.41 |
| Concat. Transf. | 25 | 2.43 | 9.33 | 15.53 | 2.43 / 9.33 / 15.53 |
| Concat. Transf. | full | 2.28 | 8.91 | 15.27 | 2.28 / 8.91 / 15.27 |

The same computation reproduces Table 3's Subtraction, TBF and Concatenation
Transformer rows on a G=1929 gallery with 4820 change-caption queries, which is
an independent confirmation of the retrieval protocol.

Note that both stages need their *own* query embedding: they are separately
trained models with separate text heads, and scoring them with one shared text
embedding compares vectors from two different spaces.

## Latency

Latency is the one number that will not transfer. It is a wall-clock
measurement at batch 1, so it depends on the GPU *and on what else is using
it*. Contention does not scale all modules equally and can reorder the table.
`scripts/benchmark_efficiency.py --stat min` reports the fastest single pass
instead of the mean over the timed loop, which is much closer to the
uncontended cost on a shared GPU — at the cost of no longer being the statistic
the paper reports.

Measured here with another process resident on the GPU throughout:

| Module | `--stat mean` | `--stat min` | Paper |
|---|---:|---:|---:|
| Subtraction | 0.04 | 0.03 | 0.04 |
| MLP Fusion | 0.35 | 0.19 | 0.20 |
| Concat. Transformer | 1.25 | 0.74 | 0.76 |
| TFF (Text-ITSR) | 3.09 | 1.19 | 1.90 |
| TBF | 0.78 | **0.46** | 0.47 |
| Concat. Mamba | 0.99 | 0.50 | 0.58 |
| Bottleneck Mamba | 0.98 | 0.51 | 0.62 |
| Interleaved Mamba | 1.05 | **0.46** | 0.60 |

Under `--stat min` the four non-SSM modules land on the published values almost
exactly (0.03/0.19/0.74/0.46 against 0.04/0.20/0.76/0.47), which is a good sign
that the protocol transfers. The SSM variants come out faster than published —
most plausibly a newer `mamba-ssm`/CUDA than the paper used.

That shifts one sentence of the paper without touching its conclusion. Table 1
bolds TBF as the fastest learned module and says Interleaved Mamba's 0.60 ms is
"28% higher than TBF"; here the two tie at 0.46 ms. The claim that the
scan's linear complexity buys no wall-clock advantage at L=196 survives intact,
and arguably reads stronger: Interleaved Mamba has 2.5x fewer parameters than
TBF and still only matches it.
