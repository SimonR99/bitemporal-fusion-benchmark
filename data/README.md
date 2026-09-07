# Datasets

Neither dataset is redistributed here — only the split indices in
[`../splits/`](../splits/). Download each from its original source and place or
symlink it under this directory. Everything in `data/` is gitignored except
this file.

```
data/
  levir-cc/output_015/    LEVIR-CC (preprocessed bundle, see below)
  dubai-cc/               Dubai-CC (released layout, no preprocessing)
```

## Dubai-CC

The quicker of the two: 500 pairs of 50x50 tiles from Landsat 7 ETM+
acquisitions of Dubai (2000 and 2010) at ~30 m/px, with the standard
300/50/150 split. Training one fusion module takes minutes.

Source: Hoxha et al., "Change Captioning: A New Paradigm for Multitemporal
Remote Sensing Image Analysis", IEEE TGRS 2022
(<https://doi.org/10.1109/TGRS.2022.3195692>).

Expected layout — this is exactly how the dataset ships, so just unpack it:

```
data/dubai-cc/
  imgs_tiles/RGB/500_2000/<tile>.tif     earlier acquisition
  imgs_tiles/RGB/500_2010/<tile>.tif     later acquisition
  captions/{Train,Validation,Test}_Dubai_CC.json
```

Change labels are derived from the captions (Dubai-CC has no label field): a
pair counts as no-change when any of its captions matches one of the fixed
no-change phrases, which is how the original code assigns them.

## LEVIR-CC

10,077 bi-temporal pairs of 256x256 tiles at 0.5 m/px over Texas, five captions
each. We use a retrieval-oriented split (all 5,038 change pairs, no-change
subsampled to 2,142) — see [`../splits/levir_cc/`](../splits/levir_cc/).

Source: Liu et al., "Remote Sensing Image Change Captioning With Dual-Branch
Transformers: A New Method and a Large Scale Dataset", IEEE TGRS 2022
(<https://github.com/Chen-Yang-Liu/RSICC>).

`bitemporal.data.LevirCC` reads the preprocessed bundle used for the paper:

```
data/levir-cc/output_015/
  {TRAIN,VAL,TEST}_IMAGES_LEVIR_CC_5_cap_per_img.hdf5      (N, 2, 3, 256, 256) uint8
  {TRAIN,VAL,TEST}_CAPTIONS_LEVIR_CC_5_cap_per_img.json    5N captions, pair-major
  {TRAIN,VAL,TEST}_FILENAMES_LEVIR_CC_5_cap_per_img.json   N filenames
```

Build it from a LEVIR-CC download with:

```bash
python scripts/prepare_levir.py --source /path/to/LEVIR-CC --out data/levir-cc/output_015
```

where `--source` contains the released `images/{train,val,test}/{A,B}/*.png` and
`LevirCCcaptions.json`. The script follows the released split order, and
`LevirCC` refuses to load a bundle whose filenames disagree with it, so a stale
or reordered bundle fails loudly rather than silently misaligning the gallery.

The bundle is ~2.7 GB. If you would rather not materialise it, adapting
`LevirCC.load_pair` to read the PNGs directly is a small change — the HDF5 is
only there to keep the 10-seed sweeps I/O-bound rather than decode-bound.
