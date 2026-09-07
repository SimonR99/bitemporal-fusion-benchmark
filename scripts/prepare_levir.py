#!/usr/bin/env python3
"""Build the LEVIR-CC bundle that ``bitemporal.data.LevirCC`` reads.

Packs the released PNGs into one HDF5 per split, in the order given by
``splits/levir_cc/``, together with the matching caption and filename files.
Reading 10k pairs from a single contiguous array keeps 10-seed sweeps I/O-bound
rather than PNG-decode-bound.

    python scripts/prepare_levir.py --source /path/to/LEVIR-CC --out data/levir-cc/output_015

``--source`` must contain the released layout::

    images/{train,val,test}/{A,B}/<pair>.png
    LevirCCcaptions.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from bitemporal.data import CAPTIONS_PER_PAIR, load_split, normalise_caption

TILE = 256


def _captions_by_file(source: Path) -> dict[str, list[str]]:
    raw = json.loads((source / "LevirCCcaptions.json").read_text())
    return {
        entry["filename"]: [
            normalise_caption(s["raw"]) for s in entry["sentences"]
        ][:CAPTIONS_PER_PAIR]
        for entry in raw["images"]
    }


def build_split(source: Path, out: Path, split: str, captions: dict[str, list[str]]) -> None:
    import h5py

    pairs = load_split("levir_cc", split)
    names = [p["file"] for p in pairs]

    missing = [n for n in names if n not in captions]
    if missing:
        raise SystemExit(f"{len(missing)} pairs have no captions, e.g. {missing[:3]}")

    stem = f"{split.upper()}_%s_LEVIR_CC_5_cap_per_img"
    out.mkdir(parents=True, exist_ok=True)

    with h5py.File(out / (stem % "IMAGES" + ".hdf5"), "w") as handle:
        handle.attrs["captions_per_image"] = CAPTIONS_PER_PAIR
        images = handle.create_dataset(
            "images", (len(names), 2, 3, TILE, TILE), dtype="uint8"
        )
        for i, name in enumerate(names):
            for j, folder in enumerate(("A", "B")):
                path = source / "images" / split / folder / name
                if not path.exists():  # some releases omit the per-split level
                    path = source / "images" / folder / name
                frame = np.asarray(Image.open(path).convert("RGB"))
                images[i, j] = frame.transpose(2, 0, 1)
            if (i + 1) % 500 == 0:
                print(f"  {split}: {i + 1}/{len(names)}", flush=True)

    (out / (stem % "FILENAMES" + ".json")).write_text(json.dumps(names))
    (out / (stem % "CAPTIONS" + ".json")).write_text(
        json.dumps([c for n in names for c in captions[n]])
    )
    (out / (stem % "LABELS" + ".json")).write_text(json.dumps([p["change"] for p in pairs]))
    print(f"{split}: {len(names)} pairs -> {out}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", required=True, type=Path, help="LEVIR-CC download root")
    p.add_argument("--out", required=True, type=Path, help="destination bundle directory")
    p.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    args = p.parse_args()

    captions = _captions_by_file(args.source)
    for split in args.splits:
        build_split(args.source, args.out, split, captions)


if __name__ == "__main__":
    main()
