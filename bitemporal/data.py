"""LEVIR-CC and Dubai-CC loading.

Both datasets expose the same items -- one per caption, so a pair with five
captions yields five items::

    (before, after, caption, pair_index, change_label)

``pair_index`` is the position of the pair in ``dataset.pairs`` and doubles as
the retrieval ground truth: the gallery is built by encoding every pair once,
in that order.

See ``data/README.md`` for how to obtain each dataset.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import torch
import torchvision.transforms.functional as TF
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

__all__ = [
    "LevirCC",
    "DubaiCC",
    "build_dataset",
    "load_split",
    "normalise_caption",
    "one_caption_per_pair",
    "UniqueImageSampler",
]

SPLITS_DIR = Path(__file__).resolve().parent.parent / "splits"
CAPTIONS_PER_PAIR = 5

# Dubai-CC marks "no change" through its captions rather than a label field.
NO_CHANGE_PHRASES = (
    "Nothing has changed",
    "No change occurred",
    "There is no change to mention",
    "There is no difference",
    "Everything remains the same",
    "The area appears the same",
    "No change in this area",
    "All remained the same",
    "No difference in this area",
)


def normalise_caption(caption: str) -> str:
    """Strip surrounding space and the trailing full stop.

    Both datasets are annotated inconsistently on this point, and the reference
    implementation normalises at load. A trailing "." is its own CLIP token, so
    leaving it in shifts every text embedding slightly away from the paper's.
    """
    return caption.strip().rstrip(".")


def load_split(dataset: str, split: str, splits_dir: Path | str = SPLITS_DIR) -> list[dict]:
    """Return ``[{"file": ..., "change": 0|1}, ...]`` for a released split."""
    payload = json.loads((Path(splits_dir) / dataset / f"{split}.json").read_text())
    return payload["pairs"]


class BiTemporalDataset(Dataset):
    """Shared item layout, augmentation and normalisation.

    Subclasses fill ``self.pairs``, ``self.captions`` and implement
    :meth:`load_pair`, which returns two ``uint8`` CHW arrays.
    """

    mean: tuple[float, float, float]
    std: tuple[float, float, float]

    def __init__(self, split: str, augment: bool | None = None, seed: int = 42):
        self.split = split
        self.augment = split == "train" if augment is None else augment
        self.rng = random.Random(seed)
        self.pairs: list[dict] = []
        self.captions: dict[str, list[str]] = {}

    def load_pair(self, index: int) -> tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError

    def __len__(self) -> int:
        return len(self.pairs) * CAPTIONS_PER_PAIR

    def pair_index(self, idx: int) -> int:
        return idx // CAPTIONS_PER_PAIR

    def _augment(self, pair: torch.Tensor) -> torch.Tensor:
        """Geometric and photometric jitter, applied identically to both frames."""
        if torch.rand(1) < 0.5:
            pair = TF.hflip(pair)
        if torch.rand(1) < 0.5:
            pair = TF.vflip(pair)
        angle = transforms.RandomRotation.get_params([-15, 15])
        pair = TF.rotate(pair, angle)

        jitter = transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05)
        order, b, c, s, h = jitter.get_params(
            jitter.brightness, jitter.contrast, jitter.saturation, jitter.hue
        )
        # These ops do not commute, and get_params returns the order to use.
        ops = {
            0: (b, TF.adjust_brightness),
            1: (c, TF.adjust_contrast),
            2: (s, TF.adjust_saturation),
            3: (h, TF.adjust_hue),
        }
        for fn_id in order:
            value, op = ops[int(fn_id)]
            if value is not None:
                pair = op(pair, value)
        return pair

    def __getitem__(self, idx: int):
        p = self.pair_index(idx)
        before, after = self.load_pair(p)

        # Stack so augmentation is guaranteed synchronised across the two frames.
        pair = torch.from_numpy(np.stack([before, after])).float() / 255.0
        if self.augment:
            pair = self._augment(pair)
        pair = TF.normalize(pair, self.mean, self.std)

        record = self.pairs[p]
        caption = self.captions[record["file"]][idx % CAPTIONS_PER_PAIR]
        return pair[0], pair[1], caption, p, record["change"]


class LevirCC(BiTemporalDataset):
    """LEVIR-CC from the preprocessed HDF5 bundle (see ``data/README.md``).

    ``<root>/{SPLIT}_IMAGES_LEVIR_CC_5_cap_per_img.hdf5`` holds an ``images``
    dataset of shape ``(N, 2, 3, 256, 256)`` aligned with the released split
    order, alongside the matching ``CAPTIONS``/``FILENAMES`` JSON files.
    """

    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)

    def __init__(self, root, split, augment=None, change_only=False, seed=42):
        super().__init__(split, augment, seed)
        import h5py

        self.root = Path(root)
        stem = f"{split.upper()}_%s_LEVIR_CC_5_cap_per_img"
        self._h5 = h5py.File(self.root / (stem % "IMAGES" + ".hdf5"), "r")
        self._images = self._h5["images"]

        names = json.loads((self.root / (stem % "FILENAMES" + ".json")).read_text())
        caps = json.loads((self.root / (stem % "CAPTIONS" + ".json")).read_text())
        released = load_split("levir_cc", split)
        if [p["file"] for p in released] != names:
            raise ValueError("HDF5 order does not match splits/levir_cc — regenerate the bundle")

        self.captions = {
            n: [
                normalise_caption(c)
                for c in caps[i * CAPTIONS_PER_PAIR : (i + 1) * CAPTIONS_PER_PAIR]
            ]
            for i, n in enumerate(names)
        }
        self._rows = list(range(len(names)))
        if change_only:
            keep = [i for i, p in enumerate(released) if p["change"]]
            self._rows = keep
            released = [released[i] for i in keep]
        self.pairs = released

    def load_pair(self, index):
        row = self._images[self._rows[index]]
        return row[0], row[1]


class DubaiCC(BiTemporalDataset):
    """Dubai-CC from the released tiles -- no preprocessing needed.

    ``<root>/imgs_tiles/RGB/500_2000`` and ``500_2010`` hold the two
    acquisitions; ``<root>/captions/*.json`` holds the annotations.
    """

    mean = (0.618, 0.456, 0.364)
    std = (0.299, 0.203, 0.172)
    CAPTION_FILES = {
        "train": "Train_Dubai_CC.json",
        "val": "Validation_Dubai_CC.json",
        "test": "Test_Dubai_CC.json",
    }

    def __init__(self, root, split, augment=None, change_only=False, seed=42, version="RGB"):
        super().__init__(split, augment, seed)
        self.root = Path(root)
        self.before_dir = self.root / "imgs_tiles" / version / "500_2000"
        self.after_dir = self.root / "imgs_tiles" / version / "500_2010"

        raw = json.loads((self.root / "captions" / self.CAPTION_FILES[split]).read_text())
        pairs, captions = [], {}
        for item in raw["images"]:
            sentences = [normalise_caption(s["raw"]) for s in item["sentences"]]
            change = int(not any(p in s for s in sentences for p in NO_CHANGE_PHRASES))
            if change_only and not change:
                continue
            pairs.append({"file": item["filename"], "change": change})
            captions[item["filename"]] = sentences[:CAPTIONS_PER_PAIR]
        self.pairs, self.captions = pairs, captions

    def load_pair(self, index):
        name = self.pairs[index]["file"]
        before = np.array(Image.open(self.before_dir / name))
        after = np.array(Image.open(self.after_dir / name))
        return before.transpose(2, 0, 1), after.transpose(2, 0, 1)


def build_dataset(name: str, root, split: str, **kwargs) -> BiTemporalDataset:
    cls = {"levir_cc": LevirCC, "dubai_cc": DubaiCC}[name]
    return cls(root, split, **kwargs)


def one_caption_per_pair(dataset: BiTemporalDataset) -> torch.utils.data.Subset:
    """Deterministic view holding the first caption of every pair.

    Used for the validation loss that drives checkpoint selection: it keeps the
    InfoNCE negatives genuine (no two captions of one pair in a batch) without
    the per-epoch randomness of :class:`UniqueImageSampler`, so the loss is
    comparable across epochs.
    """
    return torch.utils.data.Subset(
        dataset, range(0, len(dataset.pairs) * CAPTIONS_PER_PAIR, CAPTIONS_PER_PAIR)
    )


class UniqueImageSampler(torch.utils.data.Sampler):
    """One randomly chosen caption per pair, per epoch.

    Two things depend on this. A batch never contains two captions of the same
    pair, so the InfoNCE negatives are all genuine — otherwise captions of the
    same pair are pushed apart as if they were negatives. And an epoch is one
    pass over *pairs* rather than over captions, which is what makes an epoch
    count comparable to the paper's.
    """

    def __init__(self, dataset: BiTemporalDataset, shuffle: bool = True, seed: int = 42):
        self.n_pairs = len(dataset.pairs)
        self.shuffle = shuffle
        self.rng = random.Random(seed)

    def __len__(self) -> int:
        return self.n_pairs

    def __iter__(self):
        order = list(range(self.n_pairs))
        if self.shuffle:
            self.rng.shuffle(order)
        for pair in order:
            yield pair * CAPTIONS_PER_PAIR + self.rng.randrange(CAPTIONS_PER_PAIR)
