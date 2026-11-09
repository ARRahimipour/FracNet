"""Image manifests, duplicate checks and train/val/test data loaders."""
from __future__ import annotations

import hashlib
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import StratifiedGroupKFold

EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
LABELS = {"fractured": 1, "fracture": 1, "broken": 1,
          "not_fractured": 0, "not-fractured": 0, "normal": 0,
          "nonfractured": 0, "non-fractured": 0}


def digest(path):
    """Return a content hash to detect exact duplicate images across sources."""
    h = hashlib.sha256()
    with open(path, "rb") as handle:
        for data in iter(lambda: handle.read(1 << 20), b""):
            h.update(data)
    return h.hexdigest()


def scan_data(root, datasets):
    """Build a manifest from dataset/class/image folders.

    Class folders may contain nested image directories. Filename stems are
    *not* reliable patient identifiers; supply real patient IDs when available.
    """
    root = Path(root).resolve()
    rows = []
    for dataset in datasets:
        folder = root / dataset
        if not folder.is_dir():
            raise FileNotFoundError(f"Dataset folder missing: {folder}. See README_FA.md")
        for label_dir in folder.iterdir():
            if not label_dir.is_dir():
                continue
            name = label_dir.name.lower().strip().replace(" ", "_")
            if name not in LABELS:
                raise ValueError(f"Unknown label folder {label_dir}")
            for image in label_dir.rglob("*"):
                if image.is_file() and image.suffix.lower() in EXTENSIONS:
                    rows.append({"path": str(image.resolve()), "dataset": dataset,
                                 "label": LABELS[name],
                                 "patient_id": f"{dataset}:{image.stem}"})
    if not rows:
        raise ValueError("No X-ray images found under D1, D2, D3")
    warnings.warn("Filename-derived patient IDs are placeholders, not verified patient identifiers.")
    return pd.DataFrame(rows)


def validate_manifest(df, datasets, hashes=False):
    """Validate labels, image paths, unique images and patient split isolation."""
    required = {"path", "dataset", "label", "patient_id"}
    if not required.issubset(df):
        raise ValueError(f"Manifest must contain: {sorted(required)}")
    if df.empty or df[list(required)].isna().any().any():
        raise ValueError("Manifest has missing values or no rows")
    if set(df.dataset) != set(datasets):
        raise ValueError(f"Expected datasets {datasets}; got {sorted(df.dataset.unique())}")
    if set(df.label.astype(int)) != {0, 1}:
        raise ValueError("Both labels 0=normal and 1=fractured are required")
    if any(set(x.label.astype(int)) != {0, 1} for _, x in df.groupby("dataset")):
        raise ValueError("Each dataset requires positive and negative labels")
    paths = [str(Path(p).resolve()) for p in df.path]
    if len(paths) != len(set(paths)):
        raise ValueError("Duplicate file path in manifest")
    for p in paths:
        if not Path(p).is_file():
            raise FileNotFoundError(f"Missing image: {p}")
    if hashes:
        fingerprints = [digest(p) for p in paths]
        if len(fingerprints) != len(set(fingerprints)):
            raise ValueError("Exact duplicate images detected; remove them before splitting")
    if "split" in df:
        if not set(df.split) == {"train", "val", "test"}:
            raise ValueError("Expected train, val, test splits")
        groups = df.dataset.astype(str) + ":" + df.patient_id.astype(str)
        if df.assign(_group=groups).groupby("_group").split.nunique().max() > 1:
            raise ValueError("Patient leaked across splits")
        if any(set(x.split) != {"train", "val", "test"} for _, x in df.groupby("dataset")):
            raise ValueError("Every dataset requires all splits")


def group_split(df, cfg):
    """Create approximate stratified, patient-group-aware holdout splits.

    Every dataset receives separate train/validation/test folds. Split ratios
    vary with patient-group size and are an explicit reproduction assumption.
    """
    if "split" in df:
        return df.copy()
    partitions = []
    for dataset, part in df.groupby("dataset", sort=False):
        part = part.reset_index(drop=True).copy()
        outer_folds = max(2, round(1 / cfg["test_fraction"]))
        try:
            outer = StratifiedGroupKFold(n_splits=outer_folds, shuffle=True,
                                         random_state=cfg["seed"])
            train_val, test = next(outer.split(part, part.label, part.patient_id))
            subset = part.iloc[train_val].reset_index(drop=True)
            inner_folds = max(2, round((1 - cfg["test_fraction"]) / cfg["val_fraction"]))
            inner = StratifiedGroupKFold(n_splits=inner_folds, shuffle=True,
                                         random_state=cfg["seed"] + 1)
            _, valid = next(inner.split(subset, subset.label, subset.patient_id))
        except ValueError as e:
            raise ValueError(f"Insufficient patient groups / class examples in {dataset}") from e
        part["split"] = "train"
        part.loc[test, "split"] = "test"
        part.loc[train_val[valid], "split"] = "val"
        partitions.append(part)
    return pd.concat(partitions, ignore_index=True)


def manifest(cfg):
    """Load the prepared manifest before training or feature extraction."""
    path = Path(cfg["manifest"])
    if not path.is_file():
        raise FileNotFoundError(f"Missing {path}. First run: python run.py prepare --config .env --data-root data")
    df = pd.read_csv(path)
    if "split" not in df:
        raise ValueError("Manifest needs train/val/test split; run prepare first")
    validate_manifest(df, cfg["dataset_names"])
    return df


def transforms_for(size, train=False, ssl=False):
    """Use consistent preprocessing per backbone and separate train augmentations."""
    from torchvision import transforms as T
    norm = T.Normalize([0.5] * 3, [0.5] * 3) if size == 299 else T.Normalize(
        [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    if ssl:
        return T.Compose([T.RandomResizedCrop(size, scale=(0.55, 1)),
                          T.RandomHorizontalFlip(), T.RandomRotation(12),
                          T.ToTensor(), norm])
    augment = [T.RandomHorizontalFlip(), T.RandomRotation(10)] if train else []
    return T.Compose([T.Resize((size, size)), *augment, T.ToTensor(), norm])


class XRayImages:
    """Image dataset compatible with torch DataLoader, including SSL image pairs."""
    def __init__(self, paths, labels, transform, two_views=False):
        self.paths = list(map(str, paths))
        self.labels = None if labels is None else list(map(int, labels))
        self.transform = transform
        self.two_views = two_views

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        with Image.open(self.paths[i]) as image:
            image = image.convert("RGB")
            if self.two_views:
                return self.transform(image.copy()), self.transform(image.copy())
            tensor = self.transform(image)
        return tensor if self.labels is None else (tensor, self.labels[i])


def loader(paths, labels, size, cfg, train=False, ssl=False):
    """Build a DataLoader with consistent batching and CUDA pinning options."""
    import torch
    ds = XRayImages(paths, labels, transforms_for(size, train, ssl), two_views=ssl)
    return torch.utils.data.DataLoader(ds, batch_size=cfg["batch_size"],
                                       shuffle=train or ssl, num_workers=cfg["num_workers"],
                                       pin_memory=torch.cuda.is_available(), drop_last=ssl)


def ssl_paths(cfg, df):
    """Collect eligible unlabeled images without exact val/test image overlap."""
    root = Path(cfg["unlabeled_dir"])
    external = [str(p.resolve()) for p in root.rglob("*")
                if p.is_file() and p.suffix.lower() in EXTENSIONS] if root.is_dir() else []
    train = list(df.loc[df.split == "train", "path"])
    holdout_hashes = {digest(p) for p in df.loc[df.split != "train", "path"]}
    seen, usable = set(), []
    # Hash comparison catches re-used exact files even with different filenames.
    for p in external + train:
        h = digest(p)
        if h not in holdout_hashes and h not in seen:
            seen.add(h)
            usable.append(p)
    if cfg["ssl_max_images"] is not None:
        usable = usable[:cfg["ssl_max_images"]]
    if len(usable) < cfg["batch_size"]:
        raise ValueError("Insufficient unlabeled images for SSL batch; add more to data/ssl_unlabeled")
    return usable

