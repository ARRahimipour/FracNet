"""Drop-in replacement for fracnet/common.py: typed .env configuration.

Keep the same load_config, seed_everything, write_json and output_path API
used by run.py, predict.py and the earlier FracNet research modules.
"""
from __future__ import annotations

import json
import os
import random
from pathlib import Path

import numpy as np


def _dotenv_values(path: Path) -> dict[str, str | None]:
    """Use python-dotenv when installed; a strict KEY=VALUE fallback otherwise."""
    try:
        from dotenv import dotenv_values
    except ImportError:
        values: dict[str, str | None] = {}
        with path.open(encoding="utf-8-sig") as stream:
            for number, source in enumerate(stream, 1):
                line = source.strip()
                if not line or line.startswith("#"):
                    continue
                if "=" not in line:
                    raise ValueError(f"Bad .env line {number}: expected KEY=VALUE")
                key, value = (part.strip() for part in line.split("=", 1))
                if not key or not key.replace("_", "").isalnum():
                    raise ValueError(f"Bad .env key on line {number}")
                if value.startswith(("'", '"')) and value.endswith(value[0]):
                    value = value[1:-1]
                values[key] = value
        return values
    return dotenv_values(path)


_DEFAULTS = {
    "SEED": "42",
    "MANIFEST": "data/manifest.csv",
    "UNLABELED_DIR": "data/ssl_unlabeled",
    "OUTPUT_DIR": "runs/full",
    "SCENARIO": "full",
    "DATASETS": "D1,D2,D3",
    "XCEPTION_MODEL": "legacy_xception",
    "MOBILENET_MODEL": "mobilenetv2_100",
    "EFFICIENTNET_MODEL": "efficientnet_b0",
    "XCEPTION_IMAGE_SIZE": "299",
    "MOBILENET_IMAGE_SIZE": "224",
    "EFFICIENTNET_IMAGE_SIZE": "224",
    "PRETRAINED_IMAGENET": "true",
    "FREEZE_FRACTION": "0.70",
    "BATCH_SIZE": "15",
    "NUM_WORKERS": "4",
    "EPOCHS": "100",
    "SSL_EPOCHS": "50",
    "SSL_TEMPERATURE": "0.2",
    "LEARNING_RATE": "0.001",
    "WEIGHT_DECAY": "0.0001",
    "EARLY_STOPPING_PATIENCE": "12",
    "VAL_FRACTION": "0.15",
    "TEST_FRACTION": "0.15",
    "CORRELATION_THRESHOLD": "0.95",
    "Z_SCORE_THRESHOLD": "3.0",
    "OUTLIER_ROW_FEATURE_FRACTION": "0.02",
    "RF_N_ESTIMATORS": "100",
    "PCA_VARIANCE": "0.95",
    "TSNE_MAX_SAMPLES": "1500",
    "VOTE_MODELS": "xgboost,svm,knn",
    "SSL_MAX_IMAGES": "",
}


def _bool(raw: str, key: str) -> bool:
    value = raw.strip().lower()
    if value in {"1", "true", "yes", "on"}:
        return True
    if value in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"FRACNET_{key} must be true or false, got {raw!r}")


def _csv(raw: str, key: str) -> list[str]:
    result = [v.strip() for v in raw.split(",") if v.strip()]
    if not result:
        raise ValueError(f"FRACNET_{key} must not be empty")
    if len(set(result)) != len(result):
        raise ValueError(f"FRACNET_{key} contains duplicates")
    return result


def _path(raw: str, base: Path) -> str:
    value = Path(raw).expanduser()
    return str((value if value.is_absolute() else base / value).resolve())


def load_config(path: str | Path = ".env") -> dict:
    """Load the FracNet config; real process env takes precedence over .env.

    Parameters retain the exact dictionary keys expected by the earlier
    implementation, so `run.py` and `predict.py` need only default to `.env`.
    No file is modified by reading configuration.
    """
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Configuration file not found: {path}. "
                                "Copy .env.example to .env in the project root.")
    raw_file = {k.removeprefix("FRACNET_"): v for k, v in _dotenv_values(path).items()
                if k.startswith("FRACNET_") and v is not None}
    raw_os = {k.removeprefix("FRACNET_"): v for k, v in os.environ.items()
              if k.startswith("FRACNET_")}
    unknown = (set(raw_file) | set(raw_os)) - set(_DEFAULTS)
    if unknown:
        raise ValueError(f"Unknown FracNet variable(s): {sorted(unknown)}")
    data = {**_DEFAULTS, **raw_file, **raw_os}

    def intval(key):
        try:
            return int(data[key])
        except ValueError as err:
            raise ValueError(f"FRACNET_{key} requires an integer") from err

    def floatval(key):
        try:
            value = float(data[key])
        except ValueError as err:
            raise ValueError(f"FRACNET_{key} requires a number") from err
        if not np.isfinite(value):
            raise ValueError(f"FRACNET_{key} must be finite")
        return value

    scenario = data["SCENARIO"].strip()
    if scenario not in {"full", "S1", "S2", "S3"}:
        raise ValueError("FRACNET_SCENARIO must be full, S1, S2 or S3")
    cfg = {
        "seed": intval("SEED"),
        "manifest": _path(data["MANIFEST"], path.parent),
        "unlabeled_dir": _path(data["UNLABELED_DIR"], path.parent),
        "output_dir": _path(data["OUTPUT_DIR"], path.parent),
        "scenario": scenario,
        "dataset_names": _csv(data["DATASETS"], "DATASETS"),
        "models": {
            "xception": data["XCEPTION_MODEL"],
            "mobilenetv2": data["MOBILENET_MODEL"],
            "efficientnet": data["EFFICIENTNET_MODEL"],
        },
        "image_sizes": {
            "xception": intval("XCEPTION_IMAGE_SIZE"),
            "mobilenetv2": intval("MOBILENET_IMAGE_SIZE"),
            "efficientnet": intval("EFFICIENTNET_IMAGE_SIZE"),
        },
        "pretrained_imagenet": _bool(data["PRETRAINED_IMAGENET"], "PRETRAINED_IMAGENET"),
        "freeze_fraction": floatval("FREEZE_FRACTION"),
        "batch_size": intval("BATCH_SIZE"),
        "num_workers": intval("NUM_WORKERS"),
        "epochs": intval("EPOCHS"),
        "ssl_epochs": intval("SSL_EPOCHS"),
        "ssl_temperature": floatval("SSL_TEMPERATURE"),
        "learning_rate": floatval("LEARNING_RATE"),
        "weight_decay": floatval("WEIGHT_DECAY"),
        "early_stopping_patience": intval("EARLY_STOPPING_PATIENCE"),
        "val_fraction": floatval("VAL_FRACTION"),
        "test_fraction": floatval("TEST_FRACTION"),
        "correlation_threshold": floatval("CORRELATION_THRESHOLD"),
        "z_score_threshold": floatval("Z_SCORE_THRESHOLD"),
        "outlier_row_feature_fraction": floatval("OUTLIER_ROW_FEATURE_FRACTION"),
        "rf_n_estimators": intval("RF_N_ESTIMATORS"),
        "pca_variance": floatval("PCA_VARIANCE"),
        "tsne_max_samples": intval("TSNE_MAX_SAMPLES"),
        "vote_models": _csv(data["VOTE_MODELS"], "VOTE_MODELS"),
        "ssl_max_images": None if not data["SSL_MAX_IMAGES"].strip() else intval("SSL_MAX_IMAGES"),
    }
    if not 0 <= cfg["freeze_fraction"] < 1:
        raise ValueError("FRACNET_FREEZE_FRACTION must be in [0,1)")
    if not 0 < cfg["val_fraction"] < 0.5 or not 0 < cfg["test_fraction"] < 0.5:
        raise ValueError("Validation and test fractions must each be between 0 and 0.5")
    if cfg["val_fraction"] + cfg["test_fraction"] >= 1:
        raise ValueError("Validation + test fractions must total below 1")
    for key in ("batch_size", "epochs", "ssl_epochs", "early_stopping_patience", "rf_n_estimators", "tsne_max_samples"):
        if cfg[key] <= 0:
            raise ValueError(f"{key} must be positive")
    for size in cfg["image_sizes"].values():
        if size <= 0:
            raise ValueError("Image dimensions must be positive")
    if cfg["num_workers"] < 0:
        raise ValueError("num_workers cannot be negative")
    if not 0 < cfg["correlation_threshold"] < 1 or not 0 < cfg["pca_variance"] < 1:
        raise ValueError("Correlation threshold and PCA variance must be in (0,1)")
    if len(cfg["vote_models"]) % 2 != 1:
        raise ValueError("Majority voting requires an odd number of classifiers")
    return cfg


def seed_everything(seed: int):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
    except ImportError:
        return
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def write_json(path: str | Path, payload: dict):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, ensure_ascii=False, default=float)


def output_path(cfg: dict, *parts: str) -> Path:
    path = Path(cfg["output_dir"]).joinpath(*parts)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path

