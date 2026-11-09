"""FracNet training and evaluation command line, configured by .env."""
import argparse
from pathlib import Path

import pandas as pd

from fracnet.common import load_config, seed_everything
from fracnet.data import group_split, scan_data, validate_manifest


def prepare(cfg, root=None, force=False):
    """Audit source images and write a grouped manifest without silent overwrite."""
    file = Path(cfg["manifest"])
    if root:
        if file.exists() and not force:
            raise FileExistsError(f"Manifest already exists: {file}. Use --force to regenerate.")
        data = scan_data(root, cfg["dataset_names"])
    else:
        if not file.is_file():
            raise FileNotFoundError("No manifest. Supply --data-root data with D1/D2/D3 images")
        data = pd.read_csv(file)
        if "split" in data:
            validate_manifest(data, cfg["dataset_names"], hashes=True)
            print("Existing manifest checked. No changes made.")
            return
    validate_manifest(data, cfg["dataset_names"], hashes=True)
    data = group_split(data, cfg)
    validate_manifest(data, cfg["dataset_names"])
    file.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(file, index=False)
    print(f"Wrote: {file}")
    print(data.groupby(["dataset", "split", "label"]).size().to_string())


def status(cfg):
    """Print a non-training readiness check for local CPU/GPU environments."""
    print("FracNet with .env is configured")
    print("Scenario:", cfg["scenario"])
    print("Manifest:", cfg["manifest"])
    print("Output:", cfg["output_dir"])
    print("Unlabeled images:", cfg["unlabeled_dir"])
    print("Models:", ", ".join(cfg["models"]))
    if not Path(cfg["manifest"]).is_file():
        print("NEXT STEP: Download real X-ray images and run prepare --data-root data")


def main():
    p = argparse.ArgumentParser(description="FracNet (2025): reproducible research pipeline")
    p.add_argument("stage", choices=["check", "prepare", "ssl", "train", "cnn_eval", "extract",
                                      "classify", "cross_domain", "gradcam", "all"])
    p.add_argument("--config", default=".env", help="Path to .env (default: .env)")
    p.add_argument("--data-root", help="D1,D2,D3 folders each containing fractured/normal images")
    p.add_argument("--force", action="store_true", help="Allow intentional regeneration of the manifest")
    args = p.parse_args()
    cfg = load_config(args.config)
    seed_everything(cfg["seed"])
    if args.stage == "check":
        return status(cfg)
    if args.stage == "prepare":
        return prepare(cfg, args.data_root, args.force)
    if args.stage == "all":
        prepare(cfg, args.data_root, args.force)
        stages = ([] if cfg["scenario"] == "S1" else ["ssl"]) + [
            "train", "cnn_eval", "extract", "classify", "cross_domain", "gradcam"]
    else:
        stages = [args.stage]
    for stage in stages:
        print(f"\n=== {stage.upper()} ===", flush=True)
        if stage == "ssl":
            from fracnet.deep import pretrain_ssl
            pretrain_ssl(cfg)
        elif stage == "train":
            from fracnet.deep import train_cnn
            train_cnn(cfg)
        elif stage == "cnn_eval":
            from fracnet.deep import evaluate_bases
            evaluate_bases(cfg)
        elif stage == "extract":
            from fracnet.deep import extract_features
            extract_features(cfg)
        elif stage == "classify":
            from fracnet.classical import classify
            classify(cfg)
        elif stage == "cross_domain":
            from fracnet.deep import cross_domain
            cross_domain(cfg)
        elif stage == "gradcam":
            from fracnet.deep import explain
            explain(cfg)


if __name__ == "__main__":
    main()

