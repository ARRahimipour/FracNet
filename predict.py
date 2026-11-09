"""Research-only single X-ray inference; trained checkpoints are required."""
import argparse
import json
from pathlib import Path

import joblib
import numpy as np
from PIL import Image

from fracnet.common import load_config


def predict(image_path, dataset, cfg):
    """Apply one dataset-specific CNN triplet and its trained voting ensemble."""
    import torch
    from fracnet.data import transforms_for
    from fracnet.deep import _device, trained_model
    saved = Path(cfg["output_dir"]) / "evaluation" / "trained_classifiers.joblib"
    if not saved.is_file():
        raise FileNotFoundError("Train and classify first; no final classifier found")
    # joblib is pickle-based. Only load checkpoints created/trusted by you.
    pipeline = joblib.load(saved)
    with Image.open(image_path) as im:
        rgb = im.convert("RGB")
    embeddings = []
    for label in cfg["models"]:
        model = trained_model(cfg, dataset, label)
        x = transforms_for(cfg["image_sizes"][label])(rgb).unsqueeze(0).to(_device())
        with torch.inference_mode():
            embeddings.append(model.features(x).float().cpu().numpy())
    matrix = pipeline["processor"].transform(np.concatenate(embeddings, axis=1))
    classes = {k: int(v.predict(matrix)[0]) for k, v in pipeline["models"].items()}
    votes = pipeline["voters"]
    label = int(sum(classes[k] for k in votes) > len(votes) // 2)
    return {"class": label, "meaning": "fractured" if label else "normal",
            "models": classes, "warning": "Research only; not clinically validated"}


def main():
    p = argparse.ArgumentParser(description="Predict on one X-ray (research only)")
    p.add_argument("--image", required=True)
    p.add_argument("--domain", required=True, choices=["D1", "D2", "D3"])
    p.add_argument("--config", default=".env")
    args = p.parse_args()
    print(json.dumps(predict(args.image, args.domain, load_config(args.config)), indent=2))


if __name__ == "__main__":
    main()

