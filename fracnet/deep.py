"""Self-supervised pretraining, CNNs, feature extraction and explanations."""
from pathlib import Path

import numpy as np
import pandas as pd

from .common import output_path, seed_everything, write_json
from .data import loader, manifest, ssl_paths, transforms_for


def _torch():
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("PyTorch not installed. See README_FA.md for CUDA/CPU setup.") from exc
    return torch


def _device():
    torch = _torch()
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _checkpoint(cfg, dataset, label):
    return output_path(cfg, "cnn", dataset, f"{label}.pt")


def pretrain_ssl(cfg):
    """Pretrain each encoder on eligible X-rays with a contrastive SSL proxy."""
    torch = _torch()
    from .networks import FracCNN, SSLModel, contrastive_loss
    seed_everything(cfg["seed"])
    df = manifest(cfg)
    paths = ssl_paths(cfg, df)
    write_json(output_path(cfg, "ssl", "audit.json"),
               {"usable_ssl_images": len(paths), "note": "Exact val/test duplicates excluded"})
    for label, backbone in cfg["models"].items():
        size = cfg["image_sizes"][label]
        batches = loader(paths, None, size, cfg, ssl=True)
        model = SSLModel(FracCNN(backbone, pretrained=cfg["pretrained_imagenet"], attention=False)).to(_device())
        optim = torch.optim.Adam(model.parameters(), lr=cfg["learning_rate"], weight_decay=cfg["weight_decay"])
        history = []
        for epoch in range(cfg["ssl_epochs"]):
            model.train()
            losses = []
            for a, b in batches:
                optim.zero_grad(set_to_none=True)
                loss = contrastive_loss(model(a.to(_device())), model(b.to(_device())), cfg["ssl_temperature"])
                loss.backward()
                optim.step()
                losses.append(loss.item())
            if not losses:
                raise ValueError("No SSL batches produced")
            value = float(np.mean(losses))
            history.append(value)
            print(f"SSL {label}: epoch={epoch+1} loss={value:.4f}", flush=True)
        torch.save({"encoder": model.encoder.state_dict(), "backbone": backbone},
                   output_path(cfg, "ssl", f"{label}.pt"))
        write_json(output_path(cfg, "ssl", f"{label}_history.json"), {"loss": history})


def _start_model(cfg, label):
    """Construct an ablation-specific model and restore the SSL encoder."""
    torch = _torch()
    from .networks import FracCNN, freeze_fraction
    imagenet = cfg["scenario"] == "S1" and cfg["pretrained_imagenet"]
    attention = cfg["scenario"] in {"S3", "full"}
    model = FracCNN(cfg["models"][label], pretrained=imagenet, attention=attention)
    if cfg["scenario"] != "S1":
        ssl = output_path(cfg, "ssl", f"{label}.pt")
        if not ssl.exists():
            raise FileNotFoundError(f"Run SSL stage first; missing {ssl}")
        state = torch.load(ssl, map_location="cpu", weights_only=True)
        model.backbone.load_state_dict(state["encoder"])
    frozen, total = freeze_fraction(model, cfg["freeze_fraction"])
    return model, {"frozen_leaf_layers": frozen, "total_leaf_layers": total}


def _predictions(model, batches):
    torch = _torch()
    model.eval()
    y, probability = [], []
    with torch.inference_mode():
        for images, labels in batches:
            probability.extend(torch.sigmoid(model(images.to(_device()))).cpu().numpy().tolist())
            y.extend(labels.numpy().tolist())
    return np.asarray(y, dtype=int), np.asarray(probability)


def cnn_metrics(model, batches):
    from .classical import metrics
    y, probability = _predictions(model, batches)
    return metrics(y, (probability >= 0.5).astype(int), probability)


def train_cnn(cfg):
    """Fine-tune every backbone independently on each of the three datasets."""
    torch = _torch()
    seed_everything(cfg["seed"])
    df = manifest(cfg)
    for dataset in cfg["dataset_names"]:
        data = df[df.dataset == dataset]
        train = data[data.split == "train"]
        valid = data[data.split == "val"]
        for label in cfg["models"]:
            model, info = _start_model(cfg, label)
            model = model.to(_device())
            size = cfg["image_sizes"][label]
            training = loader(train.path, train.label, size, cfg, train=True)
            validation = loader(valid.path, valid.label, size, cfg)
            pos = max(1, int(train.label.sum()))
            neg = max(1, len(train) - pos)
            loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(neg / pos, device=_device()))
            optim = torch.optim.Adam((p for p in model.parameters() if p.requires_grad),
                                     lr=cfg["learning_rate"], weight_decay=cfg["weight_decay"])
            sched = torch.optim.lr_scheduler.ReduceLROnPlateau(optim, mode="max", patience=4, factor=0.5)
            best, patience, history = -1., 0, []
            for epoch in range(cfg["epochs"]):
                model.train()
                train_losses = []
                for image, y in training:
                    optim.zero_grad(set_to_none=True)
                    loss = loss_fn(model(image.to(_device())), y.to(_device()).float())
                    loss.backward()
                    optim.step()
                    train_losses.append(loss.item())
                score = cnn_metrics(model, validation)
                sched.step(score["f1"])
                history.append({"epoch": epoch + 1, "loss": float(np.mean(train_losses)), **score})
                if score["f1"] > best + 1e-7:
                    # Checkpoints are selected exclusively on validation F1.
                    best, patience = score["f1"], 0
                    torch.save({"state_dict": model.state_dict(), "name": cfg["models"][label],
                                "attention": cfg["scenario"] in {"S3", "full"}},
                               _checkpoint(cfg, dataset, label))
                else:
                    patience += 1
                print(f"Train {dataset}/{label}: epoch={epoch+1} val_f1={score['f1']:.3f}", flush=True)
                if patience >= cfg["early_stopping_patience"]:
                    break
            write_json(output_path(cfg, "cnn", dataset, f"{label}_history.json"),
                       {**info, "history": history, "best_validation_f1": best})


def trained_model(cfg, dataset, label):
    """Load a locally trained checkpoint without downloading new weights."""
    torch = _torch()
    from .networks import FracCNN
    path = _checkpoint(cfg, dataset, label)
    if not path.is_file():
        raise FileNotFoundError(f"Model weights missing: {path}. Run train stage first")
    saved = torch.load(path, map_location="cpu", weights_only=True)
    model = FracCNN(saved["name"], pretrained=False, attention=saved["attention"])
    model.load_state_dict(saved["state_dict"])
    return model.to(_device()).eval()


def evaluate_bases(cfg):
    """Produce held-out CNN metrics comparable in structure to paper Table 2."""
    df, rows = manifest(cfg), []
    for ds in cfg["dataset_names"]:
        for label in cfg["models"]:
            model = trained_model(cfg, ds, label)
            for split in ("val", "test"):
                sub = df[(df.dataset == ds) & (df.split == split)]
                values = cnn_metrics(model, loader(sub.path, sub.label, cfg["image_sizes"][label], cfg))
                rows.append({"dataset": ds, "model": label, "split": split, **values})
    pd.DataFrame(rows).to_csv(output_path(cfg, "evaluation", "cnn_base_metrics.csv"), index=False)


def extract_features(cfg):
    """Concatenate three model embeddings per image, preserving split isolation."""
    torch = _torch()
    df = manifest(cfg)
    for dataset in cfg["dataset_names"]:
        for split in ("train", "val", "test"):
            sub = df[(df.dataset == dataset) & (df.split == split)]
            blocks = []
            for label in cfg["models"]:
                model = trained_model(cfg, dataset, label)
                batches = loader(sub.path, sub.label, cfg["image_sizes"][label], cfg)
                extracted = []
                with torch.inference_mode():
                    for image, _ in batches:
                        extracted.append(model.features(image.to(_device())).cpu().float().numpy())
                blocks.append(np.concatenate(extracted))
                del model
            X = np.concatenate(blocks, axis=1).astype("float32")
            # Each dataset gets an equal-width feature representation so the
            # classical stage can vertically concatenate all three datasets.
            np.savez_compressed(output_path(cfg, "features", f"{dataset}_{split}.npz"),
                                X=X, y=sub.label.to_numpy(dtype=int),
                                path=sub.path.to_numpy(dtype=str),
                                patient=sub.patient_id.to_numpy(dtype=str),
                                dataset=np.repeat(dataset, len(sub)))
            print(f"Extracted {dataset}/{split}: {X.shape}", flush=True)


def cross_domain(cfg):
    """Measure how a CNN trained on one dataset transfers to another dataset."""
    df, records = manifest(cfg), []
    for source in cfg["dataset_names"]:
        for label in cfg["models"]:
            model = trained_model(cfg, source, label)
            for target in cfg["dataset_names"]:
                if source == target:
                    continue
                sub = df[(df.dataset == target) & (df.split == "test")]
                scores = cnn_metrics(model, loader(sub.path, sub.label, cfg["image_sizes"][label], cfg))
                records.append({"source": source, "target": target, "model": label, **scores})
    pd.DataFrame(records).to_csv(output_path(cfg, "evaluation", "cross_domain.csv"), index=False)


def explain(cfg, per_dataset=3):
    """Save qualitative Grad-CAM visualizations for held-out examples."""
    from PIL import Image
    import matplotlib.pyplot as plt
    from .networks import grad_cam
    df = manifest(cfg)
    for dataset in cfg["dataset_names"]:
        items = df[(df.dataset == dataset) & (df.split == "test")].head(per_dataset)
        for label in cfg["models"]:
            model = trained_model(cfg, dataset, label)
            size = cfg["image_sizes"][label]
            for index, row in enumerate(items.itertuples()):
                with Image.open(row.path) as im:
                    image = im.convert("RGB").resize((size, size))
                tensor = transforms_for(size)(image).unsqueeze(0).to(_device())
                heat, prob = grad_cam(model, tensor, positive=bool(row.label))
                fig, ax = plt.subplots(figsize=(4, 4))
                ax.imshow(image)
                ax.imshow(heat, cmap="jet", alpha=0.4)
                ax.set_title(f"{dataset} {label}: P(fracture)={prob:.3f}")
                ax.axis("off")
                fig.tight_layout()
                fig.savefig(output_path(cfg, "gradcam", f"{dataset}_{label}_{index}.png"), dpi=170)
                plt.close(fig)

