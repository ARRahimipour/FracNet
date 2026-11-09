"""Training-only feature selection, PCA, classifiers, voting and evaluation."""
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score,
                             ConfusionMatrixDisplay)
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from .common import output_path, write_json


def metrics(y, predicted, probability=None):
    """Compute binary metrics with fractured=1 as the positive class."""
    y, predicted = np.asarray(y, int), np.asarray(predicted, int)
    tn, fp, fn, tp = confusion_matrix(y, predicted, labels=[0, 1]).ravel()
    result = {"n": len(y), "accuracy": accuracy_score(y, predicted),
              "precision": precision_score(y, predicted, zero_division=0),
              "recall": recall_score(y, predicted, zero_division=0),
              "specificity": tn / (tn + fp) if tn + fp else None,
              "f1": f1_score(y, predicted, zero_division=0),
              "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)}
    if probability is not None and len(set(y)) == 2:
        result["roc_auc"] = roc_auc_score(y, probability)
    z, n, p = norm.ppf(0.975), len(y), result["accuracy"]
    center = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    result["accuracy_ci95_low"] = center - half
    result["accuracy_ci95_high"] = center + half
    return {k: (float(v) if isinstance(v, np.floating) else v) for k, v in result.items()}


class FeatureProcessor:
    """Fit feature selection on training only; transform all held-out rows."""
    def __init__(self, cfg):
        self.cfg = cfg
        self.select = cfg["scenario"] == "full"

    def fit_transform(self, X, y):
        """Fit correlation, outlier policy, RF ranking, scaler, and optional PCA."""
        X, y = np.asarray(X, np.float32), np.asarray(y, int)
        self.input_width = X.shape[1]
        self.training_mask = np.ones(len(X), dtype=bool)
        if self.select:
            # Remove constant and highly correlated columns before RF ranking.
            std = X.std(axis=0)
            surviving = np.flatnonzero(std > 1e-8)
            if len(surviving) == 0:
                raise ValueError("No nonconstant features")
            if len(surviving) > 1:
                corr = np.nan_to_num(np.abs(np.corrcoef(X[:, surviving], rowvar=False)))
                discard = (np.triu(corr, 1) > self.cfg["correlation_threshold"]).any(axis=0)
                surviving = surviving[~discard]
            self.correlation_columns = surviving
            X = X[:, surviving]
            self.mean, self.sd = X.mean(0), np.maximum(X.std(0), 1e-6)
            # Outlier *rows* are removed from training only; held-out samples
            # remain intact so test metrics have the correct denominator.
            outlier_ratio = np.mean(np.abs((X - self.mean) / self.sd) > self.cfg["z_score_threshold"], axis=1)
            self.training_mask = outlier_ratio <= self.cfg["outlier_row_feature_fraction"]
            if self.training_mask.sum() < 20 or len(set(y[self.training_mask])) != 2:
                raise ValueError("Outlier removal left too few training examples")
            X, y = X[self.training_mask], y[self.training_mask]
            forest = RandomForestClassifier(n_estimators=self.cfg["rf_n_estimators"],
                                            bootstrap=True, oob_score=True,
                                            n_jobs=-1, random_state=self.cfg["seed"])
            forest.fit(X, y)
            # Random-forest importance is a filter criterion; no test labels
            # influence feature ranking or the PCA projection.
            self.feature_indices = np.flatnonzero(forest.feature_importances_ > forest.feature_importances_.mean())
            if len(self.feature_indices) == 0:
                self.feature_indices = np.array([forest.feature_importances_.argmax()])
            self.oob = float(forest.oob_score_)
            X = X[:, self.feature_indices]
        self.scaler = StandardScaler().fit(X)
        X = self.scaler.transform(X)
        if self.select:
            self.pca = PCA(n_components=self.cfg["pca_variance"], svd_solver="full")
            X = self.pca.fit_transform(X)
        return X, y

    def transform(self, X):
        """Apply the frozen training transformation without fitting again."""
        X = np.asarray(X, dtype=np.float32)
        if X.shape[1] != self.input_width:
            raise ValueError("Feature width mismatch")
        if self.select:
            X = X[:, self.correlation_columns][:, self.feature_indices]
        X = self.scaler.transform(X)
        return self.pca.transform(X) if self.select else X

    def audit(self):
        out = {"input_features": self.input_width, "retained_train_rows": int(self.training_mask.sum()),
               "final_features": int(self.pca.n_components_) if self.select else self.scaler.n_features_in_}
        if self.select:
            out.update({"after_correlation": len(self.correlation_columns),
                        "after_rf_importance": len(self.feature_indices),
                        "pca_explained_variance": float(self.pca.explained_variance_ratio_.sum()),
                        "rf_oob_accuracy": self.oob})
        return out


def _load(cfg, split):
    """Vertically stack D1/D2/D3 matrices for exactly one data split."""
    blocks, target, dataset, paths, patients = [], [], [], [], []
    for ds in cfg["dataset_names"]:
        file = output_path(cfg, "features", f"{ds}_{split}.npz")
        if not file.is_file():
            raise FileNotFoundError(f"Run extract before classify; missing {file}")
        with np.load(file, allow_pickle=False) as packed:
            blocks.append(packed["X"])
            target.append(packed["y"])
            dataset.extend(packed["dataset"])
            paths.extend(packed["path"])
            patients.extend(packed["patient"])
    widths = {x.shape[1] for x in blocks}
    if len(widths) != 1:
        raise ValueError("CNN feature dimensions differ between datasets")
    return (np.vstack(blocks), np.concatenate(target), np.asarray(dataset),
            np.asarray(paths), np.asarray(patients))


def _models(cfg):
    """Instantiate the five reported families with reproducible defaults."""
    from xgboost import XGBClassifier
    return {
        "xgboost": XGBClassifier(n_estimators=300, max_depth=5, learning_rate=.05,
                                  eval_metric="logloss", tree_method="hist",
                                  random_state=cfg["seed"], n_jobs=-1),
        "svm": SVC(C=1, kernel="rbf", gamma="scale", probability=True, random_state=cfg["seed"]),
        "knn": KNeighborsClassifier(n_neighbors=5),
        "decision_tree": DecisionTreeClassifier(max_depth=10, random_state=cfg["seed"]),
        "naive_bayes": GaussianNB(),
    }


def _plot_cm(cfg, y, prediction, title):
    fig, ax = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay.from_predictions(y, prediction, labels=[0, 1],
                                            display_labels=["Normal", "Fractured"],
                                            cmap="Blues", colorbar=False, ax=ax)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(output_path(cfg, "evaluation", f"confusion_{title}.png"), dpi=180)
    plt.close(fig)


def _tsne(cfg, X, y):
    """Make an exploratory 2-D plot; never use t-SNE for model selection."""
    from sklearn.manifold import TSNE
    n = min(len(X), cfg["tsne_max_samples"])
    if n < 7:
        return
    ids = np.random.default_rng(cfg["seed"]).choice(len(X), n, replace=False)
    samples = X[ids]
    if samples.shape[1] < 2:
        samples = np.column_stack([samples[:, 0], np.zeros(n)])
    xy = TSNE(n_components=2, perplexity=min(30, max(2, (n - 1) // 3)),
              random_state=cfg["seed"], init="pca", learning_rate="auto").fit_transform(samples)
    fig, ax = plt.subplots(figsize=(7, 5))
    for label in (0, 1):
        mask = y[ids] == label
        ax.scatter(xy[mask, 0], xy[mask, 1], s=14, alpha=.7,
                   label="Fractured" if label else "Normal")
    ax.legend()
    ax.set_title("t-SNE: fused train embeddings (exploratory)")
    fig.tight_layout()
    fig.savefig(output_path(cfg, "evaluation", "tsne_fused.png"), dpi=180)
    plt.close(fig)


def classify(cfg):
    """Train five models and report a held-out three-model majority vote."""
    X_train, y_train, ds_train, path_train, person_train = _load(cfg, "train")
    X_val, y_val, ds_val, path_val, person_val = _load(cfg, "val")
    X_test, y_test, ds_test, path_test, person_test = _load(cfg, "test")
    seen = {}
    for group, split in ((X_train, "train"), (X_val, "val"), (X_test, "test")):
        if len(group) == 0:
            raise ValueError(f"Empty {split} split")
    for datasets, persons, split in [(ds_train, person_train, "train"),
                                     (ds_val, person_val, "val"),
                                     (ds_test, person_test, "test")]:
        for key in zip(datasets, persons):
            if key in seen and seen[key] != split:
                raise ValueError("Patient leakage across splits")
            seen[key] = split
    processor = FeatureProcessor(cfg)
    # The pipeline is fitted exactly once on training features and labels.
    trainX, trainY = processor.fit_transform(X_train, y_train)
    validX, testX = processor.transform(X_val), processor.transform(X_test)
    write_json(output_path(cfg, "evaluation", "features.json"), processor.audit())
    models = _models(cfg)
    rows = []
    for name, model in models.items():
        model.fit(trainX, trainY)
        for split, X, y, ds in [("val", validX, y_val, ds_val),
                                ("test", testX, y_test, ds_test)]:
            probability = model.predict_proba(X)[:, 1]
            prediction = model.predict(X)
            for dataset in ["ALL", *cfg["dataset_names"]]:
                mask = np.ones(len(y), dtype=bool) if dataset == "ALL" else ds == dataset
                rows.append({"model": name, "split": split, "dataset": dataset,
                             **metrics(y[mask], prediction[mask], probability[mask])})
        print(f"Fitted classifier {name}", flush=True)
    votes = cfg["vote_models"]
    if any(name not in models for name in votes):
        raise ValueError("Unknown model in FRACNET_VOTE_MODELS")
    for split, X, y, ds in [("val", validX, y_val, ds_val), ("test", testX, y_test, ds_test)]:
        # Hard voting uses the already fitted models; it does not refit them.
        prediction = np.stack([models[n].predict(X) for n in votes]).sum(axis=0) > len(votes) // 2
        for dataset in ["ALL", *cfg["dataset_names"]]:
            mask = np.ones(len(y), dtype=bool) if dataset == "ALL" else ds == dataset
            rows.append({"model": "majority_vote", "split": split, "dataset": dataset,
                         **metrics(y[mask], prediction[mask])})
            if split == "test":
                _plot_cm(cfg, y[mask], prediction[mask], dataset)
        if split == "test":
            pd.DataFrame({"path": path_test, "dataset": ds_test,
                          "actual": y_test, "predicted_vote": prediction.astype(int)}).to_csv(
                output_path(cfg, "evaluation", "test_predictions.csv"), index=False)
    pd.DataFrame(rows).to_csv(output_path(cfg, "evaluation", "metrics.csv"), index=False)
    _tsne(cfg, trainX, trainY)
    joblib.dump({"processor": processor, "models": models, "voters": votes},
                output_path(cfg, "evaluation", "trained_classifiers.joblib"))

