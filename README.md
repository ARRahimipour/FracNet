# FracNet: Reproducible Research Implementation

A Python/PyTorch reconstruction of the fracture-classification workflow described in:

> Alwzwazy, H. A., Alzubaidi, L., Zhao, Z., and Gu, Y. (2025). **FracNet: An end-to-end deep learning framework for bone fracture detection.** *Pattern Recognition Letters*, 190, 1–7. https://doi.org/10.1016/j.patrec.2025.01.034

**Researcher:** Alireza Rahimipour Anaraki  
**Program:** Ph.D. Candidate in Computer Engineering – Artificial Intelligence  
**Institution:** Department of Computer Engineering, Islamic Azad University, North Tehran Branch

> **Research software only.** The supplied code has been checked with synthetic, nonmedical unit tests. The original X-ray datasets, the authors' checkpoints, and the external 100-image verification set are not included. The paper's 100% accuracy has **not** been reproduced or independently validated here.

## Project overview

FracNet combines domain-adapted deep features with classical machine learning. This implementation contains:

1. Three CNN backbones: Xception, MobileNetV2, and EfficientNet-B0.
2. Domain self-supervised pretraining (an explicit SimCLR/NT-Xent proxy for the paper's insufficiently specified SSP method).
3. Spatial self-attention, Squeeze-and-Excitation channel attention, and 70% backbone-layer freezing.
4. Nine downstream training jobs: three architectures evaluated independently on three datasets.
5. Horizontal feature fusion per sample and vertical fusion across datasets.
6. Training-only correlation filtering, z-score-based outlier exclusion, random-forest importance, scaling, and PCA.
7. XGBoost, RBF-SVM, KNN, decision tree, Gaussian Naive Bayes, and majority voting.
8. Held-out evaluation, transfer between datasets, confusion matrices, t-SNE, and Grad-CAM visualizations.

Implementation details and disclosed departures from the publication are in [IMPLEMENTATION_NOTES.md](docs/IMPLEMENTATION_NOTES.md).

## Files

```text
FracNet_Complete_ENV/
  .env.example             # Safe, version-controlled template
  .env                     # Local configuration (gitignored)
  .gitignore
  requirements.txt
  run.py                   # Training and evaluation CLI
  predict.py               # Research-only image inference
  fracnet/
    common.py              # Typed .env loading and utilities
    data.py                # Manifests, grouping, deduplication and image loading
    networks.py            # CNNs, attention, SSL, Grad-CAM
    deep.py                # Deep learning stages
    classical.py           # Feature selection and classical classifiers
  tests/
    test_env_config.py
    test_classical.py
  docs/
    IMPLEMENTATION_NOTES.md
```

## Windows CMD setup

Open File Explorer in the directory **containing `run.py`**, type `cmd` in the address bar, and press Enter.

```cmd
py --version
py -m venv .venv
.venv\Scripts\activate.bat
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The official [PyTorch installation selector](https://pytorch.org/get-started/locally/) provides CPU/CUDA-specific installation commands. This pipeline automatically uses CUDA when the installed PyTorch build recognizes a compatible GPU; otherwise it uses CPU. Full pretraining and nine CNN training jobs can be prohibitively slow on CPU.

**Create your own configuration from the safe template:**

```cmd
copy .env.example .env
```

Edit `.env` for your image directories and available computing resources. Do not commit API keys, private file paths, or patient data.

## Smoke checks (no medical images needed)

```cmd
python run.py check
python -m unittest discover -s tests -v
```

`check` confirms that the `.env` values are readable. A successful unit test suite **does not** establish that real-model training was completed or that clinical predictions are accurate.

## Download and organize image datasets

The paper cites these sources:

- **D1:** [Fracture Classification Dataset on Kaggle](https://www.kaggle.com/datasets/akshayramakrishnan28/fracture-classification-dataset)
- **D2:** [Fracture Detection Using X-ray Images on Kaggle](https://www.kaggle.com/datasets/devbatrax/fracture-detection-using-x-ray-images)
- **D3:** [Wrist Fracture – X-rays on Mendeley Data](https://data.mendeley.com/datasets/xbdsnzr8ct/1)
- **Unlabeled SSL images:** e.g. [Stanford MURA](https://stanfordmlgroup.github.io/competitions/mura/) and other licensed, non-overlapping X-rays.

After checking each dataset's license, provenance, and labels, organize the files as:

```text
data/
  D1/
    fractured/*.png
    normal/*.png
  D2/
    fractured/*.jpg
    normal/*.jpg
  D3/
    fractured/*.png
    normal/*.png
  ssl_unlabeled/
    [any nested directories with eligible X-ray images]
```

Supported formats include JPG, PNG, BMP, TIFF, and WebP. The folder scanner uses filenames as **placeholder** group IDs; these cannot establish real patient separation. For patient-level evaluation, provide `data/manifest.csv` with `path,dataset,label,patient_id` using actual, verified patient IDs. Labels are `1=fractured` and `0=normal`.

## Execution

Prepare and verify the dataset manifest:

```cmd
python run.py prepare --config .env --data-root data
```

Run every stage in order:

```cmd
python run.py ssl --config .env
python run.py train --config .env
python run.py cnn_eval --config .env
python run.py extract --config .env
python run.py classify --config .env
python run.py cross_domain --config .env
python run.py gradcam --config .env
```

Alternatively, `python run.py all --config .env` runs the pipeline on a prepared manifest. The pipeline does not automatically resume interrupted neural-network training, so preserve checkpoint directories.

The following outputs are generated **after real training**:

```text
runs/full/
  ssl/                        # Encoder weights and SSL logs
  cnn/D1/ ... cnn/D3/         # Nine fine-tuned CNNs and training histories
  features/                   # Fused sample embeddings
  evaluation/
    cnn_base_metrics.csv
    cross_domain.csv
    metrics.csv
    test_predictions.csv
    confusion_*.png
    tsne_fused.png
    features.json
    trained_classifiers.joblib
  gradcam/                    # Qualitative heatmaps
```

### New-image inference

Only after training and evaluating the model:

```cmd
python predict.py --config .env --image "C:\path\to\xray.png" --domain D1
```

The `--domain` option chooses a **dataset-specific encoder triplet**. Cross-domain inference can be unreliable without independent validation. Only load `joblib` models you created or explicitly trust, because deserialization can execute arbitrary code.

## Ablation studies

Set `FRACNET_SCENARIO` in `.env`:

| Scenario | Configuration |
|---|---|
| `S1` | ImageNet-based CNNs, no domain SSL, no added attention |
| `S2` | Domain SSL, no added attention |
| `S3` | Domain SSL, attention, feature fusion, no feature selection |
| `full` | SSL, attention, feature fusion, feature selection and voting |

Assign each scenario a **different** `FRACNET_OUTPUT_DIR` to avoid overwriting checkpoints.

## Scientific and clinical limitations

- This independently written PyTorch reconstruction is **not identical** to the paper's MATLAB implementation.
- The original authors did not fully specify the SSL procedure, attention placement, dataset split, exact outlier aggregation, and all classifier hyperparameters. Explicit choices are documented.
- The code rejects exact duplicate image bytes during manifest preparation; near-duplicates and patient overlap across differently encoded files still require independent audit.
- All feature selection, scaling, and PCA are fitted on **training data only**.
- Grad-CAM shows model sensitivity, not verified fracture localization.
- Medical data must not be shared or redistributed without appropriate authorization.

