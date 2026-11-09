# FracNet Implementation Notes

## Purpose and status

This document connects the independent Python implementation to the paper and to the accompanying machine-learning lecture binder (CE-717 / CE-725, Sharif University of Technology, M. Soleymani). It describes **implemented algorithms** and **assumptions**, not verified clinical results.

## End-to-end workflow

1. **Dataset manifest:** binary labels, dataset identity, patient grouping, exact-file SHA-256 deduplication, and grouped train/validation/test folds.
2. **Domain self-supervision:** three independently trained encoders using two augmentations per image and normalized contrastive embeddings.
3. **Downstream fine-tuning:** dataset-specific CNNs with frozen early backbone layers, a trainable spatial attention block, and a Squeeze-and-Excitation block.
4. **Feature fusion:** concatenate each sample's Xception, MobileNetV2, and EfficientNet representations; then stack D1, D2, and D3 samples along the sample axis.
5. **Feature selection:** train-only correlation filtering, z-score outlier policy, random-forest importance ranking, standardization, and 95%-variance PCA.
6. **Classification:** independent XGBoost, RBF-SVM, KNN, decision tree, and Gaussian Naive Bayes estimators; majority vote among XGBoost/SVM/KNN.
7. **Reporting:** per-dataset and pooled metrics, cross-domain transfer, confusion matrices, exploratory t-SNE and qualitative Grad-CAM.

## Core equations

**Attention.** For feature map `F`, linear projections produce `Q`, `K`, and `V`. Spatial attention uses:

\[
\operatorname{Attention}(F) = \operatorname{softmax}(QK^\top / \sqrt{d})V.
\]

The code adds a learnable residual scaling parameter and then applies channel-wise SE gating.

**Feature concatenation.** For image `i`, the feature vector is:

\[
h_i = [h_i^{\mathrm{Xception}};h_i^{\mathrm{MobileNetV2}};h_i^{\mathrm{EfficientNet}}].
\]

The feature matrices of D1, D2, and D3 are concatenated by rows. Training, validation, and test matrices remain separate throughout.

**Correlation filter.** Redundant features with absolute Pearson correlation exceeding `0.95` are removed, using the training matrix only. Constant features are removed before correlation calculation.

**Outlier filter.** Training samples are excluded if more than `2%` of remaining feature z-scores exceed `3` in absolute value. This row aggregation is an implementation assumption; the original paper gives a z-score threshold but not the aggregation rule.

**Random forest.** A 100-tree forest estimates feature importance. Features with importance greater than the mean are retained. PCA preserves at least `95%` of training variance after scaling.

**Final prediction.** Each selected classifier predicts a binary label, and the final decision is the majority of three votes.

## Source-to-code mapping

| Research method | Implementation | Lecture binder connection (PDF pages) |
|---|---|---|
| Data splitting, supervised learning | `data.py`, `run.py` | 11, 20, 33–35, 455–464 |
| Deep feature learning | `networks.py`, `deep.py` | General supervised learning and optimization, 377–395 |
| Decision trees and bagging | `classical.py` | 99–149 |
| Feature selection | `classical.py` | 187–213 |
| K-nearest neighbors | `classical.py` | 254–299 |
| Kernel SVM | `classical.py` | 506–589 |
| Model validation and generalization | `data.py`, `classical.py` | 455–464 |

## Choices that cannot be verified from the paper

| Topic | Published description | Explicit implementation choice |
|---|---|---|
| SSP recipe | Domain self-supervised pretraining using more than 50,000 X-rays | SimCLR/NT-Xent proxy with configurable epochs and temperature |
| CNN architectures | Xception, MobileNetV2, EfficientNet | `legacy_xception`, `mobilenetv2_100`, `efficientnet_b0` from timm |
| Attention placement | Self-attention and SE added to the base models | Both operate on the final spatial backbone feature map |
| Frozen layers | 70% of layers frozen | 70% of parameter-owning backbone leaf modules in traversal order |
| Data split | Training, validation, testing | Approximately 70/15/15 grouped folds, with actual proportions varying |
| Optimizer | Adam, learning rate 0.001, batch size 15, up to 100 epochs | As published, plus class weighting, validation F1 checkpointing, and early stopping |
| Outlier threshold | Absolute z-score threshold 3 | More than 2% of features above threshold triggers removal of a training sample |
| Final classifier settings | Five reported classifiers, three-classifier majority vote | Documented defaults in `classical.py`, without claiming exact paper parity |

## Verification boundary

The local tests exercise environment parsing, type checking, known classification metrics, and held-out feature transformations on **synthetic data**. The following require additional real-data experiments:

- Reproducing published metrics on all three original sources.
- Repeating ablations S1, S2, S3, and full with identical patient-safe splits.
- Quantifying robustness to other hospitals and acquisition devices.
- Confirming fracture localization against expert annotations.
- Testing the additional 100 independently expert-reviewed images described in the paper; those exact files were not supplied.

**Do not convert paper-reported percentages into claimed experimental results.** Report actual training configurations, sample counts, runtime environment, uncertainty intervals, and test-set policies.

