# -*- coding: utf-8 -*-
"""
Spatial OI-Mamba: repeated stratified cross-validation and regional prediction
===========================================================================

This script expects two Excel files with English column names:

1. ``data/labeled_samples.xlsx``
   - The study used 36 labeled observations (18 positive and 18 negative).
   - Required columns: X, Y, 12 geochemical elements, mineralization_label.
   - Class labels: 1 = positive mineralization sample; 0 = pseudo-negative sample.

2. ``data/prediction_grid.xlsx``
   - The study used 14,400 grid cells for regional prediction.
   - Required columns: X, Y, and the same 12 geochemical elements.
   - An optional label column in this file is ignored during prediction.

Methodology
-----------
- Apply log1p and fold-fitted StandardScaler transformations to element values.
- Apply small Gaussian perturbations to standardized element values during training only.
- Randomly permute element tokens and their identity indices together per training sample.
- Combine learnable element-identity embeddings with 2D sinusoidal coordinate encoding.
- Encode element tokens with two Mamba layers, Pre-LayerNorm, and residual connections.
- Apply mean pooling across element-token outputs for binary classification.
- Repeat stratified five-fold cross-validation ten times (50 fitted fold models).
- Aggregate five out-of-fold (OOF) validation folds within each repeat before
  calculating metrics; report the mean and standard deviation across repeats.
- Fit a final ensemble using the median CV-selected epoch count and average predictions
  over ensemble members and multiple element permutations.

PFI, local occlusion, and Gradient x Input are implemented in the separate
``oi_mamba_explainability_oof.py`` script.

Installation
------------
Install the packages listed in ``requirements.txt``, and install ``mamba-ssm`` and
``causal-conv1d`` with versions compatible with your PyTorch/CUDA environment.

Usage
-----
    python oi_mamba_repeated_cv_train_predict.py --mode check
    python oi_mamba_repeated_cv_train_predict.py --mode cv
    python oi_mamba_repeated_cv_train_predict.py --mode all
    python oi_mamba_repeated_cv_train_predict.py --mode all \
        --labeled-file data/labeled_samples.xlsx \
        --prediction-file data/prediction_grid.xlsx \
        --output-dir results/oi_mamba_cv

Outputs
-------
Default directory: ``oi_mamba_repeated_cv_results``.
- ``config.json``: resolved run configuration.
- ``input_data_summary.json``: input validation summary.
- ``cv_results.xlsx``: fold metrics, repeat metrics, and summary statistics.
- ``cv_oof_predictions_long.xlsx``: OOF probabilities for every repeat.
- ``cv_oof_mean_predictions.xlsx``: per-sample average OOF predictions.
- ``cv_summary.json``: CV statistics and suggested final epoch count.
- ``final_models/*.pt``: weights for models fitted to all labeled samples.
- ``final_preprocessor.joblib``: fitted feature and coordinate scalers.
- ``prediction_all_with_prob.xlsx``: regional anomaly probabilities and labels.
- ``final_training_summary.xlsx``: ensemble fitting information.

Methodological notes
--------------------
1. Gaussian noise and random element orders generate training views, not new samples.
2. Each CV fold fits preprocessing transformations on its training split only.
3. Regional predictions are not used to select cross-validation results.
4. Random seeds are repeatability controls, not tuned hyperparameters.
5. The outer validation fold is also used for early stopping in this implementation.
   For a stricter evaluation, introduce a separate inner early-stopping split.
6. Order standard deviation here is computed over the configured inference orders
   (default 10), rather than the 50-order stability test described in the manuscript.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import warnings
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

# Installation of mamba-ssm varies across operating systems and CUDA versions.
# Keep --mode check usable without Mamba by recording an import error now;
# raise an actionable error only when training is requested.
try:
    from mamba_ssm import Mamba
except ImportError as exc:  # pragma: no cover - environment-dependent dependency
    Mamba = None
    MAMBA_IMPORT_ERROR = exc
else:
    MAMBA_IMPORT_ERROR = None


# =============================================================================
# 1. Configuration
# =============================================================================


@dataclass
class Config:
    """Configuration for data inputs, model, validation, and outputs."""

    # --------------------------- Input and output ---------------------------
    labeled_file: str = "data/labeled_samples.xlsx"
    prediction_file: str = "data/prediction_grid.xlsx"
    output_dir: str = "oi_mamba_repeated_cv_results"

    # --------------------------- Input column names ---------------------------
    coord_cols: Tuple[str, str] = ("X", "Y")
    feature_cols: Tuple[str, ...] = (
        "Sb",
        "Ti",
        "V",
        "Mo",
        "As",
        "Zn",
        "Pb",
        "Cu",
        "Ag",
        "Hg",
        "Au",
        "W",
    )
    label_col: str = "mineralization_label"

    # --------------------------- Cross-validation ---------------------------
    n_splits: int = 5
    n_repeats: int = 10
    cv_random_state: int = 2026
    base_training_seed: int = 1000

    # --------------------------- Model architecture ---------------------------
    d_model: int = 64
    d_state: int = 16
    d_conv: int = 4
    expand: int = 2
    num_layers: int = 2
    dropout: float = 0.30

    # --------------------------- Optimization ---------------------------
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    batch_size: int = 8
    max_epochs: int = 200
    early_stopping_patience: int = 25
    early_stopping_min_delta: float = 1e-4
    gradient_clip_norm: float = 5.0
    classification_threshold: float = 0.5

    # --------------------------- Online data augmentation ---------------------------
    # Add noise to elemental features after log1p and StandardScaler.
    # A value of 0.05 is a small perturbation in standardized feature space.
    gaussian_noise_std: float = 0.05

    # --------------------------- Permutation-averaged inference ---------------------------
    # Use fewer permutations for early stopping and more for final fold evaluation.
    early_stop_permutations: int = 3
    evaluation_permutations: int = 10

    # --------------------------- Final ensemble fitted to all labels ---------------------------
    # After CV, fit several models to all labeled observations and average probabilities.
    final_ensemble_size: int = 10
    final_seed_start: int = 20000
    prediction_batch_size: int = 1024

    # --------------------------- Runtime configuration ---------------------------
    device: str = "auto"  # auto / cuda / cpu
    num_workers: int = 0


# =============================================================================
# 2. Shared utilities
# =============================================================================


def set_global_seed(seed: int) -> None:
    """Seed the Python, NumPy, and PyTorch random number generators."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # These settings improve repeatability; some CUDA kernels may still vary slightly.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def resolve_device(requested: str) -> torch.device:
    """Select a PyTorch device based on the requested option and hardware."""

    requested = requested.lower()
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("--device cuda was requested, but CUDA is unavailable.")
        return torch.device("cuda")
    if requested == "cpu":
        return torch.device("cpu")
    raise ValueError("--device must be one of: auto, cuda, cpu.")


def require_mamba() -> None:
    """Verify that mamba-ssm is installed before fitting a model."""

    if Mamba is None:
        raise ImportError(
            "Could not import mamba_ssm.Mamba. Install compatible versions of "
            "mamba-ssm and causal-conv1d for your PyTorch/CUDA environment. Original error:"
            f"\n{MAMBA_IMPORT_ERROR}"
        )


def json_default(value):
    """Convert NumPy scalars, arrays, and paths into JSON-compatible values."""

    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"Cannot serialize object of type {type(value)}")


def save_json(path: Path, data: Dict) -> None:
    """Write a JSON file encoded in UTF-8."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2, default=json_default)


def resolve_path(path_text: str, script_dir: Path) -> Path:
    """Resolve relative paths from the script directory, not the working directory."""

    path = Path(path_text)
    if path.is_absolute():
        return path
    return script_dir / path


# =============================================================================
# 3. Input loading and validation
# =============================================================================


def _convert_columns_to_numeric(
    df: pd.DataFrame,
    columns: Sequence[str],
    file_name: str,
) -> pd.DataFrame:
    """Convert the specified columns to numeric values with clear error reporting."""

    converted = df.copy()
    for column in columns:
        converted[column] = pd.to_numeric(converted[column], errors="coerce")

    missing_counts = converted[list(columns)].isna().sum()
    bad_columns = missing_counts[missing_counts > 0]
    if not bad_columns.empty:
        detail = ", ".join(f"{col}: {int(count)}" for col, count in bad_columns.items())
        raise ValueError(f"File {file_name} has missing or invalid numeric values in: {detail}")

    values = converted[list(columns)].to_numpy(dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError(f"File {file_name} contains non-finite numeric values.")

    return converted


def load_and_validate_data(
    labeled_path: Path,
    prediction_path: Path,
    cfg: Config,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict]:
    """
    Load and validate the two Excel workbooks.

    Returns
    ----
    labeled_df:
        Labeled observations used in model fitting and validation.
    prediction_df:
        Regional grid cells for prediction.
    summary:
        Summary of the input datasets.
    """

    if not labeled_path.exists():
        raise FileNotFoundError(f"Labeled sample file not found: {labeled_path}")
    if not prediction_path.exists():
        raise FileNotFoundError(f"Prediction grid file not found: {prediction_path}")

    labeled_df = pd.read_excel(labeled_path)
    prediction_df = pd.read_excel(prediction_path)

    labeled_required = list(cfg.coord_cols) + list(cfg.feature_cols) + [cfg.label_col]
    prediction_required = list(cfg.coord_cols) + list(cfg.feature_cols)

    missing_labeled = [col for col in labeled_required if col not in labeled_df.columns]
    missing_prediction = [col for col in prediction_required if col not in prediction_df.columns]

    if missing_labeled:
        raise ValueError(f"Missing columns in labeled data: {missing_labeled}")
    if missing_prediction:
        raise ValueError(f"Missing columns in prediction data: {missing_prediction}")

    labeled_df = _convert_columns_to_numeric(
        labeled_df,
        columns=list(cfg.coord_cols) + list(cfg.feature_cols) + [cfg.label_col],
        file_name=labeled_path.name,
    )
    prediction_df = _convert_columns_to_numeric(
        prediction_df,
        columns=list(cfg.coord_cols) + list(cfg.feature_cols),
        file_name=prediction_path.name,
    )

    # Class labels must be binary: 0 or 1.
    label_values = labeled_df[cfg.label_col].to_numpy(dtype=np.float64)
    if not np.all(np.isin(label_values, [0.0, 1.0])):
        unique_values = sorted(pd.unique(labeled_df[cfg.label_col]).tolist())
        raise ValueError(f"Label column {cfg.label_col} must contain only 0/1; found: {unique_values}")
    labeled_df[cfg.label_col] = labeled_df[cfg.label_col].astype(np.int64)

    class_counts = labeled_df[cfg.label_col].value_counts().sort_index().to_dict()
    if set(class_counts.keys()) != {0, 1}:
        raise ValueError("Both negative (0) and positive (1) examples are required.")
    if min(class_counts.values()) < cfg.n_splits:
        raise ValueError(
            f"The smallest class has {min(class_counts.values())} samples, fewer than "
            f"n_splits={cfg.n_splits}; stratified cross-validation is not feasible."
        )

    # Raw geochemical concentrations should normally be nonnegative.
    # Warn if negative values are found; clip them to zero before log1p.
    labeled_negative = int((labeled_df[list(cfg.feature_cols)] < 0).sum().sum())
    prediction_negative = int((prediction_df[list(cfg.feature_cols)] < 0).sum().sum())
    if labeled_negative > 0 or prediction_negative > 0:
        warnings.warn(
            "Negative elemental concentrations were detected and will be clipped to 0 "
            f"before log1p. Labeled: {labeled_negative}; grid: {prediction_negative}."
        )

    duplicate_labeled_xy = int(labeled_df.duplicated(subset=list(cfg.coord_cols)).sum())
    duplicate_prediction_xy = int(prediction_df.duplicated(subset=list(cfg.coord_cols)).sum())
    if duplicate_labeled_xy > 0:
        warnings.warn(f"Labeled data contains {duplicate_labeled_xy} duplicate XY coordinates.")
    if duplicate_prediction_xy > 0:
        warnings.warn(f"Prediction grid contains {duplicate_prediction_xy} duplicate XY coordinates.")

    summary = {
        "labeled_file": str(labeled_path),
        "prediction_file": str(prediction_path),
        "num_labeled_samples": int(len(labeled_df)),
        "num_prediction_samples": int(len(prediction_df)),
        "class_counts": {str(k): int(v) for k, v in class_counts.items()},
        "coord_cols": list(cfg.coord_cols),
        "feature_cols": list(cfg.feature_cols),
        "label_col": cfg.label_col,
        "duplicate_labeled_xy": duplicate_labeled_xy,
        "duplicate_prediction_xy": duplicate_prediction_xy,
        "negative_feature_values_labeled": labeled_negative,
        "negative_feature_values_prediction": prediction_negative,
    }

    return labeled_df.reset_index(drop=True), prediction_df.reset_index(drop=True), summary


# =============================================================================
# 4. Preprocessing
# =============================================================================


def log1p_features(raw_features: np.ndarray) -> np.ndarray:
    """Apply log(1 + x) after clipping negative concentration values to zero."""

    clipped = np.clip(raw_features.astype(np.float64), a_min=0.0, a_max=None)
    return np.log1p(clipped)


@dataclass
class FittedPreprocessor:
    """Store the feature and coordinate StandardScaler instances."""

    feature_scaler: StandardScaler
    coord_scaler: StandardScaler
    feature_cols: Tuple[str, ...]
    coord_cols: Tuple[str, str]

    def transform(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Transform a DataFrame with matching feature and coordinate columns."""

        raw_x = df[list(self.feature_cols)].to_numpy(dtype=np.float64)
        raw_c = df[list(self.coord_cols)].to_numpy(dtype=np.float64)

        x_log = log1p_features(raw_x)
        x_scaled = self.feature_scaler.transform(x_log)
        c_scaled = self.coord_scaler.transform(raw_c)

        return x_scaled.astype(np.float32), c_scaled.astype(np.float32)


def fit_preprocessor(train_df: pd.DataFrame, cfg: Config) -> FittedPreprocessor:
    """Fit feature and coordinate scalers using only the training data."""

    raw_x = train_df[list(cfg.feature_cols)].to_numpy(dtype=np.float64)
    raw_c = train_df[list(cfg.coord_cols)].to_numpy(dtype=np.float64)

    x_log = log1p_features(raw_x)

    feature_scaler = StandardScaler().fit(x_log)
    coord_scaler = StandardScaler().fit(raw_c)

    return FittedPreprocessor(
        feature_scaler=feature_scaler,
        coord_scaler=coord_scaler,
        feature_cols=cfg.feature_cols,
        coord_cols=cfg.coord_cols,
    )


# =============================================================================
# 5. Spatial OI-Mamba model
# =============================================================================


class SpatialOIMamba(nn.Module):
    """
    Spatially aware Mamba binary classifier with reduced element-order sensitivity.

    Inputs
    ----
    features:
        Standardized geochemical array with shape (B, 12).
    coords:
        Standardized coordinate array with shape (B, 2).
    permute:
        If True, independently shuffle element values and identities per sample.
    permutation:
        Optional explicit permutation indices of shape (B, 12) for inference.
    """

    def __init__(self, cfg: Config):
        super().__init__()
        require_mamba()

        self.num_features = len(cfg.feature_cols)
        self.d_model = cfg.d_model

        if self.d_model % 4 != 0:
            raise ValueError("d_model must be divisible by 4 for 2D sinusoidal encoding.")

        # Share one scalar-to-vector value projection across all elements.
        self.feature_projection = nn.Linear(1, cfg.d_model)

        # A separate learnable embedding identifies each chemical element.
        self.element_id_embedding = nn.Embedding(self.num_features, cfg.d_model)

        # Stack Mamba blocks with Pre-LayerNorm, residual updates, and dropout.
        self.mamba_layers = nn.ModuleList(
            [
                Mamba(
                    d_model=cfg.d_model,
                    d_state=cfg.d_state,
                    d_conv=cfg.d_conv,
                    expand=cfg.expand,
                )
                for _ in range(cfg.num_layers)
            ]
        )
        self.layer_norms = nn.ModuleList(
            [nn.LayerNorm(cfg.d_model) for _ in range(cfg.num_layers)]
        )
        self.dropout = nn.Dropout(cfg.dropout)

        # Mean-pool token outputs before applying the binary classification head.
        self.classifier = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.d_model // 2),
            nn.ReLU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.d_model // 2, 2),
        )

    def build_2d_sincos_encoding(self, coords: torch.Tensor) -> torch.Tensor:
        """
        Encode sample coordinates into a d_model-dimensional sinusoidal vector.

        The encoding refers to geographic coordinates, not element-token positions.
        All element tokens from one sample receive the same spatial encoding.
        """

        frequency_dim = self.d_model // 4
        denominator = max(frequency_dim - 1, 1)

        # Frequencies vary smoothly from 1 to 1/10000.
        omega = torch.arange(
            frequency_dim,
            device=coords.device,
            dtype=coords.dtype,
        ) / denominator
        omega = 1.0 / (10000.0 ** omega)

        x_phase = coords[:, 0:1] * omega.unsqueeze(0)
        y_phase = coords[:, 1:2] * omega.unsqueeze(0)

        return torch.cat(
            [
                torch.sin(x_phase),
                torch.cos(x_phase),
                torch.sin(y_phase),
                torch.cos(y_phase),
            ],
            dim=1,
        )

    @staticmethod
    def _apply_permutation(
        features: torch.Tensor,
        element_ids: torch.Tensor,
        permutation: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Reorder element values and element IDs using the same per-sample indices."""

        if permutation.ndim != 2:
            raise ValueError("permutation must have shape (B, num_features).")
        if permutation.shape != features.shape:
            raise ValueError(
                f"Permutation shape {tuple(permutation.shape)} does not match "
                f"feature shape {tuple(features.shape)}."
            )

        features = torch.gather(features, dim=1, index=permutation)
        element_ids = torch.gather(element_ids, dim=1, index=permutation)
        return features.contiguous(), element_ids.contiguous()

    def forward(
        self,
        features: torch.Tensor,
        coords: torch.Tensor,
        permute: bool = False,
        permutation: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        batch_size, num_features = features.shape
        if num_features != self.num_features:
            raise ValueError(
                f"Expected {self.num_features} elemental features, received {num_features}."
            )

        # Element identities always refer to the order in cfg.feature_cols.
        element_ids = torch.arange(
            self.num_features,
            device=features.device,
            dtype=torch.long,
        ).unsqueeze(0).expand(batch_size, -1)

        # Training uses random per-sample orders; inference can supply fixed orders.
        if permutation is not None:
            features, element_ids = self._apply_permutation(
                features,
                element_ids,
                permutation.to(features.device),
            )
        elif permute:
            # Independent per-sample permutations from ranks of random scores.
            random_scores = torch.rand(
                batch_size,
                self.num_features,
                device=features.device,
            )
            random_permutation = torch.argsort(random_scores, dim=1)
            features, element_ids = self._apply_permutation(
                features,
                element_ids,
                random_permutation,
            )

        # Shared value projection: (B, 12) -> (B, 12, 1) -> (B, 12, d_model).
        tokens = self.feature_projection(features.unsqueeze(-1))

        # Add identity embeddings to scalar element-token projections.
        tokens = tokens + self.element_id_embedding(element_ids)

        # Broadcast the sample-level geographic encoding to every element token.
        spatial_encoding = self.build_2d_sincos_encoding(coords.to(tokens.dtype))
        tokens = tokens + spatial_encoding.unsqueeze(1)
        tokens = self.dropout(tokens)

        # Pass tokens through the stacked Mamba encoder.
        for layer_norm, mamba_layer in zip(self.layer_norms, self.mamba_layers):
            residual = tokens
            hidden = layer_norm(tokens)
            hidden = mamba_layer(hidden)
            tokens = self.dropout(hidden + residual)

        # Symmetric mean pooling avoids reliance on the last element token.
        pooled = tokens.mean(dim=1)
        return self.classifier(pooled)


# =============================================================================
# 6. Optimization, prediction, and evaluation
# =============================================================================


@dataclass
class EarlyStopping:
    """Early stop using validation F1, breaking ties with validation log loss."""

    patience: int
    min_delta: float
    best_f1: float = -math.inf
    best_loss: float = math.inf
    best_epoch: int = 0
    counter: int = 0
    best_state: Optional[Dict[str, torch.Tensor]] = None

    def step(
        self,
        f1_value: float,
        loss_value: float,
        model: nn.Module,
        epoch: int,
    ) -> bool:
        """Update early stopping; return True when training should stop."""

        f1_improved = f1_value > self.best_f1 + self.min_delta
        f1_tied = abs(f1_value - self.best_f1) <= self.min_delta
        loss_improved = loss_value < self.best_loss - self.min_delta

        if f1_improved or (f1_tied and loss_improved):
            self.best_f1 = float(f1_value)
            self.best_loss = float(loss_value)
            self.best_epoch = int(epoch)
            self.counter = 0
            self.best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
        else:
            self.counter += 1

        return self.counter >= self.patience


def build_loader(
    features: np.ndarray,
    coords: np.ndarray,
    labels: Optional[np.ndarray],
    batch_size: int,
    shuffle: bool,
    seed: int,
    num_workers: int,
) -> DataLoader:
    """Create a DataLoader from NumPy arrays."""

    x_tensor = torch.as_tensor(features, dtype=torch.float32)
    c_tensor = torch.as_tensor(coords, dtype=torch.float32)

    if labels is None:
        dataset = TensorDataset(x_tensor, c_tensor)
    else:
        y_tensor = torch.as_tensor(labels, dtype=torch.long)
        dataset = TensorDataset(x_tensor, c_tensor, y_tensor)

    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=False,
        num_workers=num_workers,
        generator=generator if shuffle else None,
        pin_memory=torch.cuda.is_available(),
    )


def train_one_epoch(
    model: SpatialOIMamba,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    gaussian_noise_std: float,
    gradient_clip_norm: float,
) -> float:
    """Train for one epoch, applying augmentation to elemental training values only."""

    model.train()
    total_loss = 0.0
    total_samples = 0

    for features, coords, labels in loader:
        features = features.to(device, non_blocking=True)
        coords = coords.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        # Fresh Gaussian noise is generated for each training batch.
        if gaussian_noise_std > 0:
            features = features + torch.randn_like(features) * gaussian_noise_std

        optimizer.zero_grad(set_to_none=True)

        # Randomly permute element values together with their identity embeddings.
        logits = model(features, coords, permute=True)
        loss = criterion(logits, labels)
        loss.backward()

        if gradient_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_norm)

        optimizer.step()

        batch_size = features.size(0)
        total_loss += float(loss.item()) * batch_size
        total_samples += batch_size

    return total_loss / max(total_samples, 1)


def make_deterministic_permutation(
    batch_size: int,
    num_features: int,
    seed: int,
    device: torch.device,
) -> torch.Tensor:
    """
    Use a local CPU random generator to build deterministic per-sample orders.

    Do not change the global PyTorch RNG state.
    """

    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    random_scores = torch.rand(
        (batch_size, num_features),
        generator=generator,
        device="cpu",
    )
    return torch.argsort(random_scores, dim=1).to(device)


@torch.no_grad()
def predict_probabilities(
    model: SpatialOIMamba,
    features: np.ndarray,
    coords: np.ndarray,
    device: torch.device,
    batch_size: int,
    num_permutations: int,
    permutation_seed: int,
    num_workers: int = 0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Predict positive-class probabilities and average over element orders.

    The first inference pass uses the original order; later passes use fixed shuffles.

    Returns
    ----
    mean_prob:
        Mean positive-class probability for each sample.
    order_std:
        Within-model standard deviation across element orders.
    """

    if num_permutations < 1:
        raise ValueError("num_permutations must be at least 1.")

    model.eval()
    loader = build_loader(
        features=features,
        coords=coords,
        labels=None,
        batch_size=batch_size,
        shuffle=False,
        seed=0,
        num_workers=num_workers,
    )

    all_pass_probabilities: List[np.ndarray] = []

    for permutation_index in range(num_permutations):
        pass_probabilities: List[np.ndarray] = []
        sample_offset = 0

        for batch in loader:
            batch_features, batch_coords = batch
            batch_features = batch_features.to(device, non_blocking=True)
            batch_coords = batch_coords.to(device, non_blocking=True)

            if permutation_index == 0:
                # The first inference pass retains the original element order.
                logits = model(batch_features, batch_coords, permute=False)
            else:
                # Use reproducible, distinct seeds for each permutation pass and batch.
                permutation = make_deterministic_permutation(
                    batch_size=batch_features.size(0),
                    num_features=model.num_features,
                    seed=permutation_seed
                    + permutation_index * 1_000_000
                    + sample_offset,
                    device=device,
                )
                logits = model(
                    batch_features,
                    batch_coords,
                    permutation=permutation,
                )

            probabilities = torch.softmax(logits, dim=1)[:, 1]
            pass_probabilities.append(probabilities.cpu().numpy())
            sample_offset += batch_features.size(0)

        all_pass_probabilities.append(np.concatenate(pass_probabilities))

    stacked = np.stack(all_pass_probabilities, axis=0)
    mean_prob = stacked.mean(axis=0)
    order_std = stacked.std(axis=0, ddof=0)
    return mean_prob.astype(np.float64), order_std.astype(np.float64)


def calculate_metrics(
    labels: np.ndarray,
    probabilities: np.ndarray,
    threshold: float,
) -> Dict[str, float]:
    """Calculate accuracy, precision, recall, F1, AUC, and log loss."""

    labels = np.asarray(labels, dtype=np.int64)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    predictions = (probabilities >= threshold).astype(np.int64)

    # Avoid taking the logarithm of exactly 0 or 1 in log loss.
    clipped_probabilities = np.clip(probabilities, 1e-7, 1.0 - 1e-7)

    return {
        "accuracy": float(accuracy_score(labels, predictions)),
        "precision": float(precision_score(labels, predictions, zero_division=0)),
        "recall": float(recall_score(labels, predictions, zero_division=0)),
        "f1": float(f1_score(labels, predictions, zero_division=0)),
        "auc": float(roc_auc_score(labels, probabilities)),
        "log_loss": float(
            log_loss(
                labels,
                np.column_stack([1.0 - clipped_probabilities, clipped_probabilities]),
                labels=[0, 1],
            )
        ),
    }


def train_fold_model(
    train_features: np.ndarray,
    train_coords: np.ndarray,
    train_labels: np.ndarray,
    val_features: np.ndarray,
    val_coords: np.ndarray,
    val_labels: np.ndarray,
    cfg: Config,
    device: torch.device,
    training_seed: int,
) -> Tuple[SpatialOIMamba, Dict[str, float], int, np.ndarray, np.ndarray]:
    """Fit one CV-fold model and return its best weights and validation predictions."""

    set_global_seed(training_seed)

    train_loader = build_loader(
        features=train_features,
        coords=train_coords,
        labels=train_labels,
        batch_size=min(cfg.batch_size, len(train_labels)),
        shuffle=True,
        seed=training_seed,
        num_workers=cfg.num_workers,
    )

    model = SpatialOIMamba(cfg).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
    )

    early_stopper = EarlyStopping(
        patience=cfg.early_stopping_patience,
        min_delta=cfg.early_stopping_min_delta,
    )

    for epoch in range(1, cfg.max_epochs + 1):
        train_loss = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            gaussian_noise_std=cfg.gaussian_noise_std,
            gradient_clip_norm=cfg.gradient_clip_norm,
        )

        # Reuse fixed inference permutations at every epoch to limit evaluation noise.
        val_probabilities, _ = predict_probabilities(
            model=model,
            features=val_features,
            coords=val_coords,
            device=device,
            batch_size=cfg.prediction_batch_size,
            num_permutations=cfg.early_stop_permutations,
            permutation_seed=training_seed + 500_000,
            num_workers=cfg.num_workers,
        )
        val_metrics = calculate_metrics(
            labels=val_labels,
            probabilities=val_probabilities,
            threshold=cfg.classification_threshold,
        )

        should_stop = early_stopper.step(
            f1_value=val_metrics["f1"],
            loss_value=val_metrics["log_loss"],
            model=model,
            epoch=epoch,
        )

        if epoch == 1 or epoch % 20 == 0 or should_stop:
            print(
                f"      epoch={epoch:3d}  train_loss={train_loss:.4f}  "
                f"val_f1={val_metrics['f1']:.4f}  val_auc={val_metrics['auc']:.4f}  "
                f"best_epoch={early_stopper.best_epoch:3d}  "
                f"patience={early_stopper.counter}/{early_stopper.patience}",
                flush=True,
            )

        if should_stop:
            break

    if early_stopper.best_state is None:
        raise RuntimeError("Early stopping did not save a valid model state.")

    model.load_state_dict(early_stopper.best_state)
    model.to(device)

    # Final fold evaluation averages more orders to reduce order-specific noise.
    final_probabilities, order_std = predict_probabilities(
        model=model,
        features=val_features,
        coords=val_coords,
        device=device,
        batch_size=cfg.prediction_batch_size,
        num_permutations=cfg.evaluation_permutations,
        permutation_seed=training_seed + 900_000,
        num_workers=cfg.num_workers,
    )
    final_metrics = calculate_metrics(
        labels=val_labels,
        probabilities=final_probabilities,
        threshold=cfg.classification_threshold,
    )

    return (
        model,
        final_metrics,
        early_stopper.best_epoch,
        final_probabilities,
        order_std,
    )


# =============================================================================
# 7. Repeated stratified five-fold cross-validation
# =============================================================================


def summarize_repeat_metrics(repeat_metrics_df: pd.DataFrame) -> pd.DataFrame:
    """Summarize repeated-CV metrics using mean, std, min, and max."""

    metric_names = ["accuracy", "precision", "recall", "f1", "auc", "log_loss"]
    rows = []
    for metric in metric_names:
        values = repeat_metrics_df[metric].to_numpy(dtype=np.float64)
        rows.append(
            {
                "metric": metric,
                "mean": float(values.mean()),
                "std": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
                "min": float(values.min()),
                "max": float(values.max()),
                "n_repeats": int(len(values)),
            }
        )
    return pd.DataFrame(rows)


def run_repeated_stratified_cv(
    labeled_df: pd.DataFrame,
    cfg: Config,
    device: torch.device,
    output_dir: Path,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, int]:
    """
    Run repeated stratified five-fold cross-validation.

    Aggregation protocol
    --------
    - Each repeat comprises five non-overlapping validation folds.
    - Each sample receives exactly one OOF prediction within each repeat.
    - Metrics are computed once from the complete OOF predictions per repeat.
    - Mean and standard deviation are reported across repeats.
    """

    output_dir.mkdir(parents=True, exist_ok=True)

    labels = labeled_df[cfg.label_col].to_numpy(dtype=np.int64)
    num_samples = len(labeled_df)

    splitter = RepeatedStratifiedKFold(
        n_splits=cfg.n_splits,
        n_repeats=cfg.n_repeats,
        random_state=cfg.cv_random_state,
    )

    # Rows correspond to repeats; columns correspond to original labeled samples.
    oof_probabilities = np.full(
        (cfg.n_repeats, num_samples),
        fill_value=np.nan,
        dtype=np.float64,
    )
    oof_order_std = np.full_like(oof_probabilities, fill_value=np.nan)

    fold_rows: List[Dict] = []
    oof_long_rows: List[Dict] = []
    best_epochs: List[int] = []

    # RepeatedStratifiedKFold yields five consecutive folds for each repeat.
    for split_index, (train_indices, val_indices) in enumerate(
        splitter.split(np.zeros(num_samples), labels)
    ):
        repeat_index = split_index // cfg.n_splits
        fold_index = split_index % cfg.n_splits
        training_seed = cfg.base_training_seed + repeat_index * 100 + fold_index

        print(
            "\n"
            + "=" * 88
            + f"\nRepeat {repeat_index + 1:02d}/{cfg.n_repeats}, "
            f"Fold {fold_index + 1}/{cfg.n_splits}, seed={training_seed}\n"
            + "=" * 88,
            flush=True,
        )

        train_df = labeled_df.iloc[train_indices].reset_index(drop=True)
        val_df = labeled_df.iloc[val_indices].reset_index(drop=True)

        # Fit scalers on the training fold to avoid leakage from validation samples.
        preprocessor = fit_preprocessor(train_df, cfg)
        train_features, train_coords = preprocessor.transform(train_df)
        val_features, val_coords = preprocessor.transform(val_df)
        train_labels = train_df[cfg.label_col].to_numpy(dtype=np.int64)
        val_labels = val_df[cfg.label_col].to_numpy(dtype=np.int64)

        (
            _model,
            fold_metrics,
            best_epoch,
            val_probabilities,
            val_order_std,
        ) = train_fold_model(
            train_features=train_features,
            train_coords=train_coords,
            train_labels=train_labels,
            val_features=val_features,
            val_coords=val_coords,
            val_labels=val_labels,
            cfg=cfg,
            device=device,
            training_seed=training_seed,
        )

        best_epochs.append(best_epoch)
        oof_probabilities[repeat_index, val_indices] = val_probabilities
        oof_order_std[repeat_index, val_indices] = val_order_std

        fold_row = {
            "repeat": repeat_index + 1,
            "fold": fold_index + 1,
            "training_seed": training_seed,
            "n_train": int(len(train_indices)),
            "n_val": int(len(val_indices)),
            "train_positive": int(labels[train_indices].sum()),
            "val_positive": int(labels[val_indices].sum()),
            "best_epoch": int(best_epoch),
            "mean_order_std": float(val_order_std.mean()),
            **fold_metrics,
        }
        fold_rows.append(fold_row)

        for local_position, original_index in enumerate(val_indices):
            source_row = labeled_df.iloc[original_index]
            probability = float(val_probabilities[local_position])
            oof_long_rows.append(
                {
                    "repeat": repeat_index + 1,
                    "fold": fold_index + 1,
                    "sample_index": int(original_index),
                    cfg.coord_cols[0]: float(source_row[cfg.coord_cols[0]]),
                    cfg.coord_cols[1]: float(source_row[cfg.coord_cols[1]]),
                    "y_true": int(source_row[cfg.label_col]),
                    "y_prob": probability,
                    "y_pred": int(probability >= cfg.classification_threshold),
                    "order_prob_std": float(val_order_std[local_position]),
                }
            )

        # Release fold models promptly to minimize GPU memory use.
        del _model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    if np.isnan(oof_probabilities).any():
        raise RuntimeError("OOF predictions still contain NaNs; some samples are missing.")

    fold_metrics_df = pd.DataFrame(fold_rows)
    oof_long_df = pd.DataFrame(oof_long_rows)

    repeat_rows: List[Dict] = []
    for repeat_index in range(cfg.n_repeats):
        repeat_metrics = calculate_metrics(
            labels=labels,
            probabilities=oof_probabilities[repeat_index],
            threshold=cfg.classification_threshold,
        )
        repeat_rows.append(
            {
                "repeat": repeat_index + 1,
                "mean_order_std": float(oof_order_std[repeat_index].mean()),
                **repeat_metrics,
            }
        )

    repeat_metrics_df = pd.DataFrame(repeat_rows)
    summary_df = summarize_repeat_metrics(repeat_metrics_df)

    # Average OOF probabilities across repeats for inspecting consensus predictions.
    oof_mean_probability = oof_probabilities.mean(axis=0)
    oof_mean_df = labeled_df.copy()
    oof_mean_df.insert(0, "sample_index", np.arange(num_samples, dtype=np.int64))
    oof_mean_df["oof_prob_mean"] = oof_mean_probability
    oof_mean_df["oof_prob_std_across_repeats"] = oof_probabilities.std(axis=0, ddof=0)
    oof_mean_df["oof_order_std_mean"] = oof_order_std.mean(axis=0)
    oof_mean_df["oof_pred"] = (
        oof_mean_probability >= cfg.classification_threshold
    ).astype(np.int64)

    # Use the median best epoch across the 50 CV folds for final full-data training.
    # The median is less sensitive to outlier folds than an extreme epoch.
    recommended_final_epochs = max(1, int(round(float(np.median(best_epochs)))))

    # Save fold, repeat, and aggregate statistics in separate workbook sheets.
    with pd.ExcelWriter(output_dir / "cv_results.xlsx", engine="openpyxl") as writer:
        fold_metrics_df.to_excel(writer, sheet_name="FoldMetrics_50", index=False)
        repeat_metrics_df.to_excel(writer, sheet_name="RepeatMetrics_10", index=False)
        summary_df.to_excel(writer, sheet_name="Summary", index=False)
        pd.DataFrame(
            {
                "recommended_final_epochs": [recommended_final_epochs],
                "median_best_epoch": [float(np.median(best_epochs))],
                "mean_best_epoch": [float(np.mean(best_epochs))],
                "std_best_epoch": [float(np.std(best_epochs, ddof=1))],
                "min_best_epoch": [int(np.min(best_epochs))],
                "max_best_epoch": [int(np.max(best_epochs))],
            }
        ).to_excel(writer, sheet_name="EpochRecommendation", index=False)

    oof_long_df.to_excel(output_dir / "cv_oof_predictions_long.xlsx", index=False)
    oof_mean_df.to_excel(output_dir / "cv_oof_mean_predictions.xlsx", index=False)

    cv_summary = {
        "protocol": {
            "n_splits": cfg.n_splits,
            "n_repeats": cfg.n_repeats,
            "total_fold_models": cfg.n_splits * cfg.n_repeats,
            "classification_threshold": cfg.classification_threshold,
            "evaluation_permutations": cfg.evaluation_permutations,
        },
        "repeat_metric_summary": summary_df.to_dict(orient="records"),
        "recommended_final_epochs": recommended_final_epochs,
        "best_epoch_statistics": {
            "median": float(np.median(best_epochs)),
            "mean": float(np.mean(best_epochs)),
            "std": float(np.std(best_epochs, ddof=1)),
            "min": int(np.min(best_epochs)),
            "max": int(np.max(best_epochs)),
        },
    }
    save_json(output_dir / "cv_summary.json", cv_summary)

    print("\nCross-validation complete. Mean +/- SD across repeats:")
    for _, row in summary_df.iterrows():
        print(f"  {row['metric']:>10s}: {row['mean']:.4f} ± {row['std']:.4f}")
    print(f"  Suggested number of final training epochs: {recommended_final_epochs}")

    return fold_metrics_df, repeat_metrics_df, summary_df, recommended_final_epochs


# =============================================================================
# 8. Final ensemble training and regional prediction
# =============================================================================


def train_full_data_model(
    full_features: np.ndarray,
    full_coords: np.ndarray,
    full_labels: np.ndarray,
    cfg: Config,
    device: torch.device,
    seed: int,
    epochs: int,
) -> Tuple[SpatialOIMamba, List[float]]:
    """Fit one final model to all labeled samples for a fixed number of epochs."""

    set_global_seed(seed)

    loader = build_loader(
        features=full_features,
        coords=full_coords,
        labels=full_labels,
        batch_size=min(cfg.batch_size, len(full_labels)),
        shuffle=True,
        seed=seed,
        num_workers=cfg.num_workers,
    )

    model = SpatialOIMamba(cfg).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
    )

    history: List[float] = []
    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(
            model=model,
            loader=loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            gaussian_noise_std=cfg.gaussian_noise_std,
            gradient_clip_norm=cfg.gradient_clip_norm,
        )
        history.append(float(train_loss))

        if epoch == 1 or epoch % 20 == 0 or epoch == epochs:
            print(
                f"      full-fit epoch={epoch:3d}/{epochs}  train_loss={train_loss:.4f}",
                flush=True,
            )

    return model, history


def run_final_training_and_prediction(
    labeled_df: pd.DataFrame,
    prediction_df: pd.DataFrame,
    cfg: Config,
    device: torch.device,
    output_dir: Path,
    final_epochs: int,
) -> pd.DataFrame:
    """
    Train an ensemble using all labeled samples and predict regional grid cells.

    Final probabilities are averaged across two sources of variation:
    1. Independently initialized ensemble members;
    2. Multiple element orders for each model.
    """

    final_models_dir = output_dir / "final_models"
    final_models_dir.mkdir(parents=True, exist_ok=True)

    # Fit the final preprocessor using all labeled samples only.
    preprocessor = fit_preprocessor(labeled_df, cfg)
    full_features, full_coords = preprocessor.transform(labeled_df)
    prediction_features, prediction_coords = preprocessor.transform(prediction_df)
    full_labels = labeled_df[cfg.label_col].to_numpy(dtype=np.int64)

    # Save the scalers and column names rather than serializing a custom class
    # defined under __main__, for easier use in independent scripts.
    joblib.dump(
        {
            "feature_scaler": preprocessor.feature_scaler,
            "coord_scaler": preprocessor.coord_scaler,
            "feature_cols": list(preprocessor.feature_cols),
            "coord_cols": list(preprocessor.coord_cols),
            "transform": "clip_nonnegative -> log1p -> StandardScaler",
        },
        output_dir / "final_preprocessor.joblib",
    )

    prediction_probabilities_by_model: List[np.ndarray] = []
    labeled_probabilities_by_model: List[np.ndarray] = []
    training_summary_rows: List[Dict] = []

    for ensemble_index in range(cfg.final_ensemble_size):
        seed = cfg.final_seed_start + ensemble_index
        print(
            "\n"
            + "-" * 88
            + f"\nFinal model {ensemble_index + 1:02d}/{cfg.final_ensemble_size}, "
            f"seed={seed}, epochs={final_epochs}\n"
            + "-" * 88,
            flush=True,
        )

        model, history = train_full_data_model(
            full_features=full_features,
            full_coords=full_coords,
            full_labels=full_labels,
            cfg=cfg,
            device=device,
            seed=seed,
            epochs=final_epochs,
        )

        # Average the original-order and randomized-order predictions per model.
        prediction_prob, prediction_order_std = predict_probabilities(
            model=model,
            features=prediction_features,
            coords=prediction_coords,
            device=device,
            batch_size=cfg.prediction_batch_size,
            num_permutations=cfg.evaluation_permutations,
            permutation_seed=seed + 1_000_000,
            num_workers=cfg.num_workers,
        )
        labeled_prob, labeled_order_std = predict_probabilities(
            model=model,
            features=full_features,
            coords=full_coords,
            device=device,
            batch_size=cfg.prediction_batch_size,
            num_permutations=cfg.evaluation_permutations,
            permutation_seed=seed + 2_000_000,
            num_workers=cfg.num_workers,
        )

        prediction_probabilities_by_model.append(prediction_prob)
        labeled_probabilities_by_model.append(labeled_prob)

        model_path = final_models_dir / f"spatial_oi_mamba_seed_{seed}.pt"
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "config": asdict(cfg),
                "seed": seed,
                "epochs": final_epochs,
                "feature_cols": list(cfg.feature_cols),
                "coord_cols": list(cfg.coord_cols),
                "label_col": cfg.label_col,
            },
            model_path,
        )

        training_summary_rows.append(
            {
                "ensemble_index": ensemble_index + 1,
                "seed": seed,
                "epochs": final_epochs,
                "initial_train_loss": history[0],
                "final_train_loss": history[-1],
                "mean_prediction_order_std": float(prediction_order_std.mean()),
                "mean_labeled_order_std": float(labeled_order_std.mean()),
                "model_file": str(model_path),
            }
        )

        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    prediction_matrix = np.stack(prediction_probabilities_by_model, axis=0)
    labeled_matrix = np.stack(labeled_probabilities_by_model, axis=0)

    prediction_mean = prediction_matrix.mean(axis=0)
    prediction_model_std = prediction_matrix.std(axis=0, ddof=0)
    labeled_mean = labeled_matrix.mean(axis=0)
    labeled_model_std = labeled_matrix.std(axis=0, ddof=0)

    prediction_output = prediction_df.copy()
    prediction_output["pred_prob_0"] = 1.0 - prediction_mean
    prediction_output["pred_prob_1"] = prediction_mean
    prediction_output["pred_prob_model_std"] = prediction_model_std
    prediction_output["pred_label"] = (
        prediction_mean >= cfg.classification_threshold
    ).astype(np.int64)
    prediction_output.to_excel(output_dir / "prediction_all_with_prob.xlsx", index=False)

    # Full-fit training predictions are diagnostic, not cross-validated estimates.
    labeled_output = labeled_df.copy()
    labeled_output["fullfit_prob_0"] = 1.0 - labeled_mean
    labeled_output["fullfit_prob_1"] = labeled_mean
    labeled_output["fullfit_prob_model_std"] = labeled_model_std
    labeled_output["fullfit_pred"] = (
        labeled_mean >= cfg.classification_threshold
    ).astype(np.int64)
    labeled_output.to_excel(output_dir / "labeled_fullfit_predictions.xlsx", index=False)

    training_summary_df = pd.DataFrame(training_summary_rows)
    training_summary_df.to_excel(output_dir / "final_training_summary.xlsx", index=False)

    final_metadata = {
        "final_epochs": final_epochs,
        "final_ensemble_size": cfg.final_ensemble_size,
        "final_seeds": [
            cfg.final_seed_start + index for index in range(cfg.final_ensemble_size)
        ],
        "num_prediction_samples": int(len(prediction_output)),
        "predicted_positive_count": int(prediction_output["pred_label"].sum()),
        "predicted_positive_ratio": float(prediction_output["pred_label"].mean()),
        "mean_predicted_probability": float(prediction_mean.mean()),
        "mean_model_probability_std": float(prediction_model_std.mean()),
        "note": (
            "Regional output averages probabilities over ensemble members and "
            "element permutations; full-fit predictions are not CV results."
        ),
    }
    save_json(output_dir / "final_prediction_metadata.json", final_metadata)

    print("\nFinal regional prediction completed:")
    print(f"  Number of prediction cells: {len(prediction_output)}")
    print(
        f"  Predicted positive cells: {int(prediction_output['pred_label'].sum())} "
        f"({prediction_output['pred_label'].mean():.2%})"
    )
    print(f"  Output file: {output_dir / 'prediction_all_with_prob.xlsx'}")

    return prediction_output


# =============================================================================
# 9. Command-line interface
# =============================================================================


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Spatial OI-Mamba repeated stratified CV training and prediction"
    )
    parser.add_argument(
        "--mode",
        choices=("check", "cv", "all"),
        default="all",
        help="check=validate inputs; cv=run cross-validation; all=CV + ensemble prediction.",
    )
    parser.add_argument(
        "--labeled-file",
        default=None,
        help="Path to the labeled-samples Excel workbook.",
    )
    parser.add_argument(
        "--prediction-file",
        default=None,
        help="Path to the unlabeled regional prediction-grid Excel workbook.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory for result files.",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cuda", "cpu"),
        default=None,
        help="Execution device (default: auto).",
    )
    parser.add_argument(
        "--n-repeats",
        type=int,
        default=None,
        help="Number of repeats of stratified five-fold CV (default: 10).",
    )
    parser.add_argument(
        "--noise-std",
        type=float,
        default=None,
        help="Gaussian noise in standardized element space (default: 0.05; 0 disables).",
    )
    parser.add_argument(
        "--final-ensemble-size",
        type=int,
        default=None,
        help="Number of final full-data ensemble members (default: 10).",
    )
    return parser


def apply_command_line_overrides(cfg: Config, args: argparse.Namespace) -> Config:
    """Override selected dataclass defaults using command-line arguments."""

    updates = {}
    if args.labeled_file is not None:
        updates["labeled_file"] = args.labeled_file
    if args.prediction_file is not None:
        updates["prediction_file"] = args.prediction_file
    if args.output_dir is not None:
        updates["output_dir"] = args.output_dir
    if args.device is not None:
        updates["device"] = args.device
    if args.n_repeats is not None:
        if args.n_repeats < 1:
            raise ValueError("--n-repeats must be at least 1.")
        updates["n_repeats"] = args.n_repeats
    if args.noise_std is not None:
        if args.noise_std < 0:
            raise ValueError("--noise-std cannot be negative.")
        updates["gaussian_noise_std"] = args.noise_std
    if args.final_ensemble_size is not None:
        if args.final_ensemble_size < 1:
            raise ValueError("--final-ensemble-size must be at least 1.")
        updates["final_ensemble_size"] = args.final_ensemble_size

    return replace(cfg, **updates)


def main() -> None:
    parser = build_argument_parser()
    args = parser.parse_args()

    cfg = apply_command_line_overrides(Config(), args)
    script_dir = Path(__file__).resolve().parent

    labeled_path = resolve_path(cfg.labeled_file, script_dir)
    prediction_path = resolve_path(cfg.prediction_file, script_dir)
    output_dir = resolve_path(cfg.output_dir, script_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Record resolved absolute paths in the saved configuration.
    cfg_for_save = asdict(cfg)
    cfg_for_save["resolved_labeled_file"] = str(labeled_path)
    cfg_for_save["resolved_prediction_file"] = str(prediction_path)
    cfg_for_save["resolved_output_dir"] = str(output_dir)
    save_json(output_dir / "config.json", cfg_for_save)

    labeled_df, prediction_df, input_summary = load_and_validate_data(
        labeled_path=labeled_path,
        prediction_path=prediction_path,
        cfg=cfg,
    )
    save_json(output_dir / "input_data_summary.json", input_summary)

    print("=" * 96)
    print("Spatial OI-Mamba: repeated stratified CV and regional prediction")
    print(f"Labeled data: {labeled_path}")
    print(f"Prediction grid: {prediction_path}")
    print(f"Labeled samples: {len(labeled_df)}; class counts: {input_summary['class_counts']}")
    print(f"Prediction cells: {len(prediction_df)}")
    print(
        f"CV protocol: {cfg.n_splits}-fold stratified CV x {cfg.n_repeats} repeats "
        f"= {cfg.n_splits * cfg.n_repeats} folds"
    )
    print(f"Online augmentation: Gaussian noise SD={cfg.gaussian_noise_std} + element shuffles")
    print(f"Output directory: {output_dir}")
    print("=" * 96)

    if args.mode == "check":
        print("Input validation passed. No Mamba model was constructed or trained.")
        return

    require_mamba()
    device = resolve_device(cfg.device)
    print(f"Execution device: {device}")

    _, _, _, recommended_final_epochs = run_repeated_stratified_cv(
        labeled_df=labeled_df,
        cfg=cfg,
        device=device,
        output_dir=output_dir,
    )

    if args.mode == "all":
        run_final_training_and_prediction(
            labeled_df=labeled_df,
            prediction_df=prediction_df,
            cfg=cfg,
            device=device,
            output_dir=output_dir,
            final_epochs=recommended_final_epochs,
        )


if __name__ == "__main__":
    main()
