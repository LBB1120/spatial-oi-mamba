# -*- coding: utf-8 -*-
"""
Spatial OI-Mamba: out-of-fold interpretability analysis
=======================================================

This script reuses the model, preprocessing, splits, training seeds, and prediction
protocol defined in ``oi_mamba_repeated_cv_train_predict.py``. It implements:

1. Permutation Feature Importance (PFI)
   - Compute feature importance on complete out-of-fold (OOF) predictions for each CV repeat.
   - Record changes in F1, AUC, accuracy, and log loss after element-wise permutations.
   - Higher (baseline - permuted) F1/AUC denotes stronger predictive dependence.

2. Local Occlusion Analysis (LOA)
   - Explain only held-out validation samples from cross-validation.
   - Replace one standardized element with zero, the training-fold mean after log1p.
   - Record changes in positive-class probability, with TP/TN/FP/FN summaries.

3. Gradient x Input (GxI)
   - Differentiate the positive-class logit with respect to standardized elements.
   - Multiply each input by its local gradient and average across fixed element orders.

Why train the CV models again?
------------------------------
The main training script does not save fold-specific models. This script retrains
those models with the same CV configuration and immediately generates OOF explanations
for each held-out fold, rather than explaining the final ensemble's training samples.

Required files
--------------
- ``oi_mamba_repeated_cv_train_predict.py``
- ``data/labeled_samples.xlsx``
- ``data/prediction_grid.xlsx`` (required by the shared input-validation routine)

Usage
-----
    python oi_mamba_explainability_oof.py --mode check
    python oi_mamba_explainability_oof.py --mode run
    python oi_mamba_explainability_oof.py --mode run --n-repeats 1 --pfi-repeats 3
    python oi_mamba_explainability_oof.py --mode run \
        --config-json oi_mamba_repeated_cv_results/config.json

Outputs
-------
Default directory: ``oi_mamba_explainability_results``.
- ``explainability_results.xlsx``: OOF, PFI, LOA, GxI, and sample-level summaries.
- ``PFI_Summary.xlsx`` and analogous per-method summary workbooks.
- ``figures/*.png`` and optional ``figures/single_samples/*.png``.

Interpretation notes
--------------------
- PFI uncertainty is summarized across CV repeats, not across independent shuffles.
- Positive LOA contribution means the observed element supports positive-class probability.
- GxI measures local logit sensitivity in standardized input space, not causality.
- All explanations use fixed element orders with the model in evaluation mode.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
import warnings
from dataclasses import asdict, dataclass, fields, replace
from pathlib import Path
from types import ModuleType
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

import matplotlib

# Use a non-interactive backend on remote servers and compute nodes.
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import RepeatedStratifiedKFold


# =============================================================================
# 1. Explainability configuration
# =============================================================================


@dataclass
class ExplainabilityConfig:
    """Settings for attribution calculations and figure generation."""

    # Keep interpretability results separate from model-training outputs.
    output_dir: str = "oi_mamba_explainability_results"

    # Number of independent feature permutations within each CV repeat.
    # Consider 20-50 repetitions for a final study; default is 30.
    pfi_repeats: int = 30

    # Optionally permute X/Y coordinates together as an additional feature group.
    # This result is recorded in tables but excluded from element-only figures.
    include_spatial_pfi: bool = True

    # Number of representative samples per TP/TN/FP/FN outcome group.
    representatives_per_group: int = 2

    # Whether to create individual LOA plots for representative samples.
    make_single_sample_figures: bool = True

    # Figure resolution in dots per inch.
    figure_dpi: int = 300

    # Use training feature order except where PFI plots sort by importance.
    group_order: Tuple[str, ...] = ("TP", "TN", "FP", "FN")


@dataclass
class FoldBundle:
    """Store the fitted model and held-out data for one CV fold."""

    repeat_index: int
    fold_index: int
    training_seed: int
    val_indices: np.ndarray
    val_features: np.ndarray
    val_coords: np.ndarray
    val_labels: np.ndarray
    baseline_probabilities: np.ndarray
    baseline_order_std: np.ndarray
    model: torch.nn.Module


# =============================================================================
# 2. Loading the shared training module
# =============================================================================


def load_training_module(script_path: Path) -> ModuleType:
    """
    Dynamically import the training module from the specified Python script.

    Reuse the same model architecture, data transformations, random seeds,
    training routine, and inference logic without duplicating implementations.
    """

    if not script_path.exists():
        raise FileNotFoundError(f"Training script not found: {script_path}")

    module_name = "oi_mamba_training_runtime"
    spec = importlib.util.spec_from_file_location(module_name, script_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load training script: {script_path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def load_training_config_from_json(
    training_module: ModuleType,
    json_path: Optional[Path],
) -> Any:
    """
    Reconstruct the training Config object from an optional config.json file.

    Ignore extra resolved_* paths, which are not dataclass fields.
    Convert feature_cols and coord_cols back to tuples.
    """

    config_class = training_module.Config
    if json_path is None:
        return config_class()

    if not json_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {json_path}")

    with json_path.open("r", encoding="utf-8") as file:
        raw = json.load(file)

    valid_names = {item.name for item in fields(config_class)}
    kwargs = {key: value for key, value in raw.items() if key in valid_names}

    if "feature_cols" in kwargs:
        kwargs["feature_cols"] = tuple(kwargs["feature_cols"])
    if "coord_cols" in kwargs:
        kwargs["coord_cols"] = tuple(kwargs["coord_cols"])

    return config_class(**kwargs)


# =============================================================================
# 3. Interpretability utilities
# =============================================================================


def outcome_group(y_true: int, y_pred: int) -> str:
    """Return TP, TN, FP, or FN for a true and predicted binary label."""

    if y_true == 1 and y_pred == 1:
        return "TP"
    if y_true == 0 and y_pred == 0:
        return "TN"
    if y_true == 0 and y_pred == 1:
        return "FP"
    return "FN"


def safe_std(values: Sequence[float]) -> float:
    """Return zero for an undefined sample standard deviation (n < 2)."""

    array = np.asarray(values, dtype=np.float64)
    if array.size < 2:
        return 0.0
    return float(array.std(ddof=1))


def confidence_interval_95(std: float, n: int) -> float:
    """Approximate the half-width of a 95% CI across CV repetitions."""

    if n <= 1:
        return 0.0
    return float(1.96 * std / math.sqrt(n))


def predict_with_same_order_set(
    training_module: ModuleType,
    bundle: FoldBundle,
    features: np.ndarray,
    coords: np.ndarray,
    train_cfg: Any,
    device: torch.device,
) -> np.ndarray:
    """
    Predict using exactly the same fixed element-order set as the baseline.

    Keeping the orders fixed ensures that differences between perturbed and
    baseline outputs arise from feature perturbations, not changed token orders.
    """

    probabilities, _ = training_module.predict_probabilities(
        model=bundle.model,
        features=features,
        coords=coords,
        device=device,
        batch_size=train_cfg.prediction_batch_size,
        num_permutations=train_cfg.evaluation_permutations,
        permutation_seed=bundle.training_seed + 900_000,
        num_workers=train_cfg.num_workers,
    )
    return probabilities


# =============================================================================
# 4. Local Occlusion Analysis (LOA)
# =============================================================================


def calculate_local_occlusion_for_fold(
    training_module: ModuleType,
    bundle: FoldBundle,
    labeled_df: pd.DataFrame,
    train_cfg: Any,
    device: torch.device,
) -> List[Dict[str, Any]]:
    """
    Compute local occlusion contributions for one held-out CV fold.

    Occlusion baseline
    --------
    Inputs have been transformed by log1p and training-fold StandardScaler.
    Replacing a standardized element with zero therefore uses its training-fold
    mean in log1p space, rather than a zero raw elemental concentration.
    """

    records: List[Dict[str, Any]] = []
    baseline = bundle.baseline_probabilities
    baseline_pred = (baseline >= train_cfg.classification_threshold).astype(np.int64)

    for feature_index, feature_name in enumerate(train_cfg.feature_cols):
        occluded_features = bundle.val_features.copy()
        occluded_features[:, feature_index] = 0.0

        occluded_prob = predict_with_same_order_set(
            training_module=training_module,
            bundle=bundle,
            features=occluded_features,
            coords=bundle.val_coords,
            train_cfg=train_cfg,
            device=device,
        )

        # Positive contributions mean the observed element raises positive probability.
        contribution_prob1 = baseline - occluded_prob

        # Contribution to confidence in the originally predicted class.
        baseline_conf = np.where(baseline_pred == 1, baseline, 1.0 - baseline)
        occluded_conf = np.where(
            baseline_pred == 1,
            occluded_prob,
            1.0 - occluded_prob,
        )
        contribution_original_class = baseline_conf - occluded_conf

        for local_index, original_index in enumerate(bundle.val_indices):
            source_row = labeled_df.iloc[int(original_index)]
            y_true = int(bundle.val_labels[local_index])
            y_pred = int(baseline_pred[local_index])

            records.append(
                {
                    "repeat": bundle.repeat_index + 1,
                    "fold": bundle.fold_index + 1,
                    "training_seed": bundle.training_seed,
                    "sample_index": int(original_index),
                    train_cfg.coord_cols[0]: float(source_row[train_cfg.coord_cols[0]]),
                    train_cfg.coord_cols[1]: float(source_row[train_cfg.coord_cols[1]]),
                    "y_true": y_true,
                    "y_pred": y_pred,
                    "outcome_group": outcome_group(y_true, y_pred),
                    "base_prob_1": float(baseline[local_index]),
                    "base_order_std": float(bundle.baseline_order_std[local_index]),
                    "feature": feature_name,
                    "feature_index": feature_index,
                    "occlusion_baseline": "training_fold_standardized_mean_0",
                    "occluded_prob_1": float(occluded_prob[local_index]),
                    "contribution_prob_1": float(contribution_prob1[local_index]),
                    "abs_contribution_prob_1": float(abs(contribution_prob1[local_index])),
                    "base_original_class_confidence": float(baseline_conf[local_index]),
                    "occluded_original_class_confidence": float(occluded_conf[local_index]),
                    "contribution_original_class": float(
                        contribution_original_class[local_index]
                    ),
                    "abs_contribution_original_class": float(
                        abs(contribution_original_class[local_index])
                    ),
                }
            )

    return records


# =============================================================================
# 5. Gradient x Input attribution
# =============================================================================


def calculate_gradient_x_input_for_fold(
    training_module: ModuleType,
    bundle: FoldBundle,
    labeled_df: pd.DataFrame,
    train_cfg: Any,
    device: torch.device,
) -> List[Dict[str, Any]]:
    """
    Attribute the positive-class logit using Gradient x Input for a CV fold.

    Average attributions across the same fixed element orders used in evaluation.
    The first order is the original feature order; remaining orders are fixed
    permutations. Autograd traces torch.gather back to original element columns.
    """

    bundle.model.eval()
    num_samples, num_features = bundle.val_features.shape
    attribution_passes: List[np.ndarray] = []

    coords_tensor = torch.as_tensor(
        bundle.val_coords,
        dtype=torch.float32,
        device=device,
    )

    for permutation_index in range(train_cfg.evaluation_permutations):
        # Make a fresh leaf tensor to isolate automatic differentiation for each pass.
        features_tensor = torch.as_tensor(
            bundle.val_features,
            dtype=torch.float32,
            device=device,
        ).clone().detach().requires_grad_(True)

        bundle.model.zero_grad(set_to_none=True)

        if permutation_index == 0:
            logits = bundle.model(features_tensor, coords_tensor, permute=False)
        else:
            permutation = training_module.make_deterministic_permutation(
                batch_size=num_samples,
                num_features=num_features,
                seed=bundle.training_seed
                + 900_000
                + permutation_index * 1_000_000,
                device=device,
            )
            logits = bundle.model(
                features_tensor,
                coords_tensor,
                permutation=permutation,
            )

        # Sum logits across independent samples to obtain per-sample gradients.
        positive_logit_sum = logits[:, 1].sum()
        gradients = torch.autograd.grad(
            outputs=positive_logit_sum,
            inputs=features_tensor,
            retain_graph=False,
            create_graph=False,
        )[0]

        attribution = features_tensor * gradients
        attribution_passes.append(attribution.detach().cpu().numpy())

    stacked = np.stack(attribution_passes, axis=0)
    mean_attribution = stacked.mean(axis=0)
    std_attribution = stacked.std(axis=0, ddof=0)

    baseline_pred = (
        bundle.baseline_probabilities >= train_cfg.classification_threshold
    ).astype(np.int64)

    records: List[Dict[str, Any]] = []
    for local_index, original_index in enumerate(bundle.val_indices):
        source_row = labeled_df.iloc[int(original_index)]
        y_true = int(bundle.val_labels[local_index])
        y_pred = int(baseline_pred[local_index])

        for feature_index, feature_name in enumerate(train_cfg.feature_cols):
            value = float(mean_attribution[local_index, feature_index])
            records.append(
                {
                    "repeat": bundle.repeat_index + 1,
                    "fold": bundle.fold_index + 1,
                    "training_seed": bundle.training_seed,
                    "sample_index": int(original_index),
                    train_cfg.coord_cols[0]: float(source_row[train_cfg.coord_cols[0]]),
                    train_cfg.coord_cols[1]: float(source_row[train_cfg.coord_cols[1]]),
                    "y_true": y_true,
                    "y_pred": y_pred,
                    "outcome_group": outcome_group(y_true, y_pred),
                    "base_prob_1": float(bundle.baseline_probabilities[local_index]),
                    "feature": feature_name,
                    "feature_index": feature_index,
                    "gradient_x_input": value,
                    "abs_gradient_x_input": float(abs(value)),
                    "order_attribution_std": float(
                        std_attribution[local_index, feature_index]
                    ),
                }
            )

    return records


# =============================================================================
# 6. Permutation Feature Importance (PFI)
# =============================================================================


def calculate_pfi_for_repeat(
    training_module: ModuleType,
    bundles: Sequence[FoldBundle],
    labels: np.ndarray,
    baseline_oof_probabilities: np.ndarray,
    train_cfg: Any,
    explain_cfg: ExplainabilityConfig,
    device: torch.device,
) -> List[Dict[str, Any]]:
    """
    Compute permutation importance from one complete repeat of OOF predictions.

    Combine the five validation folds before computing F1/AUC rather than
    computing separate metrics from small folds with only 7-8 observations.
    This reduces discretization due to small validation-fold sizes.
    """

    baseline_metrics = training_module.calculate_metrics(
        labels=labels,
        probabilities=baseline_oof_probabilities,
        threshold=train_cfg.classification_threshold,
    )

    records: List[Dict[str, Any]] = []
    feature_specs: List[Tuple[str, str, Optional[int]]] = [
        (name, "element", index)
        for index, name in enumerate(train_cfg.feature_cols)
    ]
    if explain_cfg.include_spatial_pfi:
        feature_specs.append(("Spatial_XY", "spatial_group", None))

    for feature_name, feature_type, feature_index in feature_specs:
        for permutation_repeat in range(explain_cfg.pfi_repeats):
            permuted_oof = np.full_like(baseline_oof_probabilities, np.nan)

            for bundle in bundles:
                # Unique reproducible RNG seed for each fold, feature, and PFI repeat.
                rng_seed = (
                    73_000_000
                    + (bundle.repeat_index + 1) * 1_000_000
                    + (bundle.fold_index + 1) * 100_000
                    + (0 if feature_index is None else feature_index + 1) * 1_000
                    + permutation_repeat
                )
                rng = np.random.default_rng(rng_seed)

                permuted_features = bundle.val_features.copy()
                permuted_coords = bundle.val_coords.copy()

                if feature_type == "element":
                    row_permutation = rng.permutation(len(bundle.val_indices))
                    permuted_features[:, int(feature_index)] = bundle.val_features[
                        row_permutation, int(feature_index)
                    ]
                else:
                    # Permute X/Y as coordinate pairs to preserve valid pairings.
                    row_permutation = rng.permutation(len(bundle.val_indices))
                    permuted_coords = bundle.val_coords[row_permutation].copy()

                fold_prob = predict_with_same_order_set(
                    training_module=training_module,
                    bundle=bundle,
                    features=permuted_features,
                    coords=permuted_coords,
                    train_cfg=train_cfg,
                    device=device,
                )
                permuted_oof[bundle.val_indices] = fold_prob

            if np.isnan(permuted_oof).any():
                raise RuntimeError("Permuted PFI OOF probabilities contain NaNs.")

            permuted_metrics = training_module.calculate_metrics(
                labels=labels,
                probabilities=permuted_oof,
                threshold=train_cfg.classification_threshold,
            )

            records.append(
                {
                    "repeat": bundles[0].repeat_index + 1,
                    "feature": feature_name,
                    "feature_type": feature_type,
                    "feature_index": feature_index,
                    "permutation_repeat": permutation_repeat + 1,
                    "baseline_accuracy": baseline_metrics["accuracy"],
                    "baseline_f1": baseline_metrics["f1"],
                    "baseline_auc": baseline_metrics["auc"],
                    "baseline_log_loss": baseline_metrics["log_loss"],
                    "permuted_accuracy": permuted_metrics["accuracy"],
                    "permuted_f1": permuted_metrics["f1"],
                    "permuted_auc": permuted_metrics["auc"],
                    "permuted_log_loss": permuted_metrics["log_loss"],
                    "delta_accuracy": baseline_metrics["accuracy"]
                    - permuted_metrics["accuracy"],
                    "delta_f1": baseline_metrics["f1"] - permuted_metrics["f1"],
                    "delta_auc": baseline_metrics["auc"] - permuted_metrics["auc"],
                    # Increasing log loss is worse: use permuted minus baseline.
                    "delta_log_loss": permuted_metrics["log_loss"]
                    - baseline_metrics["log_loss"],
                }
            )

    return records


# =============================================================================
# 7. Aggregation and summary functions
# =============================================================================


def build_oof_prediction_tables(
    labeled_df: pd.DataFrame,
    labels: np.ndarray,
    oof_probabilities: np.ndarray,
    oof_order_std: np.ndarray,
    train_cfg: Any,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Build per-repeat OOF records and per-sample consensus OOF predictions."""

    long_rows: List[Dict[str, Any]] = []
    for repeat_index in range(oof_probabilities.shape[0]):
        for sample_index in range(oof_probabilities.shape[1]):
            probability = float(oof_probabilities[repeat_index, sample_index])
            y_true = int(labels[sample_index])
            y_pred = int(probability >= train_cfg.classification_threshold)
            source_row = labeled_df.iloc[sample_index]

            long_rows.append(
                {
                    "repeat": repeat_index + 1,
                    "sample_index": sample_index,
                    train_cfg.coord_cols[0]: float(source_row[train_cfg.coord_cols[0]]),
                    train_cfg.coord_cols[1]: float(source_row[train_cfg.coord_cols[1]]),
                    "y_true": y_true,
                    "y_prob": probability,
                    "y_pred": y_pred,
                    "outcome_group": outcome_group(y_true, y_pred),
                    "order_prob_std": float(oof_order_std[repeat_index, sample_index]),
                }
            )

    long_df = pd.DataFrame(long_rows)

    mean_probability = oof_probabilities.mean(axis=0)
    mean_df = labeled_df.copy()
    mean_df.insert(0, "sample_index", np.arange(len(labeled_df), dtype=np.int64))
    mean_df["oof_prob_mean"] = mean_probability
    mean_df["oof_prob_std_across_repeats"] = oof_probabilities.std(axis=0, ddof=0)
    mean_df["oof_order_std_mean"] = oof_order_std.mean(axis=0)
    mean_df["oof_pred"] = (
        mean_probability >= train_cfg.classification_threshold
    ).astype(np.int64)
    mean_df["outcome_group"] = [
        outcome_group(int(y), int(pred))
        for y, pred in zip(labels, mean_df["oof_pred"].to_numpy())
    ]
    mean_df["is_correct"] = (
        mean_df[train_cfg.label_col].to_numpy(dtype=np.int64)
        == mean_df["oof_pred"].to_numpy(dtype=np.int64)
    )
    # Retain the configured input schema for downstream explanation summaries.
    mean_df.attrs["coordinate_columns"] = tuple(train_cfg.coord_cols)
    mean_df.attrs["label_column"] = train_cfg.label_col

    return long_df, mean_df


def summarize_pfi(
    pfi_raw_df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Average PFI shuffles within each CV repeat before summarizing repeats.

    The CI sample size is the number of CV repeats, not the number of shuffles.
    """

    metric_columns = ["delta_accuracy", "delta_f1", "delta_auc", "delta_log_loss"]

    by_repeat = (
        pfi_raw_df.groupby(
            ["repeat", "feature", "feature_type", "feature_index"],
            dropna=False,
            as_index=False,
        )[metric_columns]
        .mean()
    )

    rows: List[Dict[str, Any]] = []
    for keys, group in by_repeat.groupby(
        ["feature", "feature_type", "feature_index"],
        dropna=False,
    ):
        feature, feature_type, feature_index = keys
        row: Dict[str, Any] = {
            "feature": feature,
            "feature_type": feature_type,
            "feature_index": feature_index,
            "n_cv_repeats": int(group["repeat"].nunique()),
        }

        for metric in metric_columns:
            values = group[metric].to_numpy(dtype=np.float64)
            mean_value = float(values.mean())
            std_value = safe_std(values)
            row[f"mean_{metric}"] = mean_value
            row[f"std_{metric}"] = std_value
            row[f"ci95_{metric}"] = confidence_interval_95(std_value, len(values))
            row[f"min_{metric}"] = float(values.min())
            row[f"max_{metric}"] = float(values.max())

        rows.append(row)

    summary = pd.DataFrame(rows).sort_values(
        "mean_delta_f1",
        ascending=False,
    ).reset_index(drop=True)
    return by_repeat, summary


def aggregate_sample_feature_explanations(
    raw_df: pd.DataFrame,
    oof_mean_df: pd.DataFrame,
    method: str,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Aggregate repeat-level explanations by unique sample and feature, then group."""

    if method == "loa":
        aggregation = {
            "contribution_prob_1": ["mean", "std"],
            "abs_contribution_prob_1": "mean",
            "contribution_original_class": ["mean", "std"],
            "abs_contribution_original_class": "mean",
            "base_prob_1": "mean",
        }
    elif method == "gxi":
        aggregation = {
            "gradient_x_input": ["mean", "std"],
            "abs_gradient_x_input": "mean",
            "order_attribution_std": "mean",
            "base_prob_1": "mean",
        }
    else:
        raise ValueError("method must be either loa or gxi.")

    grouped = raw_df.groupby(
        ["sample_index", "feature", "feature_index"],
        as_index=False,
    ).agg(aggregation)

    # Flatten hierarchical column names for Excel and plotting.
    flattened: List[str] = []
    for column in grouped.columns:
        if isinstance(column, tuple):
            left, right = column
            flattened.append(left if right == "" else f"{left}_{right}")
        else:
            flattened.append(str(column))
    grouped.columns = flattened

    sample_meta_cols = [
        "sample_index",
        "oof_prob_mean",
        "oof_prob_std_across_repeats",
        "oof_order_std_mean",
        "oof_pred",
        "outcome_group",
        "is_correct",
    ]
    # Preserve coordinates and true class labels.
    for candidate in list(oof_mean_df.attrs.get("coordinate_columns", ("X", "Y"))) + [oof_mean_df.attrs.get("label_column", "mineralization_label")]:
        if candidate in oof_mean_df.columns and candidate not in sample_meta_cols:
            sample_meta_cols.append(candidate)

    grouped = grouped.merge(
        oof_mean_df[sample_meta_cols],
        on="sample_index",
        how="left",
        validate="many_to_one",
    )

    # Build All, Correct, Incorrect, and TP/TN/FP/FN analysis groups.
    expanded_rows: List[Dict[str, Any]] = []
    for row in grouped.to_dict(orient="records"):
        memberships = ["All", str(row["outcome_group"])]
        memberships.append("Correct" if bool(row["is_correct"]) else "Incorrect")
        for group_name in memberships:
            expanded = dict(row)
            expanded["analysis_group"] = group_name
            expanded_rows.append(expanded)

    expanded_df = pd.DataFrame(expanded_rows)

    if method == "loa":
        signed_col = "contribution_prob_1_mean"
        abs_col = "abs_contribution_prob_1_mean"
    else:
        signed_col = "gradient_x_input_mean"
        abs_col = "abs_gradient_x_input_mean"

    summary_rows: List[Dict[str, Any]] = []
    for (group_name, feature, feature_index), subset in expanded_df.groupby(
        ["analysis_group", "feature", "feature_index"],
        as_index=False,
    ):
        signed_values = subset[signed_col].to_numpy(dtype=np.float64)
        abs_values = subset[abs_col].to_numpy(dtype=np.float64)
        summary_rows.append(
            {
                "analysis_group": group_name,
                "feature": feature,
                "feature_index": feature_index,
                "n_unique_samples": int(subset["sample_index"].nunique()),
                "mean_signed": float(signed_values.mean()),
                "std_signed": safe_std(signed_values),
                "mean_absolute": float(abs_values.mean()),
                "std_absolute": safe_std(abs_values),
            }
        )

    group_summary = pd.DataFrame(summary_rows)
    return grouped, group_summary


def select_representative_samples(
    loa_sample_feature_df: pd.DataFrame,
    oof_mean_df: pd.DataFrame,
    explain_cfg: ExplainabilityConfig,
) -> pd.DataFrame:
    """
    Select representative samples with the largest total absolute LOA contribution.

    Apply a predefined criterion to avoid cherry-picking element directions.
    """

    strength = (
        loa_sample_feature_df.groupby("sample_index", as_index=False)[
            "abs_contribution_prob_1_mean"
        ]
        .sum()
        .rename(columns={"abs_contribution_prob_1_mean": "total_abs_occlusion"})
    )

    label_column = oof_mean_df.attrs.get("label_column", "mineralization_label")
    candidates = oof_mean_df[
        ["sample_index", "outcome_group", "oof_prob_mean", "oof_pred", label_column]
    ].merge(strength, on="sample_index", how="left")

    selected_parts: List[pd.DataFrame] = []
    for group_name in explain_cfg.group_order:
        subset = candidates[candidates["outcome_group"] == group_name].copy()
        if subset.empty:
            continue
        subset = subset.sort_values("total_abs_occlusion", ascending=False).head(
            explain_cfg.representatives_per_group
        )
        selected_parts.append(subset)

    if selected_parts:
        selected = pd.concat(selected_parts, ignore_index=True)
    else:
        selected = candidates.sort_values("total_abs_occlusion", ascending=False).head(
            min(8, len(candidates))
        )

    # Omit outcome groups that have no samples; never manufacture FP/FN cases.
    selected["selection_rank"] = np.arange(1, len(selected) + 1)
    return selected


# =============================================================================
# 8. Plotting functions
# =============================================================================


def configure_matplotlib() -> None:
    """Set common font fallbacks for English figure labels."""

    plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial", "Liberation Sans"]
    plt.rcParams["axes.unicode_minus"] = False


def plot_pfi_delta_f1(
    pfi_summary_df: pd.DataFrame,
    figure_path: Path,
    dpi: int,
) -> None:
    """Plot element permutation importance (delta F1) with horizontal bars."""

    data = pfi_summary_df[pfi_summary_df["feature_type"] == "element"].copy()
    data = data.sort_values("mean_delta_f1", ascending=True)

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.barh(
        data["feature"],
        data["mean_delta_f1"],
        xerr=data["std_delta_f1"],
        capsize=3,
    )
    ax.axvline(0.0, linewidth=1)
    ax.set_xlabel("Mean decrease in F1 after feature permutation")
    ax.set_ylabel("Element feature")
    ax.set_title("Permutation importance ranked by ΔF1")
    ax.grid(axis="x", linestyle="--", alpha=0.4)
    fig.tight_layout()
    fig.savefig(figure_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def plot_pfi_f1_auc_comparison(
    pfi_summary_df: pd.DataFrame,
    figure_path: Path,
    dpi: int,
) -> None:
    """Compare mean element-wise delta F1 and delta AUC."""

    data = pfi_summary_df[pfi_summary_df["feature_type"] == "element"].copy()
    data = data.sort_values("mean_delta_f1", ascending=False)

    x = np.arange(len(data))
    width = 0.38

    fig, ax = plt.subplots(figsize=(11, 6))
    ax.bar(
        x - width / 2,
        data["mean_delta_f1"],
        width,
        yerr=data["std_delta_f1"],
        capsize=2,
        label="ΔF1",
    )
    ax.bar(
        x + width / 2,
        data["mean_delta_auc"],
        width,
        yerr=data["std_delta_auc"],
        capsize=2,
        label="ΔAUC",
    )
    ax.axhline(0.0, linewidth=1)
    ax.set_xticks(x)
    ax.set_xticklabels(data["feature"], rotation=45, ha="right")
    ax.set_ylabel("Mean metric decrease after permutation")
    ax.set_title("Permutation importance comparison: ΔF1 vs ΔAUC")
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    fig.tight_layout()
    fig.savefig(figure_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def plot_local_occlusion_heatmap(
    loa_sample_feature_df: pd.DataFrame,
    representative_df: pd.DataFrame,
    feature_order: Sequence[str],
    figure_path: Path,
    dpi: int,
) -> None:
    """Plot signed LOA contributions for representative OOF samples."""

    selected_ids = representative_df["sample_index"].astype(int).tolist()
    subset = loa_sample_feature_df[
        loa_sample_feature_df["sample_index"].isin(selected_ids)
    ].copy()

    matrix = subset.pivot(
        index="sample_index",
        columns="feature",
        values="contribution_prob_1_mean",
    ).reindex(index=selected_ids, columns=list(feature_order))

    meta = representative_df.set_index("sample_index")
    row_labels = [
        f"id={sample_id} {meta.loc[sample_id, 'outcome_group']}"
        for sample_id in selected_ids
    ]

    values = matrix.to_numpy(dtype=np.float64)
    max_abs = float(np.nanmax(np.abs(values))) if values.size else 1.0
    if not np.isfinite(max_abs) or max_abs == 0:
        max_abs = 1.0

    fig_height = max(4.5, 0.55 * len(selected_ids) + 1.8)
    fig, ax = plt.subplots(figsize=(11, fig_height))
    image = ax.imshow(
        values,
        aspect="auto",
        cmap="coolwarm",
        vmin=-max_abs,
        vmax=max_abs,
    )
    ax.set_xticks(np.arange(len(feature_order)))
    ax.set_xticklabels(feature_order, rotation=45, ha="right")
    ax.set_yticks(np.arange(len(row_labels)))
    ax.set_yticklabels(row_labels)
    ax.set_xlabel("Element feature")
    ax.set_ylabel("Representative OOF validation samples")
    ax.set_title(
        "Local occlusion heatmap\n"
        "color = base positive probability - occluded positive probability"
    )
    colorbar = fig.colorbar(image, ax=ax, pad=0.02)
    colorbar.set_label("Occlusion contribution to positive probability")
    fig.tight_layout()
    fig.savefig(figure_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def plot_occlusion_group_abs_contribution(
    loa_group_summary_df: pd.DataFrame,
    feature_order: Sequence[str],
    group_order: Sequence[str],
    figure_path: Path,
    dpi: int,
) -> None:
    """Plot absolute occlusion contributions by TP/TN/FP/FN outcome groups."""

    available_groups = [
        group
        for group in group_order
        if group in set(loa_group_summary_df["analysis_group"])
    ]
    if not available_groups:
        warnings.warn("No TP/TN/FP/FN groups available for the LOA group plot.")
        return

    data = loa_group_summary_df[
        loa_group_summary_df["analysis_group"].isin(available_groups)
    ].copy()

    # Sort by absolute contribution in the All group, or use the given order.
    all_data = loa_group_summary_df[
        loa_group_summary_df["analysis_group"] == "All"
    ].set_index("feature")
    if not all_data.empty:
        ordered_features = (
            all_data.reindex(feature_order)["mean_absolute"]
            .sort_values(ascending=False)
            .index.tolist()
        )
    else:
        ordered_features = list(feature_order)

    x = np.arange(len(ordered_features))
    width = 0.8 / len(available_groups)

    fig, ax = plt.subplots(figsize=(12, 6))
    for group_index, group_name in enumerate(available_groups):
        group_data = data[data["analysis_group"] == group_name].set_index("feature")
        values = group_data.reindex(ordered_features)["mean_absolute"].fillna(0.0)
        offset = (group_index - (len(available_groups) - 1) / 2) * width
        ax.bar(x + offset, values, width, label=group_name)

    ax.set_xticks(x)
    ax.set_xticklabels(ordered_features, rotation=45, ha="right")
    ax.set_ylabel("Mean absolute occlusion contribution")
    ax.set_title("Occlusion contribution by OOF outcome group")
    ax.legend(title="Group")
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    fig.tight_layout()
    fig.savefig(figure_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def plot_gradient_x_input_overall(
    gxi_sample_feature_df: pd.DataFrame,
    feature_order: Sequence[str],
    figure_path: Path,
    dpi: int,
) -> None:
    """Plot signed Gradient x Input values across unique OOF samples."""

    overall = (
        gxi_sample_feature_df.groupby("feature", as_index=False)[
            "gradient_x_input_mean"
        ]
        .mean()
        .set_index("feature")
        .reindex(feature_order)
        .reset_index()
    )
    overall["abs_value"] = overall["gradient_x_input_mean"].abs()
    overall = overall.sort_values("gradient_x_input_mean", ascending=True)

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.barh(overall["feature"], overall["gradient_x_input_mean"])
    ax.axvline(0.0, linewidth=1)
    ax.set_xlabel("Mean Gradient × Input attribution for positive logit")
    ax.set_ylabel("Element feature")
    ax.set_title("Directional Gradient × Input attribution")
    ax.grid(axis="x", linestyle="--", alpha=0.4)
    fig.tight_layout()
    fig.savefig(figure_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def plot_gradient_x_input_by_group(
    gxi_group_summary_df: pd.DataFrame,
    feature_order: Sequence[str],
    group_order: Sequence[str],
    figure_path: Path,
    dpi: int,
) -> None:
    """Plot signed Gradient x Input by classification outcome group."""

    available_groups = [
        group
        for group in group_order
        if group in set(gxi_group_summary_df["analysis_group"])
    ]
    if not available_groups:
        return

    x = np.arange(len(feature_order))
    width = 0.8 / len(available_groups)

    fig, ax = plt.subplots(figsize=(12, 6))
    for group_index, group_name in enumerate(available_groups):
        group_data = gxi_group_summary_df[
            gxi_group_summary_df["analysis_group"] == group_name
        ].set_index("feature")
        values = group_data.reindex(feature_order)["mean_signed"].fillna(0.0)
        offset = (group_index - (len(available_groups) - 1) / 2) * width
        ax.bar(x + offset, values, width, label=group_name)

    ax.axhline(0.0, linewidth=1)
    ax.set_xticks(x)
    ax.set_xticklabels(feature_order, rotation=45, ha="right")
    ax.set_ylabel("Mean Gradient × Input attribution")
    ax.set_title("Gradient × Input attribution by OOF outcome group")
    ax.legend(title="Group")
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    fig.tight_layout()
    fig.savefig(figure_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def plot_single_sample_loa(
    loa_sample_feature_df: pd.DataFrame,
    representative_df: pd.DataFrame,
    feature_order: Sequence[str],
    output_dir: Path,
    dpi: int,
) -> None:
    """Write separate signed LOA bar plots for representative samples."""

    output_dir.mkdir(parents=True, exist_ok=True)
    rep_meta = representative_df.set_index("sample_index")

    for sample_index in representative_df["sample_index"].astype(int):
        data = loa_sample_feature_df[
            loa_sample_feature_df["sample_index"] == sample_index
        ].set_index("feature").reindex(feature_order).reset_index()

        group_name = str(rep_meta.loc[sample_index, "outcome_group"])
        probability = float(rep_meta.loc[sample_index, "oof_prob_mean"])

        fig, ax = plt.subplots(figsize=(8, 5.5))
        ax.barh(data["feature"], data["contribution_prob_1_mean"])
        ax.axvline(0.0, linewidth=1)
        ax.set_xlabel("Base probability - occluded probability")
        ax.set_ylabel("Element feature")
        ax.set_title(
            f"Sample {sample_index} ({group_name}), mean OOF p1={probability:.3f}"
        )
        ax.grid(axis="x", linestyle="--", alpha=0.4)
        fig.tight_layout()
        fig.savefig(
            output_dir / f"single_sample_{sample_index:04d}_{group_name}.png",
            dpi=dpi,
            bbox_inches="tight",
        )
        plt.close(fig)


# =============================================================================
# 9. Cross-validated interpretability workflow
# =============================================================================


def run_cross_validated_explainability(
    training_module: ModuleType,
    labeled_df: pd.DataFrame,
    train_cfg: Any,
    explain_cfg: ExplainabilityConfig,
    device: torch.device,
    methods: Set[str],
    output_dir: Path,
) -> Dict[str, pd.DataFrame]:
    """
    Compute OOF explanations under the same repeated stratified CV protocol.

    Memory management
    --------
    Retain only the five fold models for the current repeat; after PFI, release
    them before starting the next repeat to avoid holding 50 models in GPU memory.
    """

    labels = labeled_df[train_cfg.label_col].to_numpy(dtype=np.int64)
    num_samples = len(labeled_df)

    splitter = RepeatedStratifiedKFold(
        n_splits=train_cfg.n_splits,
        n_repeats=train_cfg.n_repeats,
        random_state=train_cfg.cv_random_state,
    )
    all_splits = list(splitter.split(np.zeros(num_samples), labels))

    oof_probabilities = np.full(
        (train_cfg.n_repeats, num_samples),
        np.nan,
        dtype=np.float64,
    )
    oof_order_std = np.full_like(oof_probabilities, np.nan)

    pfi_records: List[Dict[str, Any]] = []
    loa_records: List[Dict[str, Any]] = []
    gxi_records: List[Dict[str, Any]] = []
    fold_metric_rows: List[Dict[str, Any]] = []

    for repeat_index in range(train_cfg.n_repeats):
        print("\n" + "#" * 100)
        print(
            f"Explainability repeat {repeat_index + 1:02d}/{train_cfg.n_repeats}"
        )
        print("#" * 100, flush=True)

        bundles: List[FoldBundle] = []
        repeat_oof_prob = np.full(num_samples, np.nan, dtype=np.float64)

        for fold_index in range(train_cfg.n_splits):
            split_index = repeat_index * train_cfg.n_splits + fold_index
            train_indices, val_indices = all_splits[split_index]
            training_seed = (
                train_cfg.base_training_seed + repeat_index * 100 + fold_index
            )

            print(
                f"\n  Fold {fold_index + 1}/{train_cfg.n_splits}, "
                f"seed={training_seed}",
                flush=True,
            )

            train_df = labeled_df.iloc[train_indices].reset_index(drop=True)
            val_df = labeled_df.iloc[val_indices].reset_index(drop=True)

            preprocessor = training_module.fit_preprocessor(train_df, train_cfg)
            train_features, train_coords = preprocessor.transform(train_df)
            val_features, val_coords = preprocessor.transform(val_df)
            train_labels = train_df[train_cfg.label_col].to_numpy(dtype=np.int64)
            val_labels = val_df[train_cfg.label_col].to_numpy(dtype=np.int64)

            (
                model,
                fold_metrics,
                best_epoch,
                val_probabilities,
                val_order_std,
            ) = training_module.train_fold_model(
                train_features=train_features,
                train_coords=train_coords,
                train_labels=train_labels,
                val_features=val_features,
                val_coords=val_coords,
                val_labels=val_labels,
                cfg=train_cfg,
                device=device,
                training_seed=training_seed,
            )

            bundle = FoldBundle(
                repeat_index=repeat_index,
                fold_index=fold_index,
                training_seed=training_seed,
                val_indices=np.asarray(val_indices, dtype=np.int64),
                val_features=val_features,
                val_coords=val_coords,
                val_labels=val_labels,
                baseline_probabilities=val_probabilities,
                baseline_order_std=val_order_std,
                model=model,
            )
            bundles.append(bundle)

            oof_probabilities[repeat_index, val_indices] = val_probabilities
            oof_order_std[repeat_index, val_indices] = val_order_std
            repeat_oof_prob[val_indices] = val_probabilities

            fold_metric_rows.append(
                {
                    "repeat": repeat_index + 1,
                    "fold": fold_index + 1,
                    "training_seed": training_seed,
                    "n_train": len(train_indices),
                    "n_val": len(val_indices),
                    "best_epoch": best_epoch,
                    "mean_order_std": float(val_order_std.mean()),
                    **fold_metrics,
                }
            )

            if "loa" in methods:
                loa_records.extend(
                    calculate_local_occlusion_for_fold(
                        training_module=training_module,
                        bundle=bundle,
                        labeled_df=labeled_df,
                        train_cfg=train_cfg,
                        device=device,
                    )
                )

            if "gxi" in methods:
                gxi_records.extend(
                    calculate_gradient_x_input_for_fold(
                        training_module=training_module,
                        bundle=bundle,
                        labeled_df=labeled_df,
                        train_cfg=train_cfg,
                        device=device,
                    )
                )

        if np.isnan(repeat_oof_prob).any():
            raise RuntimeError(
                f"Repeat {repeat_index + 1}: OOF probabilities contain NaNs."
            )

        if "pfi" in methods:
            print("\n  Calculating repeat-level PFI ...", flush=True)
            pfi_records.extend(
                calculate_pfi_for_repeat(
                    training_module=training_module,
                    bundles=bundles,
                    labels=labels,
                    baseline_oof_probabilities=repeat_oof_prob,
                    train_cfg=train_cfg,
                    explain_cfg=explain_cfg,
                    device=device,
                )
            )

        # Release the five fold models for the completed repeat.
        for bundle in bundles:
            del bundle.model
        del bundles
        if device.type == "cuda":
            torch.cuda.empty_cache()

    if np.isnan(oof_probabilities).any():
        raise RuntimeError("Explainability OOF probabilities still contain NaNs.")

    oof_long_df, oof_mean_df = build_oof_prediction_tables(
        labeled_df=labeled_df,
        labels=labels,
        oof_probabilities=oof_probabilities,
        oof_order_std=oof_order_std,
        train_cfg=train_cfg,
    )

    results: Dict[str, pd.DataFrame] = {
        "FoldMetrics": pd.DataFrame(fold_metric_rows),
        "OOF_Predictions": oof_long_df,
        "OOF_Mean": oof_mean_df,
    }

    if "pfi" in methods:
        pfi_raw_df = pd.DataFrame(pfi_records)
        pfi_by_repeat_df, pfi_summary_df = summarize_pfi(pfi_raw_df)
        results.update(
            {
                "PFI_Raw": pfi_raw_df,
                "PFI_ByRepeat": pfi_by_repeat_df,
                "PFI_Summary": pfi_summary_df,
            }
        )

    if "loa" in methods:
        loa_raw_df = pd.DataFrame(loa_records)
        loa_sample_feature_df, loa_group_summary_df = (
            aggregate_sample_feature_explanations(
                raw_df=loa_raw_df,
                oof_mean_df=oof_mean_df,
                method="loa",
            )
        )
        representative_df = select_representative_samples(
            loa_sample_feature_df=loa_sample_feature_df,
            oof_mean_df=oof_mean_df,
            explain_cfg=explain_cfg,
        )
        results.update(
            {
                "LOA_Raw": loa_raw_df,
                "LOA_SampleFeature": loa_sample_feature_df,
                "LOA_GroupSummary": loa_group_summary_df,
                "RepresentativeSamples": representative_df,
            }
        )

    if "gxi" in methods:
        gxi_raw_df = pd.DataFrame(gxi_records)
        gxi_sample_feature_df, gxi_group_summary_df = (
            aggregate_sample_feature_explanations(
                raw_df=gxi_raw_df,
                oof_mean_df=oof_mean_df,
                method="gxi",
            )
        )
        results.update(
            {
                "GxI_Raw": gxi_raw_df,
                "GxI_SampleFeature": gxi_sample_feature_df,
                "GxI_GroupSummary": gxi_group_summary_df,
            }
        )

    save_explainability_results(
        results=results,
        train_cfg=train_cfg,
        explain_cfg=explain_cfg,
        methods=methods,
        output_dir=output_dir,
    )
    return results


# =============================================================================
# 10. Saving tables and plots
# =============================================================================


def save_explainability_results(
    results: Dict[str, pd.DataFrame],
    train_cfg: Any,
    explain_cfg: ExplainabilityConfig,
    methods: Set[str],
    output_dir: Path,
) -> None:
    """Save Excel workbooks, JSON metadata, and candidate publication figures."""

    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    # Excel limits sheet names to 31 characters; all names below comply.
    workbook_path = output_dir / "explainability_results.xlsx"
    with pd.ExcelWriter(workbook_path, engine="openpyxl") as writer:
        sheet_map = {
            "FoldMetrics": "FoldMetrics",
            "OOF_Predictions": "OOF_Predictions",
            "OOF_Mean": "OOF_Mean",
            "PFI_Raw": "PFI_Raw",
            "PFI_ByRepeat": "PFI_ByRepeat",
            "PFI_Summary": "PFI_Summary",
            "LOA_Raw": "LOA_Raw",
            "LOA_SampleFeature": "LOA_SampleFeature",
            "LOA_GroupSummary": "LOA_GroupSummary",
            "GxI_Raw": "GxI_Raw",
            "GxI_SampleFeature": "GxI_SampleFeature",
            "GxI_GroupSummary": "GxI_GroupSummary",
            "RepresentativeSamples": "RepresentativeSamples",
        }
        for key, dataframe in results.items():
            if key in sheet_map:
                dataframe.to_excel(writer, sheet_name=sheet_map[key], index=False)

    # Export key summary sheets separately for use in plotting or reporting.
    for key in [
        "PFI_Summary",
        "LOA_SampleFeature",
        "LOA_GroupSummary",
        "GxI_SampleFeature",
        "GxI_GroupSummary",
        "RepresentativeSamples",
        "OOF_Mean",
    ]:
        if key in results:
            results[key].to_excel(output_dir / f"{key}.xlsx", index=False)

    summary_json: Dict[str, Any] = {
        "methods": sorted(methods),
        "training_protocol": {
            "n_splits": train_cfg.n_splits,
            "n_repeats": train_cfg.n_repeats,
            "total_fold_models": train_cfg.n_splits * train_cfg.n_repeats,
            "evaluation_permutations": train_cfg.evaluation_permutations,
            "classification_threshold": train_cfg.classification_threshold,
        },
        "explainability_config": asdict(explain_cfg),
        "output_workbook": str(workbook_path),
    }
    if "PFI_Summary" in results:
        summary_json["top_elements_by_delta_f1"] = (
            results["PFI_Summary"]
            .query("feature_type == 'element'")
            .sort_values("mean_delta_f1", ascending=False)
            .head(12)[["feature", "mean_delta_f1", "std_delta_f1", "mean_delta_auc"]]
            .to_dict(orient="records")
        )

    with (output_dir / "explainability_summary.json").open(
        "w", encoding="utf-8"
    ) as file:
        json.dump(summary_json, file, ensure_ascii=False, indent=2)

    configure_matplotlib()

    if "pfi" in methods:
        plot_pfi_delta_f1(
            results["PFI_Summary"],
            figures_dir / "fig01_permutation_delta_f1_bar.png",
            explain_cfg.figure_dpi,
        )
        plot_pfi_f1_auc_comparison(
            results["PFI_Summary"],
            figures_dir / "fig02_permutation_f1_auc_comparison.png",
            explain_cfg.figure_dpi,
        )

    if "loa" in methods:
        plot_local_occlusion_heatmap(
            loa_sample_feature_df=results["LOA_SampleFeature"],
            representative_df=results["RepresentativeSamples"],
            feature_order=train_cfg.feature_cols,
            figure_path=figures_dir / "fig03_local_occlusion_heatmap.png",
            dpi=explain_cfg.figure_dpi,
        )
        plot_occlusion_group_abs_contribution(
            loa_group_summary_df=results["LOA_GroupSummary"],
            feature_order=train_cfg.feature_cols,
            group_order=explain_cfg.group_order,
            figure_path=figures_dir / "fig04_occlusion_group_abs_contribution.png",
            dpi=explain_cfg.figure_dpi,
        )
        if explain_cfg.make_single_sample_figures:
            plot_single_sample_loa(
                loa_sample_feature_df=results["LOA_SampleFeature"],
                representative_df=results["RepresentativeSamples"],
                feature_order=train_cfg.feature_cols,
                output_dir=figures_dir / "single_samples",
                dpi=explain_cfg.figure_dpi,
            )

    if "gxi" in methods:
        plot_gradient_x_input_overall(
            gxi_sample_feature_df=results["GxI_SampleFeature"],
            feature_order=train_cfg.feature_cols,
            figure_path=figures_dir / "fig05_gradient_x_input_diverging_bar.png",
            dpi=explain_cfg.figure_dpi,
        )
        plot_gradient_x_input_by_group(
            gxi_group_summary_df=results["GxI_GroupSummary"],
            feature_order=train_cfg.feature_cols,
            group_order=explain_cfg.group_order,
            figure_path=figures_dir / "fig06_gradient_x_input_by_group.png",
            dpi=explain_cfg.figure_dpi,
        )

    print("\nExplainability results saved:")
    print(f"  Excel: {workbook_path}")
    print(f"  Figures: {figures_dir}")


# =============================================================================
# 11. Command-line interface
# =============================================================================


def parse_methods(text: str) -> Set[str]:
    """Parse the --methods argument, e.g., pfi,loa,gxi."""

    methods = {item.strip().lower() for item in text.split(",") if item.strip()}
    valid = {"pfi", "loa", "gxi"}
    unknown = methods - valid
    if unknown:
        raise ValueError(f"Unknown methods: {sorted(unknown)}; choose pfi,loa,gxi.")
    if not methods:
        raise ValueError("Select at least one explanation method.")
    return methods


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Cross-validated explainability for Spatial OI-Mamba"
    )
    parser.add_argument(
        "--mode",
        choices=("check", "run"),
        default="run",
        help="check=validate inputs; run=retrain CV folds and calculate explanations.",
    )
    parser.add_argument(
        "--training-script",
        default="oi_mamba_repeated_cv_train_predict.py",
        help="Path to the shared model-training script.",
    )
    parser.add_argument(
        "--config-json",
        default=None,
        help="Optional config.json created by the training script.",
    )
    parser.add_argument("--labeled-file", default=None)
    parser.add_argument("--prediction-file", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default=None)
    parser.add_argument("--n-repeats", type=int, default=None)
    parser.add_argument(
        "--methods",
        default="pfi,loa,gxi",
        help="Comma-separated methods (default: pfi,loa,gxi).",
    )
    parser.add_argument("--pfi-repeats", type=int, default=None)
    parser.add_argument(
        "--no-spatial-pfi",
        action="store_true",
        help="Disable grouped X/Y permutation feature importance.",
    )
    parser.add_argument("--representatives-per-group", type=int, default=None)
    parser.add_argument(
        "--no-single-sample-figures",
        action="store_true",
        help="Do not create per-sample LOA figures.",
    )
    return parser


def main() -> None:
    parser = build_argument_parser()
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    training_script_path = Path(args.training_script)
    if not training_script_path.is_absolute():
        training_script_path = script_dir / training_script_path

    training_module = load_training_module(training_script_path)

    config_json_path: Optional[Path] = None
    if args.config_json is not None:
        config_json_path = Path(args.config_json)
        if not config_json_path.is_absolute():
            config_json_path = script_dir / config_json_path

    train_cfg = load_training_config_from_json(
        training_module=training_module,
        json_path=config_json_path,
    )

    # Override selected training-config paths, device, and repeat count from CLI.
    train_updates: Dict[str, Any] = {}
    if args.labeled_file is not None:
        train_updates["labeled_file"] = args.labeled_file
    if args.prediction_file is not None:
        train_updates["prediction_file"] = args.prediction_file
    if args.device is not None:
        train_updates["device"] = args.device
    if args.n_repeats is not None:
        if args.n_repeats < 1:
            raise ValueError("--n-repeats must be at least 1.")
        train_updates["n_repeats"] = args.n_repeats
    if train_updates:
        train_cfg = replace(train_cfg, **train_updates)

    explain_cfg = ExplainabilityConfig()
    explain_updates: Dict[str, Any] = {}
    if args.output_dir is not None:
        explain_updates["output_dir"] = args.output_dir
    if args.pfi_repeats is not None:
        if args.pfi_repeats < 1:
            raise ValueError("--pfi-repeats must be at least 1.")
        explain_updates["pfi_repeats"] = args.pfi_repeats
    if args.no_spatial_pfi:
        explain_updates["include_spatial_pfi"] = False
    if args.representatives_per_group is not None:
        if args.representatives_per_group < 1:
            raise ValueError("--representatives-per-group must be at least 1.")
        explain_updates["representatives_per_group"] = (
            args.representatives_per_group
        )
    if args.no_single_sample_figures:
        explain_updates["make_single_sample_figures"] = False
    if explain_updates:
        explain_cfg = replace(explain_cfg, **explain_updates)

    methods = parse_methods(args.methods)

    labeled_path = Path(train_cfg.labeled_file)
    prediction_path = Path(train_cfg.prediction_file)
    if not labeled_path.is_absolute():
        labeled_path = script_dir / labeled_path
    if not prediction_path.is_absolute():
        prediction_path = script_dir / prediction_path

    output_dir = Path(explain_cfg.output_dir)
    if not output_dir.is_absolute():
        output_dir = script_dir / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    labeled_df, prediction_df, input_summary = training_module.load_and_validate_data(
        labeled_path=labeled_path,
        prediction_path=prediction_path,
        cfg=train_cfg,
    )

    # Save the resolved paths and configuration used in this analysis.
    run_config = {
        "training_script": str(training_script_path),
        "config_json": None if config_json_path is None else str(config_json_path),
        "resolved_labeled_file": str(labeled_path),
        "resolved_prediction_file": str(prediction_path),
        "training_config": asdict(train_cfg),
        "explainability_config": asdict(explain_cfg),
        "methods": sorted(methods),
        "input_summary": input_summary,
    }
    with (output_dir / "explainability_run_config.json").open(
        "w", encoding="utf-8"
    ) as file:
        json.dump(run_config, file, ensure_ascii=False, indent=2)

    print("=" * 100)
    print("Spatial OI-Mamba cross-validated explainability")
    print(f"Training script: {training_script_path}")
    print(f"Labeled data:   {labeled_path}")
    print(f"Samples:        {len(labeled_df)}")
    print(
        f"CV:             {train_cfg.n_splits} folds × "
        f"{train_cfg.n_repeats} repeats"
    )
    print(f"Methods:        {sorted(methods)}")
    print(f"PFI repeats:    {explain_cfg.pfi_repeats}")
    print(f"Output:         {output_dir}")
    print("=" * 100)

    if args.mode == "check":
        print("Input validation passed. No model training or explanations were run.")
        return

    training_module.require_mamba()
    device = training_module.resolve_device(train_cfg.device)
    print(f"Device: {device}")

    run_cross_validated_explainability(
        training_module=training_module,
        labeled_df=labeled_df,
        train_cfg=train_cfg,
        explain_cfg=explain_cfg,
        device=device,
        methods=methods,
        output_dir=output_dir,
    )


if __name__ == "__main__":
    main()
