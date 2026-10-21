"""Lightweight tests that do not require the Mamba backend or geochemical study data."""

from pathlib import Path
import importlib.util
import sys

import numpy as np
import pandas as pd
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import oi_mamba_repeated_cv_train_predict as training


def example_frames():
    """Construct small synthetic records strictly for schema testing."""
    cfg = training.Config()
    rng = np.random.default_rng(24)
    n = 12  # At least five samples per class for five-fold CV.
    labeled = pd.DataFrame({
        "X": np.arange(n, dtype=float) + 1_000,
        "Y": np.arange(n, dtype=float) + 2_000,
        **{feature: rng.random(n) for feature in cfg.feature_cols},
        cfg.label_col: np.array([0, 1] * (n // 2), dtype=int),
    })
    grid = labeled.drop(columns=[cfg.label_col]).iloc[:4].copy()
    return cfg, labeled, grid


def test_english_schema_and_input_validation(tmp_path):
    cfg, labeled, grid = example_frames()
    assert cfg.label_col == "mineralization_label"
    assert cfg.labeled_file == "data/labeled_samples.xlsx"
    assert cfg.prediction_file == "data/prediction_grid.xlsx"

    labeled_path = tmp_path / "labeled_samples.xlsx"
    grid_path = tmp_path / "prediction_grid.xlsx"
    labeled.to_excel(labeled_path, index=False)
    grid.to_excel(grid_path, index=False)

    loaded_labeled, loaded_grid, summary = training.load_and_validate_data(
        labeled_path, grid_path, cfg
    )
    assert len(loaded_labeled) == 12
    assert len(loaded_grid) == 4
    assert summary["class_counts"] == {"0": 6, "1": 6}
    assert summary["label_col"] == "mineralization_label"


def test_missing_english_label_is_rejected(tmp_path):
    cfg, labeled, grid = example_frames()
    labeled.rename(columns={"mineralization_label": "missing_label"}).to_excel(
        tmp_path / "labeled_samples.xlsx", index=False
    )
    grid.to_excel(tmp_path / "prediction_grid.xlsx", index=False)
    with pytest.raises(ValueError, match="Missing columns in labeled data"):
        training.load_and_validate_data(
            tmp_path / "labeled_samples.xlsx", tmp_path / "prediction_grid.xlsx", cfg
        )


def test_preprocessing_uses_training_fold_only():
    cfg, labeled, _ = example_frames()
    fitted = training.fit_preprocessor(labeled.iloc[:8], cfg)
    scaled_train, coords_train = fitted.transform(labeled.iloc[:8])
    assert scaled_train.shape == (8, 12)
    assert coords_train.shape == (8, 2)
    np.testing.assert_allclose(scaled_train.mean(axis=0), 0.0, atol=1e-6)
    np.testing.assert_allclose(coords_train.mean(axis=0), 0.0, atol=1e-6)


def test_fixed_permutations_are_reproducible():
    p1 = training.make_deterministic_permutation(4, 12, 42, torch.device("cpu"))
    p2 = training.make_deterministic_permutation(4, 12, 42, torch.device("cpu"))
    assert torch.equal(p1, p2)
    assert all(torch.equal(torch.sort(row).values, torch.arange(12)) for row in p1)


def test_explainability_module_imports_and_defaults():
    import oi_mamba_explainability_oof as explain

    cfg = explain.ExplainabilityConfig()
    assert cfg.pfi_repeats == 30
    assert explain.parse_methods("pfi,loa,gxi") == {"pfi", "loa", "gxi"}
    with pytest.raises(ValueError, match="Unknown methods"):
        explain.parse_methods("invalid")
