# Spatial Order-Insensitive Mamba (Spatial OI-Mamba)

**Spatially aware, order-robust Mamba modeling and out-of-fold explanations for mineralization-related geochemical anomaly identification.**

This repository provides the Python implementation accompanying the manuscript **“Order-Insensitive Mamba with Spatial Encoding and Multilevel Explanations for Mineralization-Related Geochemical Anomaly Identification.”** It covers repeated stratified cross-validation, final regional prediction, and three complementary post hoc explainability methods.

> **Data availability:** The original stream-sediment measurements, labeled observations, and interpolated prediction grid are **not included** in this repository. The English data schema is documented in [`data/README.md`](data/README.md). Users must supply their own appropriately licensed Excel datasets before running the workflow.

## Methods

The model represents each chemical element as a distinct token with a shared concentration projection and a learnable element-identity embedding. It combines per-sample synchronized permutations, a two-layer selective state-space (Mamba) encoder, mean pooling, and two-dimensional sinusoidal coordinate encoding. Multiple fixed element orders are averaged at inference time to reduce sensitivity to an arbitrary element order.

The explainability script implements:

- **Permutation Feature Importance (PFI):** changes in complete-repeat, out-of-fold classification metrics when one feature is independently permuted within each held-out fold; optionally includes a grouped X/Y coordinate permutation.
- **Local Occlusion Analysis (LOA):** changes in predicted positive-class probability when a standardized element is replaced by its training-fold reference value (zero in standardized log1p space).
- **Gradient × Input (GxI):** signed local sensitivity of the positive-class logit, averaged over a fixed collection of element orders.

“Order-insensitive” means **empirically reduced order sensitivity**, not mathematical permutation invariance. The feature-importance and attribution methods describe model behavior; they do not establish causal geological effects.

## Project structure

```text
spatial-oi-mamba/
├── README.md
├── requirements.txt
├── requirements-mamba.txt
├── .gitignore
├── oi_mamba_repeated_cv_train_predict.py
├── oi_mamba_explainability_oof.py
├── data/
│   ├── README.md
│   ├── labeled_samples_columns.csv
│   └── prediction_grid_columns.csv
└── tests/
    └── test_repository.py
```

## Installation

Python 3.10+ is recommended. Model training requires **PyTorch** and a working **`mamba-ssm`** installation. An NVIDIA CUDA environment is recommended; check the official installation requirements of your selected PyTorch, `causal-conv1d`, and `mamba-ssm` versions.

```bash
git clone https://github.com/YOUR-USERNAME/spatial-oi-mamba.git
cd spatial-oi-mamba
python -m venv .venv
# Activate the environment using the appropriate command for your operating system.
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Install an appropriate CUDA-enabled or CPU-enabled build of PyTorch for your environment, then install the Mamba extensions. For example, in a compatible build environment:

```bash
python -m pip install causal-conv1d mamba-ssm --no-build-isolation
```

The optional [`requirements-mamba.txt`](requirements-mamba.txt) lists both Mamba packages, but their installation may require environment-specific wheels or source builds. **Input checks** and unit tests do not require `mamba-ssm`; actual Mamba training and prediction do.

## Input data format

Prepare two Excel files (`.xlsx`) with **English column headers** in the `data/` directory:

| File | Description | Required headers |
| --- | --- | --- |
| `data/labeled_samples.xlsx` | Positive and negative training observations | `X`, `Y`, `Sb`, `Ti`, `V`, `Mo`, `As`, `Zn`, `Pb`, `Cu`, `Ag`, `Hg`, `Au`, `W`, `mineralization_label` |
| `data/prediction_grid.xlsx` | Grid cells for regional prediction | `X`, `Y`, `Sb`, `Ti`, `V`, `Mo`, `As`, `Zn`, `Pb`, `Cu`, `Ag`, `Hg`, `Au`, `W` |

- `mineralization_label`: binary integer label; **1** indicates a positive sample associated with a known mineral deposit, and **0** indicates a selected pseudo-negative sample.
- `X` and `Y`: numeric **projected planar coordinates** appropriate for the study area (use the same coordinate reference system in both files).
- Element columns: raw nonnegative numeric concentrations, **not** preprocessed values. Units should be consistent between labeled and prediction datasets.
- The 12 chemical elements are read in the fixed order defined by `Config.feature_cols` regardless of Excel column order. During model training, values and their corresponding element identities are permuted together.
- All required fields must be present and contain finite numeric values. Negative concentrations are warned about and clipped to zero before log1p transformation.

The manuscript case study used **18 positive and 18 pseudo-negative labeled samples** and **14,400 prediction grid cells**. The scripts can validate other datasets, provided both classes contain enough observations for the requested number of CV folds.

The two header-only CSV files under `data/` are **schema templates**, not usable measurements. Open them in a spreadsheet editor, populate with your own observations, and save them as the `.xlsx` files above. See [`data/README.md`](data/README.md) for additional guidance.

## Running the training and prediction workflow

### 1. Validate your Excel files without fitting Mamba

```bash
python oi_mamba_repeated_cv_train_predict.py --mode check
```

### 2. Repeated stratified cross-validation only

```bash
python oi_mamba_repeated_cv_train_predict.py --mode cv
```

By default, the workflow performs **10 × stratified five-fold CV (50 fold models)**. Each repeat aggregates one OOF prediction per labeled observation before calculating accuracy, precision, recall, F1, AUC, and log loss. Fold-level preprocessing is fitted on the training split only.

### 3. Cross-validation followed by final ensemble prediction

```bash
python oi_mamba_repeated_cv_train_predict.py --mode all
```

The model ensemble is retrained on all labeled observations using the median CV-selected epoch count. Probabilities for every prediction-grid cell are averaged across **10 final models** and **10 element orders per model** by default.

To override inputs or use a smaller development run:

```bash
python oi_mamba_repeated_cv_train_predict.py --mode cv \
  --labeled-file data/labeled_samples.xlsx \
  --prediction-file data/prediction_grid.xlsx \
  --output-dir results/development_cv \
  --n-repeats 1 --device auto
```

Run `python oi_mamba_repeated_cv_train_predict.py --help` for all supported options.

## Running the explainability workflow

After preparing both Excel files, run:

```bash
python oi_mamba_explainability_oof.py --mode run
```

The explainability workflow **retrains** the same CV fold models, because the training script does not save individual fold checkpoints. It analyzes each held-out fold using the same preprocessing, training seeds, and fixed element-order protocol. The full default run may be computationally expensive, especially PFI.

For a quick development run (not for publication-level results):

```bash
python oi_mamba_explainability_oof.py --mode run \
  --n-repeats 1 --pfi-repeats 3 --methods pfi,loa,gxi
```

To reuse an existing training configuration:

```bash
python oi_mamba_explainability_oof.py --mode run \
  --config-json oi_mamba_repeated_cv_results/config.json
```

Note that `--config-json` may contain input locations from the original run; override them with `--labeled-file` and `--prediction-file` when necessary. The explainability script currently expects both workbooks because it shares the training script's input-validation routine, although the attribution calculations use only the labeled data.

Run `python oi_mamba_explainability_oof.py --help` for available options, including `--methods`, `--no-spatial-pfi`, and figure-generation settings.

## Output files

### Training and regional mapping (`oi_mamba_repeated_cv_results/`)

| Output | Content |
| --- | --- |
| `config.json` | Actual configuration and resolved paths |
| `input_data_summary.json` | Input checks, class counts, and column names |
| `cv_results.xlsx` | Fold metrics, repeat metrics, summary, and epoch recommendation |
| `cv_oof_predictions_long.xlsx` | OOF predictions for each repeat and validation sample |
| `cv_oof_mean_predictions.xlsx` | Per-sample averages across CV repeats |
| `cv_summary.json` | CV metric summaries and final epoch recommendation |
| `final_models/*.pt` | Final ensemble model state dictionaries |
| `final_preprocessor.joblib` | Preprocessing scalers and schema |
| `prediction_all_with_prob.xlsx` | Regional positive-class probabilities, ensemble dispersion, and binary predictions |
| `labeled_fullfit_predictions.xlsx` | Predictions on the full data used for final fitting (**not CV metrics**) |
| `final_training_summary.xlsx` | Seeds and training losses for final models |
| `final_prediction_metadata.json` | Regional prediction summary |

### Explainability (`oi_mamba_explainability_results/`)

`explainability_results.xlsx` contains `FoldMetrics`, `OOF_Predictions`, `OOF_Mean`, `PFI_Raw`, `PFI_ByRepeat`, `PFI_Summary`, `LOA_Raw`, `LOA_SampleFeature`, `LOA_GroupSummary`, `GxI_Raw`, `GxI_SampleFeature`, `GxI_GroupSummary`, and `RepresentativeSamples`. Additional summary spreadsheets, JSON run metadata, and PNG figures are also created.

## Reproducibility and methodological boundaries

1. **Cross-validation and early stopping.** Per-fold scalers are fitted only on training samples. The **outer validation fold is also used for early stopping and final evaluation** in the provided scripts; this may induce optimistic model-selection bias. A nested or independent inner split is preferable for an unbiased external estimate.
2. **Element-order sensitivity.** The training script averages **10 element orders** by default and reports within-model order probability standard deviation across those 10 orders. The manuscript separately describes a **50-order, unaveraged stability test and label-consistency statistic**, which these two scripts do **not** implement as a standalone analysis. Do not treat the 10-order dispersion as the manuscript's 50-order result.
3. **Scope of implementation.** These files implement the proposed Spatial OI-Mamba workflow and its three explanation methods. They do **not** include the manuscript's alternative baseline algorithms, ablation variants, or external geological target delineation procedure.
4. **Data interpretation.** Regional outputs indicate model-relative favorability, not calibrated probabilities of an ore body being present. Observed agreement with deposits used for training is not independent discovery validation.
5. **Repeatability.** Seeds, repeated splits, and element-order draws are controlled; low-level CUDA operations may nevertheless vary across environments.
6. **Data privacy and licensing.** Do not publish private coordinates, unreleased exploration data, or datasets with redistribution restrictions without appropriate authorization.

## Tests

The repository includes small tests for schema checks, preprocessing, and deterministic permutations (they do not train the real Mamba backend):

```bash
python -m pip install pytest
python -m pytest -q
```

## Citation and licensing

If you use the code in research, cite the associated manuscript once its bibliographic details become available:

*Order-Insensitive Mamba with Spatial Encoding and Multilevel Explanations for Mineralization-Related Geochemical Anomaly Identification*.

No software license has been selected here. **Choose and add an appropriate `LICENSE` file before advertising the repository as open-source.**
