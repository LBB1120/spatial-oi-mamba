# Input data and column schema

The real geochemical measurements are **not distributed** with this code repository. Both runtime inputs must be user-provided Excel (`.xlsx`) files, with numeric values and **English** column names.

## 1. Labeled samples: `labeled_samples.xlsx`

Required column order in the provided template (the script selects columns by name and does not require this physical order):

```text
X,Y,Sb,Ti,V,Mo,As,Zn,Pb,Cu,Ag,Hg,Au,W,mineralization_label
```

- `X`, `Y`: projected planar coordinates in the same CRS as the prediction grid.
- `Sb`, `Ti`, `V`, `Mo`, `As`, `Zn`, `Pb`, `Cu`, `Ag`, `Hg`, `Au`, `W`: raw concentrations in consistent units and numeric cells, not normalized or logarithmically transformed.
- `mineralization_label`: integer `1` for a positive deposit-associated sample and `0` for a pseudo-negative sample.
- For the manuscript's case study, 18 positives and 18 pseudo-negatives were used; the original labeled observations are not included here.

## 2. Prediction grid: `prediction_grid.xlsx`

Required headers:

```text
X,Y,Sb,Ti,V,Mo,As,Zn,Pb,Cu,Ag,Hg,Au,W
```

- Each row is one regional grid cell or point at which an anomaly score should be predicted.
- The input coordinates must be expressed in the same projected CRS as the labeled samples.
- The element measurement units and detection-limit handling must be consistent with those used for labeled data.
- Do not provide missing, infinite, or nonnumeric feature values. An optional label column is ignored by the prediction routine.

## Creating your own workbooks

1. Open `labeled_samples_columns.csv` and `prediction_grid_columns.csv` in Excel or another spreadsheet editor. They contain **headers only**; no placeholder values are presented as real data.
2. Add observations in rows beneath the headers, preserving the English names exactly (case sensitive).
3. Save each workbook as `labeled_samples.xlsx` and `prediction_grid.xlsx` in this directory.
4. Check the data without fitting a model:

   ```bash
   python oi_mamba_repeated_cv_train_predict.py --mode check
   ```

The original study files may use different names or non-English column headings. Rename the corresponding columns and the output workbooks to the English schema above before using these scripts. Do **not** substitute arbitrary generated numbers for scientific observations when reporting research results.

The code clips negative element values to zero before log1p, with a warning; investigate negative measurements or detection-limit representations rather than relying blindly on this behavior.
