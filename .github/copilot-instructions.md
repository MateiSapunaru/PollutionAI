# Copilot instructions for PollutionAI

Purpose: Help AI coding agents become productive quickly in this repository.

Quick start (Windows PowerShell)

```
python -m venv .venv; .\\.venv\\Scripts\\Activate.ps1
pip install -r requirements.txt
# Train models and generate artifacts
python src/concentration_ai.py
python src/disease_ai.py
python src/visualization.py
# Run the Streamlit app
streamlit run app.py
```

High-level architecture

- `app.py` — Streamlit frontend and entrypoint. Loads model packages from `models/` and visualization PNGs from `visualizations/`.
- `src/concentration_ai.py` — training pipeline for pollutant concentration (regression). Produces `models/concentration_model.pkl` and model visualizations in `models/`.
- `src/disease_ai.py` — training pipeline for disease-type classification. Produces `models/disease_type_model.pkl` and visualizations.
- `src/visualization.py` — dataset analysis and plot generation saved to `visualizations/` (expected filenames documented below).
- `src/utils.py` — shared helpers (encoding, scaling, metrics, plotting).
- `data/soil_contamination_scientific.csv` — canonical dataset used by training and visualizations.

Important conventions & patterns (project-specific)

- Model artifact shape: training scripts save a dictionary (joblib) with keys used by `app.py`:
  - Regression (`concentration_model.pkl`): `{'model','model_name','encoders','scaler','feature_names'}`
  - Classification (`disease_type_model.pkl`): same keys plus `'target_encoder'`.
- Encoders are `sklearn.preprocessing.LabelEncoder`; `scaler` is a `StandardScaler` — the app applies `encoder.transform()` and `scaler.transform()` at runtime. Keep training & runtime encoders in sync.
- Feature order matters: `model_package['feature_names']` defines the exact column order expected by the model. Do not reorder or rename those columns lightly.
- Categorical label stability: `app.py` includes a `DATASET_VALUES` map (fixed lists like `pollutants`, `countries`, `soil_textures`) that must match the labels used during training — changing these will break runtime `encoder.transform()` calls.
- Random seeds: training scripts use `random_state=42` for reproducibility.

Files and artifact names the app expects

- Models: `models/concentration_model.pkl`, `models/disease_type_model.pkl`
- Visualization images (checked by `app.py`): `visualizations/ph_bioavailability_analysis.png`, `visualizations/soil_texture_analysis.png`, `visualizations/distance_decay_analysis.png`, `visualizations/pollutant_analysis.png`, `visualizations/age_vulnerability_analysis.png`, `visualizations/industry_patterns.png`
- Model visualizations saved under `models/`: `concentration_model_comparison.png`, `concentration_feature_importance.png`, `concentration_residual_analysis.png`, `disease_type_model_comparison.png`, `disease_type_confusion_matrices.png`

Developer workflows & debugging tips

- If a model fails to load in `app.py`, verify the file exists and is a joblib-dumped dict with the keys above. Example quick check:

```python
import joblib
pkg = joblib.load('models/concentration_model.pkl')
print(pkg.keys())
print(len(pkg['feature_names']))
```

- To reproduce training exactly, run the scripts from the repository root (they read `data/soil_contamination_scientific.csv`):

```
python src/concentration_ai.py
python src/disease_ai.py
```

- Example minimal prediction snippet (match `feature_names` order and encode categories):

```python
import joblib
import pandas as pd

pkg = joblib.load('models/concentration_model.pkl')
inp = pd.DataFrame([{
  k: 'SOME_VALUE' if k.startswith('Pollutant') else 0.0 for k in pkg['feature_names']
}])
# apply encoders and scaler in the same way training did
for col, enc in pkg['encoders'].items():
    if col in inp.columns:
        inp[col] = enc.transform(inp[col].astype(str))
X = pkg['scaler'].transform(inp[pkg['feature_names']])
pred = pkg['model'].predict(X)
```

When modifying model inputs

- Update `DATASET_VALUES` in `app.py` when you intentionally change categorical labels — but prefer updating training data and re-training so encoders remain consistent.
- When you change feature columns, update `feature_cols` lists in `src/concentration_ai.py` and `src/disease_ai.py` and retrain both models.

External dependencies and heavy libraries

- This repo depends on: `pandas`, `numpy`, `scikit-learn`, `xgboost`, `lightgbm`, `joblib`, `streamlit`, `matplotlib`, `seaborn`, `Pillow` (see `requirements.txt`). Expect long training times on CPU for tree ensembles.

If anything is unclear or you want me to include test snippets, examples for CI, or a short README we can add, tell me which sections to expand. 
