# ML Algorithm Comparison and Insight Tool

A Flask web app for automated EDA and quick ML model comparison with server‑rendered Plotly charts.

## Key features

- Upload CSV/XLSX/XLS and get instant EDA: size, dtypes, missingness, duplicate count, memory footprint
- Smart target detection and problem-type inference (binary/multiclass/classification vs regression)
- One‑click training and comparison of multiple models:
  - Classification: Logistic Regression, Random Forest, Gradient Boosting, SVC, Decision Tree
  - Regression: Linear Regression, Random Forest, Gradient Boosting, SVR, Decision Tree
- Task‑specific metrics: Accuracy/Precision/Recall/F1 or R²/MSE, plus training time
- Best model selection, CSV export of all results, and downloadable model package (model + preprocessors)
- Secure server‑generated data preview and chart HTML (reduced client‑side attack surface)

## What’s new (Aug 26, 2025)

- Switched charting to Python/Plotly rendered on the server and injected as HTML
  - Endpoints: `/quality_chart`, `/types_chart`, `/missing_chart`, `/feature_importance_chart`, `/performance_chart`
- Moved validations and analysis from JS to Python for consistency and reliability
  - Upload/file validation, target analysis, split percentage, training config checks
  - New/updated endpoints: `/analyze_target`, `/get_data_preview_html`, `/calculate_split`, `/validate_training_config`
- Improved EDA: memory usage, capped correlations, data‑quality score, JSON‑safe outputs
- Robust preprocessing: ColumnTransformer (impute+scale numeric; impute+one‑hot categorical), LabelEncoder for target
- AutoMLComparer: unified pipeline with sensible defaults and timeouts per model; SVM auto‑scales
- Feature importance with fallbacks: native importances/coefficients or permutation importance for the best model
- Session hygiene and caching
  - Large EDA/training artifacts stored in a server‑side in‑process cache keyed by `session_id`
  - Cookie bloat avoided; session lifetime set to 1h; uploads cleaned up on shutdown

## Tech stack

- Backend: Flask, pandas, numpy, scikit‑learn, joblib
- Visualization: Plotly (server‑rendered HTML snippets; Plotly CDN loaded in layout)
- Frontend: HTML/CSS/vanilla JS (no framework)
- File support: pandas + openpyxl (.xlsx) + xlrd 1.2.0 (.xls)

## Project structure

```
├── app.py               # Flask routes: upload, EDA, training, charts, exports
├── model_utils.py       # Backward‑compat API mapping to ml_utils/* modules
├── ml_utils/
│   ├── config.py        # MLConfig, typed results, preview HTML helper
│   ├── models.py        # AutoMLComparer: train/score/select + importance
│   ├── preprocessing.py # ColumnTransformer pipelines + fallback
│   ├── eda.py           # Minimal+enhanced EDA and correlations
│   ├── charts.py        # Plotly chart builders (server‑side)
│   └── utils.py         # JSON safety, validations, target detection
├── templates/           # Jinja (layout + index)
├── static/              # style.css, app.js
└── requirements.txt     # Python dependencies
```

## Setup

1) Create a virtual environment (macOS/Linux)
```
python -m venv .venv
source .venv/bin/activate
```
2) Install dependencies
```
pip install -r requirements.txt
```

## Run

```
python app.py
```
Then open http://127.0.0.1:5002/ in your browser.

Notes
- Max upload size: 50MB; supported: .csv, .xlsx, .xls
- Secret key is randomized per run; sessions reset when the server restarts

## Usage workflow

1) Upload a dataset to see a safe, server‑generated preview
2) Click “Analyze Dataset” to run EDA and view Plotly charts
3) Select a target; the app auto‑detects problem type (you can override it)
4) Validate config and start training; models are compared automatically
5) Inspect performance and feature importance; export CSV or download the model package

## API endpoints (selected)

- Upload/EDA
  - POST `/` — Upload file (AJAX)
  - POST `/process_eda` — Run EDA and return stats
  - GET  `/eda` — EDA JSON
  - GET  `/data_preview` — Preview JSON
  - GET  `/get_data_preview_html` — Secure HTML preview
- Training/results
  - POST `/validate_training_config` — Validate config
  - POST `/train` — Train and return metrics/results/importance
  - GET  `/best_model`, `/metrics`, `/model_comparison`, `/feature_importance`
- Charts (server‑rendered Plotly)
  - GET  `/quality_chart`, `/types_chart`, `/missing_chart`
  - GET  `/feature_importance_chart`, `/performance_chart`
- Utilities
  - POST `/analyze_target`, POST `/calculate_split`
  - GET  `/download_model` (ZIP: model.joblib + preprocessors.joblib + README)
  - GET  `/export_results` (CSV of all models)

## Security and data handling

- Server‑side HTML escaping in previews; chart HTML generated on the server
- In‑process cache for large artifacts; no large blobs in cookies
- Uploads saved under `uploads/` for the session; removed on shutdown

## Troubleshooting

- `.xls` requires `xlrd==1.2.0`; for `.xlsx` only, you can remove xlrd
- Very large/wide datasets: correlations are capped to reduce memory pressure
- If a model exceeds the time budget, it’s skipped; consider sampling or simpler models

## License

MIT
