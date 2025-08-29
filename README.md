# ML Algorithm Comparison and Insight Tool

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![Flask](https://img.shields.io/badge/Flask-3.0+-green.svg)](https://flask.palletsprojects.com/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.4+-orange.svg)](https://scikit-learn.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A Flask web app for automated Exploratory Data Analysis (EDA) and quick ML model comparison, with secure server-rendered previews and exports.

## 📋 Table of Contents

- Features
- Quick Start
- Installation
- Usage
- Project Structure
- API
- Supported Algorithms
- Configuration
- Troubleshooting
- Contributing
- License
- Changelog

## 🚀 Features

- Automated EDA: dataset size, dtypes, missingness, duplicates, memory usage
- Smart target detection and problem-type inference (binary/multiclass vs regression)
- One-click training and comparison across multiple sklearn models
- Feature importance visualization and downloadable model package (model + preprocessors)
- CSV export of model comparison results
- Secure, server-generated HTML for data preview to minimize XSS attack surface

## 🏃 Quick Start

### macOS/Linux (zsh) — copy & paste

```bash
# 1) Clone the repository (replace with your repo URL)

git clone https://github.com/BoddapuLokesh/ML-Algorithm-Comparison.git
cd ML-Algorithm-Comparison

# 2) Create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate

# 3) Install dependencies
python -m pip install --upgrade pip
pip install -r requirements.txt

# 4) Run the application
python app.py
# App will start at http://127.0.0.1:5002/
```

If you already have the folder locally, start from step 2 inside the project directory.

### Windows (PowerShell)

```powershell
# 1) Clone the repository (replace with your repo URL)
git clone https://github.com/BoddapuLokesh/ML-Algorithm-Comparison.git
cd ML-Algorithm-Comparison

# 2) Create and activate a virtual environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# 3) Install dependencies
python -m pip install --upgrade pip
pip install -r requirements.txt

# 4) Run the application
python app.py
# App will start at http://127.0.0.1:5002/
```

## 📦 Installation

Prerequisites:
- Python 3.8+
- pip

Steps:
1) Clone the repository
2) Create and activate a virtual environment
3) Install dependencies: `pip install -r requirements.txt`
4) Run the app: `python app.py`

Optional verification:
```bash
python -c "import flask, pandas, sklearn; print('Deps OK')"
```

## 💡 Usage

1) Upload a CSV/XLSX/XLS (max 50MB) to see a safe, server-generated preview
2) Click Analyze to compute EDA (stats, missingness, correlations)
3) Choose the target and confirm problem type (auto-detected; you can override)
4) Start training to compare models; the best model is selected automatically
5) Review metrics and feature importance; export CSV or download the model package

## 🏗️ Project Structure

```
ML-Algorithm-Comparison/
├── app.py                # Flask routes: upload, EDA, training, exports
├── app_helpers.py        # JSON envelope, guards, upload/EDA/train handlers
├── model_utils.py        # Back-compat facade to ml_utils/*
├── ml_utils/
│   ├── config.py         # MLConfig, typed results, preview HTML
│   ├── eda.py            # Minimal+enhanced EDA
│   ├── models.py         # AutoMLComparer (fit, score, select, importance)
│   ├── preprocessing.py  # ColumnTransformer pipelines + fallback
│   └── utils.py          # JSON safety, validations, detection
├── templates/            # Jinja templates (layout/index)
├── static/               # style.css, app.js
└── requirements.txt      # Python dependencies
```

## 📚 API

File upload and EDA
- POST `/` — Upload dataset (AJAX)
- POST `/process_eda` — Run EDA and return stats/auto target/type
- GET `/eda` — EDA JSON (server-side cached)
- GET `/data_preview` — Preview JSON
- GET `/get_data_preview_html` — Secure HTML preview

Training and results
- POST `/validate_training_config` — Validate target/type/split
- POST `/train` — Train, compare, and return metrics/results/importance
- GET `/metrics` — Best model metrics
- GET `/best_model` — Best model name + metrics
- GET `/model_comparison` — All trained models and metrics
- GET `/feature_importance` — Feature importance data

Utilities
- POST `/analyze_target` — Inspect a chosen target column
- POST `/calculate_split` — Convert split ratio to percentages
- GET `/download_model` — ZIP: model.joblib + preprocessors.joblib + README
- GET `/export_results` — CSV export of all models
- GET `/debug_session` — Inspect session keys (debug)
- POST `/reset_session` — Clear session/caches (debug)

Notes
- JSON shapes vary by endpoint; on errors you’ll receive `{ "success": false, "error": "..." }`.

## 🤖 Supported Algorithms

Classification: Logistic Regression, Random Forest, Gradient Boosting, SVC, Decision Tree

Regression: Linear Regression, Random Forest, Gradient Boosting, SVR, Decision Tree

Metrics
- Classification: Accuracy, Precision, Recall, F1, Training Time
- Regression: R², MSE, Training Time

## ⚙️ Configuration

Environment (optional)
```bash
export FLASK_ENV=development
export FLASK_DEBUG=1
```

Runtime settings
- Max upload size: 50MB
- Session lifetime: 1 hour
- Model timeout (default): 300s per model (see `MLConfig`)

Customization
- Adjust preprocessing or defaults in `ml_utils/*.py`

## 🔧 Troubleshooting

- `.xls` files require `xlrd==1.2.0` (installed via requirements.txt)
- Very large/wide datasets: correlations are capped to reduce memory use
- If a model hits the time budget, it’s skipped; consider sampling or simpler models

## 🤝 Contributing

Small PRs are welcome. Please open an issue first if the change is substantial.

## 📄 License

MIT

## 📝 Changelog

August 2025
- Server-side preview HTML and consolidated Python validations
- AutoMLComparer pipelines and improved EDA (memory usage, quality score)
- Model export (model + preprocessors) and CSV results export
