# ML Algorithm Comparison and Insight Tool

A Flask web app that performs automated EDA, compares multiple ML algorithms, and explains results with interactive charts.

## Key features

- Upload CSV, XLSX, or XLS and get instant EDA with data quality, missingness, and type breakdowns
- Smart target detection and problem-type inference (classification vs regression)
- Train and compare multiple models out of the box:
  - Classification: Logistic Regression, Random Forest, Gradient Boosting, SVC, Decision Tree
  - Regression: Linear Regression, Random Forest, Gradient Boosting, SVR, Decision Tree
- Clear metrics per task (Accuracy/Precision/Recall/F1 for classification, R²/MSE for regression)
- Best model selection and downloadable serialized model (.joblib)
- Secure, server-generated data preview and backend validations
- Interactive visualizations powered by Chart.js on the frontend

## What changed recently (merged from enhancement notes)

- Moved validation, target analysis, and preview generation from JavaScript to Python for consistency and security
- Added endpoints: `/analyze_target`, `/get_data_preview_html`, `/calculate_split`, `/validate_training_config`
- Enhanced `/process_eda` and `/train` with better errors, EDA stats, and configuration checks
- Hardened XSS protection via server-side HTML escaping for previews

## Tech stack

- Backend: Flask, pandas, scikit-learn, joblib
- Frontend: HTML/CSS, Chart.js (CDN)
- File support: pandas + openpyxl (xlsx) + xlrd (xls)

## Project structure

```
├── app.py              # Flask app and API endpoints
├── model_utils.py      # EDA, preprocessing, training, validations
├── templates/          # Jinja templates (layout + index)
├── static/             # CSS and JS (Chart.js via CDN)
├── requirements.txt    # Python dependencies
└── ENHANCEMENT_SUMMARY.md  # Historical notes (merged into this README)
```

## Setup

1) Clone and enter the folder
- git clone <repository-url>
- cd ML-Algorithm-Comparison-and-Insight-Tool

2) Create and activate a virtual environment (macOS/Linux)
- python -m venv .venv
- source .venv/bin/activate

3) Install dependencies
- pip install -r requirements.txt

## Run

- python app.py
- Open http://127.0.0.1:5002 in your browser

Notes
- Default max upload is 50MB; supported formats: .csv, .xlsx, .xls
- The app generates a random secret key on each run, so sessions reset on restart

## Usage workflow

1) Upload a dataset and you will see a preview with all columns and some rows.
2) Click Analyze Dataset to run EDA and get dataset stats and charts.
3) Pick target column for prediction.
4) The app suggests problem type automatically, which can be overriden by user if needed.
5) Validate and start training.
6) The application compare models and metrics and results in comparision of all models and feature importance of attributes.
7) Export results CSV or download the best model (.joblib).

## API endpoints (selected)

- POST `/`                          — Upload file (AJAX)
- POST `/process_eda`               — Run EDA and return stats
- POST `/train`                     — Train selected task and return metrics
- GET  `/data_preview`              — JSON preview data
- GET  `/get_data_preview_html`     — Secure HTML preview
- POST `/analyze_target`            — Analyze a target column
- POST `/validate_training_config`  — Validate config before training
- POST `/calculate_split`           — Convert ratio to train/test percentages
- GET  `/best_model`, `/metrics`, `/feature_importance`, `/model_comparison`

## Troubleshooting

- Reading .xls requires xlrd. If you only need .xlsx, you can drop xlrd.
- If a model times out on very large datasets, the app will skip it. Consider sampling or using linear kernels.

## License

MIT
