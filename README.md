# ML Algorithm Comparison and Insight Tool

A Flask web application for automated machine learning model comparison and analysis.

## Features

- **Automated EDA**: Upload a dataset and get comprehensive exploratory data analysis
- **Smart Target Detection**: Automatically identifies target variables and problem types (classification/regression)
- **Multi-Model Training**: Trains and compares multiple ML algorithms:
  - Logistic Regression / Linear Regression
  - Random Forest
  - Gradient Boosting
  - Support Vector Machine
  - Decision Tree
- **Performance Metrics**: Comprehensive evaluation metrics for each model
- **Interactive Visualizations**: Plotly-powered charts and graphs
- **Model Export**: Download the best performing model

## Installation

1. Clone the repository:
```bash
git clone https://github.com/BoddapuLokesh/ML-Algorithm-Comparison-and-Insight-Tool.git
cd ML-Algorithm-Comparison-and-Insight-Tool
```

2. Create a virtual environment:

For Mac or Linux:
```bash
python -m venv venv
source venv/bin/activate  
```
For Windows:
```
python -m venv venv
venv\Scripts\activate
```

3. Install dependencies:
```bash
pip3 install -r requirements.txt ##For Mac or Linux
```

```
pip install -r requirements.txt ##For Windows
```

## Usage

1. Start the Flask application:
```bash
python3 app.py ##For Mac or Linux
```
```
python app.py ##For Windows
```

2. Open your browser and navigate to `http://localhost:5000`

3. Upload a CSV, Excel, or XLS file containing your dataset

4. Follow the guided workflow:
   - Review EDA results
   - Confirm target variable and problem type
   - Train models and compare performance
   - Download the best model

## Supported File Formats

- CSV (.csv)
- Excel (.xlsx, .xls)

## Project Structure

```
├── app.py              # Main Flask application
├── model_utils.py      # ML utilities and model training functions
├── requirements.txt    # Python dependencies
├── static/            # CSS, JavaScript, and static assets
├── templates/         # HTML templates
└── .gitignore        # Git ignore rules
```

## Requirements

- Python 3.8+
- Flask 3.0+
- scikit-learn 1.3+
- pandas 2.1+
- plotly 5.17+

## License

This project is open source and available under the [MIT License](LICENSE).
