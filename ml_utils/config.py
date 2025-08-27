"""Configuration and typed result structures for ML utilities."""

import html
from dataclasses import dataclass
from typing import Dict, List, Optional, Union, Any, TypedDict


@dataclass
class MLConfig:
    """Configuration object for ML model training and evaluation."""
    target_column: str
    problem_type: str
    test_size: float = 0.2
    random_state: int = 42
    timeout_seconds: int = 300


class ModelMetrics(TypedDict, total=False):
    """Type-safe structure for model metrics."""
    # Classification metrics
    Accuracy: float
    Precision: float
    Recall: float
    F1: float
    
    # Regression metrics
    MSE: float
    R2: float
    
    # Common metrics
    Training_Time: float


class EDAResults(TypedDict):
    """Type-safe structure for EDA results."""
    stats: Dict[str, Any]
    plot_data: List[Dict[str, Any]]
    correlation_data: Dict[str, Any]


class MLResults(TypedDict):
    """Type-safe structure for ML training results."""
    models: Dict[str, Any]
    best_model_name: str
    metrics: ModelMetrics
    feature_importance: Dict[str, Dict[str, float]]
    all_results: List[Dict[str, Any]]
    preprocessing_artifacts: Dict[str, Any]


def generate_data_preview_html(preview_data: List[Dict], columns: List[str]) -> str:
    """Generate HTML table server-side for better security."""
    if not preview_data or not columns:
        return '<div class="text-center text-gray-500">No data available for preview</div>'
    
    # Generate HTML server-side (safer than client-side JS)
    html_content = '<table class="data-table"><thead><tr>'
    
    # Add headers with proper escaping
    for col in columns:
        html_content += f'<th>{html.escape(str(col))}</th>'
    html_content += '</tr></thead><tbody>'
    
    # Add data rows with proper escaping
    for row in preview_data:
        html_content += '<tr>'
        for col in columns:
            value = row.get(col, '') if row else ''
            if value is None or str(value).lower() == 'nan':
                value = ''
            html_content += f'<td>{html.escape(str(value))}</td>'
        html_content += '</tr>'
    html_content += '</tbody></table>'
    
    return html_content


def calculate_split_percentages(split_ratio: float) -> tuple[int, int]:
    """Calculate train/test split percentages."""
    train_percent = int(split_ratio * 100)
    test_percent = 100 - train_percent
    return train_percent, test_percent


# ---------------------------------------------------------------------------
# Features implemented in this module
# - MLConfig dataclass for training parameters
# - Typed dictionaries for metrics, EDA results, and training results
# - HTML-safe data preview generator
# - Utility to convert split ratio into train/test percentages
# ---------------------------------------------------------------------------
