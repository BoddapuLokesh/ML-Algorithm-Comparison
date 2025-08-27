"""
Minimalistic ML utilities for algorithm comparison.
Modernized with sklearn pipelines, type hints, and clean architecture.
"""

from .config import MLConfig, ModelMetrics, generate_data_preview_html, calculate_split_percentages
from .preprocessing import create_preprocessing_pipeline, apply_preprocessing_minimal
from .eda import perform_eda_minimal, get_enhanced_eda_stats_minimal
from .models import AutoMLComparer
from .utils import safe_json_convert, validate_file_upload_minimal, detect_target_and_problem_type, validate_training_config_minimal

__all__ = [
    'MLConfig',
    'ModelMetrics', 
    'create_preprocessing_pipeline',
    'apply_preprocessing_minimal',
    'perform_eda_minimal',
    'get_enhanced_eda_stats_minimal',
    'AutoMLComparer',
    'safe_json_convert',
    'validate_file_upload_minimal',
    'detect_target_and_problem_type',
    'validate_training_config_minimal',
    'generate_data_preview_html',
    'calculate_split_percentages'
]
