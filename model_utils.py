"""
Optimized model_utils.py using minimalistic modular approach.
Reduces the original 616 lines to ~150 lines while preserving all functionality.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple

# Import from our optimized modules
from ml_utils import (
    MLConfig,
    AutoMLComparer,
    perform_eda_minimal,
    get_enhanced_eda_stats_minimal,
    detect_target_and_problem_type,
    validate_file_upload_minimal,
    validate_training_config_minimal,
    generate_data_preview_html,
    calculate_split_percentages,
    apply_preprocessing_minimal,
    safe_json_convert
)

# ============================================================================
# MAIN API FUNCTIONS (Backward compatibility with original model_utils.py)
# ============================================================================

def allowed_file(filename: str) -> bool:
    """Check if file extension is allowed."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'csv', 'xlsx', 'xls'}


def perform_eda(df: pd.DataFrame) -> Dict[str, Any]:
    """Perform EDA using optimized minimal approach."""
    return perform_eda_minimal(df)


def detect_target_and_type(df: pd.DataFrame) -> Tuple[str, str]:
    """Detect target column and problem type."""
    return detect_target_and_problem_type(df)


def preprocess_data(df: pd.DataFrame, target: str) -> Tuple[pd.DataFrame, pd.Series, Dict[str, Any]]:
    """Preprocess data using sklearn pipelines."""
    from ml_utils.preprocessing import preprocess_data_minimal
    return preprocess_data_minimal(df, target)


def train_models(df: pd.DataFrame, target: str, problem_type: str, split_ratio: float) -> Tuple[Dict, str, Dict, Dict, list, Dict]:
    """Train multiple models using the optimized AutoML comparer.
    
    Returns same format as original for backward compatibility.
    """
    # Create configuration
    config = MLConfig(
        target_column=target,
        problem_type=problem_type,
        test_size=1-split_ratio
    )
    
    # Initialize and run AutoML comparison
    automl = AutoMLComparer()
    results = automl.fit_compare(df, config)
    
    # Return in original format for backward compatibility
    return (
        results['models'],
        results['best_model_name'], 
        results['metrics'],
        results['feature_importance'],
        results['all_results'],
        results['preprocessing_artifacts']
    )


def validate_file_upload(file) -> Tuple[bool, str]:
    """Validate file upload."""
    return validate_file_upload_minimal(file)


def analyze_target_column(df: pd.DataFrame, target_column: str) -> Dict[str, Any]:
    """Enhanced target analysis for compatibility."""
    if target_column not in df.columns:
        return {
            'success': False,
            'error': f"Column '{target_column}' not found in dataset"
        }
        
    target_values = df[target_column].dropna()
    unique_values = target_values.nunique()
    total_values = len(df[target_column])
    missing_count = df[target_column].isnull().sum()
    
    # Auto-detection logic
    if pd.api.types.is_numeric_dtype(df[target_column]):
        if unique_values <= 10 and all(target_values.dropna().apply(lambda x: float(x).is_integer())):
            detected_type = 'classification'
        else:
            detected_type = 'regression'
    else:
        detected_type = 'classification'
    
    return {
        'success': True,
        'column_name': target_column,
        'unique_count': safe_json_convert(unique_values),
        'missing_count': safe_json_convert(missing_count),
        'total_rows': safe_json_convert(total_values),
        'data_type': str(df[target_column].dtype),
        'sample_values': [str(v) for v in target_values.head(10).tolist()],
        'missing_percentage': round((missing_count / total_values * 100), 2) if total_values > 0 else 0,
        'detected_type': detected_type,
        'type_confidence': 'high' if unique_values <= 10 else 'medium'
    }


def apply_preprocessing(df: pd.DataFrame, preprocessing_artifacts: Dict[str, Any]) -> pd.DataFrame:
    """Apply preprocessing artifacts to new data."""
    return apply_preprocessing_minimal(df, preprocessing_artifacts)


def validate_training_config(df: pd.DataFrame, target: str, problem_type: str, split_ratio: float) -> Dict[str, Any]:
    """Validate training configuration."""
    return validate_training_config_minimal(df, target, problem_type, split_ratio)


def get_enhanced_eda_stats(df: pd.DataFrame) -> Dict[str, Any]:
    """Get enhanced EDA statistics."""
    return get_enhanced_eda_stats_minimal(df)


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

# Export utility functions that were in original model_utils.py
__all__ = [
    'allowed_file',
    'safe_json_convert', 
    'perform_eda',
    'detect_target_and_type',
    'preprocess_data',
    'train_models',
    'validate_file_upload',
    'analyze_target_column', 
    'apply_preprocessing',
    'generate_data_preview_html',
    'calculate_split_percentages',
    'validate_training_config',
    'get_enhanced_eda_stats'
]
