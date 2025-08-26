"""Utility functions for data processing and validation."""

import numpy as np
import pandas as pd
from typing import Union, Dict, Any


def safe_json_convert(obj) -> Union[int, float, list, Dict[str, Any], str, None]:
    """Convert numpy types and NaN values to JSON-serializable types.
    
    Improved version that doesn't mask missing data with zeros.
    """
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        if np.isnan(obj):
            return None  # Don't mask missing data
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif hasattr(obj, 'item'):  # numpy scalars
        return obj.item()
    elif hasattr(obj, 'tolist'):  # numpy arrays/series
        return obj.tolist()
    elif hasattr(obj, '__iter__') and not isinstance(obj, (str, dict)):
        # Handle iterables like pandas Series, lists
        try:
            return [safe_json_convert(item) for item in obj]
        except:
            pass
    
    # Check for NaN after other type checks to avoid array ambiguity
    try:
        if pd.isna(obj):
            return None  # Don't mask pandas NaN
    except (ValueError, TypeError):
        # pd.isna might fail on some types
        pass
    
    return obj


def validate_file_upload_minimal(file) -> tuple[bool, str]:
    """Enhanced file validation with minimal code."""
    if not file or not file.filename:
        return False, "No file selected"
    
    # File type validation
    allowed_extensions = {'csv', 'xlsx', 'xls'}
    if not ('.' in file.filename and 
            file.filename.rsplit('.', 1)[1].lower() in allowed_extensions):
        return False, "Invalid file type. Only CSV, XLSX, and XLS files are supported."
    
    return True, "File validation passed"


def detect_target_and_problem_type(df: pd.DataFrame) -> tuple[str, str]:
    """Smart target detection and problem type inference.
    
    Combines the functionality of detect_target_and_type and analyze_target_column.
    """
    # Remove completely empty columns first
    df_clean = df.dropna(axis=1, how='all')
    
    # Heuristic: look for column names (common target column names)
    target_patterns = ["target", "label", "class", "y", "output", "prediction", "outcome", "diagnosis"]
    candidates = [c for c in df_clean.columns if c.lower() in target_patterns]
    
    if not candidates:
        # Look for binary/categorical columns that might be targets
        potential_targets = []
        for col in df_clean.columns:
            if df_clean[col].dtype == 'object' or df_clean[col].nunique() <= 10:
                potential_targets.append(col)
        
        if potential_targets:
            candidates = [potential_targets[0]]  # Use first potential target
        else:
            # fallback: last non-empty column
            candidates = [df_clean.columns[-1]]
    
    target = candidates[0]
    
    # Ensure target column has some non-null values
    if df[target].isna().all():
        # If target is all null, find first column with actual values
        for col in df_clean.columns:
            if not df[col].isna().all():
                target = col
                break
    
    # Problem type detection
    unique_values = df[target].nunique(dropna=True)
    dtype = df[target].dtype
    
    if pd.api.types.is_numeric_dtype(dtype):
        if unique_values == 2:
            problem_type = "binary"
        elif 2 < unique_values < 20:
            problem_type = "multiclass"
        else:
            problem_type = "regression"
    else:
        if unique_values == 2:
            problem_type = "binary"
        else:
            problem_type = "multiclass"
    
    print(f"Detected target: {target}, type: {problem_type}, unique values: {unique_values}")
    return target, problem_type


def validate_training_config_minimal(df: pd.DataFrame, target: str, problem_type: str, split_ratio: float) -> Dict[str, Any]:
    """Validate training configuration with comprehensive checks."""
    errors = []
    
    # Validate target column
    if target not in df.columns:
        errors.append(f"Target column '{target}' not found in dataset")
    elif df[target].isnull().all():
        errors.append(f"Target column '{target}' contains no valid values")
    
    # Validate problem type
    valid_types = ['classification', 'regression', 'binary', 'multiclass']
    if problem_type not in valid_types:
        errors.append(f"Invalid problem type '{problem_type}'. Must be one of {valid_types}")
    
    # Validate split ratio
    try:
        ratio = float(split_ratio)
        if ratio <= 0 or ratio >= 1:
            errors.append("Split ratio must be between 0 and 1")
    except (ValueError, TypeError):
        errors.append("Split ratio must be a valid number")
    
    # Check minimum data requirements
    if len(df) < 10:
        errors.append("Dataset too small. Need at least 10 rows for training")
    
    # Check target values for classification
    if target in df.columns:
        unique_target_values = df[target].nunique()
        if problem_type in ['classification', 'binary', 'multiclass'] and unique_target_values < 2:
            errors.append(f"Classification requires at least 2 unique target values, found {unique_target_values}")
    
    return {
        'valid': len(errors) == 0,
        'errors': errors
    }
