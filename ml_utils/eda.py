"""Exploratory Data Analysis helpers using pandas built-ins.

Produces compact stats, light-weight plot payloads, and capped correlation
matrices suitable for rendering in the front end.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any

from .utils import safe_json_convert


def perform_eda_minimal(df: pd.DataFrame) -> Dict[str, Any]:
    """Minimalistic EDA using pandas built-ins instead of manual calculations.
    
    Replaces the 130+ line perform_eda function with a clean 30-line version.
    """
    # Basic statistics using pandas built-ins
    basic_stats = df.describe(include='all').fillna(0)
    
    # Essential info
    info_stats = {
        "rows": df.shape[0],
        "cols": df.shape[1],
        "columns": list(df.columns),
        "dtypes": df.dtypes.astype(str).to_dict(),
        "missing": df.isnull().sum().to_dict(),
        "missing_percentage": (df.isnull().sum() / len(df) * 100).round(2).to_dict(),
        "duplicate_rows": df.duplicated().sum(),
        "numerics": list(df.select_dtypes(include=['number']).columns),
        "categoricals": list(df.select_dtypes(include=['object', 'category']).columns),
        "data_completeness": round((1 - df.isnull().sum().sum() / (df.shape[0] * df.shape[1])) * 100, 2)
    }
    
    # Convert to JSON-safe format
    for key, value in info_stats.items():
        info_stats[key] = safe_json_convert(value)
    
    # Simplified plot data generation
    plot_data = _generate_plot_data_minimal(df)
    
    # Correlation data using pandas built-in
    correlation_data = _generate_correlation_data_minimal(df)
    
    return {
        "stats": info_stats,
        "plot_data": plot_data,
        "correlation_data": correlation_data
    }


def _generate_plot_data_minimal(df: pd.DataFrame) -> List[Dict[str, Any]]:
    """Generate plot data with optimized sampling."""
    plot_data = []
    
    # Histogram data for numeric columns (top 3) - optimized sampling
    numeric_cols = df.select_dtypes(include=['number']).columns[:3]
    for col in numeric_cols:
        if not df[col].isnull().all():
            # Optimized: sample first, then convert
            available_data = df[col].dropna()
            sample_size = min(100, len(available_data))
            if sample_size > 0:
                hist_data = available_data.sample(sample_size, random_state=42).tolist()
                plot_data.append({
                    "type": "histogram",
                    "column": col,
                    "data": [safe_json_convert(x) for x in hist_data],
                    "title": f"Distribution of {col}"
                })
    
    # Bar chart data for categorical columns (top 3) - optimized counting
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns[:3]
    for col in categorical_cols:
        if not df[col].isnull().all():
            # Optimized: use value_counts with limit
            counts = df[col].value_counts().head(10)
            plot_data.append({
                "type": "bar",
                "column": col,
                "labels": [str(x) for x in counts.index],
                "values": [safe_json_convert(x) for x in counts.values],
                "title": f"Top Categories in {col}"
            })
    
    return plot_data


def _generate_correlation_data_minimal(df: pd.DataFrame) -> Dict[str, Any]:
    """Generate correlation data using pandas built-in corr()."""
    numeric_cols = df.select_dtypes(include=['number']).columns
    
    if len(numeric_cols) < 2:
        return {}
    
    try:
        # Optimize: limit columns to prevent memory issues
        capped_cols = list(numeric_cols[:25])
        corr_matrix = df[capped_cols].corr()
        
        correlation_data = {
            'columns': capped_cols,
            'values': [[safe_json_convert(x) for x in row] for row in corr_matrix.values.tolist()]
        }
        
        if len(numeric_cols) > 25:
            correlation_data['truncated'] = True
            
        return correlation_data
        
    except Exception as e:
        print(f"Error calculating correlation: {e}")
        return {}


def get_enhanced_eda_stats_minimal(df: pd.DataFrame) -> Dict[str, Any]:
    """Enhanced EDA stats with minimal additional computation.
    
    Replaces get_enhanced_eda_stats by extending the basic EDA results.
    """
    basic_eda = perform_eda_minimal(df)
    enhanced_stats = basic_eda['stats'].copy()
    
    # Add memory usage efficiently
    try:
        memory_usage = df.memory_usage(deep=True).sum()
        enhanced_stats['memory_usage_mb'] = round(memory_usage / (1024 * 1024), 2)
    except Exception:
        enhanced_stats['memory_usage_mb'] = 0
    
    # Column type distribution (already computed)
    enhanced_stats['column_type_distribution'] = {
        'numeric': len(enhanced_stats['numerics']),
        'categorical': len(enhanced_stats['categoricals']),
        'total': enhanced_stats['cols']
    }
    
    # Data quality score
    total_cells = enhanced_stats['rows'] * enhanced_stats['cols']
    missing_cells = sum(enhanced_stats['missing'].values())
    enhanced_stats['data_quality_score'] = round((1 - missing_cells / total_cells) * 100, 2) if total_cells > 0 else 100
    
    return {
        'stats': enhanced_stats,
        'plot_data': basic_eda['plot_data'],
        'correlation_data': basic_eda['correlation_data']
    }


# ---------------------------------------------------------------------------
# Features implemented in this module
# - perform_eda_minimal: summary stats, dtypes, missingness, duplicates
# - plot data generation: sampled histograms and top-k category bars
# - correlation matrix with column cap and truncation flag
# - enhanced stats: memory usage, type distribution, quality score
# ---------------------------------------------------------------------------
