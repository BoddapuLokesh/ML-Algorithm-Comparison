"""Preprocessing pipelines for tabular data.

Provides ColumnTransformer-based pipelines for numeric/categorical features,
fallback encoders, and helpers to apply saved preprocessors to new data.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from scipy import sparse
from scipy.sparse import spmatrix

from .utils import safe_json_convert


def create_preprocessing_pipeline(X: pd.DataFrame) -> ColumnTransformer:
    """Create a comprehensive preprocessing pipeline using sklearn.
    
    Replaces the manual preprocessing logic with sklearn's optimized implementations.
    """
    numeric_features = X.select_dtypes(include=['number']).columns.tolist()
    categorical_features = X.select_dtypes(include=['object']).columns.tolist()
    
    # Define transformers for different column types
    transformers = []
    
    if numeric_features:
        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='mean')),
            ('scaler', StandardScaler())  # Will be conditional based on model
        ])
        transformers.append(('num', numeric_transformer, numeric_features))
    
    if categorical_features:
        # Use OneHotEncoder with max_categories to prevent feature explosion
        from sklearn.preprocessing import OneHotEncoder
        try:
            # Try with max_categories (sklearn >= 1.2)
            categorical_transformer = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('encoder', OneHotEncoder(
                    drop='first', 
                    sparse_output=False, 
                    handle_unknown='ignore',
                    max_categories=10  # Limit categories per feature to prevent explosion
                ))
            ])
        except TypeError:
            # Fallback for older sklearn versions
            categorical_transformer = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('encoder', OneHotEncoder(
                    drop='first', 
                    sparse_output=False, 
                    handle_unknown='ignore'
                ))
            ])
        transformers.append(('cat', categorical_transformer, categorical_features))
    
    if not transformers:
        raise ValueError("No valid columns found for preprocessing")
    
    preprocessor = ColumnTransformer(
        transformers=transformers,
        remainder='drop'  # Drop columns not specified
    )
    
    return preprocessor


def preprocess_data_minimal(df: pd.DataFrame, target: str) -> Tuple[pd.DataFrame, pd.Series, Dict[str, Any]]:
    """Minimal preprocessing using sklearn pipelines.
    
    Replaces the complex manual preprocessing with clean sklearn implementation.
    """
    # Separate features and target
    X = df.drop(columns=[target])
    y = df[target]
    
    # Remove completely empty columns
    X = X.dropna(axis=1, how='all')
    print(f"After removing empty columns, X shape: {X.shape}")
    
    # Check for high cardinality categorical features that might cause explosion
    categorical_features = X.select_dtypes(include=['object']).columns
    for col in categorical_features:
        unique_count = X[col].nunique()
        if unique_count > 50:  # Arbitrary threshold
            print(f"Warning: Column '{col}' has {unique_count} unique values, might cause feature explosion")
            # Convert high cardinality categorical to ordinal encoding
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))
    
    # Handle target encoding if categorical
    target_encoder = None
    if y.dtype == 'object':
        target_encoder = LabelEncoder()
        y_values = target_encoder.fit_transform(y.astype(str))
        y = pd.Series(data=np.array(y_values), index=y.index)
    
    # Create and apply preprocessing pipeline
    preprocessor = None
    try:
        preprocessor = create_preprocessing_pipeline(X)
        X_processed = preprocessor.fit_transform(X)
        
        # Get correct feature names from the fitted pipeline
        try:
            feature_names = list(preprocessor.get_feature_names_out())
        except AttributeError:
            # Fallback for older sklearn versions or custom transformers
            feature_names = [f"feature_{i}" for i in range(X_processed.shape[1])]
        
        # Convert sparse matrix to dense if needed
        def _to_dense(arr: Any) -> np.ndarray:
            try:
                # spmatrix supports .toarray() at runtime; guard with try
                return arr.toarray()  # type: ignore[attr-defined]
            except Exception:
                return np.asarray(arr)

        if sparse.issparse(X_processed):
            X_processed = _to_dense(X_processed)
        elif not isinstance(X_processed, np.ndarray):
            X_processed = np.asarray(X_processed)
        
        # Ensure it's a numpy array
        X_processed = np.asarray(X_processed)
        
        X_processed_df = pd.DataFrame(
            data=X_processed, 
            columns=feature_names,
            index=X.index
        )
        
    except Exception as e:
        print(f"Pipeline preprocessing failed, falling back to simple approach: {e}")
        # Fallback to simple preprocessing
        X_processed_df = _simple_preprocessing_fallback(X)
    
    print(f"Final preprocessed X shape: {X_processed_df.shape}")
    
    # Store preprocessing artifacts
    preprocessing_artifacts = {
        'preprocessor': preprocessor,
        'target_encoder': target_encoder,
        'feature_columns': list(X_processed_df.columns),
        'original_columns': list(X.columns)
    }
    
    return X_processed_df, y, preprocessing_artifacts


def _simple_preprocessing_fallback(X: pd.DataFrame) -> pd.DataFrame:
    """Simple fallback preprocessing when pipeline fails."""
    X_processed = X.copy()
    
    # Handle categorical columns
    for col in X_processed.select_dtypes(include='object'):
        if X_processed[col].notna().any():
            try:
                encoder = LabelEncoder()
                X_processed[col] = encoder.fit_transform(X_processed[col].astype(str))
            except Exception:
                # If encoding fails, drop the column
                X_processed = X_processed.drop(columns=[col])
    
    # Handle missing values
    if X_processed.select_dtypes(include='number').shape[1] > 0:
        numeric_cols = X_processed.select_dtypes(include='number').columns
        X_processed[numeric_cols] = X_processed[numeric_cols].fillna(X_processed[numeric_cols].mean())
    
    # Fill any remaining missing values
    X_processed = X_processed.fillna(0)
    
    return X_processed


def apply_preprocessing_minimal(df: pd.DataFrame, preprocessing_artifacts: Dict[str, Any]) -> pd.DataFrame:
    """Apply saved preprocessing pipeline to new data.
    
    Simplified version of apply_preprocessing using stored pipeline.
    """
    if 'preprocessor' in preprocessing_artifacts:
        try:
            # Use the stored preprocessor
            preprocessor = preprocessing_artifacts['preprocessor']
            X_processed = preprocessor.transform(df)
            
            # Convert back to DataFrame
            feature_names = preprocessing_artifacts.get('feature_columns', [])
            X_processed_df = pd.DataFrame(X_processed, columns=feature_names, index=df.index)
            
            return X_processed_df
            
        except Exception as e:
            print(f"Pipeline transform failed: {e}")
            return _simple_preprocessing_fallback(df)
    else:
        # Fallback to simple preprocessing
        return _simple_preprocessing_fallback(df)


# ---------------------------------------------------------------------------
# Features implemented in this module
# - Create sklearn ColumnTransformer with imputers, scalers, one-hot encoder
# - Encode high-cardinality categoricals defensively; label-encode fallback
# - Provide preprocess -> (X_processed, y, artifacts) and simple fallback
# - Re-apply stored preprocessing artifacts to incoming data
# ---------------------------------------------------------------------------
