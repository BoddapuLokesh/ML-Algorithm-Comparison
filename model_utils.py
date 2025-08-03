import pandas as pd
import numpy as np
import time
import json
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.metrics import accuracy_score, mean_squared_error, precision_score, recall_score, f1_score, r2_score

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'csv', 'xlsx', 'xls'}

def safe_json_convert(obj):
    """Convert numpy types and NaN values to JSON-serializable types."""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        if np.isnan(obj):
            return 0  # Replace NaN with 0
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif pd.isna(obj):
        return 0  # Replace pandas NaN with 0
    return obj

def perform_eda(df):
    # Basic stats
    n_rows, n_cols = df.shape
    cols = list(df.columns)
    dtypes = {k: str(v) for k, v in df.dtypes.astype(str).to_dict().items()}
    missing = {k: safe_json_convert(v) for k, v in df.isnull().sum().to_dict().items()}
    duplicate_rows = safe_json_convert(df.duplicated().sum())
    numeric_cols = list(df.select_dtypes(include=['number']).columns)
    categorical_cols = list(df.select_dtypes(include=['object', 'category']).columns)

    # Calculate more detailed statistics with safe JSON conversion
    numeric_stats = {}
    for col in numeric_cols:
        if not df[col].isnull().all():
            numeric_stats[col] = {
                'mean': safe_json_convert(df[col].mean()),
                'std': safe_json_convert(df[col].std()),
                'min': safe_json_convert(df[col].min()),
                'max': safe_json_convert(df[col].max()),
                'median': safe_json_convert(df[col].median())
            }
        else:
            numeric_stats[col] = {
                'mean': 0, 'std': 0, 'min': 0, 'max': 0, 'median': 0
            }
    
    categorical_stats = {}
    for col in categorical_cols:
        if not df[col].isnull().all():
            value_counts = df[col].value_counts()
            categorical_stats[col] = {
                'unique_count': safe_json_convert(df[col].nunique()),
                'most_frequent': str(value_counts.index[0]) if len(value_counts) > 0 else 'N/A',
                'most_frequent_count': safe_json_convert(value_counts.iloc[0]) if len(value_counts) > 0 else 0,
                'top_values': {str(k): safe_json_convert(v) for k, v in zip(value_counts.head(5).index, value_counts.head(5).values)}
            }
        else:
            categorical_stats[col] = {
                'unique_count': 0,
                'most_frequent': 'N/A',
                'most_frequent_count': 0,
                'top_values': {}
            }

    # Store plot data for visualizations
    plot_data = []
    
    # Histogram data for numeric columns (top 3)
    for col in numeric_cols[:3]:
        if not df[col].isnull().all():
            hist_data = [safe_json_convert(x) for x in df[col].dropna().tolist()[:100]]
            plot_data.append({
                "type": "histogram", 
                "column": col, 
                "data": hist_data,
                "title": f"Distribution of {col}"
            })
    
    # Bar chart data for categorical columns (top 3)
    for col in categorical_cols[:3]:
        if not df[col].isnull().all():
            counts = df[col].value_counts().head(10)
            plot_data.append({
                "type": "bar", 
                "column": col, 
                "labels": [str(x) for x in counts.index], 
                "values": [safe_json_convert(x) for x in counts.values],
                "title": f"Top Categories in {col}"
            })

    # Correlation data for numeric columns
    correlation_data = {}
    if len(numeric_cols) >= 2:
        try:
            corr_matrix = df[numeric_cols].corr()
            # Convert to format suitable for heatmap with safe JSON conversion
            correlation_data = {
                'columns': numeric_cols,
                'values': [[safe_json_convert(x) for x in row] for row in corr_matrix.values.tolist()]
            }
        except Exception as e:
            print(f"Error calculating correlation: {e}")
            correlation_data = {}

    # Safe JSON conversion for missing percentage
    missing_percentage = {}
    for k, v in missing.items():
        if n_rows > 0:
            missing_percentage[k] = round((v/n_rows)*100, 2)
        else:
            missing_percentage[k] = 0

    stats = {
        "rows": safe_json_convert(n_rows), 
        "cols": safe_json_convert(n_cols), 
        "columns": cols,
        "dtypes": dtypes, 
        "missing": missing,
        "duplicate_rows": duplicate_rows, 
        "numerics": numeric_cols, 
        "categoricals": categorical_cols,
        "numeric_stats": numeric_stats,
        "categorical_stats": categorical_stats,
        "missing_percentage": missing_percentage,
        "total_missing_values": safe_json_convert(sum(missing.values())),
        "data_completeness": round(((n_rows * n_cols - sum(missing.values())) / (n_rows * n_cols)) * 100, 2) if n_rows * n_cols > 0 else 100
    }

    return {
        "stats": stats, 
        "plot_data": plot_data,
        "correlation_data": correlation_data
    }

def detect_target_and_type(df):
    # Remove completely empty columns first
    df_clean = df.dropna(axis=1, how='all')
    
    # Heuristic: look for column names (common target column names)
    candidates = [c for c in df_clean.columns if c.lower() in ["target", "label", "class", "y", "output", "prediction", "outcome", "diagnosis"]]
    
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
    
    unique_values = df[target].nunique(dropna=True)  # Don't count NaN as unique value
    dtype = df[target].dtype
    
    if pd.api.types.is_numeric_dtype(dtype):
        if unique_values == 2:
            ptype = "binary"
        elif 2 < unique_values < 20:
            ptype = "multiclass"
        else:
            ptype = "regression"
    else:
        if unique_values == 2:
            ptype = "binary"
        else:
            ptype = "multiclass"
    
    print(f"Detected target: {target}, type: {ptype}, unique values: {unique_values}")
    return target, ptype

def preprocess_data(df, target):
    # Remove target column to get features
    X = df.drop(columns=[target])
    y = df[target]
    
    # Remove completely empty columns (all NaN/null)
    X = X.dropna(axis=1, how='all')
    print(f"After removing empty columns, X shape: {X.shape}")
    
    # Encode categorical columns
    X = X.copy()
    categorical_columns = []
    for col in X.select_dtypes(include='object'):
        if X[col].notna().any():  # Only encode if column has non-null values
            X[col] = LabelEncoder().fit_transform(X[col].astype(str))
            categorical_columns.append(col)
    
    # Encode target if it's categorical
    if y.dtype == 'object':
        y = LabelEncoder().fit_transform(y.astype(str))
    
    # Handle missing values with SimpleImputer
    if X.select_dtypes(include='number').shape[1] > 0:  # If there are numeric columns
        # Separate numeric and categorical columns for different imputation strategies
        numeric_cols = X.select_dtypes(include='number').columns
        categorical_cols = X.select_dtypes(exclude='number').columns
        
        if len(numeric_cols) > 0:
            # Impute numeric columns with mean
            X_numeric = SimpleImputer(strategy='mean').fit_transform(X[numeric_cols])
            X_numeric_df = pd.DataFrame(X_numeric, columns=numeric_cols, index=X.index)
        else:
            X_numeric_df = pd.DataFrame(index=X.index)
            
        if len(categorical_cols) > 0:
            # Impute categorical columns with most frequent value
            X_categorical = SimpleImputer(strategy='most_frequent').fit_transform(X[categorical_cols])
            X_categorical_df = pd.DataFrame(X_categorical, columns=categorical_cols, index=X.index)
        else:
            X_categorical_df = pd.DataFrame(index=X.index)
        
        # Combine numeric and categorical columns
        if not X_numeric_df.empty and not X_categorical_df.empty:
            X = pd.concat([X_numeric_df, X_categorical_df], axis=1)
        elif not X_numeric_df.empty:
            X = X_numeric_df
        elif not X_categorical_df.empty:
            X = X_categorical_df
        else:
            raise ValueError("No valid columns found after preprocessing")
    else:
        # If no numeric columns, use most_frequent strategy for all
        X = pd.DataFrame(SimpleImputer(strategy='most_frequent').fit_transform(X), 
                        columns=X.columns, index=X.index)
    
    print(f"Final preprocessed X shape: {X.shape}")
    print("Preprocessing completed successfully")
    
    return X, y

def train_models(df, target, problem_type, split_ratio):
    try:
        # Validate input data
        print(f"Starting training with target: {target}, problem_type: {problem_type}")
        if target not in df.columns:
            raise ValueError(f"Target column '{target}' not found in dataset")
        
        if df[target].isna().all():
            raise ValueError(f"Target column '{target}' has no valid values")
        
        X, y = preprocess_data(df, target)
        
        # Validate preprocessed data
        if X.empty:
            raise ValueError("No features available after preprocessing")
        
        # Check unique values in target for validation
        try:
            unique_y_count = len(set(y))  # type: ignore
        except (TypeError, ValueError):
            # If y is not hashable or iterable, try converting
            try:
                if hasattr(y, 'tolist'):
                    unique_y_count = len(set(y.tolist()))  # type: ignore
                else:
                    unique_y_count = len(set(list(y)))  # type: ignore
            except:
                unique_y_count = 2  # Assume it's valid if we can't check
        
        if unique_y_count < 2 and problem_type in ['binary', 'multiclass']:
            raise ValueError(f"Target has only {unique_y_count} unique values, cannot perform classification")
        
        print(f"Training data: X shape: {X.shape}, y unique values: {unique_y_count}")
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1-split_ratio, random_state=42)
        results = []
        models = {}

        # Select models based on problem type
        if problem_type == 'regression':
            model_configs = [
                ("LinearRegression", LinearRegression()),
                ("RandomForestRegressor", RandomForestRegressor(random_state=42, n_estimators=10)),
                ("GradientBoostingRegressor", GradientBoostingRegressor(random_state=42, n_estimators=10)),
                ("SVR", SVR()),
                ("DecisionTreeRegressor", DecisionTreeRegressor(random_state=42, max_depth=5))
            ]
        else:
            model_configs = [
                ("LogisticRegression", LogisticRegression(max_iter=1000, random_state=42)),
                ("RandomForestClassifier", RandomForestClassifier(random_state=42, n_estimators=10)),
                ("GradientBoostingClassifier", GradientBoostingClassifier(random_state=42, n_estimators=10)),
                ("SVC", SVC(random_state=42, probability=True)),
                ("DecisionTreeClassifier", DecisionTreeClassifier(random_state=42, max_depth=5))
            ]

        best_score = -np.inf
        best_model_name = None
        all_metrics = {}

        for name, model in model_configs:
            try:
                print(f"Training {name}...")
                
                # Track training time
                start_time = time.time()
                model.fit(X_train, y_train)
                training_time = time.time() - start_time
                
                preds = model.predict(X_test)
                
                if problem_type == 'regression':
                    score = r2_score(y_test, preds)  # Use R² as primary metric
                    metrics = {
                        "MSE": safe_json_convert(mean_squared_error(y_test, preds)),
                        "R^2": safe_json_convert(score),
                        "Training_Time": round(training_time, 3)
                    }
                else:
                    score = accuracy_score(y_test, preds)
                    metrics = {
                        "Accuracy": safe_json_convert(score),
                        "Precision": safe_json_convert(precision_score(y_test, preds, average='weighted', zero_division=0)),
                        "Recall": safe_json_convert(recall_score(y_test, preds, average='weighted', zero_division=0)),
                        "F1": safe_json_convert(f1_score(y_test, preds, average='weighted', zero_division=0)),
                        "Training_Time": round(training_time, 3)
                    }
                    
                results.append((name, model, score))
                models[name] = model
                all_metrics[name] = metrics
                print(f"{name} trained successfully with score: {score:.3f} (Time: {training_time:.3f}s)")

                if score > best_score:
                    best_score = score
                    best_model_name = name
                    
            except Exception as e:
                print(f"Error training {name}: {str(e)}")
                continue

        if not models:
            raise Exception("No models were successfully trained")

        # Feature importance (for tree/random forest) with safe JSON conversion
        feature_importance = {}
        for name, model in models.items():
            try:
                if hasattr(model, "feature_importances_"):
                    imp = model.feature_importances_
                    feature_importance[name] = {col: safe_json_convert(importance) for col, importance in zip(X.columns, imp)}
                elif hasattr(model, "coef_"):
                    vals = model.coef_
                    coefs = vals[0] if len(vals.shape) > 1 else vals
                    feature_importance[name] = {col: safe_json_convert(abs(coef)) for col, coef in zip(X.columns, coefs)}
                else:
                    feature_importance[name] = {}
            except Exception as e:
                print(f"Error extracting feature importance for {name}: {str(e)}")
                feature_importance[name] = {}

        # Prepare results for frontend
        all_results = [{"model": k, "metrics": v} for k, v in all_metrics.items()]
        metrics = all_metrics[best_model_name] if best_model_name else {}

        return models, best_model_name, metrics, feature_importance, all_results
        
    except Exception as e:
        print(f"Error in train_models: {str(e)}")
        raise
