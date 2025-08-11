import pandas as pd
import numpy as np
import time
import json
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
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
        
        # For very large datasets (>100k samples), provide option to sample for faster training
        if X_train.shape[0] > 100000:
            print(f"Large dataset detected ({X_train.shape[0]} samples). Using sampling for faster training.")
            sample_size = min(50000, X_train.shape[0])  # Sample up to 50k rows
            sample_indices = np.random.choice(X_train.shape[0], sample_size, replace=False)
            X_train_sampled = X_train.iloc[sample_indices] if hasattr(X_train, 'iloc') else X_train[sample_indices]
            y_train_sampled = y_train.iloc[sample_indices] if hasattr(y_train, 'iloc') else y_train[sample_indices]
            print(f"Training on sampled dataset: {X_train_sampled.shape[0]} samples")
        else:
            X_train_sampled = X_train
            y_train_sampled = y_train
            
        results = []
        models = {}

        # For large datasets (>50k samples), use optimized configurations
        is_large_dataset = X_train.shape[0] > 50000
        
        # Select models based on problem type
        if problem_type == 'regression':
            if is_large_dataset:
                # Use linear SVR for large datasets - much faster
                model_configs = [
                    ("LinearRegression", LinearRegression()),
                    ("RandomForestRegressor", RandomForestRegressor(random_state=42, n_estimators=10, max_depth=10)),
                    ("GradientBoostingRegressor", GradientBoostingRegressor(random_state=42, n_estimators=10, max_depth=3)),
                    ("LinearSVR", SVR(kernel='linear', C=1.0, max_iter=1000)),
                    ("DecisionTreeRegressor", DecisionTreeRegressor(random_state=42, max_depth=5))
                ]
            else:
                # Use RBF SVR for smaller datasets
                model_configs = [
                    ("LinearRegression", LinearRegression()),
                    ("RandomForestRegressor", RandomForestRegressor(random_state=42, n_estimators=10)),
                    ("GradientBoostingRegressor", GradientBoostingRegressor(random_state=42, n_estimators=10)),
                    ("SVR", SVR(kernel='rbf', C=1.0, gamma='scale')),
                    ("DecisionTreeRegressor", DecisionTreeRegressor(random_state=42, max_depth=5))
                ]
        else:
            if is_large_dataset:
                # Use linear SVC for large datasets - much faster
                model_configs = [
                    ("LogisticRegression", LogisticRegression(max_iter=1000, random_state=42)),
                    ("RandomForestClassifier", RandomForestClassifier(random_state=42, n_estimators=10, max_depth=10)),
                    ("GradientBoostingClassifier", GradientBoostingClassifier(random_state=42, n_estimators=10, max_depth=3)),
                    ("LinearSVC", SVC(kernel='linear', C=1.0, random_state=42, probability=True, max_iter=1000)),
                    ("DecisionTreeClassifier", DecisionTreeClassifier(random_state=42, max_depth=5))
                ]
            else:
                # Use RBF SVC for smaller datasets
                model_configs = [
                    ("LogisticRegression", LogisticRegression(max_iter=1000, random_state=42)),
                    ("RandomForestClassifier", RandomForestClassifier(random_state=42, n_estimators=10)),
                    ("GradientBoostingClassifier", GradientBoostingClassifier(random_state=42, n_estimators=10)),
                    ("SVC", SVC(kernel='rbf', C=1.0, gamma='scale', random_state=42, probability=True)),
                    ("DecisionTreeClassifier", DecisionTreeClassifier(random_state=42, max_depth=5))
                ]

        best_score = -np.inf
        best_model_name = None
        all_metrics = {}

        for name, model in model_configs:
            try:
                print(f"Training {name}...")
                
                # Scale features for SVR/SVC models (they are sensitive to feature scales)
                if 'SVR' in name or 'SVC' in name:
                    scaler = StandardScaler()
                    X_train_scaled = scaler.fit_transform(X_train_sampled)
                    X_test_scaled = scaler.transform(X_test)
                    X_train_model = X_train_scaled
                    X_test_model = X_test_scaled
                    y_train_model = y_train_sampled
                else:
                    X_train_model = X_train_sampled
                    X_test_model = X_test
                    y_train_model = y_train_sampled
                
                # Track training time with timeout
                start_time = time.time()
                
                # Set timeout based on dataset size
                timeout = 60 if is_large_dataset else 120  # 1-2 minutes max
                
                try:
                    model.fit(X_train_model, y_train_model)
                    training_time = time.time() - start_time
                    
                    # Skip if training took too long
                    if training_time > timeout:
                        print(f"{name} training timeout ({training_time:.1f}s > {timeout}s), skipping...")
                        continue
                        
                except Exception as fit_error:
                    print(f"Error fitting {name}: {str(fit_error)}")
                    continue
                
                preds = model.predict(X_test_model)
                
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

def validate_file_upload(file):
    """Enhanced file validation moved from JavaScript"""
    if not file or not file.filename:
        return False, "No file selected"
    
    # File type validation (moved from JS)
    allowed_extensions = {'csv', 'xlsx', 'xls'}
    if not ('.' in file.filename and 
            file.filename.rsplit('.', 1)[1].lower() in allowed_extensions):
        return False, "Invalid file type. Only CSV, XLSX, and XLS files are supported."
    
    return True, "File validation passed"

def analyze_target_column(df, target_column):
    """Enhanced target analysis that JavaScript currently does"""
    if target_column not in df.columns:
        return {
            'success': False,
            'error': f"Column '{target_column}' not found in dataset"
        }
        
    target_values = df[target_column].dropna()
    unique_values = target_values.nunique()
    total_values = len(df[target_column])
    missing_count = df[target_column].isnull().sum()
    
    # Enhanced logic from JavaScript
    analysis = {
        'success': True,
        'column_name': target_column,
        'unique_count': safe_json_convert(unique_values),
        'missing_count': safe_json_convert(missing_count),
        'total_rows': safe_json_convert(total_values),
        'data_type': str(df[target_column].dtype),
        'sample_values': [str(v) for v in target_values.head(10).tolist()],
        'missing_percentage': round((missing_count / total_values * 100), 2) if total_values > 0 else 0
    }
    
    # Auto-detection logic (moved from JavaScript)
    if pd.api.types.is_numeric_dtype(df[target_column]):
        # For numeric columns, check if it looks like classification or regression
        if unique_values <= 10 and all(target_values.dropna().apply(lambda x: float(x).is_integer())):
            detected_type = 'classification'
        else:
            detected_type = 'regression'
    else:
        # For non-numeric columns, always classification
        detected_type = 'classification'
    
    analysis['detected_type'] = detected_type
    analysis['type_confidence'] = 'high' if unique_values <= 10 else 'medium'
    
    return analysis

def generate_data_preview_html(preview_data, columns):
    """Generate HTML table server-side for better security"""
    import html  # Local import - only used in this function
    
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
            if pd.isna(value) or value is None:
                value = ''
            html_content += f'<td>{html.escape(str(value))}</td>'
        html_content += '</tr>'
    html_content += '</tbody></table>'
    
    return html_content

def calculate_split_percentages(split_ratio):
    """Calculate train/test split percentages"""
    train_percent = int(split_ratio * 100)
    test_percent = 100 - train_percent
    return train_percent, test_percent

def validate_training_config(df, target, problem_type, split_ratio):
    """Validate training configuration before starting training"""
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
    
    if target in df.columns:
        unique_target_values = df[target].nunique()
        if problem_type in ['classification', 'binary', 'multiclass'] and unique_target_values < 2:
            errors.append(f"Classification requires at least 2 unique target values, found {unique_target_values}")
    
    return {
        'valid': len(errors) == 0,
        'errors': errors
    }

def get_enhanced_eda_stats(df):
    """Get enhanced EDA statistics with additional insights"""
    basic_eda = perform_eda(df)
    
    # Add additional statistics
    enhanced_stats = basic_eda['stats'].copy()
    
    # Data quality score
    total_cells = enhanced_stats['rows'] * enhanced_stats['cols']
    missing_cells = enhanced_stats['total_missing_values']
    quality_score = ((total_cells - missing_cells) / total_cells * 100) if total_cells > 0 else 100
    enhanced_stats['data_quality_score'] = round(quality_score, 2)
    
    # Memory usage
    try:
        memory_usage = df.memory_usage(deep=True).sum()
        enhanced_stats['memory_usage_mb'] = round(memory_usage / (1024 * 1024), 2)
    except:
        enhanced_stats['memory_usage_mb'] = 0
    
    # Column type distribution
    enhanced_stats['column_type_distribution'] = {
        'numeric': len(enhanced_stats['numerics']),
        'categorical': len(enhanced_stats['categoricals']),
        'total': enhanced_stats['cols']
    }
    
    return {
        'stats': enhanced_stats,
        'plot_data': basic_eda['plot_data'],
        'correlation_data': basic_eda['correlation_data']
    }
