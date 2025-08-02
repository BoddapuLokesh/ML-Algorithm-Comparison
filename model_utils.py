import pandas as pd
import numpy as np
import time
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.metrics import accuracy_score, mean_squared_error, classification_report, precision_score, recall_score, f1_score, r2_score
import plotly.graph_objs as go

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'csv', 'xlsx', 'xls'}

def perform_eda(df):
    # Basic stats
    n_rows, n_cols = df.shape
    cols = list(df.columns)
    dtypes = df.dtypes.astype(str).to_dict()
    missing = df.isnull().sum().to_dict()
    duplicate_rows = int(df.duplicated().sum())
    numeric_cols = list(df.select_dtypes(include='number').columns)
    categorical_cols = list(df.select_dtypes(include='object').columns)

    # Store plot data instead of HTML (much smaller)
    plot_data = []
    # Store just the data for first 3 numeric/categorical columns
    for col in numeric_cols[:3]:
        hist_data = df[col].dropna().tolist()
        plot_data.append({"type": "histogram", "column": col, "data": hist_data[:100]})  # Limit to 100 points
    for col in categorical_cols[:3]:
        counts = df[col].value_counts().head(10)  # Limit to top 10 categories
        plot_data.append({"type": "bar", "column": col, "labels": counts.index.astype(str).tolist(), "values": counts.values.tolist()})

    stats = {"rows": n_rows, "cols": n_cols, "dtypes": dtypes, "missing": missing,
             "duplicate_rows": duplicate_rows, "numerics": numeric_cols, "categoricals": categorical_cols}

    return {"stats": stats, "plot_data": plot_data}

def detect_target_and_type(df):
    # Heuristic: look for column names
    candidates = [c for c in df.columns if c.lower() in ["target", "label", "class", "y", "output", "prediction", "outcome"]]
    if not candidates:
        # fallback: last column
        candidates = [df.columns[-1]]
    target = candidates[0]
    unique_values = df[target].nunique(dropna=False)
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
    return target, ptype

def preprocess_data(df, target):
    X = df.drop(columns=[target])
    y = df[target]
    # Encode categoricals
    X = X.copy()
    for col in X.select_dtypes(include='object'):
        X[col] = LabelEncoder().fit_transform(X[col].astype(str))
    if y.dtype == 'object':
        y = LabelEncoder().fit_transform(y.astype(str))
    X = pd.DataFrame(SimpleImputer(strategy='mean').fit_transform(X), columns=X.columns)
    return X, y

def train_models(df, target, problem_type, split_ratio):
    try:
        X, y = preprocess_data(df, target)
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
                        "MSE": mean_squared_error(y_test, preds),
                        "R^2": score,
                        "Training_Time": round(training_time, 3)
                    }
                else:
                    score = accuracy_score(y_test, preds)
                    metrics = {
                        "Accuracy": score,
                        "Precision": precision_score(y_test, preds, average='weighted', zero_division=0),
                        "Recall": recall_score(y_test, preds, average='weighted', zero_division=0),
                        "F1": f1_score(y_test, preds, average='weighted', zero_division=0),
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

        # Feature importance (for tree/random forest)
        feature_importance = {}
        for name, model in models.items():
            try:
                if hasattr(model, "feature_importances_"):
                    imp = model.feature_importances_
                    feature_importance[name] = dict(zip(X.columns, imp))
                elif hasattr(model, "coef_"):
                    vals = model.coef_
                    coefs = vals[0] if len(vals.shape) > 1 else vals
                    feature_importance[name] = dict(zip(X.columns, np.abs(coefs)))
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

def get_model_metrics():
    # (optional, not used as everything is returned in one go)
    pass
