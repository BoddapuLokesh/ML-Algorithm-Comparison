"""Modern AutoML model comparison using sklearn best practices."""

import time
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple, Optional
from sklearn.base import BaseEstimator
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.metrics import accuracy_score, mean_squared_error, precision_score, recall_score, f1_score, r2_score
from sklearn.preprocessing import StandardScaler

from .config import MLConfig, MLResults, ModelMetrics
from .preprocessing import preprocess_data_minimal
from .utils import safe_json_convert


class AutoMLComparer(BaseEstimator):
    """Minimalistic AutoML model comparison using sklearn best practices.
    
    Replaces the complex train_models function with a clean, pipeline-based approach.
    """
    
    MODELS = {
        'classification': [
            ('LogisticRegression', LogisticRegression(max_iter=1000, random_state=42)),
            ('RandomForest', RandomForestClassifier(n_estimators=100, random_state=42)),
            ('GradientBoosting', GradientBoostingClassifier(n_estimators=100, random_state=42)),
            ('SVC', SVC(kernel='rbf', C=1.0, gamma='scale', probability=True, random_state=42)),
            ('DecisionTree', DecisionTreeClassifier(random_state=42))
        ],
        'regression': [
            ('LinearRegression', LinearRegression()),
            ('RandomForest', RandomForestRegressor(n_estimators=100, random_state=42)),
            ('GradientBoosting', GradientBoostingRegressor(n_estimators=100, random_state=42)),
            ('SVR', SVR(kernel='rbf', C=1.0, gamma='scale')),
            ('DecisionTree', DecisionTreeRegressor(random_state=42))
        ]
    }
    
    def __init__(self, timeout_seconds: int = 300):
        self.timeout_seconds = timeout_seconds
        self.models_ = {}
        self.results_ = {}
        self.best_model_name_ = None
        self.preprocessing_artifacts_ = None
    
    def fit_compare(self, df: pd.DataFrame, config: MLConfig) -> Dict[str, Any]:
        """Train and compare multiple models with clean pipeline approach."""
        print(f"Starting training with target: {config.target_column}, type: {config.problem_type}")
        
        # Validate inputs
        self._validate_inputs(df, config)
        
        # Preprocess data
        X, y, preprocessing_artifacts = preprocess_data_minimal(df, config.target_column)
        self.preprocessing_artifacts_ = preprocessing_artifacts
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=config.test_size, random_state=config.random_state
        )
        
        # Map problem types to model categories
        model_category = self._map_problem_type(config.problem_type)
        
        # Train models
        results = self._train_models(X_train, X_test, y_train, y_test, model_category)
        
        # Extract best model
        best_model_name, best_metrics = self._find_best_model(results, model_category)
        
        # Extract feature importance (with fallback for models lacking native importance)
        feature_importance = self._extract_feature_importance(
            feature_names=X.columns,
            best_model_name=best_model_name,
            X_test=X_test,
            y_test=y_test,
            model_category=model_category
        )
        
        # Format results
        all_results = [{"model": name, "metrics": metrics} for name, metrics in results.items()]
        
        return {
            'models': self.models_,
            'best_model_name': best_model_name,
            'metrics': best_metrics,
            'feature_importance': feature_importance,
            'all_results': all_results,
            'preprocessing_artifacts': preprocessing_artifacts
        }
    
    def _validate_inputs(self, df: pd.DataFrame, config: MLConfig) -> None:
        """Validate input data and configuration."""
        if config.target_column not in df.columns:
            raise ValueError(f"Target column '{config.target_column}' not found in dataset")
        
        if df[config.target_column].isna().all():
            raise ValueError(f"Target column '{config.target_column}' has no valid values")
        
        if len(df) < 10:
            raise ValueError("Dataset too small. Need at least 10 rows for training")
    
    def _map_problem_type(self, problem_type: str) -> str:
        """Map problem type to model category."""
        if problem_type in ['binary', 'multiclass']:
            return 'classification'
        elif problem_type == 'regression':
            return 'regression'
        else:
            # Default mapping
            return 'classification' if problem_type == 'classification' else 'regression'
    
    def _train_models(self, X_train: pd.DataFrame, X_test: pd.DataFrame, 
                     y_train: pd.Series, y_test: pd.Series, model_category: str) -> Dict[str, Dict[str, float]]:
        """Train models with timeout protection and proper scaling."""
        results = {}
        scaler = None
        
        for name, base_model in self.MODELS[model_category]:
            try:
                start_time = time.time()
                
                # Create pipeline with conditional scaling
                pipeline_steps = []
                
                # Add scaling for SVM models
                if any(svm_type in name for svm_type in ['SVC', 'SVR']):
                    if scaler is None:
                        scaler = StandardScaler()
                        scaler.fit(X_train)
                    pipeline_steps.append(('scaler', scaler))
                
                pipeline_steps.append(('model', base_model))
                pipeline = Pipeline(pipeline_steps)
                
                # Train model
                pipeline.fit(X_train, y_train)
                
                # Check timeout
                elapsed_time = time.time() - start_time
                if elapsed_time > self.timeout_seconds:
                    print(f"{name} exceeded timeout ({elapsed_time:.1f}s) - skipped")
                    continue
                
                # Make predictions
                y_pred = pipeline.predict(X_test)
                
                # Calculate metrics
                metrics = self._calculate_metrics(y_test, y_pred, model_category, elapsed_time)
                
                # Store results
                self.models_[name] = pipeline
                results[name] = metrics
                
                print(f"{name} completed - Score: {metrics.get('Accuracy', metrics.get('R2', 0)):.3f}, Time: {elapsed_time:.2f}s")
                
            except Exception as e:
                print(f"Error training {name}: {e}")
                continue
        
        if not results:
            raise RuntimeError("No models trained successfully")
        
        return results
    
    def _calculate_metrics(self, y_test: pd.Series, y_pred: Any, 
                          model_category: str, training_time: float) -> Dict[str, float]:
        """Calculate appropriate metrics based on problem type."""
        metrics = {'Training_Time': round(training_time, 3)}
        
        try:
            if model_category == 'classification':
                acc = accuracy_score(y_test, y_pred)
                prec = precision_score(y_test, y_pred, average='weighted', zero_division=0)
                rec = recall_score(y_test, y_pred, average='weighted', zero_division=0)
                f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
                
                metrics.update({
                    'Accuracy': float(acc),
                    'Precision': float(prec), 
                    'Recall': float(rec),
                    'F1': float(f1)
                })
            else:  # regression
                mse = mean_squared_error(y_test, y_pred)
                r2 = r2_score(y_test, y_pred)
                
                metrics.update({
                    'MSE': float(mse),
                    'R2': float(r2)
                })
        except Exception as e:
            print(f"Error calculating metrics: {e}")
            
        return metrics
    
    def _find_best_model(self, results: Dict[str, Dict[str, float]], model_category: str) -> Tuple[str, Dict[str, float]]:
        """Find the best performing model."""
        if not results:
            raise ValueError("No models to compare")
        
        # Determine scoring metric
        score_key = 'Accuracy' if model_category == 'classification' else 'R2'
        
        # Find best model
        best_name = max(results.keys(), key=lambda x: results[x].get(score_key, -np.inf))
        
        return best_name, results[best_name]
    
    def _extract_feature_importance(
        self,
        feature_names: pd.Index,
        best_model_name: Optional[str] = None,
        X_test: Optional[pd.DataFrame] = None,
        y_test: Optional[pd.Series] = None,
        model_category: Optional[str] = None,
    ) -> Dict[str, Dict[str, float]]:
        """Extract feature importance for trained models with sensible fallbacks.

        - Uses .feature_importances_ or absolute coefficients when available.
        - Falls back to permutation_importance for the best model if native importance is absent.
        """
        feature_importance: Dict[str, Dict[str, float]] = {}

        # Helper to convert array of importances into dict
        def to_dict(importances: np.ndarray) -> Dict[str, float]:
            return {str(feature): float(imp) for feature, imp in zip(feature_names, importances)}

        # First, try native importances for all models
        for name, pipeline in self.models_.items():
            try:
                model = pipeline.named_steps.get('model', pipeline)
                if hasattr(model, 'feature_importances_'):
                    feature_importance[name] = to_dict(np.array(model.feature_importances_))
                elif hasattr(model, 'coef_'):
                    coefs = np.array(model.coef_)
                    # For multiclass, aggregate over classes
                    if coefs.ndim > 1:
                        importances = np.mean(np.abs(coefs), axis=0)
                    else:
                        importances = np.abs(coefs)
                    feature_importance[name] = to_dict(importances)
                else:
                    feature_importance[name] = {}
            except Exception as e:
                print(f"Feature importance extraction failed for {name}: {e}")
                feature_importance[name] = {}

        # If best model has empty importance and we have test data, use permutation importance
        if best_model_name and best_model_name in self.models_:
            if not feature_importance.get(best_model_name) and X_test is not None and y_test is not None:
                try:
                    from sklearn.inspection import permutation_importance
                    pipeline = self.models_[best_model_name]
                    # Use default scoring for estimator; limit repeats for speed
                    result = permutation_importance(
                        pipeline, X_test, y_test, n_repeats=5, random_state=42, n_jobs=1
                    )
                    importances = getattr(result, 'importances_mean', None)
                    if importances is None:
                        try:
                            importances = result["importances_mean"]  # type: ignore[index]
                        except Exception:
                            importances = np.zeros(len(feature_names))
                    feature_importance[best_model_name] = to_dict(np.array(importances))
                    print(f"Permutation importance computed for {best_model_name}")
                except Exception as e:
                    print(f"Permutation importance failed for {best_model_name}: {e}")
                    # keep empty dict

        return feature_importance
