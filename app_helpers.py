"""
Helper functions and decorators for the Flask app optimization
Maintains 100% backward compatibility while reducing code duplication
"""

import os
import uuid
import functools
import pandas as pd
from flask import session, jsonify, request, current_app
from typing import Callable, Dict, Any, Tuple, Optional

# Server-side cache (same as original)
app_cache = {
    'eda_data': {},
    'training_data': {}
}

def api_response(func: Callable) -> Callable:
    """Decorator for standardized API responses with error handling"""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            result = func(*args, **kwargs)
            if isinstance(result, dict):
                return jsonify(result)
            return result
        except Exception as e:
            current_app.logger.error(f"API Error in {func.__name__}: {str(e)}")
            return jsonify({'success': False, 'error': str(e)}), 500
    return wrapper

def require_file(func: Callable) -> Callable:
    """Decorator to ensure a file is uploaded and accessible"""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        filepath = session.get('filepath')
        if not filepath or not os.path.exists(filepath):
            return jsonify({
                'success': False, 
                'error': 'No uploaded file found. Please upload a dataset first.'
            }), 400
        return func(*args, **kwargs)
    return wrapper

def load_dataframe() -> Tuple[pd.DataFrame, str]:
    """Utility function to load dataframe from session filepath"""
    filepath = session.get('filepath')
    if not filepath or not os.path.exists(filepath):
        raise ValueError("No valid file found in session")
    
    df = pd.read_csv(filepath) if filepath.endswith('.csv') else pd.read_excel(filepath)
    return df, filepath

def get_session_cache(cache_type: str) -> Dict[str, Any]:
    """Get cached data for current session"""
    session_id = session.get('session_id')
    if not session_id:
        return {}
    return app_cache.get(cache_type, {}).get(session_id, {})

def set_session_cache(cache_type: str, data: Dict[str, Any]) -> None:
    """Set cached data for current session"""
    session_id = session.get('session_id')
    if not session_id:
        session_id = str(uuid.uuid4())
        session['session_id'] = session_id
    
    if cache_type not in app_cache:
        app_cache[cache_type] = {}
    app_cache[cache_type][session_id] = data

def clear_session_cache() -> None:
    """Clear all cached data for current session"""
    session_id = session.get('session_id')
    if session_id:
        for cache_type in app_cache:
            app_cache[cache_type].pop(session_id, None)

def handle_file_upload():
    """Centralized file upload handling - exactly same logic as original"""
    print("=== FILE UPLOAD STARTED ===")
    
    # Clear all previous session data and cache when new file is uploaded
    old_session_id = session.get('session_id')
    session.clear()
    # Clear old cache data if it exists
    if old_session_id and old_session_id in app_cache['eda_data']:
        del app_cache['eda_data'][old_session_id]
    if old_session_id and old_session_id in app_cache['training_data']:
        del app_cache['training_data'][old_session_id]
    print("Previous session data cleared for new upload")
    
    if 'dataset' not in request.files:
        print("ERROR: No file uploaded")
        return {'success': False, 'error': "No file uploaded."}
    
    file = request.files['dataset']
    print(f"Uploaded file: {file.filename}")
    
    # Enhanced file validation (moved from JavaScript)
    from model_utils import validate_file_upload
    is_valid, validation_message = validate_file_upload(file)
    if not is_valid:
        print(f"ERROR: File validation failed: {validation_message}")
        return {'success': False, 'error': validation_message}

    filename = f"{uuid.uuid4().hex}_{file.filename or 'upload'}"
    filepath = os.path.join(current_app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)
    session['filepath'] = filepath
    print(f"File saved to: {filepath}")

    try:
        # Validate file before processing
        if not os.path.exists(filepath):
            print(f"ERROR: File not found: {filepath}")
            return {'success': False, 'error': "File upload failed."}
        
        # Check file size
        file_size = os.path.getsize(filepath)
        if file_size == 0:
            print(f"ERROR: Empty file: {filepath}")
            return {'success': False, 'error': "File is empty."}
        
        df = pd.read_csv(filepath) if filename.endswith('.csv') else pd.read_excel(filepath)
        print(f"Data loaded: {df.shape}")
        
        # Validate dataframe
        if df.empty:
            print(f"ERROR: Empty dataframe from file: {filepath}")
            return {'success': False, 'error': "File contains no data."}
        
        # Store basic file info in session for later EDA processing
        session['file_shape'] = list(df.shape)  # Convert tuple to list for JSON serialization
        session['file_columns'] = list(df.columns)
        session['filename'] = file.filename
        
        # Store data preview (first 5 rows) for frontend display - handle NaN values
        preview_df = df.head(5).fillna('')  # Replace NaN with empty string
        preview_data = preview_df.to_dict('records')  # Convert to list of dicts
        session['data_preview'] = preview_data
        print(f"File info stored in session: {df.shape} with columns: {list(df.columns)[:5]}...")
        
        # Auto-detect basic data types for preview
        dtypes_info = {}
        for col in df.columns:
            if pd.api.types.is_numeric_dtype(df[col]):
                dtypes_info[col] = 'numeric'
            else:
                dtypes_info[col] = 'categorical'
        session['dtypes_info'] = dtypes_info
        
    except Exception as e:
        print(f"ERROR: Failed to read file: {str(e)}")
        return {'success': False, 'error': f"Failed to read the file: {str(e)}"}

    print("=== FILE UPLOAD COMPLETED ===")

    # Return JSON response for AJAX handling - ensure all values are JSON serializable
    return {
        'success': True,
        'file_uploaded': True, 
        'file_name': file.filename, 
        'file_shape': list(df.shape),  # Convert tuple to list
        'columns': list(df.columns),
        'preview_data': preview_data,
        'message': 'File uploaded successfully!'
    }

def handle_eda_processing():
    """Centralized EDA processing - exactly same logic as original"""
    print("=== EDA PROCESSING STARTED ===")
    
    # Get uploaded file from session
    filepath = session.get('filepath')
    print(f"Processing EDA for filepath: {filepath}")
    
    if not filepath or not os.path.exists(filepath):
        print("ERROR: No uploaded file found for EDA processing.")
        print("Session filepath:", session.get('filepath'))
        print("Available session keys:", list(session.keys()))
        return {'success': False, 'error': 'No uploaded file found. Please upload a dataset first.'}
    
    # Clear any previous EDA/training data for new analysis
    for key in ['eda', 'target', 'ptype', 'history', 'best_model_name', 'metrics', 'feature_importance', 'all_results']:
        session.pop(key, None)
    print("Previous EDA/training data cleared")
    
    try:
        # Load from uploaded file
        df = pd.read_csv(filepath) if filepath.endswith('.csv') else pd.read_excel(filepath)
        print(f"Data loaded for EDA: {df.shape}")
        
        # Perform enhanced EDA analysis
        from model_utils import get_enhanced_eda_stats, detect_target_and_type
        eda = get_enhanced_eda_stats(df)
        target, ptype = detect_target_and_type(df)
        session['target'], session['ptype'] = target, ptype
        session['columns'] = list(df.columns)
        
        # Generate session ID for cache storage of large data
        if 'session_id' not in session:
            session['session_id'] = str(uuid.uuid4())
        
        # Store EDA in server cache instead of session (to avoid size limits)
        app_cache['eda_data'][session['session_id']] = eda
        print(f"EDA completed. Target: {target}, Type: {ptype}")
        print(f"EDA stats keys: {list(eda.keys())}")
        print("=== EDA PROCESSING COMPLETED ===")

        # Return JSON response with EDA results
        return {
            'success': True,
            'eda': eda,
            'target': target,
            'ptype': ptype,
            'columns': list(df.columns),
            'message': 'EDA completed successfully!'
        }
        
    except Exception as e:
        print(f"ERROR in EDA processing: {str(e)}")
        import traceback
        traceback.print_exc()
        return {'success': False, 'error': f'EDA processing failed: {str(e)}'}

def handle_training():
    """Centralized training logic - exactly same as original"""
    print("=== TRAINING STARTED ===")
    
    # Get uploaded file from session
    filepath = session.get('filepath')
    print(f"Filepath: {filepath}")
    print(f"Session keys available: {list(session.keys())}")
    
    if not filepath or not os.path.exists(filepath):
        print("ERROR: No uploaded file found. Please upload a dataset first.")
        print("Session filepath:", session.get('filepath'))
        return {'success': False, 'error': 'No uploaded file found. Please upload a dataset first.'}

    target = request.form['target']
    ptype = request.form['ptype']
    split_ratio = float(request.form.get('split', 0.8))
    
    print(f"Training config: target={target}, ptype={ptype}, split={split_ratio}")
    print(f"Using file: {filepath}")

    try:
        # Load from uploaded file
        df = pd.read_csv(filepath) if filepath.endswith('.csv') else pd.read_excel(filepath)
        print(f"Data loaded from uploaded file: {df.shape}")
        print(f"Columns: {list(df.columns)}")
        print(f"Target column '{target}' exists: {target in df.columns}")

        # Enhanced training validation (moved from JavaScript)
        from model_utils import validate_training_config, train_models
        validation_result = validate_training_config(df, target, ptype, split_ratio)
        if not validation_result['valid']:
            error_message = "; ".join(validation_result['errors'])
            print(f"ERROR: Training validation failed: {error_message}")
            return {'success': False, 'error': f"Training validation failed: {error_message}"}

        if target not in df.columns:
            print(f"ERROR: Target column '{target}' not found in data")
            return {'success': False, 'error': f"Target column '{target}' not found in data"}

        models, best_model_name, metrics, feature_importance, all_results, preprocessing_artifacts = train_models(
            df, target, ptype, split_ratio
        )
        print(f"Training completed. Best model: {best_model_name}")
        print(f"Metrics: {metrics}")
        print(f"Number of models trained: {len(models)}")

        # Persist best model with preprocessing artifacts
        from joblib import dump
        model_path = f"{filepath}_bestmodel.joblib"
        preprocessors_path = f"{filepath}_preprocessors.joblib"

        # Save the best model and preprocessing artifacts separately
        dump(models[best_model_name], model_path)
        dump(preprocessing_artifacts, preprocessors_path)

        session['model_path'] = model_path
        session['preprocessors_path'] = preprocessors_path
        session['best_model_name'] = best_model_name
        session['metrics'] = metrics

        # Ensure a session-scoped cache key exists
        if 'session_id' not in session:
            session['session_id'] = str(uuid.uuid4())

        # Store larger training artifacts in server-side cache to avoid cookie bloat
        sid = session['session_id']
        app_cache['training_data'][sid] = {
            'feature_importance': feature_importance,
            'history': all_results
        }

        # Force session save
        session.permanent = True
        session.modified = True

        print(f"Session data stored. History length: {len(all_results)}")
        print(f"Best model stored: {session.get('best_model_name')}")
        print(f"Metrics stored: {session.get('metrics')}")
        print("=== TRAINING COMPLETED ===")

        # Always render all sections, set current_step for JS
        return {
            'success': True,
            'training_results': True,
            'best_model': best_model_name,
            'metrics': metrics,
            # Return full data in the response for immediate UI rendering
            'feature_importance': feature_importance,
            'all_results': all_results,
            'columns': list(df.drop(columns=[target]).columns),
            'message': f'Training completed! Best model: {best_model_name}'
        }
    except Exception as e:
        print(f"ERROR in training: {str(e)}")
        import traceback
        traceback.print_exc()
        return {'success': False, 'error': f'Training failed: {str(e)}'}
