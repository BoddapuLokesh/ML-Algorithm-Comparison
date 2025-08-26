import os
import uuid
import pandas as pd
from flask import Flask, render_template, request, session, send_file, jsonify
from werkzeug.utils import secure_filename
from model_utils import (
    detect_target_and_type, train_models, validate_file_upload, analyze_target_column, generate_data_preview_html,
    get_enhanced_eda_stats, validate_training_config, calculate_split_percentages
)
from joblib import dump

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'csv', 'xlsx', 'xls'}

# Server-side cache for large data that doesn't fit in session
app_cache = {
    'eda_data': {},
    'training_data': {}
}

app = Flask(__name__)
app.secret_key = os.urandom(24)  # Generate random secret key
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB max file size

# Ensure directories exist before configuring session
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(os.path.join(os.getcwd(), 'flask_session'), exist_ok=True)

# Configure session settings for better persistence
app.config['SESSION_PERMANENT'] = True
app.config['PERMANENT_SESSION_LIFETIME'] = 3600  # 1 hour

@app.route('/', methods=['GET', 'POST'])
def index():
    """Main page: handles file upload and initial EDA."""
    if request.method == 'POST':
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
            return jsonify({'success': False, 'error': "No file uploaded."})
        file = request.files['dataset']
        print(f"Uploaded file: {file.filename}")
        
        # Enhanced file validation (moved from JavaScript)
        is_valid, validation_message = validate_file_upload(file)
        if not is_valid:
            print(f"ERROR: File validation failed: {validation_message}")
            return jsonify({'success': False, 'error': validation_message})

        filename = f"{uuid.uuid4().hex}_{secure_filename(file.filename or 'upload')}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        session['filepath'] = filepath
        print(f"File saved to: {filepath}")

        try:
            # Validate file before processing
            if not os.path.exists(filepath):
                print(f"ERROR: File not found: {filepath}")
                return jsonify({'success': False, 'error': "File upload failed."})
            
            # Check file size
            file_size = os.path.getsize(filepath)
            if file_size == 0:
                print(f"ERROR: Empty file: {filepath}")
                return jsonify({'success': False, 'error': "File is empty."})
            
            df = pd.read_csv(filepath) if filename.endswith('.csv') else pd.read_excel(filepath)
            print(f"Data loaded: {df.shape}")
            
            # Validate dataframe
            if df.empty:
                print(f"ERROR: Empty dataframe from file: {filepath}")
                return jsonify({'success': False, 'error': "File contains no data."})
            
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
            return jsonify({'success': False, 'error': f"Failed to read the file: {str(e)}"})

        print("=== FILE UPLOAD COMPLETED ===")

        # Return JSON response for AJAX handling - ensure all values are JSON serializable
        return jsonify({
            'success': True,
            'file_uploaded': True, 
            'file_name': file.filename, 
            'file_shape': list(df.shape),  # Convert tuple to list
            'columns': list(df.columns),
            'preview_data': preview_data,
            'message': 'File uploaded successfully!'
        })

    return render_template('index.html', current_step='upload')

@app.route('/process_eda', methods=['POST'])
def process_eda():
    """Process EDA analysis when user requests it."""
    print("=== EDA PROCESSING STARTED ===")
    
    # Get uploaded file from session
    filepath = session.get('filepath')
    print(f"Processing EDA for filepath: {filepath}")
    
    if not filepath or not os.path.exists(filepath):
        print("ERROR: No uploaded file found for EDA processing.")
        print("Session filepath:", session.get('filepath'))
        print("Available session keys:", list(session.keys()))
        return jsonify({'success': False, 'error': 'No uploaded file found. Please upload a dataset first.'})
    
    # Clear any previous EDA/training data for new analysis
    for key in ['eda', 'target', 'ptype', 'history', 'best_model_name', 'metrics', 'feature_importance', 'all_results']:
        session.pop(key, None)
    print("Previous EDA/training data cleared")
    
    try:
        # Load from uploaded file
        df = pd.read_csv(filepath) if filepath.endswith('.csv') else pd.read_excel(filepath)
        print(f"Data loaded for EDA: {df.shape}")
        
        # Perform enhanced EDA analysis
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
        return jsonify({
            'success': True,
            'eda': eda,
            'target': target,
            'ptype': ptype,
            'columns': list(df.columns),
            'message': 'EDA completed successfully!'
        })
        
    except Exception as e:
        print(f"ERROR in EDA processing: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': f'EDA processing failed: {str(e)}'})

@app.route('/train', methods=['POST'])
def train():
    """Train models and store results for comparison."""
    print("=== TRAINING STARTED ===")
    
    # Get uploaded file from session
    filepath = session.get('filepath')
    print(f"Filepath: {filepath}")
    print(f"Session keys available: {list(session.keys())}")
    
    if not filepath or not os.path.exists(filepath):
        print("ERROR: No uploaded file found. Please upload a dataset first.")
        print("Session filepath:", session.get('filepath'))
        return jsonify({'success': False, 'error': 'No uploaded file found. Please upload a dataset first.'})

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
        validation_result = validate_training_config(df, target, ptype, split_ratio)
        if not validation_result['valid']:
            error_message = "; ".join(validation_result['errors'])
            print(f"ERROR: Training validation failed: {error_message}")
            return jsonify({'success': False, 'error': f"Training validation failed: {error_message}"})

        if target not in df.columns:
            print(f"ERROR: Target column '{target}' not found in data")
            return jsonify({'success': False, 'error': f"Target column '{target}' not found in data"})

        models, best_model_name, metrics, feature_importance, all_results, preprocessing_artifacts = train_models(
            df, target, ptype, split_ratio
        )
        print(f"Training completed. Best model: {best_model_name}")
        print(f"Metrics: {metrics}")
        print(f"Number of models trained: {len(models)}")

        # Persist best model with preprocessing artifacts
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
        return jsonify({
            'success': True,
            'training_results': True,
            'best_model': best_model_name,
            'metrics': metrics,
            # Return full data in the response for immediate UI rendering
            'feature_importance': feature_importance,
            'all_results': all_results,
            'columns': list(df.drop(columns=[target]).columns),
            'message': f'Training completed! Best model: {best_model_name}'
        })
    except Exception as e:
        print(f"ERROR in training: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': f'Training failed: {str(e)}'})

@app.route('/metrics', methods=['GET'])
def get_metrics():
    """AJAX endpoint: returns metrics for best model as JSON."""
    return jsonify(session.get('metrics', {}))

@app.route('/best_model', methods=['GET'])
def get_best_model():
    """AJAX endpoint: returns best model name and details."""
    best_model_name = session.get('best_model_name', '')
    metrics = session.get('metrics', {})
    return jsonify({
        'name': best_model_name,
        'metrics': metrics
    })

@app.route('/feature_importance', methods=['GET'])
def get_feature_importance():
    """AJAX endpoint: returns feature importance data."""
    # Prefer server-side cache (session cookies can overflow)
    session_id = session.get('session_id')
    cached = app_cache['training_data'].get(session_id, {}) if session_id else {}
    if cached and 'feature_importance' in cached:
        return jsonify(cached['feature_importance'])
    return jsonify(session.get('feature_importance', {}))

# New: AJAX endpoint for EDA results
@app.route('/eda', methods=['GET'])
def get_eda():
    """AJAX endpoint: returns EDA results as JSON."""
    # Get EDA data from server cache instead of session
    session_id = session.get('session_id')
    eda_data = app_cache['eda_data'].get(session_id, {}) if session_id else {}
    print(f"EDA endpoint called. Returning: {bool(eda_data)}")
    if eda_data:
        print(f"EDA data keys: {list(eda_data.keys())}")
    return jsonify(eda_data)

# New: AJAX endpoint for data preview
@app.route('/data_preview', methods=['GET'])
def get_data_preview():
    """AJAX endpoint: returns data preview for table display."""
    preview_data = session.get('data_preview', [])
    columns = session.get('file_columns', [])
    filepath = session.get('filepath', 'None')
    
    # Ensure all values in preview_data are JSON serializable
    clean_preview_data = []
    for row in preview_data:
        clean_row = {}
        for key, value in row.items():
            if pd.isna(value) or value is None:
                clean_row[key] = ''  # Replace NaN/None with empty string
            else:
                clean_row[key] = str(value)  # Convert to string to be safe
        clean_preview_data.append(clean_row)
    
    print(f"Data preview requested - Filepath: {filepath}, Columns: {len(columns)}, Rows: {len(clean_preview_data)}")
    return jsonify({
        'columns': columns,
        'data': clean_preview_data,
        'filepath': filepath  # For debugging
    })

# New: AJAX endpoint for model comparison (historical)
@app.route('/model_comparison', methods=['GET'])
def model_comparison():
    """AJAX endpoint: returns all trained model metrics for comparison."""
    session_id = session.get('session_id')
    cached = app_cache['training_data'].get(session_id, {}) if session_id else {}
    if cached and 'history' in cached:
        return jsonify(cached['history'])
    return jsonify(session.get('history', []))

# New: Enhanced target analysis endpoint (moved from JavaScript)
@app.route('/analyze_target', methods=['POST'])
def analyze_target():
    """Enhanced target analysis endpoint"""
    try:
        data = request.get_json()
        target_column = data.get('target') if data else request.form.get('target')
        
        if not target_column:
            return jsonify({'success': False, 'error': 'Target column not specified'})
            
        filepath = session.get('filepath')
        if not filepath or not os.path.exists(filepath):
            return jsonify({'success': False, 'error': 'No file found. Please upload a dataset first.'})
        
        # Load data
        df = pd.read_csv(filepath) if filepath.endswith('.csv') else pd.read_excel(filepath)
        
        # Perform analysis
        analysis = analyze_target_column(df, target_column)
        return jsonify(analysis)
        
    except Exception as e:
        print(f"Error in target analysis: {str(e)}")
        return jsonify({'success': False, 'error': f'Target analysis failed: {str(e)}'})

# New: Server-side HTML generation for data preview
@app.route('/get_data_preview_html', methods=['GET'])
def get_data_preview_html():
    """Generate HTML table server-side for better security"""
    try:
        preview_data = session.get('data_preview', [])
        columns = session.get('file_columns', [])
        
        if not preview_data or not columns:
            return jsonify({'html': '<div class="text-center text-gray-500">No data available for preview</div>'})
        
        # Generate HTML server-side using the helper function
        html_content = generate_data_preview_html(preview_data, columns)
        
        return jsonify({
            'success': True,
            'html': html_content,
            'columns_count': len(columns),
            'rows_count': len(preview_data)
        })
        
    except Exception as e:
        print(f"Error generating preview HTML: {str(e)}")
        return jsonify({'success': False, 'error': f'Failed to generate preview: {str(e)}'})

# New: Split ratio calculation endpoint
@app.route('/calculate_split', methods=['POST'])
def calculate_split():
    """Calculate train/test split percentages"""
    try:
        data = request.get_json()
        split_ratio = float(data.get('split_ratio', 0.8))
        
        train_percent, test_percent = calculate_split_percentages(split_ratio)
        
        return jsonify({
            'success': True,
            'train_percent': train_percent,
            'test_percent': test_percent,
            'split_ratio': split_ratio
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': f'Failed to calculate split: {str(e)}'})

# New: Training configuration validation endpoint
@app.route('/validate_training_config', methods=['POST'])
def validate_training_config_endpoint():
    """Validate training configuration before starting training"""
    try:
        # Handle both form and JSON data
        if request.is_json:
            data = request.get_json() or {}
            target = data.get('target')
            ptype = data.get('ptype')
            split_ratio = float(data.get('split_ratio', 0.8))
        else:
            target = request.form.get('target')
            ptype = request.form.get('ptype')
            split_ratio = float(request.form.get('split', 0.8))
        
        # Validate required parameters
        if not target:
            return jsonify({'success': False, 'error': 'Target column is required'})
        if not ptype:
            return jsonify({'success': False, 'error': 'Problem type is required'})
        
        filepath = session.get('filepath')
        if not filepath or not os.path.exists(filepath):
            return jsonify({'success': False, 'error': 'No file found'})
        
        # Load data
        df = pd.read_csv(filepath) if filepath.endswith('.csv') else pd.read_excel(filepath)
        
        # Validate configuration
        validation_result = validate_training_config(df, target, ptype, split_ratio)
        
        return jsonify({
            'success': True,
            'validation_result': validation_result
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': f'Validation failed: {str(e)}'})

@app.route('/debug_session', methods=['GET'])
def debug_session():
    """Debug endpoint to check session data."""
    return jsonify({
        'session_keys': list(session.keys()),
        'filepath': session.get('filepath'),
        'file_columns': session.get('file_columns'),
        'file_shape': session.get('file_shape'),
        'best_model_name': session.get('best_model_name'),
        'metrics': session.get('metrics'),
        'feature_importance': session.get('feature_importance'),
        'history': session.get('history'),
        'all_results': session.get('all_results'),
        'eda_available': bool(session.get('eda')),
        'target': session.get('target'),
        'ptype': session.get('ptype')
    })

@app.route('/reset_session', methods=['POST'])
def reset_session():
    """Reset session for debugging."""
    # Clear cache data for this session
    old_session_id = session.get('session_id')
    if old_session_id and old_session_id in app_cache['eda_data']:
        del app_cache['eda_data'][old_session_id]
    if old_session_id and old_session_id in app_cache['training_data']:
        del app_cache['training_data'][old_session_id]
    session.clear()
    return jsonify({'message': 'Session cleared successfully'})

@app.route('/download_model')
def download_model():
    """Download the best trained model with preprocessing artifacts."""
    import zipfile
    import tempfile
    import os
    
    model_path = session.get('model_path')
    preprocessors_path = session.get('preprocessors_path')
    
    if not model_path or not preprocessors_path:
        return "No complete model package available.", 404
    
    # Create a temporary zip file containing both model and preprocessors
    try:
        temp_dir = tempfile.mkdtemp()
        zip_path = os.path.join(temp_dir, 'complete_model.zip')
        
        with zipfile.ZipFile(zip_path, 'w') as zipf:
            zipf.write(model_path, 'model.joblib')
            zipf.write(preprocessors_path, 'preprocessors.joblib')
            
            # Add a readme file with usage instructions
            readme_content = """# AutoML Model Package

This package contains:
- model.joblib: The trained machine learning model
- preprocessors.joblib: All preprocessing artifacts (encoders, imputers)

## Usage:
```python
from joblib import load
import pandas as pd

# Load the model and preprocessors
model = load('model.joblib')
preprocessors = load('preprocessors.joblib')

# To make predictions on new data:
# 1. Apply the same preprocessing steps using the saved artifacts
# 2. Use the model to predict
```
"""
            zipf.writestr('README.md', readme_content)
        
        return send_file(zip_path, as_attachment=True, download_name='complete_model.zip')
        
    except Exception as e:
        print(f"Error creating model package: {str(e)}")
        return f"Error creating model package: {str(e)}", 500

@app.route('/export_results')
def export_results():
    """Export training results as CSV."""
    try:
        import csv
        import io
        from flask import make_response
        
        # Get all results from session
        all_results = session.get('history', [])
        best_model = session.get('best_model_name', 'Unknown')
        
        if not all_results:
            return "No training results available for export.", 404
        
        # Create CSV content
        output = io.StringIO()
        writer = csv.writer(output)
        
        # Write header
        if all_results:
            metrics_keys = list(all_results[0].get('metrics', {}).keys())
            header = ['Model', 'Best_Model'] + metrics_keys
            writer.writerow(header)
            
            # Write data
            for result in all_results:
                model_name = result.get('model', 'Unknown')
                is_best = 'Yes' if model_name == best_model else 'No'
                metrics = result.get('metrics', {})
                row = [model_name, is_best] + [metrics.get(key, '') for key in metrics_keys]
                writer.writerow(row)
        
        # Create response
        response = make_response(output.getvalue())
        response.headers['Content-Type'] = 'text/csv'
        response.headers['Content-Disposition'] = 'attachment; filename=automl_results.csv'
        
        return response
        
    except Exception as e:
        print(f"Error exporting results: {str(e)}")
        return f"Error exporting results: {str(e)}", 500

# Clean up uploads on shutdown
import atexit
@atexit.register
def cleanup_uploads():
    """Cleanup uploads folder on shutdown."""
    import shutil
    shutil.rmtree(UPLOAD_FOLDER, ignore_errors=True)

if __name__ == "__main__":
    app.run(host='127.0.0.1', port=5002)

