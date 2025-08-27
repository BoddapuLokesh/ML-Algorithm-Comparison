"""
Optimized Flask App - Minimalistic approach while preserving 100% functionality
Reduces code duplication while maintaining all existing routes and behavior
Original: 602 lines → Optimized: ~280 lines (53% reduction)
"""

import os
import uuid
import pandas as pd
import zipfile
import tempfile
import csv
import io
from flask import Flask, render_template, request, session, send_file, jsonify, make_response
from werkzeug.utils import secure_filename
from model_utils import (
    detect_target_and_type, train_models, validate_file_upload, analyze_target_column, 
    generate_data_preview_html, get_enhanced_eda_stats, validate_training_config, 
    calculate_split_percentages
)
from joblib import dump
from app_helpers import (
    app_cache, api_response, require_file, load_dataframe, get_session_cache, 
    set_session_cache, clear_session_cache, handle_file_upload, handle_eda_processing, 
    handle_training
)

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'csv', 'xlsx', 'xls'}

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
        result = handle_file_upload()
        return jsonify(result)
    return render_template('index.html', current_step='upload')

@app.route('/process_eda', methods=['POST'])
@api_response
def process_eda():
    """Process EDA analysis when user requests it."""
    return handle_eda_processing()

@app.route('/train', methods=['POST'])
@api_response
def train():
    """Train models and store results for comparison."""
    return handle_training()

# Simplified API endpoints using decorators and helpers
@app.route('/metrics', methods=['GET'])
@api_response
def get_metrics():
    """AJAX endpoint: returns metrics for best model as JSON."""
    return session.get('metrics', {})

@app.route('/best_model', methods=['GET'])
@api_response
def get_best_model():
    """AJAX endpoint: returns best model name and details."""
    return {
        'name': session.get('best_model_name', ''),
        'metrics': session.get('metrics', {})
    }

@app.route('/feature_importance', methods=['GET'])
@api_response
def get_feature_importance():
    """AJAX endpoint: returns feature importance data."""
    cached = get_session_cache('training_data')
    return cached.get('feature_importance', session.get('feature_importance', {}))

@app.route('/eda', methods=['GET'])
@api_response
def get_eda():
    """AJAX endpoint: returns EDA results as JSON."""
    eda_data = get_session_cache('eda_data')
    print(f"EDA endpoint called. Returning: {bool(eda_data)}")
    return eda_data

@app.route('/data_preview', methods=['GET'])
@api_response
def get_data_preview():
    """AJAX endpoint: returns data preview for table display."""
    preview_data = session.get('data_preview', [])
    columns = session.get('file_columns', [])
    
    # Clean preview data for JSON serialization
    clean_preview_data = []
    for row in preview_data:
        clean_row = {k: '' if pd.isna(v) or v is None else str(v) for k, v in row.items()}
        clean_preview_data.append(clean_row)
    
    return {
        'columns': columns,
        'data': clean_preview_data,
        'filepath': session.get('filepath', 'None')
    }

@app.route('/model_comparison', methods=['GET'])
@api_response
def model_comparison():
    """AJAX endpoint: returns all trained model metrics for comparison."""
    cached = get_session_cache('training_data')
    return cached.get('history', session.get('history', []))

@app.route('/analyze_target', methods=['POST'])
@api_response
@require_file
def analyze_target():
    """Enhanced target analysis endpoint"""
    data = request.get_json()
    target_column = data.get('target') if data else request.form.get('target')
    
    if not target_column:
        return {'success': False, 'error': 'Target column not specified'}
    
    df, _ = load_dataframe()
    return analyze_target_column(df, target_column)

@app.route('/get_data_preview_html', methods=['GET'])
@api_response
def get_data_preview_html():
    """Generate HTML table server-side for better security"""
    preview_data = session.get('data_preview', [])
    columns = session.get('file_columns', [])
    
    if not preview_data or not columns:
        return {'html': '<div class="text-center text-gray-500">No data available for preview</div>'}
    
    html_content = generate_data_preview_html(preview_data, columns)
    return {
        'success': True,
        'html': html_content,
        'columns_count': len(columns),
        'rows_count': len(preview_data)
    }

@app.route('/calculate_split', methods=['POST'])
@api_response
def calculate_split():
    """Calculate train/test split percentages"""
    data = request.get_json()
    split_ratio = float(data.get('split_ratio', 0.8))
    train_percent, test_percent = calculate_split_percentages(split_ratio)
    return {
        'success': True,
        'train_percent': train_percent,
        'test_percent': test_percent,
        'split_ratio': split_ratio
    }

@app.route('/validate_training_config', methods=['POST'])
@api_response
@require_file
def validate_training_config_endpoint():
    """Validate training configuration before starting training"""
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
    
    if not target or not ptype:
        return {'success': False, 'error': 'Missing required parameters: target and ptype must be provided'}
    
    df, _ = load_dataframe()
    validation_result = validate_training_config(df, target, ptype, split_ratio)
    return {'success': True, 'validation_result': validation_result}

@app.route('/debug_session', methods=['GET'])
@api_response
def debug_session():
    """Debug endpoint to check session data."""
    return {
        'session_keys': list(session.keys()),
        'filepath': session.get('filepath'),
        'file_columns': session.get('file_columns'),
        'file_shape': session.get('file_shape'),
        'best_model_name': session.get('best_model_name'),
        'target': session.get('target'),
        'ptype': session.get('ptype')
    }

@app.route('/reset_session', methods=['POST'])
@api_response
def reset_session():
    """Reset session for debugging."""
    clear_session_cache()
    session.clear()
    return {'message': 'Session cleared successfully'}

@app.route('/download_model')
def download_model():
    """Download the best trained model with preprocessing artifacts."""
    model_path = session.get('model_path')
    preprocessors_path = session.get('preprocessors_path')
    
    if not model_path or not preprocessors_path:
        return "No complete model package available.", 404
    
    try:
        temp_dir = tempfile.mkdtemp()
        zip_path = os.path.join(temp_dir, 'complete_model.zip')
        
        with zipfile.ZipFile(zip_path, 'w') as zipf:
            zipf.write(model_path, 'model.joblib')
            zipf.write(preprocessors_path, 'preprocessors.joblib')
            zipf.writestr('README.md', '''# AutoML Model Package

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
```
''')
        
        return send_file(zip_path, as_attachment=True, download_name='complete_model.zip')
        
    except Exception as e:
        print(f"Error creating model package: {str(e)}")
        return f"Error creating model package: {str(e)}", 500

@app.route('/export_results')
def export_results():
    """Export training results as CSV."""
    try:
        all_results = session.get('history', [])
        best_model = session.get('best_model_name', 'Unknown')
        
        if not all_results:
            return "No training results available for export.", 404
        
        output = io.StringIO()
        writer = csv.writer(output)
        
        if all_results:
            metrics_keys = list(all_results[0].get('metrics', {}).keys())
            header = ['Model', 'Best_Model'] + metrics_keys
            writer.writerow(header)
            
            for result in all_results:
                model_name = result.get('model', 'Unknown')
                is_best = 'Yes' if model_name == best_model else 'No'
                metrics = result.get('metrics', {})
                row = [model_name, is_best] + [metrics.get(key, '') for key in metrics_keys]
                writer.writerow(row)
        
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

