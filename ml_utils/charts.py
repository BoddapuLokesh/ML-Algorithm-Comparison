"""
Complete chart generation utilities using Plotly for Python-based visualizations.
Replaces all Chart.js functionality with Python/Plotly equivalents.
"""
import plotly.graph_objects as go
import plotly.express as px
from typing import Dict, Any, List
import json


def create_feature_importance_chart(feature_importance: Dict[str, Dict[str, float]], 
                                  best_model_name: str) -> str:
    """Create a feature importance chart using Plotly."""
    if not feature_importance or best_model_name not in feature_importance:
        return "<div class='chart-placeholder'>No feature importance data available</div>"
    
    importance = feature_importance[best_model_name]
    if not importance:
        return "<div class='chart-placeholder'>No feature importance data for the best model</div>"
    
    sorted_features = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:10]
    features = [item[0] for item in sorted_features]
    values = [item[1] for item in sorted_features]
    
    fig = go.Figure(data=[
        go.Bar(
            x=values,
            y=features,
            orientation='h',
            marker=dict(color='#1FB8CD'),
            text=[f'{v:.4f}' for v in values],
            textposition='auto',
        )
    ])
    
    fig.update_layout(
        title=f'Feature Importance - {best_model_name}',
        xaxis_title='Importance Score',
        yaxis_title='Features',
        height=400,
        margin=dict(l=120, r=50, t=50, b=50),
        font=dict(size=12),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
    )
    
    config = {
        'displayModeBar': True,
        'displaylogo': False,
        'modeBarButtonsToRemove': ['pan2d', 'lasso2d', 'select2d'],
        'responsive': True
    }
    
    return fig.to_html(full_html=False, include_plotlyjs=False, config=config, div_id="featureImportanceChart")


def create_model_performance_chart(all_results: Dict[str, Dict[str, Any]]) -> str:
    """Create a model performance comparison chart."""
    if not all_results:
        return "<div class='chart-placeholder'>No model performance data available</div>"
    
    models = []
    accuracy_scores = []
    colors = ['#1FB8CD', '#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FECA57', '#FF9FF3']
    
    for model_name, metrics in all_results.items():
        models.append(model_name)
        score = metrics.get('Accuracy', metrics.get('R2', 0))
        accuracy_scores.append(score)
    
    fig = go.Figure(data=[
        go.Bar(
            x=models,
            y=accuracy_scores,
            marker=dict(color=colors[:len(models)]),
            text=[f'{score:.3f}' for score in accuracy_scores],
            textposition='auto',
        )
    ])
    
    fig.update_layout(
        title='Model Performance Comparison',
        xaxis_title='Models',
        yaxis_title='Score',
        height=400,
        margin=dict(l=50, r=50, t=50, b=100),
        font=dict(size=12),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        xaxis=dict(tickangle=45)
    )
    
    config = {
        'displayModeBar': True,
        'displaylogo': False,
        'modeBarButtonsToRemove': ['pan2d', 'lasso2d', 'select2d'],
        'responsive': True
    }
    
    return fig.to_html(full_html=False, include_plotlyjs=False, config=config, div_id="performanceChart")


def create_data_quality_chart(stats: Dict[str, Any]) -> str:
    """Create EDA data quality chart."""
    if not stats:
        return "<div class='chart-placeholder'>No data quality information available</div>"
    
    # Get data from the actual EDA stats structure
    total_rows = stats.get('rows', 0)
    total_cols = stats.get('cols', 0)
    total_values = total_rows * total_cols if total_rows and total_cols else 0
    
    # Calculate missing values from the missing dict
    missing_dict = stats.get('missing', {})
    total_missing = sum(missing_dict.values()) if missing_dict else 0
    complete_values = total_values - total_missing
    
    if total_values == 0:
        return "<div class='chart-placeholder'>No data to display</div>"
    
    labels = ['Complete Data', 'Missing Values']
    values = [complete_values, total_missing]
    colors = ['#1FB8CD', '#FF6B6B']
    
    fig = go.Figure(data=[go.Pie(
        labels=labels,
        values=values,
        hole=0.3,
        marker=dict(colors=colors),
        textinfo='label+percent+value',
        textposition='auto'
    )])
    
    fig.update_layout(
        title=f'Data Quality Overview<br><sub>{total_rows:,} rows × {total_cols} columns = {total_values:,} total values</sub>',
        height=350,
        font=dict(size=12),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=20, r=20, t=70, b=20)
    )
    
    config = {
        'displayModeBar': True,
        'displaylogo': False,
        'modeBarButtonsToRemove': ['pan2d', 'lasso2d', 'select2d'],
        'responsive': True
    }
    
    return fig.to_html(full_html=False, include_plotlyjs=False, config=config, div_id="qualityChart")


def create_column_types_chart(stats: Dict[str, Any]) -> str:
    """Create column types distribution chart."""
    if not stats:
        return "<div class='chart-placeholder'>No column type information available</div>"
    
    # Get data from the actual EDA stats structure
    numerics = stats.get('numerics', [])
    categoricals = stats.get('categoricals', [])
    
    if not numerics and not categoricals:
        return "<div class='chart-placeholder'>No column type data available</div>"
    
    labels = []
    values = []
    colors = ['#1FB8CD', '#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']
    
    if numerics:
        labels.append('Numeric')
        values.append(len(numerics))
    
    if categoricals:
        labels.append('Categorical')
        values.append(len(categoricals))
    
    fig = go.Figure(data=[go.Bar(
        x=labels,
        y=values,
        marker=dict(color=colors[:len(labels)]),
        text=values,
        textposition='auto',
    )])
    
    fig.update_layout(
        title=f'Column Types Distribution<br><sub>Total: {len(numerics) + len(categoricals)} columns</sub>',
        xaxis_title='Data Types',
        yaxis_title='Count',
        height=350,
        font=dict(size=12),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=50, r=50, t=70, b=50)
    )
    
    config = {
        'displayModeBar': True,
        'displaylogo': False,
        'modeBarButtonsToRemove': ['pan2d', 'lasso2d', 'select2d'],
        'responsive': True
    }
    
    return fig.to_html(full_html=False, include_plotlyjs=False, config=config, div_id="typesChart")


def create_missing_values_chart(stats: Dict[str, Any]) -> str:
    """Create missing values by column chart."""
    if not stats:
        return "<div class='chart-placeholder'>No missing values information available</div>"
    
    # Get missing values per column from stats
    missing_dict = stats.get('missing', {})
    
    if not missing_dict:
        return "<div class='chart-placeholder'>No missing values data available</div>"
    
    # Filter to only show columns with missing values
    columns_with_missing = {col: count for col, count in missing_dict.items() if count > 0}
    
    if not columns_with_missing:
        fig = go.Figure(data=[go.Bar(
            x=['Complete Dataset'],
            y=[0],
            marker=dict(color='#1FB8CD'),
            text=['No Missing Values ✓'],
            textposition='auto',
        )])
        
        fig.update_layout(
            title='Missing Values by Column<br><sub>All columns are complete!</sub>',
            xaxis_title='Status',
            yaxis_title='Missing Count',
            height=350,
            font=dict(size=12),
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            margin=dict(l=50, r=50, t=70, b=50)
        )
    else:
        # Show columns with missing values
        columns = list(columns_with_missing.keys())
        missing_counts = list(columns_with_missing.values())
        
        fig = go.Figure(data=[go.Bar(
            x=columns,
            y=missing_counts,
            marker=dict(color='#FF6B6B'),
            text=[f'{count} missing' for count in missing_counts],
            textposition='auto',
        )])
        
        total_missing = sum(missing_counts)
        fig.update_layout(
            title=f'Missing Values by Column<br><sub>Total missing: {total_missing:,} values across {len(columns)} columns</sub>',
            xaxis_title='Columns',
            yaxis_title='Missing Count',
            height=350,
            font=dict(size=12),
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            margin=dict(l=50, r=50, t=70, b=100),
            xaxis=dict(tickangle=45)
        )
    
    config = {
        'displayModeBar': True,
        'displaylogo': False,
        'modeBarButtonsToRemove': ['pan2d', 'lasso2d', 'select2d'],
        'responsive': True
    }
    
    return fig.to_html(full_html=False, include_plotlyjs=False, config=config, div_id="missingChart")
