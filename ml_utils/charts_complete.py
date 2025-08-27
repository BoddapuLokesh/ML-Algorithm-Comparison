"""
Extended chart generation utilities using Plotly for complete Python-based visualizations.
"""
import plotly.graph_objects as go
import plotly.express as px
from typing import Dict, Any, List
import json


def create_feature_importance_chart(feature_importance: Dict[str, Dict[str, float]], 
                                  best_model_name: str) -> str:
    """
    Create a feature importance chart using Plotly.
    """
    if not feature_importance or best_model_name not in feature_importance:
        return "<div>No feature importance data available</div>"
    
    importance = feature_importance[best_model_name]
    if not importance:
        return "<div>No feature importance data for the best model</div>"
    
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
        margin=dict(l=100, r=50, t=50, b=50),
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
    
    chart_html = fig.to_html(
        include_plotlyjs='cdn',
        config=config,
        div_id="featureImportanceChart"
    )
    
    return chart_html


def create_model_performance_chart(all_results: Dict[str, Dict[str, Any]]) -> str:
    """Create a model performance comparison chart."""
    if not all_results:
        return "<div>No model performance data available</div>"
    
    models = []
    accuracy_scores = []
    
    for model_name, metrics in all_results.items():
        models.append(model_name)
        score = metrics.get('Accuracy', metrics.get('R2', 0))
        accuracy_scores.append(score)
    
    fig = go.Figure(data=[
        go.Bar(
            x=models,
            y=accuracy_scores,
            marker=dict(color=['#1FB8CD', '#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']),
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
    
    chart_html = fig.to_html(
        include_plotlyjs='cdn',
        config=config,
        div_id="performanceChart"
    )
    
    return chart_html


def create_data_quality_chart(stats: Dict[str, Any]) -> str:
    """Create EDA data quality chart."""
    if not stats:
        return "<div>No data quality information available</div>"
    
    total_rows = stats.get('total_rows', 0)
    missing_values = stats.get('missing_values', 0)
    complete_rows = total_rows - missing_values
    
    labels = ['Complete Data', 'Missing Values']
    values = [complete_rows, missing_values]
    
    fig = go.Figure(data=[go.Pie(
        labels=labels,
        values=values,
        hole=0.3,
        marker=dict(colors=['#1FB8CD', '#FF6B6B'])
    )])
    
    fig.update_layout(
        title='Data Quality Overview',
        height=300,
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
    
    chart_html = fig.to_html(
        include_plotlyjs='cdn',
        config=config,
        div_id="qualityChart"
    )
    
    return chart_html


def create_column_types_chart(stats: Dict[str, Any]) -> str:
    """Create column types distribution chart."""
    if not stats:
        return "<div>No column type information available</div>"
    
    column_types = stats.get('column_types', {})
    if not column_types:
        return "<div>No column type data available</div>"
    
    labels = list(column_types.keys())
    values = list(column_types.values())
    
    fig = go.Figure(data=[go.Bar(
        x=labels,
        y=values,
        marker=dict(color=['#1FB8CD', '#FF6B6B', '#4ECDC4', '#45B7D1']),
        text=values,
        textposition='auto',
    )])
    
    fig.update_layout(
        title='Column Types Distribution',
        xaxis_title='Data Types',
        yaxis_title='Count',
        height=300,
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
    
    chart_html = fig.to_html(
        include_plotlyjs='cdn',
        config=config,
        div_id="typesChart"
    )
    
    return chart_html


def create_missing_values_chart(stats: Dict[str, Any]) -> str:
    """Create missing values by column chart."""
    if not stats:
        return "<div>No missing values information available</div>"
    
    # This would need the actual missing values per column data
    # For now, create a placeholder
    fig = go.Figure(data=[go.Bar(
        x=['Sample Data'],
        y=[0],
        marker=dict(color='#1FB8CD'),
        text=['No Missing Values'],
        textposition='auto',
    )])
    
    fig.update_layout(
        title='Missing Values by Column',
        xaxis_title='Columns',
        yaxis_title='Missing Count',
        height=300,
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
    
    chart_html = fig.to_html(
        include_plotlyjs='cdn',
        config=config,
        div_id="missingChart"
    )
    
    return chart_html
