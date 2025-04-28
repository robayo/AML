from sklearn.metrics import (
    classification_report, confusion_matrix, f1_score,
    roc_auc_score, average_precision_score, precision_score, recall_score
)
import numpy as np
import pandas as pd
import datetime
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from sklearn.metrics import f1_score, precision_score, recall_score
import xgboost as xgb
import time
from src.models.train_model import  add_anomaly_scores
# Enhanced evaluation function with multiple metrics
def evaluate_model(model, X, y, model_name, threshold=0.5, is_isolation_forest=False, anomaly_score_col=None):
    """
    Comprehensive evaluation with multiple metrics
    """
    if is_isolation_forest:
        # For isolation forest: -1 for anomalies (illicit), 1 for normal (licit)
        raw_scores = model.decision_function(X)
        # Convert to probability-like scores (higher = more likely to be illicit)
        y_proba = (raw_scores.max() - raw_scores) / (raw_scores.max() - raw_scores.min())
        y_pred = model.predict(X)
        # Convert from 1/-1 to 0/1 (0=licit, 1=illicit)
        y_pred = (y_pred == -1).astype(int)
    else:
        if anomaly_score_col:
            # Use X with anomaly score column
            X_with_score = X.copy()
            dmatrix = xgb.DMatrix(X_with_score)
        else:
            # Standard XGBoost prediction
            dmatrix = xgb.DMatrix(X)
        
        y_proba = model.predict(dmatrix)
        y_pred = (y_proba > threshold).astype(int)

    # Calculate comprehensive metrics
    tn, fp, fn, tp = confusion_matrix(y, y_pred).ravel()
    precision = precision_score(y, y_pred)
    recall = recall_score(y, y_pred)
    f1 = f1_score(y, y_pred)
    specificity = tn / (tn + fp)
    
    # Micro and macro averages
    precision_micro = precision_score(y, y_pred, average='micro')
    recall_micro = recall_score(y, y_pred, average='micro')
    f1_micro = f1_score(y, y_pred, average='micro')
    
    precision_macro = precision_score(y, y_pred, average='macro')
    recall_macro = recall_score(y, y_pred, average='macro')
    f1_macro = f1_score(y, y_pred, average='macro')
    
    # AUC and Average Precision
    auc_score = roc_auc_score(y, y_proba)
    ap_score = average_precision_score(y, y_proba)
    
    # Class-specific metrics
    class_report = classification_report(y, y_pred)
    conf_matrix = confusion_matrix(y, y_pred)

    print(f"\n{model_name} Results (threshold={threshold:.3f}):")
    print(f"Overall Metrics:")
    print(f"  F1 Score: {f1:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall (Sensitivity): {recall:.4f}")
    print(f"  Specificity: {specificity:.4f}")
    print(f"  AUC: {auc_score:.4f}")
    print(f"  Average Precision: {ap_score:.4f}")
    print(f"\nMicro & Macro Averages:")
    print(f"  Micro F1: {f1_micro:.4f}, Precision: {precision_micro:.4f}, Recall: {recall_micro:.4f}")
    print(f"  Macro F1: {f1_macro:.4f}, Precision: {precision_macro:.4f}, Recall: {recall_macro:.4f}")
    print(f"\nConfusion Matrix:")
    print(conf_matrix)
    print(f"\nDetailed Classification Report:")
    print(class_report)
    
    # Create illicit detection metrics specifically
    illicit_precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    illicit_recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    illicit_f1 = 2 * illicit_precision * illicit_recall / (illicit_precision + illicit_recall) if (illicit_precision + illicit_recall) > 0 else 0
    
    print(f"\nIllicit Detection Performance:")
    print(f"  Precision (% of flagged transactions that are truly illicit): {illicit_precision:.4f}")
    print(f"  Recall (% of illicit transactions successfully caught): {illicit_recall:.4f}")
    print(f"  F1 Score: {illicit_f1:.4f}")
    
    return {
        'threshold': threshold,
        'f1': f1,
        'precision': precision,
        'recall': recall,
        'specificity': specificity,
        'auc': auc_score,
        'ap': ap_score,
        'f1_micro': f1_micro,
        'f1_macro': f1_macro,
        'illicit_precision': illicit_precision,
        'illicit_recall': illicit_recall,
        'illicit_f1': illicit_f1,
        'y_pred': y_pred,
        'y_proba': y_proba,
        'conf_matrix': conf_matrix,
        'tp': tp, 'fp': fp, 'tn': tn, 'fn': fn
    }
    

def prediction_pipeline(new_data, iso_forest_model, xgb_model, threshold, explainer=None):
    """
    Complete prediction pipeline for new data with optional explanation
    
    Parameters:
    -----------
    new_data : DataFrame
        New data to predict on (should have the same features as training data)
    iso_forest_model : IsolationForest
        Trained isolation forest model
    xgb_model : XGBoost Booster
        Trained XGBoost model
    threshold : float
        Classification threshold
    explainer : SHAP TreeExplainer, optional
        If provided, will generate explanations for predictions
        
    Returns:
    --------
    DataFrame with predictions, probabilities, and explanations if requested
    """
    start_time = time.time()
    
    # Add anomaly scores
    new_data_with_scores = add_anomaly_scores(new_data, iso_forest_model)
    
    # Make predictions
    dmatrix = xgb.DMatrix(new_data_with_scores)
    probabilities = xgb_model.predict(dmatrix)
    predictions = (probabilities > threshold).astype(int)
    
    # Create results dataframe
    results = pd.DataFrame({
        'probability': probabilities,
        'prediction': ['illicit' if p else 'licit' for p in predictions],
        'anomaly_score': new_data_with_scores['anomaly_score']
    }, index=new_data.index)
    
   # Add explanations if explainer is provided
    if explainer is not None:
        # For efficiency, only explain a subset if dataset is large
        if len(new_data) > 1000:
            print("Warning: Large dataset detected. Only providing explanations for illicit predictions.")
            explain_indices = results[results['prediction'] == 'illicit'].index
            if len(explain_indices) > 100:
                explain_indices = np.random.choice(explain_indices, 100, replace=False)
        else:
            explain_indices = new_data.index
        
        # Generate explanations
        explanations = {}
        for idx in explain_indices:
            transaction = new_data_with_scores.loc[idx:idx]
            shap_values = explainer.shap_values(transaction)
            
            # Get feature contributions
            contributions = []
            for i, col in enumerate(transaction.columns):
                contributions.append({
                    'feature': col,
                    'value': float(transaction[col].values[0]),
                    'shap_value': float(shap_values[0][i]),
                    'direction': 'increases' if shap_values[0][i] > 0 else 'decreases'
                })
            
            # Sort by absolute contribution
            contributions.sort(key=lambda x: abs(x['shap_value']), reverse=True)
            explanations[idx] = {
                'top_features': contributions[:5],
                'all_features': contributions
            }
        
        results['explanation'] = pd.Series(explanations)
    
    end_time = time.time()
    processing_time = end_time - start_time
    
    # Add processing metadata
    processing_metadata = {
        'total_transactions': len(new_data),
        'illicit_predictions': (results['prediction'] == 'illicit').sum(),
        'processing_time_seconds': processing_time,
        'transactions_per_second': len(new_data) / processing_time
    }
    
    return results, processing_metadata, new_data_with_scores


# Create a function to explain individual predictions
def explain_prediction(transaction_data, model, explainer, final_threshold=0.8):
    """
    Explain a single prediction with SHAP values
    
    Parameters:
    -----------
    transaction_data : DataFrame row
        Single transaction data
    model : XGBoost model
        Trained model
    explainer : SHAP explainer
        Initialized SHAP explainer
    
    Returns:
    --------
    Dictionary with prediction and explanation
    """
    # Ensure transaction_data is a DataFrame
    if isinstance(transaction_data, pd.Series):
        transaction_data = transaction_data.to_frame().T
    
    # Make prediction
    dmatrix = xgb.DMatrix(transaction_data)
    probability = model.predict(dmatrix)[0]
    prediction = 'illicit' if probability > final_threshold else 'licit'
    
    # Get SHAP values
    shap_values = explainer.shap_values(transaction_data)
    
    # Get top contributing features
    feature_contributions = []
    for i in range(len(transaction_data.columns)):
        feature_name = transaction_data.columns[i]
        shap_value = shap_values[0][i]
        feature_contributions.append((feature_name, shap_value))
    
    # Sort by absolute contribution
    feature_contributions.sort(key=lambda x: abs(x[1]), reverse=True)
    
    return {
        'prediction': prediction,
        'probability': probability,
        'top_features': feature_contributions[:5],  # Top 5 contributing features
        'all_contributions': feature_contributions
    }

def batch_prediction_pipeline(data, iso_forest_model, xgb_model, threshold, 
                             batch_size=10000, n_jobs=-1, explain=False):
    """
    Process large datasets in batches for memory efficiency and parallelization
    
    Parameters:
    -----------
    data : DataFrame or path to parquet files
        Data to process, either as DataFrame or directory with parquet files
    iso_forest_model : IsolationForest
        Trained isolation forest model
    xgb_model : XGBoost Booster
        Trained XGBoost model
    threshold : float
        Classification threshold
    batch_size : int
        Number of transactions to process per batch
    n_jobs : int
        Number of parallel jobs for anomaly detection
    explain : bool
        Whether to generate SHAP explanations (significantly slower)
        
    Returns:
    --------
    DataFrame with predictions and processing metadata
    """
    start_time = time.time()
    all_results = []
    total_transactions = 0
    illicit_count = 0
    
    # Initialize explainer if needed
    explainer = None
    if explain:
        explainer = shap.TreeExplainer(xgb_model)
    
    # Handle directory of parquet files
    if isinstance(data, str) and os.path.isdir(data):
        files = [f for f in os.listdir(data) if f.endswith('.parquet')]
        print(f"Found {len(files)} parquet files in {data}")
        
        for i, file in enumerate(files):
            print(f"Processing file {i+1}/{len(files)}: {file}")
            df = pd.read_parquet(os.path.join(data, file))
            
            # Process this file in batches
            for batch_start in range(0, len(df), batch_size):
                batch_end = min(batch_start + batch_size, len(df))
                batch = df.iloc[batch_start:batch_end]
                
                batch_results, batch_meta = prediction_pipeline(
                    batch, iso_forest_model, xgb_model, threshold, 
                    explainer if explain else None
                )
                
                all_results.append(batch_results)
                total_transactions += batch_meta['total_transactions']
                illicit_count += batch_meta['illicit_predictions']
                
                # Print progress
                print(f"  Batch {batch_start//batch_size + 1}: {batch_meta['illicit_predictions']} illicit out of {batch_meta['total_transactions']} transactions")
    
    # Handle DataFrame input
    else:
        if not isinstance(data, pd.DataFrame):
            raise ValueError("data must be a DataFrame or a directory path")
        
        # Process DataFrame in batches
        for batch_start in range(0, len(data), batch_size):
            batch_end = min(batch_start + batch_size, len(data))
            batch = data.iloc[batch_start:batch_end]
            
            batch_results, batch_meta = prediction_pipeline(
                batch, iso_forest_model, xgb_model, threshold,
                explainer if explain else None
            )
            
            all_results.append(batch_results)
            total_transactions += batch_meta['total_transactions']
            illicit_count += batch_meta['illicit_predictions']
            
            # Print progress
            print(f"Batch {batch_start//batch_size + 1}: {batch_meta['illicit_predictions']} illicit out of {batch_meta['total_transactions']} transactions")
    
    # Combine all results
    combined_results = pd.concat(all_results)
    end_time = time.time()
    total_time = end_time - start_time
    
    # Final metadata
    final_metadata = {
        'total_transactions': total_transactions,
        'illicit_predictions': illicit_count,
        'illicit_percentage': illicit_count / total_transactions * 100,
        'total_processing_time': total_time,
        'transactions_per_second': total_transactions / total_time
    }
    
    print(f"\nProcessed {final_metadata['total_transactions']} transactions in {final_metadata['total_processing_time']:.2f} seconds")
    print(f"Overall speed: {final_metadata['transactions_per_second']:.2f} transactions per second")
    print(f"Flagged {final_metadata['illicit_predictions']} transactions as potentially illicit ({final_metadata['illicit_percentage']:.2f}%)")
    
    return combined_results, final_metadata



# Function to evaluate illicit detection metrics by time period
def evaluate_by_time(data, y_true, y_pred_proba, threshold, time_column, time_freq='D'):
    """
    Evaluate model performance across time periods
    
    Parameters:
    -----------
    data : DataFrame
        DataFrame containing the data
    y_true : Series
        True labels (0 for licit, 1 for illicit)
    y_pred_proba : array-like
        Predicted probabilities for illicit class
    threshold : float
        Decision threshold for classification
    time_column : str
        Name of the column containing timestamps
    time_freq : str
        Time frequency for grouping ('D' for day, 'W' for week, 'M' for month)
    
    Returns:
    --------
    DataFrame with performance metrics by time period
    """
    # Convert predictions to binary using threshold
    y_pred = (y_pred_proba > threshold).astype(int)
    
    # Create a DataFrame with actual labels, predictions, and time
    eval_df = pd.DataFrame({
        'time': data[time_column],
        'y_true': y_true,
        'y_pred': y_pred,
        'y_prob': y_pred_proba
    })
    
    
    # Group by time period and calculate metrics
    results = []
    
    # For each time period
    for period, group in eval_df.groupby('time'):
        # Skip periods with too few samples
        if len(group) < 10 or sum(group['y_true']) < 2:
            continue
            
        # Calculate metrics
        period_metrics = {
            'period': period,
            'samples': len(group),
            'illicit_count': sum(group['y_true']),
            'illicit_pct': sum(group['y_true']) / len(group) * 100,
            'precision': precision_score(group['y_true'], group['y_pred']),
            'recall': recall_score(group['y_true'], group['y_pred']),
            'f1': f1_score(group['y_true'], group['y_pred']),
            'avg_prob': group['y_prob'].mean()
        }
        results.append(period_metrics)
    
    # Convert to DataFrame
    results_df = pd.DataFrame(results)
    
    # Sort by time period
    results_df = results_df.sort_values('period')
    
    return results_df
# Function to plot performance over time using Plotly
def plot_performance_over_time(time_metrics, metric='f1', 
                              title=None, plot_illicit_pct=True):
    """
    Plot model performance metrics over time using Plotly
    
    Parameters:
    -----------
    time_metrics : DataFrame
        DataFrame from evaluate_by_time function
    metric : str
        Metric to plot ('f1', 'precision', or 'recall')
    title : str, optional
        Plot title
    plot_illicit_pct : bool
        Whether to plot illicit percentage as secondary axis
    """
    # Create figure with secondary y-axis
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    # Add main metric trace
    fig.add_trace(
        go.Scatter(
            x=time_metrics['period'],
            y=time_metrics[metric],
            mode='lines+markers',
            name=f'Illicit {metric.capitalize()}',
            marker=dict(size=8, color='blue'),
            line=dict(width=2, color='blue'),
            hovertemplate='<b>Date</b>: %{x|%Y-%m-%d}<br>'
                         f'<b>Illicit {metric.capitalize()}</b>: %{{y:.4f}}<br>'
                         '<extra></extra>'
        ),
        secondary_y=False
    )
    
    # Add illicit percentage if requested
    if plot_illicit_pct:
        fig.add_trace(
            go.Scatter(
                x=time_metrics['period'],
                y=time_metrics['illicit_pct'],
                mode='lines+markers',
                name='Illicit %',
                marker=dict(size=6, color='red'),
                line=dict(width=1.5, dash='dash', color='red'),
                hovertemplate='<b>Date</b>: %{x|%Y-%m-%d}<br>'
                             '<b>Illicit %</b>: %{y:.2f}%<br>'
                             '<b>Count</b>: ' + time_metrics['illicit_count'].astype(str) + ' / ' + 
                             time_metrics['samples'].astype(str) + 
                             '<extra></extra>'
            ),
            secondary_y=True
        )
    
    # Set figure layout
    if title:
        fig.update_layout(title=title)
    else:
        fig.update_layout(title=f'Illicit {metric.capitalize()} Score Over Time')
    
    fig.update_layout(
        hovermode="x unified",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        template="plotly_white"
    )
    
    # Set y-axes titles
    fig.update_yaxes(title_text=f"Illicit {metric.capitalize()} Score", range=[0, 1], secondary_y=False)
    if plot_illicit_pct:
        fig.update_yaxes(title_text="Illicit %", range=[0, max(time_metrics['illicit_pct'])*1.2], secondary_y=True)
    
    # Update x-axis
    fig.update_xaxes(title_text='Time Period')
    
    # Add grid
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
    
    return fig

# Function to compare multiple models over time using Plotly
def compare_models_over_time(data, y_true, time_column, model_preds, 
                            thresholds, model_names, time_freq='D',
                            metric='f1'):
    """
    Compare multiple models' performance over time using Plotly
    
    Parameters:
    -----------
    data : DataFrame
        DataFrame containing the data
    y_true : Series
        True labels (0 for licit, 1 for illicit)
    time_column : str
        Name of the column containing timestamps
    model_preds : list of arrays
        List of predicted probabilities from different models
    thresholds : list of floats
        Decision thresholds for each model
    model_names : list of str
        Names for each model
    time_freq : str
        Time frequency for grouping ('D' for day, 'W' for week, 'M' for month)
    metric : str
        Metric to plot ('f1', 'precision', or 'recall')
    """
    # Get metrics for each model
    model_metrics = []
    
    for i, (preds, threshold, name) in enumerate(zip(model_preds, thresholds, model_names)):
        metrics = evaluate_by_time(data, y_true, preds, threshold, time_column, time_freq)
        metrics['model'] = name
        model_metrics.append(metrics)
    
    # Combine metrics
    all_metrics = pd.concat(model_metrics)
    
    # Create figure with secondary y-axis
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    # Plot each model's metric
    colors = ['blue', 'green', 'red', 'purple', 'orange', 'brown']
    
    for i, name in enumerate(model_names):
        model_data = all_metrics[all_metrics['model'] == name]
        color = colors[i % len(colors)]
        
        fig.add_trace(
            go.Scatter(
                x=model_data['period'],
                y=model_data[metric],
                mode='lines+markers',
                name=name,
                marker=dict(size=8, color=color),
                line=dict(width=2, color=color),
                hovertemplate='<b>Model</b>: ' + name + '<br>'
                             '<b>Date</b>: %{x|%Y-%m-%d}<br>'
                             f'<b>{metric.capitalize()}</b>: %{{y:.4f}}<br>'
                             '<extra></extra>'
            ),
            secondary_y=False
        )
    
    # Add illicit percentage line (using the first model's data)
    first_model = model_metrics[0]
    fig.add_trace(
        go.Scatter(
            x=first_model['period'],
            y=first_model['illicit_pct'],
            mode='lines',
            name='Illicit %',
            line=dict(width=1.5, dash='dash', color='gray'),
            opacity=0.6,
            hovertemplate='<b>Date</b>: %{x|%Y-%m-%d}<br>'
                         '<b>Illicit %</b>: %{y:.2f}%<br>'
                         '<extra></extra>'
        ),
        secondary_y=True
    )
    
    # Set figure layout
    fig.update_layout(
        title=f'Comparison of Illicit {metric.capitalize()} Score Over Time',
        hovermode="x unified",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        template="plotly_white"
    )
    
    # Set y-axes titles
    fig.update_yaxes(title_text=f"Illicit {metric.capitalize()} Score", range=[0, 1], secondary_y=False)
    fig.update_yaxes(title_text="Illicit %", range=[0, max(first_model['illicit_pct'])*1.2], secondary_y=True)
    
    # Update x-axis
    fig.update_xaxes(title_text='Time Period')
    
    # Add grid
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
    
    return fig



# Function to analyze performance drift using Plotly
def analyze_performance_drift(time_metrics, metric='f1', window=3, threshold=0.1):
    """
    Analyze performance drift over time and detect significant changes
    
    Parameters:
    -----------
    time_metrics : DataFrame
        DataFrame from evaluate_by_time function
    metric : str
        Metric to analyze ('f1', 'precision', or 'recall')
    window : int
        Rolling window size for trend analysis
    threshold : float
        Threshold for significant change detection
    
    Returns:
    --------
    DataFrame with drift analysis and Plotly figure
    """
    # Need at least window+1 data points
    if len(time_metrics) <= window:
        print(f"Not enough time periods for drift analysis (need >{window}, got {len(time_metrics)})")
        return time_metrics, None
    
    # Calculate rolling statistics
    time_metrics['rolling_mean'] = time_metrics[metric].rolling(window=window, min_periods=1).mean()
    
    # Calculate changes
    time_metrics['change'] = time_metrics[metric] - time_metrics['rolling_mean'].shift(1)
    
    # Detect significant changes
    time_metrics['significant_change'] = abs(time_metrics['change']) > threshold
    time_metrics['improvement'] = (time_metrics['change'] > threshold)
    time_metrics['degradation'] = (time_metrics['change'] < -threshold)
    
    # Find periods with notable changes
    if sum(time_metrics['significant_change']) > 0:
        print("\nTime periods with significant performance changes:")
        for idx, row in time_metrics[time_metrics['significant_change']].iterrows():
            change_type = "improvement" if row['improvement'] else "degradation"
            print(f"  Period {row['period']}: {change_type} ({row[metric]:.3f}, change: {row['change']:.3f})")
    
    # Create Plotly figure for drift analysis
    fig = go.Figure()
    
    # Add main metric line
    fig.add_trace(
        go.Scatter(
            x=time_metrics['period'],
            y=time_metrics[metric],
            mode='lines+markers',
            name=f'{metric.capitalize()} Score',
            marker=dict(size=8, color='blue'),
            line=dict(width=2, color='blue'),
            hovertemplate='<b>Date</b>: %{x|%Y-%m-%d}<br>'
                         f'<b>{metric.capitalize()}</b>: %{{y:.4f}}<br>'
                         '<extra></extra>'
        )
    )
    
    # Add rolling average
    fig.add_trace(
        go.Scatter(
            x=time_metrics['period'],
            y=time_metrics['rolling_mean'],
            mode='lines',
            name='Rolling Average',
            line=dict(width=1.5, dash='dot', color='gray'),
            hovertemplate='<b>Date</b>: %{x|%Y-%m-%d}<br>'
                         '<b>Rolling Avg</b>: %{y:.4f}<br>'
                         '<extra></extra>'
        )
    )
    
    # Add improvements as markers
    improvements = time_metrics[time_metrics['improvement']]
    if len(improvements) > 0:
        fig.add_trace(
            go.Scatter(
                x=improvements['period'],
                y=improvements[metric],
                mode='markers',
                name='Improvement',
                marker=dict(
                    symbol='triangle-up',
                    size=12,
                    color='green',
                    line=dict(width=1, color='darkgreen')
                ),
                hovertemplate='<b>Date</b>: %{x|%Y-%m-%d}<br>'
                             f'<b>{metric.capitalize()}</b>: %{{y:.4f}}<br>'
                             '<b>Change</b>: +%{text:.4f}<br>'
                             '<extra></extra>',
                text=improvements['change'].abs()
            )
        )
    
    # Add degradations as markers
    degradations = time_metrics[time_metrics['degradation']]
    if len(degradations) > 0:
        fig.add_trace(
            go.Scatter(
                x=degradations['period'],
                y=degradations[metric],
                mode='markers',
                name='Degradation',
                marker=dict(
                    symbol='triangle-down',
                    size=12,
                    color='red',
                    line=dict(width=1, color='darkred')
                ),
                hovertemplate='<b>Date</b>: %{x|%Y-%m-%d}<br>'
                             f'<b>{metric.capitalize()}</b>: %{{y:.4f}}<br>'
                             '<b>Change</b>: -%{text:.4f}<br>'
                             '<extra></extra>',
                text=degradations['change'].abs()
            )
        )
    
    # Set figure layout
    fig.update_layout(
        title=f'{metric.capitalize()} Score Drift Analysis Over Time',
        xaxis_title='Time Period',
        yaxis_title=f'{metric.capitalize()} Score',
        yaxis=dict(range=[0, 1]),
        hovermode="x unified",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        template="plotly_white"
    )
    
    # Add grid
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
    
    return time_metrics, fig