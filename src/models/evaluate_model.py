from sklearn.metrics import (
    classification_report, confusion_matrix, f1_score,
    roc_auc_score, average_precision_score, precision_score, recall_score
)
import time
import pandas as pd
import xgboost as xgb
import numpy as np
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

# Example of using batch processing (commented out)
print("""
# Example usage of batch processing
batch_results, batch_metadata = batch_prediction_pipeline(
    'data/large_blockchain_data/',  # Directory with parquet files
    loaded_iso_forest,
    loaded_xgb_model,
    config['threshold'],
    batch_size=50000,
    explain=False  # Set to True for explanations (slower)
)
""")

print("\n=== PIPELINE COMPLETE ===")