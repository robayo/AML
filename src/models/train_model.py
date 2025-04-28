import numpy as np
import pandas as pd
import xgboost as xgb
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
from sklearn.metrics import (f1_score, precision_recall_curve,
    roc_auc_score, average_precision_score
)
# Function to add anomaly scores to a dataset
def add_anomaly_scores(X_data, iso_forest_model):
    anomaly_scores = iso_forest_model.decision_function(X_data)
    # Normalize scores to [0,1] where higher means more likely to be illicit
    normalized_scores = (anomaly_scores.max() - anomaly_scores) / (anomaly_scores.max() - anomaly_scores.min())
    X_with_scores = X_data.copy()
    X_with_scores['anomaly_score'] = normalized_scores
    return X_with_scores

# Function to find optimal F1 threshold
def find_optimal_f1_threshold(y_true, y_proba):
    """Find threshold that maximizes F1 score"""
    precision, recall, thresholds = precision_recall_curve(y_true, y_proba)
    # Add a small epsilon to avoid division by zero
    f1_scores = 2 * precision * recall / (precision + recall + 1e-7)
    # Handle the last precision/recall point
    f1_scores = np.nan_to_num(f1_scores)
    optimal_idx = np.argmax(f1_scores)
    # The thresholds array has one less element than precision/recall
    if optimal_idx >= len(thresholds):
        optimal_threshold = 0.5  # Default if we can't determine
    else:
        optimal_threshold = thresholds[optimal_idx]

    # Plot the F1 scores vs thresholds
    plt.figure(figsize=(10, 6))
    plt.plot(thresholds, f1_scores[:-1], 'b-', label='F1 Score')
    plt.axvline(x=optimal_threshold, color='r', linestyle='--', label=f'Optimal Threshold: {optimal_threshold:.3f}')
    plt.xlabel('Threshold')
    plt.ylabel('F1 Score')
    plt.title('F1 Score vs Threshold')
    plt.legend()
    plt.grid(True)
    plt.show()

    return optimal_threshold, f1_scores[optimal_idx]


# Function for XGBoost cross-validation
def xgb_cv(params, dtrain, dval,  num_boost_round=1000, early_stopping_rounds=50):
    results = {}
    y_train = dtrain.get_label()
    # Fixed parameters
    fixed_params = {
        'objective': 'binary:logistic',
        'eval_metric': 'auc',
        'scale_pos_weight': (y_train == 0).sum() / (y_train == 1).sum()
    }
    
    # Combine with variable parameters
    full_params = {**fixed_params, **params}
    
    # Train with early stopping
    model = xgb.train(
        full_params,
        dtrain,
        num_boost_round=num_boost_round,
        evals=[(dtrain, 'train'), (dval, 'val')],
        early_stopping_rounds=early_stopping_rounds,
        verbose_eval=False
    )
    
    # Predict on validation set
    y_pred_proba = model.predict(dval)
    
    # Calculate metrics
    y_val = dval.get_label()
    auc = roc_auc_score(y_val, y_pred_proba)
    ap = average_precision_score(y_val, y_pred_proba)
    
    # Find optimal threshold for F1
    precision, recall, thresholds = precision_recall_curve(y_val, y_pred_proba)
    f1_scores = 2 * recall * precision / (recall + precision + 1e-7)
    f1_scores = np.nan_to_num(f1_scores)
    optimal_idx = np.argmax(f1_scores)
    if optimal_idx < len(thresholds):
        optimal_threshold = thresholds[optimal_idx]
    else:
        optimal_threshold = 0.5
    
    # Calculate F1 with optimal threshold
    y_pred = (y_pred_proba > optimal_threshold).astype(int)
    f1 = f1_score(y_val, y_pred)
    
    results['auc'] = auc
    results['ap'] = ap
    results['f1'] = f1
    results['threshold'] = optimal_threshold
    results['model'] = model
    results['best_iteration'] = model.best_iteration
    
    return results
