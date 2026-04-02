"""
AutoML Training Module using PyCaret

This module handles:
- Automatic problem type detection (Classification vs Regression)
- Automatic model selection and training
- Model serialization to pickle format
"""

import pandas as pd
import numpy as np
from typing import Tuple, Dict, Any
import pickle
from pathlib import Path
import sys
import os
from contextlib import redirect_stdout, redirect_stderr
from pycaret.classification import setup as clf_setup, compare_models as clf_compare_models
from pycaret.regression import setup as reg_setup, compare_models as reg_compare_models
import warnings

warnings.filterwarnings('ignore')


def detect_problem_type(df: pd.DataFrame, target_column: str) -> str:
    """
    Automatically detect if the problem is Classification or Regression.
    
    Args:
        df: Input dataframe
        target_column: Name of the target column
        
    Returns:
        'classification' or 'regression'
    """
    target = df[target_column]
    
    # Get basic statistics
    n_unique = target.nunique()
    n_samples = len(target)
    
    # Heuristics for detection
    # 1. Check data type
    if target.dtype == 'object':
        return 'classification'
    
    # 2. Check if values are floats with decimals
    if target.dtype in ['float64', 'float32']:
        if (target % 1).sum() > 0:  # Has decimal values
            return 'regression'
    
    # 3. Check cardinality ratio
    cardinality_ratio = n_unique / n_samples
    
    # If very few unique values relative to samples, likely classification
    if cardinality_ratio < 0.05 or n_unique < 20:
        return 'classification'
    
    # Otherwise, regression
    return 'regression'


def train_automl_model(
    df: pd.DataFrame,
    target_column: str,
    problem_type: str = None,
    test_size: float = 0.2,
    verbose: bool = False
) -> Tuple[Dict[str, Any], str]:
    """
    Train an AutoML model using PyCaret.
    
    Args:
        df: Input dataframe (should be cleaned)
        target_column: Name of the target column
        problem_type: 'classification' or 'regression'. If None, will auto-detect.
        test_size: Train-test split ratio
        verbose: Whether to show PyCaret output
        
    Returns:
        Tuple of (results_dict, problem_type)
    """
    
    # Detect problem type if not provided
    if problem_type is None:
        problem_type = detect_problem_type(df, target_column)
    
    results = {
        'problem_type': problem_type,
        'target_column': target_column,
        'n_samples': len(df),
        'n_features': len(df.columns) - 1,
        'best_model': None,
        'best_model_name': None,
        'model_metrics': {}
    }
    
    try:
        # Suppress all output during training
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        sys.stdout = open(os.devnull, 'w')
        sys.stderr = open(os.devnull, 'w')
        
        try:
            if problem_type == 'classification':
                # Setup for classification
                clf_setup(
                    data=df,
                    target=target_column,
                    session_id=42,
                    verbose=False
                )
                
                # Compare models and get best
                best_model = clf_compare_models(n_select=1)
                if isinstance(best_model, list):
                    best_model = best_model[0]  # Get the first (and only) model from the list
                
                # Get model name
                best_model_name = type(best_model).__name__
                results['best_model_name'] = best_model_name
                results['best_model'] = best_model
                
                # Get metrics from the best model
                from pycaret.classification import pull
                metrics_df = pull()
                results['model_metrics'] = metrics_df.to_dict()
                
            else:  # regression
                # Setup for regression
                reg_setup(
                    data=df,
                    target=target_column,
                    session_id=42,
                    verbose=False
                )
                
                # Compare models and get best
                best_model = reg_compare_models(n_select=1)
                if isinstance(best_model, list):
                    best_model = best_model[0]  # Get the first (and only) model from the list
                
                # Get model name
                best_model_name = type(best_model).__name__
                results['best_model_name'] = best_model_name
                results['best_model'] = best_model
                
                # Get metrics from the best model
                from pycaret.regression import pull
                metrics_df = pull()
                results['model_metrics'] = metrics_df.to_dict()
        finally:
            # Restore stdout and stderr
            sys.stdout = old_stdout
            sys.stderr = old_stderr
    
    except Exception as e:
        results['error'] = str(e)
        # Provide more specific error messages
        error_str = str(e).lower()
        if 'memory' in error_str or 'ram' in error_str:
            raise Exception("Dataset too large for training. Try reducing the number of features or rows.")
        elif 'convergence' in error_str:
            raise Exception("Model training failed to converge. Try cleaning the data or using a different target column.")
        elif 'invalid' in error_str or 'value' in error_str:
            raise Exception("Invalid data detected. Please ensure all features are numeric and clean the data first.")
        elif 'target' in error_str:
            raise Exception("Issue with target column. Please ensure it contains valid values for the problem type.")
        else:
            raise Exception(f"Model training failed: {str(e)}")
    
    return results, problem_type


def save_model_pickle(model: Any, model_name: str, filepath: str) -> bool:
    """
    Save trained model to pickle format.
    
    Args:
        model: Trained model object
        model_name: Name of the model
        filepath: Path to save the pickle file
        
    Returns:
        True if successful, False otherwise
    """
    try:
        with open(filepath, 'wb') as f:
            pickle.dump(model, f)
        return True
    except Exception as e:
        print(f"Error saving model: {str(e)}")
        return False


def load_model_pickle(filepath: str) -> Any:
    """
    Load a trained model from pickle format.
    
    Args:
        filepath: Path to the pickle file
        
    Returns:
        Loaded model object
    """
    try:
        with open(filepath, 'rb') as f:
            model = pickle.load(f)
        return model
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        return None


def get_model_summary(results: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract a summary of model training results for display.
    
    Args:
        results: Results dictionary from train_automl_model
        
    Returns:
        Summary dictionary
    """
    summary = {
        'problem_type': results.get('problem_type', 'Unknown'),
        'target_column': results.get('target_column', 'Unknown'),
        'n_samples': results.get('n_samples', 0),
        'n_features': results.get('n_features', 0),
        'best_model': results.get('best_model_name', 'Unknown'),
        'status': 'Success' if 'error' not in results else 'Failed'
    }
    
    if 'error' in results:
        summary['error'] = results['error']
    
    return summary


def make_predictions(model: Any, input_data: pd.DataFrame) -> Dict[str, Any]:
    """
    Make predictions using a trained model.
    
    Args:
        model: Trained model object
        input_data: DataFrame with feature data (without target column)
        
    Returns:
        Dictionary with predictions and probabilities (if classification)
    """
    try:
        # Make predictions
        predictions = model.predict(input_data)
        
        result = {
            'predictions': predictions.tolist() if hasattr(predictions, 'tolist') else list(predictions),
            'n_predictions': len(predictions)
        }
        
        # If classification model, try to get prediction probabilities
        if hasattr(model, 'predict_proba'):
            try:
                probabilities = model.predict_proba(input_data)
                result['probabilities'] = probabilities.tolist()
                result['classes'] = model.classes_.tolist() if hasattr(model, 'classes_') else None
            except:
                pass
        
        return result
        
    except Exception as e:
        raise Exception(f"Prediction failed: {str(e)}")


def get_feature_columns(model: Any) -> list:
    """
    Extract feature column names from a trained model.
    
    Args:
        model: Trained model object
        
    Returns:
        List of feature column names
    """
    try:
        if hasattr(model, 'feature_names_in_'):
            return model.feature_names_in_.tolist()
        elif hasattr(model, 'n_features_in_'):
            # If we don't have feature names, return generic names
            return [f"feature_{i}" for i in range(model.n_features_in_)]
        else:
            return []
    except:
        return []
