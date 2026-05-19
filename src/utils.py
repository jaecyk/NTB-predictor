"""
Utility Functions
Helper functions for data processing and model operations
"""

import os
import json
import pickle
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_directory(path: str) -> Path:
    """
    Create directory if it doesn't exist.
    
    Args:
        path: Directory path
        
    Returns:
        Path object
    """
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def save_pickle(obj, filepath: str):
    """
    Save object as pickle file.
    
    Args:
        obj: Object to save
        filepath: Output file path
    """
    with open(filepath, 'wb') as f:
        pickle.dump(obj, f)
    logger.info(f'Object saved to {filepath}')


def load_pickle(filepath: str):
    """
    Load object from pickle file.
    
    Args:
        filepath: Path to pickle file
        
    Returns:
        Loaded object
    """
    with open(filepath, 'rb') as f:
        obj = pickle.load(f)
    logger.info(f'Object loaded from {filepath}')
    return obj


def save_json(data: dict, filepath: str):
    """
    Save dictionary as JSON file.
    
    Args:
        data: Dictionary to save
        filepath: Output file path
    """
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=4)
    logger.info(f'Data saved to {filepath}')


def load_json(filepath: str) -> dict:
    """
    Load dictionary from JSON file.
    
    Args:
        filepath: Path to JSON file
        
    Returns:
        Loaded dictionary
    """
    with open(filepath, 'r') as f:
        data = json.load(f)
    logger.info(f'Data loaded from {filepath}')
    return data


def get_timestamp() -> str:
    """
    Get current timestamp as string.
    
    Returns:
        Timestamp string (YYYY-MM-DD HH:MM:SS)
    """
    return datetime.now().strftime('%Y-%m-%d %H:%M:%S')


def calculate_percentage_change(current: float, previous: float) -> float:
    """
    Calculate percentage change between two values.
    
    Args:
        current: Current value
        previous: Previous value
        
    Returns:
        Percentage change
    """
    if previous == 0:
        return 0
    return ((current - previous) / previous) * 100


def normalize_data(data: np.ndarray, method: str = 'minmax') -> np.ndarray:
    """
    Normalize data using specified method.
    
    Args:
        data: Input array
        method: 'minmax' or 'zscore'
        
    Returns:
        Normalized array
    """
    if method == 'minmax':
        return (data - np.min(data)) / (np.max(data) - np.min(data))
    elif method == 'zscore':
        return (data - np.mean(data)) / np.std(data)
    else:
        raise ValueError(f'Unknown normalization method: {method}')


def create_lag_features(df: pd.DataFrame, column: str, lags: list) -> pd.DataFrame:
    """
    Create lagged features from a column.
    
    Args:
        df: Input DataFrame
        column: Column to create lags from
        lags: List of lag periods
        
    Returns:
        DataFrame with lag features
    """
    for lag in lags:
        df[f'{column}_lag_{lag}'] = df[column].shift(lag)
    return df


def print_config(config: dict):
    """
    Pretty print configuration dictionary.
    
    Args:
        config: Configuration dictionary
    """
    print('\n' + '='*50)
    print('CONFIGURATION')
    print('='*50)
    for key, value in config.items():
        print(f'{key:<25} : {value}')
    print('='*50 + '\n')


if __name__ == '__main__':
    # Test utility functions
    print('Testing utility functions...')
    print(f'Current timestamp: {get_timestamp()}')
    print(f'Percentage change from 100 to 120: {calculate_percentage_change(120, 100):.2f}%')
    
    # Test normalization
    data = np.array([1, 2, 3, 4, 5])
    normalized = normalize_data(data, 'minmax')
    print(f'Normalized data (minmax): {normalized}')