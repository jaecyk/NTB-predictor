"""
NTB Stop Rates Model Training Pipeline
======================================

This script trains XGBoost models for predicting NTB stop rates for each tenor.
It loads your NTB auction data, engineers features, trains models, and saves them as .pkl files.

Usage:
    python train_models.py --data data/Primary_Market_in_Excel_2.csv
    
    Or place your CSV in data/raw/ and run:
    python train_models.py
"""

import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from datetime import datetime
import logging
from typing import Tuple, Dict
import warnings

# ML imports
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

warnings.filterwarnings('ignore')

# =========================================================
# LOGGING SETUP
# =========================================================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# =========================================================
# CONSTANTS
# =========================================================
TENORS = [91, 182, 364]
OUTPUT_DIR = Path('models')
DATA_DIR = Path('data/raw')

FEATURE_ORDER = [
    "lag1_stop",
    "lag2_stop",
    "lag3_stop",
    "ma3_stop",
    "delta_stop_1",
    "offer_amt",
    "offer_change",
    "prev_bid_cover",
    "sec_rate",
    "sec_rate_change_5d",
    "sec_minus_lag1",
    "system_liquidity",
    "mpr",
    "inflation",
]

# =========================================================
# DATA LOADING & PREPROCESSING
# =========================================================
def load_ntb_data(filepath: str = None) -> pd.DataFrame:
    """Load NTB auction data from CSV."""
    if filepath is None:
        # Find CSV files in data/raw directory
        csv_files = list(DATA_DIR.glob('*.csv'))
        if not csv_files:
            logger.error(f"No CSV files found in {DATA_DIR}")
            return None
        filepath = csv_files[0]
    
    logger.info(f"Loading data from {filepath}")
    df = pd.read_csv(filepath)
    
    # Standardize column names
    df.columns = df.columns.str.lower().str.strip()
    
    logger.info(f"Loaded {len(df)} records with columns: {df.columns.tolist()}")
    return df

def preprocess_ntb_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean and prepare NTB data."""
    logger.info("Preprocessing data...")
    
    # Parse dates
    df['auctiondate'] = pd.to_datetime(df['auctiondate'], format='%d/%m/%Y', errors='coerce')
    
    # Parse tenor to days
    def extract_tenor_days(tenor_str):
        if pd.isna(tenor_str):
            return np.nan
        tenor_str = str(tenor_str).upper().strip()
        days = ''.join(filter(str.isdigit, tenor_str))
        return int(days) if days else np.nan
    
    df['tenor_days'] = df['tenor'].apply(extract_tenor_days)
    
    # Rename key columns
    column_mapping = {
        'rate': 'stop_rate',
        'successfulbidrates': 'stop_rate_range',
        'amtoffered': 'amount_offered',
        'totalsuccessful': 'amount_subscribed',
    }
    
    df = df.rename(columns=column_mapping)
    
    # Convert numeric columns
    numeric_cols = ['stop_rate', 'amount_offered', 'amount_subscribed', 'tenor_days']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Remove rows with missing stop rates
    df = df.dropna(subset=['stop_rate', 'tenor_days'])
    
    # Remove outliers (stop rates should be reasonable - e.g., 0-50%)
    df = df[(df['stop_rate'] > 0) & (df['stop_rate'] < 50)]
    
    # Sort by date
    df = df.sort_values('auctiondate').reset_index(drop=True)
    
    logger.info(f"Preprocessed data shape: {df.shape}")
    return df

# =========================================================
# FEATURE ENGINEERING
# =========================================================
def engineer_features_for_tenor(df: pd.DataFrame, tenor: int) -> pd.DataFrame:
    """
    Engineer features for a specific tenor.
    Creates lagged, rolling, and derived features.
    """
    tenor_df = df[df['tenor_days'] == tenor].copy().reset_index(drop=True)
    
    if len(tenor_df) < 10:
        logger.warning(f"Only {len(tenor_df)} records for {tenor}D tenor - insufficient for training")
        return None
    
    logger.info(f"Engineering features for {tenor}D tenor ({len(tenor_df)} records)")
    
    # Target variable
    tenor_df['target'] = tenor_df['stop_rate']
    
    # Lagged features
    for lag in [1, 2, 3]:
        tenor_df[f'lag{lag}_stop'] = tenor_df['stop_rate'].shift(lag)
    
    # Moving averages
    tenor_df['ma3_stop'] = tenor_df['stop_rate'].rolling(window=3, min_periods=1).mean()
    tenor_df['ma7_stop'] = tenor_df['stop_rate'].rolling(window=7, min_periods=1).mean()
    
    # Momentum
    tenor_df['delta_stop_1'] = tenor_df['stop_rate'].diff(1)
    tenor_df['delta_stop_3'] = tenor_df['stop_rate'].diff(3)
    
    # Amount offered features
    tenor_df['offer_amt'] = pd.to_numeric(tenor_df['amount_offered'], errors='coerce')
    tenor_df['offer_change'] = tenor_df['offer_amt'].diff(1)
    tenor_df['offer_ma3'] = tenor_df['offer_amt'].rolling(window=3, min_periods=1).mean()
    
    # Demand features
    tenor_df['bid_cover'] = (tenor_df['amount_subscribed'] / tenor_df['amount_offered']).fillna(1.0)
    tenor_df['prev_bid_cover'] = tenor_df['bid_cover'].shift(1)
    
    # Synthetic market features (using secondary market proxy)
    tenor_df['sec_rate'] = tenor_df['stop_rate'].rolling(window=3, min_periods=1).mean() + np.random.normal(0, 0.1, len(tenor_df))
    tenor_df['sec_rate'] = tenor_df['sec_rate'].clip(0, 50)
    
    tenor_df['sec_rate_change_5d'] = tenor_df['sec_rate'].diff(5)
    tenor_df['sec_minus_lag1'] = tenor_df['sec_rate'] - tenor_df['lag1_stop']
    
    # Macro features (create synthetic/default values if not available)
    tenor_df['system_liquidity'] = np.random.normal(2780, 300, len(tenor_df)).clip(500, 5000)
    tenor_df['mpr'] = np.random.normal(26.5, 1.0, len(tenor_df)).clip(15, 35)
    tenor_df['inflation'] = np.random.normal(15.0, 2.0, len(tenor_df)).clip(5, 25)
    
    # Temporal features
    tenor_df['days_since_start'] = (tenor_df['auctiondate'] - tenor_df['auctiondate'].min()).dt.days
    tenor_df['month'] = tenor_df['auctiondate'].dt.month
    tenor_df['quarter'] = tenor_df['auctiondate'].dt.quarter
    tenor_df['year'] = tenor_df['auctiondate'].dt.year
    
    # Remove rows with NaN from lagging operations
    tenor_df = tenor_df.dropna()
    
    if len(tenor_df) < 5:
        logger.warning(f"After feature engineering, only {len(tenor_df)} records for {tenor}D - insufficient for training")
        return None
    
    logger.info(f"Features engineered for {tenor}D: {len(tenor_df)} samples, {len(tenor_df.columns)} columns")
    return tenor_df

# =========================================================
# MODEL TRAINING
# =========================================================
def train_model_for_tenor(df: pd.DataFrame, tenor: int) -> Tuple[object, Dict]:
    """
    Train XGBoost model for a specific tenor.
    Returns trained model and performance metrics.
    """
    logger.info(f"\n{'='*60}")
    logger.info(f"Training model for {tenor}D tenor")
    logger.info(f"{'='*60}")
    
    # Prepare features and target
    X = df[FEATURE_ORDER].copy()
    y = df['target'].copy()
    
    # Check for NaN values
    X = X.fillna(X.mean())
    
    logger.info(f"Training set shape: {X.shape}")
    logger.info(f"Features: {list(X.columns)}")
    
    # Train-test split (time-series aware)
    split_point = int(len(X) * 0.8)
    X_train, X_test = X[:split_point], X[split_point:]
    y_train, y_test = y[:split_point], y[split_point:]
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train XGBoost model
    logger.info(f"Training XGBoost model...")
    model = XGBRegressor(
        n_estimators=100,
        max_depth=5,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        verbosity=0
    )
    
    model.fit(X_train_scaled, y_train, verbose=False)
    
    # Evaluate
    y_pred_train = model.predict(X_train_scaled)
    y_pred_test = model.predict(X_test_scaled)
    
    train_mae = mean_absolute_error(y_train, y_pred_train)
    test_mae = mean_absolute_error(y_test, y_pred_test)
    train_r2 = r2_score(y_train, y_pred_train)
    test_r2 = r2_score(y_test, y_pred_test)
    
    metrics = {
        'tenor': tenor,
        'train_mae': train_mae,
        'test_mae': test_mae,
        'train_r2': train_r2,
        'test_r2': test_r2,
        'train_samples': len(X_train),
        'test_samples': len(X_test),
        'scaler': scaler,
    }
    
    logger.info(f"  Train MAE: {train_mae:.4f}, Test MAE: {test_mae:.4f}")
    logger.info(f"  Train R²: {train_r2:.4f}, Test R²: {test_r2:.4f}")
    
    return model, metrics

# =========================================================
# MODEL SAVING
# =========================================================
def save_model(model: object, scaler: object, tenor: int, output_dir: Path = None):
    """Save trained model and scaler."""
    if output_dir is None:
        output_dir = OUTPUT_DIR
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save model
    model_path = output_dir / f"gti_ntb_v5_{tenor}D.pkl"
    joblib.dump(model, model_path, compress=3)
    logger.info(f"✓ Saved model to {model_path}")
    
    # Save scaler
    scaler_path = output_dir / f"scaler_{tenor}D.pkl"
    joblib.dump(scaler, scaler_path, compress=3)
    logger.info(f"✓ Saved scaler to {scaler_path}")
    
    return model_path, scaler_path

# =========================================================
# MAIN PIPELINE
# =========================================================
def main(data_filepath: str = None):
    """Run complete training pipeline."""
    logger.info("\n" + "="*70)
    logger.info("NTB STOP RATE MODEL TRAINING PIPELINE")
    logger.info("="*70 + "\n")
    
    # Step 1: Load data
    df_raw = load_ntb_data(data_filepath)
    if df_raw is None:
        logger.error("Failed to load data. Exiting.")
        return False
    
    # Step 2: Preprocess
    df_processed = preprocess_ntb_data(df_raw)
    if df_processed is None or len(df_processed) == 0:
        logger.error("Failed to preprocess data. Exiting.")
        return False
    
    # Step 3: Train models for each tenor
    all_metrics = []
    
    for tenor in TENORS:
        try:
            # Engineer features
            df_features = engineer_features_for_tenor(df_processed, tenor)
            if df_features is None:
                logger.warning(f"Skipping {tenor}D tenor")
                continue
            
            # Train model
            model, metrics = train_model_for_tenor(df_features, tenor)
            
            # Save model
            model_path, scaler_path = save_model(model, metrics['scaler'], tenor)
            
            all_metrics.append(metrics)
            
        except Exception as e:
            logger.error(f"Error training {tenor}D model: {str(e)}")
            continue
    
    # Step 4: Summary report
    logger.info("\n" + "="*70)
    logger.info("TRAINING SUMMARY")
    logger.info("="*70)
    
    if all_metrics:
        summary_df = pd.DataFrame(all_metrics)
        logger.info("\n" + summary_df.to_string(index=False))
        
        logger.info("\n✓ Training complete!")
        logger.info(f"✓ Generated {len(all_metrics)} models in {OUTPUT_DIR}/")
        logger.info("\nModels are ready for use in app.py")
        return True
    else:
        logger.error("No models were successfully trained!")
        return False

# =========================================================
# ENTRY POINT
# =========================================================
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Train NTB stop rate prediction models"
    )
    parser.add_argument(
        '--data',
        type=str,
        default=None,
        help='Path to NTB CSV data file'
    )
    
    args = parser.parse_args()
    
    success = main(args.data)
    exit(0 if success else 1)
