"""
Modeling Module
Implements various prediction models for NTB stop rates
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import VotingRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam

import logging
from typing import Dict, Tuple, List, Optional
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelPipeline:
    """
    Manages model training, evaluation, and comparison.
    """

    def __init__(self, test_size: float = 0.2, random_state: int = 42):
        """
        Initialize the model pipeline.
        
        Args:
            test_size: Proportion of data to use for testing
            random_state: Random seed for reproducibility
        """
        self.test_size = test_size
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.scaler_target = StandardScaler()
        self.models = {}
        self.results = {}

    def prepare_data(self, df: pd.DataFrame, target: str = 'stop_rate', 
                    test_size: Optional[float] = None, use_time_split: bool = True) -> Tuple:
        """
        Prepare data for modeling.
        
        Args:
            df: DataFrame with features
            target: Target column name
            test_size: Proportion for test set
            use_time_split: Use time-series split instead of random split
            
        Returns:
            Tuple of (X_train, X_test, y_train, y_test)
        """
        if target not in df.columns:
            raise ValueError(f'Target column {target} not found')
        
        if test_size is None:
            test_size = self.test_size
        
        # Separate features and target
        y = df[target].values
        X = df.drop(columns=[target, 'auction_date'], errors='ignore')
        
        # Handle datetime columns
        X = X.select_dtypes(include=[np.number])
        
        if use_time_split:
            # Time-series split
            split_point = int(len(X) * (1 - test_size))
            X_train, X_test = X[:split_point], X[split_point:]
            y_train, y_test = y[:split_point], y[split_point:]
        else:
            # Random split
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=self.random_state
            )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        logger.info(f'Data prepared. Train: {X_train.shape}, Test: {X_test.shape}')
        return X_train_scaled, X_test_scaled, y_train, y_test

    def train_linear_regression(self, X_train, y_train) -> LinearRegression:
        """
        Train linear regression model.
        
        Args:
            X_train: Training features
            y_train: Training target
            
        Returns:
            Trained model
        """
        model = LinearRegression()
        model.fit(X_train, y_train)
        logger.info('Linear Regression model trained')
        return model

    def train_ridge_regression(self, X_train, y_train, alpha: float = 1.0) -> Ridge:
        """
        Train Ridge regression model.
        
        Args:
            X_train: Training features
            y_train: Training target
            alpha: Regularization strength
            
        Returns:
            Trained model
        """
        model = Ridge(alpha=alpha, random_state=self.random_state)
        model.fit(X_train, y_train)
        logger.info(f'Ridge Regression model trained (alpha={alpha})')
        return model

    def train_random_forest(self, X_train, y_train, n_estimators: int = 100) -> RandomForestRegressor:
        """
        Train Random Forest model.
        
        Args:
            X_train: Training features
            y_train: Training target
            n_estimators: Number of trees
            
        Returns:
            Trained model
        """
        model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=15,
            min_samples_split=5,
            random_state=self.random_state,
            n_jobs=-1
        )
        model.fit(X_train, y_train)
        logger.info(f'Random Forest model trained ({n_estimators} trees)')
        return model

    def train_xgboost(self, X_train, y_train) -> XGBRegressor:
        """
        Train XGBoost model.
        
        Args:
            X_train: Training features
            y_train: Training target
            
        Returns:
            Trained model
        """
        model = XGBRegressor(
            n_estimators=200,
            max_depth=5,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=self.random_state
        )
        model.fit(X_train, y_train, verbose=False)
        logger.info('XGBoost model trained')
        return model

    def train_lightgbm(self, X_train, y_train) -> LGBMRegressor:
        """
        Train LightGBM model.
        
        Args:
            X_train: Training features
            y_train: Training target
            
        Returns:
            Trained model
        """
        model = LGBMRegressor(
            n_estimators=200,
            max_depth=5,
            learning_rate=0.1,
            num_leaves=31,
            random_state=self.random_state,
            verbose=-1
        )
        model.fit(X_train, y_train)
        logger.info('LightGBM model trained')
        return model

    def train_ensemble(self, X_train, y_train) -> VotingRegressor:
        """
        Train ensemble model combining multiple base learners.
        
        Args:
            X_train: Training features
            y_train: Training target
            
        Returns:
            Trained ensemble model
        """
        lr = LinearRegression()
        rf = RandomForestRegressor(n_estimators=100, random_state=self.random_state, n_jobs=-1)
        xgb = XGBRegressor(n_estimators=100, random_state=self.random_state, verbose=False)
        
        ensemble = VotingRegressor([
            ('lr', lr),
            ('rf', rf),
            ('xgb', xgb)
        ])
        ensemble.fit(X_train, y_train)
        logger.info('Ensemble model trained')
        return ensemble

    def train_lstm(self, X_train, y_train, epochs: int = 50, batch_size: int = 32) -> Sequential:
        """
        Train LSTM model for sequence prediction.
        
        Args:
            X_train: Training features
            y_train: Training target
            epochs: Number of training epochs
            batch_size: Batch size
            
        Returns:
            Trained LSTM model
        """
        # Reshape for LSTM (samples, timesteps, features)
        X_train_reshaped = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
        
        model = Sequential([
            LSTM(64, activation='relu', input_shape=(1, X_train.shape[1])),
            Dropout(0.2),
            Dense(32, activation='relu'),
            Dropout(0.2),
            Dense(1)
        ])
        
        model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])
        model.fit(X_train_reshaped, y_train, epochs=epochs, batch_size=batch_size, verbose=0)
        logger.info('LSTM model trained')
        return model

    def evaluate_model(self, model, X_test, y_test, model_name: str = 'Model') -> Dict:
        """
        Evaluate model performance.
        
        Args:
            model: Trained model
            X_test: Test features
            y_test: Test target
            model_name: Name of the model
            
        Returns:
            Dictionary with evaluation metrics
        """
        # Make predictions
        if hasattr(model, 'predict'):
            y_pred = model.predict(X_test)
        else:
            # For LSTM
            X_test_reshaped = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))
            y_pred = model.predict(X_test_reshaped, verbose=0).flatten()
        
        # Calculate metrics
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
        r2 = r2_score(y_test, y_pred)
        
        results = {
            'Model': model_name,
            'MAE': mae,
            'RMSE': rmse,
            'MAPE': mape,
            'R2': r2,
            'Predictions': y_pred
        }
        
        logger.info(f'{model_name} - MAE: {mae:.4f}, RMSE: {rmse:.4f}, MAPE: {mape:.2f}%, R2: {r2:.4f}')
        return results

    def train_all_models(self, df: pd.DataFrame, target: str = 'stop_rate') -> Dict:
        """
        Train all available models.
        
        Args:
            df: DataFrame with features
            target: Target column name
            
        Returns:
            Dictionary with results for all models
        """
        logger.info('Training all models...')
        
        # Prepare data
        X_train, X_test, y_train, y_test = self.prepare_data(df, target)
        
        results = {}
        
        # Train Linear Regression
        try:
            model = self.train_linear_regression(X_train, y_train)
            results['Linear Regression'] = self.evaluate_model(model, X_test, y_test, 'Linear Regression')
        except Exception as e:
            logger.error(f'Error training Linear Regression: {str(e)}')
        
        # Train Ridge
        try:
            model = self.train_ridge_regression(X_train, y_train)
            results['Ridge'] = self.evaluate_model(model, X_test, y_test, 'Ridge')
        except Exception as e:
            logger.error(f'Error training Ridge: {str(e)}')
        
        # Train Random Forest
        try:
            model = self.train_random_forest(X_train, y_train)
            results['Random Forest'] = self.evaluate_model(model, X_test, y_test, 'Random Forest')
        except Exception as e:
            logger.error(f'Error training Random Forest: {str(e)}')
        
        # Train XGBoost
        try:
            model = self.train_xgboost(X_train, y_train)
            results['XGBoost'] = self.evaluate_model(model, X_test, y_test, 'XGBoost')
        except Exception as e:
            logger.error(f'Error training XGBoost: {str(e)}')
        
        # Train LightGBM
        try:
            model = self.train_lightgbm(X_train, y_train)
            results['LightGBM'] = self.evaluate_model(model, X_test, y_test, 'LightGBM')
        except Exception as e:
            logger.error(f'Error training LightGBM: {str(e)}')
        
        # Train Ensemble
        try:
            model = self.train_ensemble(X_train, y_train)
            results['Ensemble'] = self.evaluate_model(model, X_test, y_test, 'Ensemble')
        except Exception as e:
            logger.error(f'Error training Ensemble: {str(e)}')
        
        self.results = results
        logger.info(f'Model training complete. {len(results)} models trained.')
        return results

    def compare_results(self, results: Optional[Dict] = None) -> pd.DataFrame:
        """
        Compare results across all models.
        
        Args:
            results: Results dictionary (uses self.results if not provided)
            
        Returns:
            DataFrame comparing model performance
        """
        if results is None:
            results = self.results
        
        if not results:
            logger.warning('No results to compare')
            return pd.DataFrame()
        
        # Create comparison DataFrame
        comparison = pd.DataFrame([
            {
                'Model': r['Model'],
                'MAE': r['MAE'],
                'RMSE': r['RMSE'],
                'MAPE': r['MAPE'],
                'R2': r['R2']
            }
            for r in results.values()
        ])
        
        comparison = comparison.sort_values('RMSE')
        logger.info('\nModel Comparison:')
        logger.info(comparison.to_string())
        return comparison

    def save_model(self, model, model_name: str, directory: str = 'models') -> str:
        """
        Save trained model.
        
        Args:
            model: Trained model
            model_name: Name for the model
            directory: Directory to save to
            
        Returns:
            Path to saved model
        """
        import pickle
        Path(directory).mkdir(parents=True, exist_ok=True)
        filepath = Path(directory) / f'{model_name}.pkl'
        with open(filepath, 'wb') as f:
            pickle.dump(model, f)
        logger.info(f'Model saved to {filepath}')
        return str(filepath)


if __name__ == '__main__':
    # Example usage
    pipeline = ModelPipeline()
    
    # Load featured data
    df = pd.read_csv('data/processed/ntb_featured.csv')
    df['auction_date'] = pd.to_datetime(df['auction_date'])
    
    # Train all models
    results = pipeline.train_all_models(df)
    
    # Compare results
    comparison = pipeline.compare_results(results)
    comparison.to_csv('results/model_comparison.csv', index=False)