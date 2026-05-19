"""
Feature Engineering Module
Creates advanced features for NTB stop rate prediction
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
from typing import Dict, List

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FeatureEngineer:
    """
    Creates features for NTB prediction models.
    """

    def __init__(self):
        self.rolling_windows = [7, 14, 30, 60, 90]  # Days for rolling calculations

    def create_lagged_features(self, df: pd.DataFrame, column: str, lags: List[int] = None) -> pd.DataFrame:
        """
        Create lagged features from a column.
        
        Args:
            df: Input DataFrame
            column: Column to create lags from
            lags: List of lag periods
            
        Returns:
            DataFrame with lagged features
        """
        if lags is None:
            lags = [1, 2, 3, 5, 7, 14]
        
        if column not in df.columns:
            logger.warning(f'Column {column} not found')
            return df
        
        for lag in lags:
            df[f'{column}_lag_{lag}'] = df[column].shift(lag)
        
        logger.info(f'Created {len(lags)} lagged features for {column}')
        return df

    def create_rolling_features(self, df: pd.DataFrame, column: str, windows: List[int] = None) -> pd.DataFrame:
        """
        Create rolling window features (mean, std, min, max).
        
        Args:
            df: Input DataFrame
            column: Column to create rolling features from
            windows: List of window sizes
            
        Returns:
            DataFrame with rolling features
        """
        if windows is None:
            windows = self.rolling_windows
        
        if column not in df.columns:
            logger.warning(f'Column {column} not found')
            return df
        
        for window in windows:
            df[f'{column}_rolling_mean_{window}'] = df[column].rolling(window=window).mean()
            df[f'{column}_rolling_std_{window}'] = df[column].rolling(window=window).std()
            df[f'{column}_rolling_min_{window}'] = df[column].rolling(window=window).min()
            df[f'{column}_rolling_max_{window}'] = df[column].rolling(window=window).max()
        
        logger.info(f'Created rolling features for {column} with windows {windows}')
        return df

    def create_momentum_features(self, df: pd.DataFrame, column: str, periods: List[int] = None) -> pd.DataFrame:
        """
        Create momentum and rate of change features.
        
        Args:
            df: Input DataFrame
            column: Column to create momentum from
            periods: List of period lengths
            
        Returns:
            DataFrame with momentum features
        """
        if periods is None:
            periods = [1, 7, 14, 30]
        
        if column not in df.columns:
            logger.warning(f'Column {column} not found')
            return df
        
        for period in periods:
            # Rate of change
            df[f'{column}_roc_{period}'] = df[column].pct_change(periods=period)
            # Simple momentum
            df[f'{column}_momentum_{period}'] = df[column].diff(periods=period)
        
        logger.info(f'Created momentum features for {column}')
        return df

    def create_volatility_features(self, df: pd.DataFrame, column: str, windows: List[int] = None) -> pd.DataFrame:
        """
        Create volatility features.
        
        Args:
            df: Input DataFrame
            column: Column to calculate volatility from
            windows: List of window sizes
            
        Returns:
            DataFrame with volatility features
        """
        if windows is None:
            windows = [7, 14, 30]
        
        if column not in df.columns:
            logger.warning(f'Column {column} not found')
            return df
        
        for window in windows:
            # Log returns volatility
            log_returns = np.log(df[column] / df[column].shift(1))
            df[f'{column}_volatility_{window}'] = log_returns.rolling(window=window).std()
        
        logger.info(f'Created volatility features for {column}')
        return df

    def create_spread_features(self, df: pd.DataFrame, column_high: str, column_low: str) -> pd.DataFrame:
        """
        Create spread features (bid-ask, term structure).
        
        Args:
            df: Input DataFrame
            column_high: Higher rate column
            column_low: Lower rate column
            
        Returns:
            DataFrame with spread features
        """
        if column_high in df.columns and column_low in df.columns:
            df['spread'] = df[column_high] - df[column_low]
            df['spread_pct'] = (df[column_high] - df[column_low]) / df[column_low]
            logger.info(f'Created spread features from {column_high} and {column_low}')
        else:
            logger.warning('Columns for spread calculation not found')
        
        return df

    def create_demand_features(self, df: pd.DataFrame, 
                              offered_col: str = 'amount_offered',
                              subscribed_col: str = 'amount_subscribed') -> pd.DataFrame:
        """
        Create market demand features.
        
        Args:
            df: Input DataFrame
            offered_col: Column with amount offered
            subscribed_col: Column with amount subscribed
            
        Returns:
            DataFrame with demand features
        """
        if offered_col in df.columns and subscribed_col in df.columns:
            # Subscription ratio (oversubscription multiple)
            df['subscription_ratio'] = df[subscribed_col] / df[offered_col]
            df['subscription_ratio_lag1'] = df['subscription_ratio'].shift(1)
            
            logger.info('Created demand features')
        else:
            logger.warning('Columns for demand calculation not found')
        
        return df

    def create_temporal_features(self, df: pd.DataFrame, date_column: str = 'auction_date') -> pd.DataFrame:
        """
        Create temporal features from date.
        
        Args:
            df: Input DataFrame
            date_column: Date column name
            
        Returns:
            DataFrame with temporal features
        """
        if date_column not in df.columns:
            logger.warning(f'Date column {date_column} not found')
            return df
        
        # Ensure datetime format
        df[date_column] = pd.to_datetime(df[date_column])
        
        # Extract temporal features
        df['year'] = df[date_column].dt.year
        df['month'] = df[date_column].dt.month
        df['quarter'] = df[date_column].dt.quarter
        df['week'] = df[date_column].dt.isocalendar().week
        df['day_of_week'] = df[date_column].dt.dayofweek
        df['day_of_month'] = df[date_column].dt.day
        df['days_in_month'] = df[date_column].dt.daysinmonth
        
        # Create cyclical features for month and day
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
        df['day_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
        df['day_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
        
        # Days since start
        df['days_since_start'] = (df[date_column] - df[date_column].min()).dt.days
        
        logger.info('Created temporal features')
        return df

    def create_exponential_moving_average(self, df: pd.DataFrame, column: str, spans: List[int] = None) -> pd.DataFrame:
        """
        Create exponential moving average features.
        
        Args:
            df: Input DataFrame
            column: Column to calculate EMA from
            spans: List of span values
            
        Returns:
            DataFrame with EMA features
        """
        if spans is None:
            spans = [7, 14, 30]
        
        if column not in df.columns:
            logger.warning(f'Column {column} not found')
            return df
        
        for span in spans:
            df[f'{column}_ema_{span}'] = df[column].ewm(span=span, adjust=False).mean()
        
        logger.info(f'Created EMA features for {column}')
        return df

    def create_all_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create comprehensive feature set for modeling.
        
        Args:
            df: Preprocessed DataFrame
            
        Returns:
            DataFrame with all engineered features
        """
        logger.info('Creating comprehensive feature set')
        
        # Ensure required columns exist
        required_cols = ['stop_rate']
        if 'stop_rate' not in df.columns:
            logger.error('stop_rate column not found')
            return df
        
        # Sort by date
        if 'auction_date' in df.columns:
            df = df.sort_values('auction_date').reset_index(drop=True)
        
        # Temporal features
        if 'auction_date' in df.columns:
            df = self.create_temporal_features(df)
        
        # Lagged features
        df = self.create_lagged_features(df, 'stop_rate', lags=[1, 2, 3, 5, 7, 14, 21, 30])
        
        # Rolling features
        df = self.create_rolling_features(df, 'stop_rate')
        
        # Momentum features
        df = self.create_momentum_features(df, 'stop_rate')
        
        # Volatility features
        df = self.create_volatility_features(df, 'stop_rate')
        
        # EMA features
        df = self.create_exponential_moving_average(df, 'stop_rate')
        
        # Demand features
        if 'amount_offered' in df.columns and 'amount_subscribed' in df.columns:
            df = self.create_demand_features(df)
        
        # Remove initial rows with NaN (due to lags and rolling windows)
        initial_rows = len(df)
        df = df.dropna()
        logger.info(f'Removed {initial_rows - len(df)} rows with NaN values')
        
        logger.info(f'Feature creation complete. Total features: {len(df.columns)}')
        return df

    def get_feature_importance_correlation(self, df: pd.DataFrame, target: str = 'stop_rate') -> pd.Series:
        """
        Calculate feature correlations with target variable.
        
        Args:
            df: DataFrame with features
            target: Target column name
            
        Returns:
            Series with correlation values
        """
        if target not in df.columns:
            logger.error(f'Target column {target} not found')
            return pd.Series()
        
        numeric_df = df.select_dtypes(include=[np.number])
        correlations = numeric_df.corr()[target].sort_values(ascending=False)
        logger.info(f'Calculated correlations for {len(correlations)} features')
        return correlations


if __name__ == '__main__':
    # Example usage
    fe = FeatureEngineer()
    
    # Load processed data
    df = pd.read_csv('data/processed/ntb_processed.csv')
    df['auction_date'] = pd.to_datetime(df['auction_date'])
    
    # Create features
    df_features = fe.create_all_features(df)
    
    # Save featured data
    df_features.to_csv('data/processed/ntb_featured.csv', index=False)
    
    # Display feature importance
    print('\nTop correlated features with stop_rate:')
    print(fe.get_feature_importance_correlation(df_features).head(15))