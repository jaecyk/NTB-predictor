"""
Data Preprocessing Module
Cleaning, validation, and preparation of NTB data
"""

import pandas as pd
import numpy as np
from datetime import datetime
import logging
from pathlib import Path
from typing import Tuple, Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataPreprocessor:
    """
    Preprocesses raw NTB auction data for modeling.
    """

    def __init__(self):
        self.missing_value_threshold = 0.5  # Drop columns with >50% missing
        self.duplicate_threshold = 0.9  # Flag if >90% duplicates

    def load_data(self, filepath: str) -> pd.DataFrame:
        """
        Load data from CSV or Excel file.
        
        Args:
            filepath: Path to data file
            
        Returns:
            Loaded DataFrame
        """
        try:
            if filepath.endswith('.csv'):
                df = pd.read_csv(filepath)
            elif filepath.endswith(('.xlsx', '.xls')):
                df = pd.read_excel(filepath)
            else:
                raise ValueError('Unsupported file format')
            
            logger.info(f'Loaded {len(df)} records from {filepath}')
            return df
        except Exception as e:
            logger.error(f'Error loading file: {str(e)}')
            raise

    def standardize_column_names(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Standardize column names for consistency.
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with standardized column names
        """
        # Convert to lowercase and replace spaces with underscores
        df.columns = df.columns.str.lower().str.replace(' ', '_')
        
        # Map common variations to standard names
        column_mapping = {
            'auction_date': 'auction_date',
            'date': 'auction_date',
            'tenor': 'tenor',
            'tenor_days': 'tenor',
            'maturity': 'tenor',
            'stop_rate': 'stop_rate',
            'marginal_rate': 'stop_rate',
            'rate': 'stop_rate',
            'amount_offered': 'amount_offered',
            'offered': 'amount_offered',
            'amount_subscribed': 'amount_subscribed',
            'subscribed': 'amount_subscribed',
            'bid_amount_range': 'bid_range',
            'bid_range': 'bid_range',
            'bid_high': 'bid_high',
            'bid_low': 'bid_low',
        }
        
        df = df.rename(columns=column_mapping)
        logger.info(f'Standardized column names: {df.columns.tolist()}')
        return df

    def parse_dates(self, df: pd.DataFrame, date_column: str = 'auction_date') -> pd.DataFrame:
        """
        Parse date columns.
        
        Args:
            df: Input DataFrame
            date_column: Name of date column
            
        Returns:
            DataFrame with parsed dates
        """
        if date_column not in df.columns:
            logger.warning(f'Date column {date_column} not found')
            return df
        
        try:
            df[date_column] = pd.to_datetime(df[date_column], infer_datetime_format=True)
            logger.info(f'Parsed {date_column} as datetime')
        except Exception as e:
            logger.error(f'Error parsing dates: {str(e)}')
        
        return df

    def handle_missing_values(self, df: pd.DataFrame, strategy: str = 'drop') -> pd.DataFrame:
        """
        Handle missing values in the dataset.
        
        Args:
            df: Input DataFrame
            strategy: 'drop' to remove rows, 'forward_fill', 'interpolate'
            
        Returns:
            DataFrame with handled missing values
        """
        initial_rows = len(df)
        missing_pct = df.isnull().sum() / len(df)
        
        # Drop columns with too many missing values
        cols_to_drop = missing_pct[missing_pct > self.missing_value_threshold].index
        if len(cols_to_drop) > 0:
            logger.info(f'Dropping columns with >50% missing: {cols_to_drop.tolist()}')
            df = df.drop(columns=cols_to_drop)
        
        # Handle remaining missing values
        if strategy == 'drop':
            df = df.dropna()
            logger.info(f'Dropped {initial_rows - len(df)} rows with missing values')
        
        elif strategy == 'forward_fill':
            df = df.fillna(method='ffill')
            df = df.fillna(method='bfill')
            logger.info('Applied forward/backward fill for missing values')
        
        elif strategy == 'interpolate':
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                df[col] = df[col].interpolate(method='linear', limit_direction='both')
            logger.info('Applied interpolation for numeric missing values')
        
        return df

    def parse_tenor(self, df: pd.DataFrame, tenor_column: str = 'tenor') -> pd.DataFrame:
        """
        Parse tenor field and extract days as integer.
        
        Args:
            df: Input DataFrame
            tenor_column: Name of tenor column
            
        Returns:
            DataFrame with parsed tenor
        """
        if tenor_column not in df.columns:
            logger.warning(f'Tenor column {tenor_column} not found')
            return df
        
        def extract_days(tenor_str):
            """Extract days from tenor string."""
            if pd.isna(tenor_str):
                return np.nan
            
            tenor_str = str(tenor_str).lower().strip()
            
            # Extract numeric part
            days = ''.join(filter(str.isdigit, tenor_str))
            return int(days) if days else np.nan
        
        df['tenor_days'] = df[tenor_column].apply(extract_days)
        logger.info('Parsed tenor field to days')
        return df

    def remove_duplicates(self, df: pd.DataFrame, subset: Optional[list] = None) -> pd.DataFrame:
        """
        Remove duplicate records.
        
        Args:
            df: Input DataFrame
            subset: Columns to consider for duplicate detection
            
        Returns:
            DataFrame with duplicates removed
        """
        initial_rows = len(df)
        
        if subset is None:
            subset = ['auction_date', 'tenor_days', 'stop_rate']
        
        # Only use columns that exist
        subset = [col for col in subset if col in df.columns]
        
        df = df.drop_duplicates(subset=subset, keep='first')
        logger.info(f'Removed {initial_rows - len(df)} duplicate records')
        return df

    def remove_outliers(self, df: pd.DataFrame, column: str, method: str = 'iqr', threshold: float = 1.5) -> pd.DataFrame:
        """
        Remove or flag outliers in numeric columns.
        
        Args:
            df: Input DataFrame
            column: Column to check for outliers
            method: 'iqr' for interquartile range, 'zscore' for z-score
            threshold: Threshold for outlier detection
            
        Returns:
            DataFrame with outliers removed
        """
        if column not in df.columns:
            logger.warning(f'Column {column} not found')
            return df
        
        initial_rows = len(df)
        
        if method == 'iqr':
            Q1 = df[column].quantile(0.25)
            Q3 = df[column].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - threshold * IQR
            upper_bound = Q3 + threshold * IQR
            df = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
        
        elif method == 'zscore':
            z_scores = np.abs((df[column] - df[column].mean()) / df[column].std())
            df = df[z_scores <= threshold]
        
        logger.info(f'Removed {initial_rows - len(df)} outliers from {column}')
        return df

    def sort_by_date(self, df: pd.DataFrame, date_column: str = 'auction_date') -> pd.DataFrame:
        """
        Sort data by date for time-series analysis.
        
        Args:
            df: Input DataFrame
            date_column: Date column to sort by
            
        Returns:
            Sorted DataFrame
        """
        if date_column not in df.columns:
            logger.warning(f'Date column {date_column} not found')
            return df
        
        df = df.sort_values(by=date_column).reset_index(drop=True)
        logger.info(f'Sorted data by {date_column}')
        return df

    def create_tenor_datasets(self, df: pd.DataFrame) -> dict:
        """
        Create separate datasets for each tenor.
        
        Args:
            df: Input DataFrame with tenor_days column
            
        Returns:
            Dictionary with datasets for each tenor
        """
        tenor_datasets = {}
        
        if 'tenor_days' in df.columns:
            for tenor in df['tenor_days'].unique():
                if pd.notna(tenor):
                    tenor_df = df[df['tenor_days'] == tenor].copy()
                    tenor_datasets[f'{int(tenor)}_day'] = tenor_df
                    logger.info(f'Created dataset for {int(tenor)}-day tenor: {len(tenor_df)} records')
        
        return tenor_datasets

    def prepare_for_modeling(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Complete preprocessing pipeline.
        
        Args:
            df: Raw DataFrame
            
        Returns:
            Cleaned and prepared DataFrame
        """
        logger.info('Starting preprocessing pipeline')
        
        # Step 1: Standardize columns
        df = self.standardize_column_names(df)
        
        # Step 2: Parse dates
        df = self.parse_dates(df)
        
        # Step 3: Parse tenor
        df = self.parse_tenor(df)
        
        # Step 4: Handle missing values
        df = self.handle_missing_values(df, strategy='drop')
        
        # Step 5: Remove duplicates
        df = self.remove_duplicates(df)
        
        # Step 6: Remove outliers from stop_rate
        if 'stop_rate' in df.columns:
            df = self.remove_outliers(df, 'stop_rate', method='iqr')
        
        # Step 7: Sort by date
        df = self.sort_by_date(df)
        
        logger.info(f'Preprocessing complete. Final shape: {df.shape}')
        return df

    def save_processed_data(self, df: pd.DataFrame, filepath: str = 'data/processed/ntb_processed.csv') -> str:
        """
        Save processed data to file.
        
        Args:
            df: DataFrame to save
            filepath: Output file path
            
        Returns:
            Path to saved file
        """
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(filepath, index=False)
        logger.info(f'Processed data saved to {filepath}')
        return filepath


if __name__ == '__main__':
    # Example usage
    preprocessor = DataPreprocessor()
    
    # Load and process data
    df = preprocessor.load_data('data/raw/ntb_raw.csv')
    df_processed = preprocessor.prepare_for_modeling(df)
    
    # Save processed data
    preprocessor.save_processed_data(df_processed)
    
    # Create tenor-specific datasets
    tenor_datasets = preprocessor.create_tenor_datasets(df_processed)
    print(f'\nCreated datasets for {len(tenor_datasets)} tenors')
    for tenor, data in tenor_datasets.items():
        print(f'{tenor}: {len(data)} records')