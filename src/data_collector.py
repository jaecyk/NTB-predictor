"""
Data Collection Module
Collects NTB auction data from CBN and other sources
"""

import requests
import pandas as pd
import numpy as np
from bs4 import BeautifulSoup
from datetime import datetime, timedelta
import json
import logging
from pathlib import Path
from typing import Optional, List, Dict

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CBNDataCollector:
    """
    Collects Nigerian Treasury Bills (NTB) auction data from CBN and other sources.
    """

    def __init__(self, output_dir: str = 'data/raw'):
        """
        Initialize the data collector.
        
        Args:
            output_dir: Directory to save downloaded files
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.cbn_url = 'https://www.cbn.gov.ng/rates/GovtSecurities.html'
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }

    def scrape_cbn_rates(self) -> Optional[pd.DataFrame]:
        """
        Scrape NTB rates from CBN website.
        
        Returns:
            DataFrame with auction data or None if scraping fails
        """
        try:
            logger.info(f'Fetching data from {self.cbn_url}')
            response = requests.get(self.cbn_url, headers=self.headers, timeout=10)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Find tables containing government securities data
            tables = soup.find_all('table')
            
            if not tables:
                logger.warning('No tables found on CBN page')
                return None
            
            # Extract data from tables
            dfs = []
            for table in tables:
                try:
                    df = pd.read_html(str(table))[0]
                    dfs.append(df)
                except Exception as e:
                    logger.warning(f'Could not parse table: {str(e)}')
                    continue
            
            if dfs:
                combined_df = pd.concat(dfs, ignore_index=True)
                logger.info(f'Successfully scraped {len(combined_df)} records')
                return combined_df
            else:
                logger.warning('No data extracted from tables')
                return None
                
        except requests.RequestException as e:
            logger.error(f'Error fetching data from CBN: {str(e)}')
            return None
        except Exception as e:
            logger.error(f'Unexpected error during scraping: {str(e)}')
            return None

    def load_from_excel(self, filepath: str) -> pd.DataFrame:
        """
        Load NTB data from Excel file (manual download from CBN website).
        
        Args:
            filepath: Path to Excel file
            
        Returns:
            DataFrame with auction data
        """
        try:
            logger.info(f'Loading data from {filepath}')
            df = pd.read_excel(filepath)
            logger.info(f'Loaded {len(df)} records from Excel')
            return df
        except Exception as e:
            logger.error(f'Error loading Excel file: {str(e)}')
            raise

    def load_from_csv(self, filepath: str) -> pd.DataFrame:
        """
        Load NTB data from CSV file.
        
        Args:
            filepath: Path to CSV file
            
        Returns:
            DataFrame with auction data
        """
        try:
            logger.info(f'Loading data from {filepath}')
            df = pd.read_csv(filepath)
            logger.info(f'Loaded {len(df)} records from CSV')
            return df
        except Exception as e:
            logger.error(f'Error loading CSV file: {str(e)}')
            raise

    def combine_multiple_files(self, directory: str) -> pd.DataFrame:
        """
        Combine multiple NTB data files from a directory.
        
        Args:
            directory: Directory containing CSV/Excel files
            
        Returns:
            Combined DataFrame
        """
        dir_path = Path(directory)
        all_files = list(dir_path.glob('*.csv')) + list(dir_path.glob('*.xlsx'))
        
        if not all_files:
            logger.warning(f'No CSV or Excel files found in {directory}')
            return pd.DataFrame()
        
        dfs = []
        for file in all_files:
            try:
                if file.suffix == '.csv':
                    df = pd.read_csv(file)
                else:
                    df = pd.read_excel(file)
                dfs.append(df)
                logger.info(f'Loaded {len(df)} records from {file.name}')
            except Exception as e:
                logger.warning(f'Could not load {file.name}: {str(e)}')
                continue
        
        combined = pd.concat(dfs, ignore_index=True)
        logger.info(f'Combined {len(combined)} total records from {len(dfs)} files')
        return combined

    def validate_data_structure(self, df: pd.DataFrame) -> bool:
        """
        Validate that the loaded data has expected columns.
        
        Args:
            df: DataFrame to validate
            
        Returns:
            True if valid structure, False otherwise
        """
        expected_columns = ['Auction Date', 'Tenor', 'Stop Rate', 'Amount Offered']
        
        # Check for at least some of the expected columns (case-insensitive)
        df_columns_lower = [col.lower() for col in df.columns]
        expected_lower = [col.lower() for col in expected_columns]
        
        found_columns = [col for col in expected_lower if col in df_columns_lower]
        
        if len(found_columns) >= 2:
            logger.info(f'Data structure validated. Found columns: {found_columns}')
            return True
        else:
            logger.warning(f'Unexpected data structure. Columns: {df.columns.tolist()}')
            return False

    def save_raw_data(self, df: pd.DataFrame, filename: str = 'ntb_raw.csv') -> str:
        """
        Save raw data to file.
        
        Args:
            df: DataFrame to save
            filename: Output filename
            
        Returns:
            Path to saved file
        """
        filepath = self.output_dir / filename
        try:
            df.to_csv(filepath, index=False)
            logger.info(f'Raw data saved to {filepath}')
            return str(filepath)
        except Exception as e:
            logger.error(f'Error saving data: {str(e)}')
            raise

    def get_sample_data(self) -> pd.DataFrame:
        """
        Generate sample data for testing (in case live data not available).
        
        Returns:
            Sample DataFrame with realistic NTB data
        """
        logger.info('Generating sample data for testing')
        
        # Create sample dates
        start_date = datetime(2021, 1, 1)
        dates = [start_date + timedelta(days=7*i) for i in range(200)]
        
        # Create sample data
        tenors = ['91-day', '182-day', '364-day']
        data = []
        
        for date in dates:
            for tenor in tenors:
                stop_rate = np.random.uniform(2.5, 12.0)  # Realistic NTB rates
                amount_offered = np.random.uniform(50, 200) * 1e9  # Naira
                amount_subscribed = amount_offered * np.random.uniform(1.5, 4.0)
                
                data.append({
                    'Auction Date': date,
                    'Tenor': tenor,
                    'Stop Rate': round(stop_rate, 2),
                    'Amount Offered': amount_offered,
                    'Amount Subscribed': amount_subscribed,
                    'Bid Amount Range': f'{round(stop_rate - 0.5, 2)}-{round(stop_rate + 0.5, 2)}',
                })
        
        df = pd.DataFrame(data)
        logger.info(f'Generated sample data with {len(df)} records')
        return df


if __name__ == '__main__':
    # Example usage
    collector = CBNDataCollector()
    
    # Try to scrape live data
    df = collector.scrape_cbn_rates()
    
    # If scraping fails, use sample data
    if df is None:
        logger.info('Using sample data instead')
        df = collector.get_sample_data()
    
    # Validate and save
    if collector.validate_data_structure(df):
        collector.save_raw_data(df)
        print(f'\nData collection complete. Shape: {df.shape}')
        print(f'\nFirst few rows:\n{df.head()}')
    else:
        print('Data validation failed')