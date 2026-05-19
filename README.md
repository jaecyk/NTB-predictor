# NTB Stop Rates Predictor

A machine learning and time-series forecasting model to predict Nigerian Treasury Bills (NTB) auction stop rates.

## Overview

This project builds predictive models for NTB stop rates across different tenors (91-day, 182-day, 364-day) using:
- **Historical auction data** from the Central Bank of Nigeria (CBN)
- **Time-series models** (ARIMA, SARIMA, Exponential Smoothing)
- **Machine learning** (Random Forest, XGBoost, LightGBM)
- **Deep learning** (LSTM, GRU)
- **Ensemble methods** for improved predictions

## Features

✅ Automated data collection from CBN website  
✅ Data cleaning and validation  
✅ Advanced feature engineering  
✅ Multiple forecasting models  
✅ Backtesting framework  
✅ Performance evaluation and comparison  

## Project Structure

```
ntb-predictor/
├── data/
│   ├── raw/                 # CBN Excel/CSV downloads
│   └── processed/           # Cleaned, formatted data
├── notebooks/
│   ├── 01_eda.ipynb        # Exploratory data analysis
│   ├── 02_data_prep.ipynb  # Data preparation
│   └── 03_modeling.ipynb    # Model building & evaluation
├── src/
│   ├── __init__.py
│   ├── data_collector.py    # Web scraper for CBN data
│   ├── preprocessor.py      # Data cleaning & validation
│   ├── features.py          # Feature engineering
│   ├── models.py            # Model implementations
│   ├── evaluation.py        # Evaluation metrics & backtesting
│   └── utils.py             # Helper functions
├── tests/
│   ├── test_preprocessor.py
│   ├── test_features.py
│   └── test_models.py
├── requirements.txt         # Python dependencies
├── setup.py                 # Package setup
└── README.md                # This file
```

## Data Sources

1. **Central Bank of Nigeria (CBN)** - Primary source
   - URL: https://www.cbn.gov.ng/rates/GovtSecurities.html
   - Data: Stop rates, bid ranges, amounts offered, tenors

2. **Secondary Sources**
   - Nairametrics
   - Financial Nigeria
   - Nigeria Galleria Finance

## Installation

```bash
# Clone the repository
git clone https://github.com/jaecyk/ntb-predictor.git
cd ntb-predictor

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Quick Start

### 1. Collect Data
```python
from src.data_collector import CBNDataCollector

collector = CBNDataCollector()
collector.scrape_ntb_rates()
```

### 2. Preprocess Data
```python
from src.preprocessor import DataPreprocessor

preprocessor = DataPreprocessor()
df_clean = preprocessor.load_and_clean('data/raw/ntb_rates.csv')
df_processed = preprocessor.prepare_for_modeling(df_clean)
```

### 3. Engineer Features
```python
from src.features import FeatureEngineer

fe = FeatureEngineer()
df_features = fe.create_features(df_processed)
```

### 4. Train Models
```python
from src.models import ModelPipeline

pipeline = ModelPipeline()
results = pipeline.train_all_models(df_features)
pipeline.compare_results(results)
```

## Model Details

### Time Series Models
- **ARIMA/SARIMA** - Seasonal AutoRegressive Integrated Moving Average
- **ExponentialSmoothing** - Triple exponential smoothing (Holt-Winters)

### Machine Learning
- **Random Forest** - Ensemble tree-based model
- **XGBoost** - Gradient boosting framework
- **LightGBM** - Fast gradient boosting

### Deep Learning
- **LSTM** - Long Short-Term Memory networks
- **GRU** - Gated Recurrent Unit networks

## Evaluation Metrics

- **MAE** - Mean Absolute Error
- **RMSE** - Root Mean Squared Error
- **MAPE** - Mean Absolute Percentage Error
- **Directional Accuracy** - % of correct direction predictions
- **Sharpe Ratio** - Risk-adjusted returns

## Backtesting

Walk-forward validation approach:
1. Train on historical data
2. Predict next period
3. Move forward in time
4. Repeat until end of data

## Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Commit changes
4. Push to the branch
5. Open a Pull Request

## License

MIT License - see LICENSE file for details

## Author

Jaecyk - Building predictive financial models for the Nigerian market

## Contact

For questions or suggestions, please open an issue on GitHub.