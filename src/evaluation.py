"""
Evaluation Module
Backtesting and performance evaluation for NTB prediction models
"""

import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error
import logging
from typing import Dict, Tuple, List
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelEvaluator:
    """
    Comprehensive evaluation and backtesting of prediction models.
    """

    def __init__(self):
        self.results = {}

    def calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, 
                         model_name: str = 'Model') -> Dict:
        """
        Calculate comprehensive evaluation metrics.
        
        Args:
            y_true: Actual values
            y_pred: Predicted values
            model_name: Name of the model
            
        Returns:
            Dictionary with evaluation metrics
        """
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mape = mean_absolute_percentage_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        
        # Directional accuracy
        actual_direction = np.diff(y_true) > 0
        pred_direction = np.diff(y_pred) > 0
        directional_accuracy = np.mean(actual_direction == pred_direction) * 100
        
        # Mean Absolute Scaled Error (MASE)
        n = len(y_true)
        numerator = np.sum(np.abs(y_true - y_pred)) / n
        denominator = np.sum(np.abs(np.diff(y_true))) / (n - 1)
        mase = numerator / denominator if denominator != 0 else np.inf
        
        metrics = {
            'Model': model_name,
            'MAE': mae,
            'RMSE': rmse,
            'MAPE': mape,
            'R2': r2,
            'Directional Accuracy': directional_accuracy,
            'MASE': mase
        }
        
        return metrics

    def walk_forward_validation(self, df: pd.DataFrame, model, train_size: int = 100,
                               test_size: int = 10, step: int = 5) -> Dict:
        """
        Perform walk-forward validation (backtesting).
        
        Args:
            df: DataFrame with features
            model: Trained model or model class
            train_size: Initial training window size
            test_size: Test window size
            step: Steps to move forward each iteration
            
        Returns:
            Dictionary with validation results
        """
        results = []
        predictions = []
        actuals = []
        
        n = len(df)
        
        for i in range(0, n - train_size - test_size, step):
            # Split data
            train_end = i + train_size
            test_end = train_end + test_size
            
            if test_end > n:
                break
            
            train_data = df.iloc[i:train_end]
            test_data = df.iloc[train_end:test_end]
            
            # Make predictions
            try:
                y_pred = model.predict(test_data)
                y_actual = test_data['stop_rate'].values
                
                predictions.extend(y_pred)
                actuals.extend(y_actual)
                
                # Calculate metrics for this window
                window_metrics = self.calculate_metrics(y_actual, y_pred, f'Window {len(results)}')
                results.append(window_metrics)
            except Exception as e:
                logger.warning(f'Error in walk-forward validation window: {str(e)}')
                continue
        
        logger.info(f'Walk-forward validation complete. {len(results)} windows tested.')
        
        return {
            'window_results': pd.DataFrame(results),
            'predictions': np.array(predictions),
            'actuals': np.array(actuals)
        }

    def calculate_prediction_intervals(self, y_true: np.ndarray, y_pred: np.ndarray,
                                     confidence: float = 0.95) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate prediction intervals based on prediction error distribution.
        
        Args:
            y_true: Actual values
            y_pred: Predicted values
            confidence: Confidence level (0-1)
            
        Returns:
            Tuple of (lower_bound, upper_bound)
        """
        errors = y_true - y_pred
        std_error = np.std(errors)
        
        # Z-score for confidence level
        z_score = 1.96 if confidence == 0.95 else 1.645
        
        margin = z_score * std_error
        lower_bound = y_pred - margin
        upper_bound = y_pred + margin
        
        return lower_bound, upper_bound

    def backtesting_returns(self, y_true: np.ndarray, y_pred: np.ndarray,
                           trading_rule: str = 'simple') -> Dict:
        """
        Simulate trading returns based on model predictions.
        
        Args:
            y_true: Actual values
            y_pred: Predicted values
            trading_rule: 'simple' or 'momentum'
            
        Returns:
            Dictionary with trading metrics
        """
        # Calculate returns (assuming inverse relationship: rate down = bond price up)
        actual_returns = -np.diff(y_true)  # Negative because rates and prices are inverse
        
        # Generate trading signals
        if trading_rule == 'simple':
            # Buy if predicted rate will decrease
            predicted_returns = -np.diff(y_pred)
            signals = (predicted_returns > 0).astype(int)
        elif trading_rule == 'momentum':
            # Use rate of change
            momentum = np.diff(y_pred) / (y_pred[:-1] + 1e-8)
            signals = (momentum < 0).astype(int)  # Decrease is positive
        else:
            signals = np.ones(len(y_true) - 1)
        
        # Calculate strategy returns
        strategy_returns = actual_returns * signals
        
        # Metrics
        cumulative_returns = np.cumsum(strategy_returns)
        total_return = cumulative_returns[-1] if len(cumulative_returns) > 0 else 0
        avg_return = np.mean(strategy_returns)
        std_return = np.std(strategy_returns)
        sharpe_ratio = avg_return / std_return if std_return > 0 else 0
        max_drawdown = np.min(cumulative_returns) if len(cumulative_returns) > 0 else 0
        
        return {
            'Total Return': total_return,
            'Average Return': avg_return,
            'Std Dev': std_return,
            'Sharpe Ratio': sharpe_ratio,
            'Max Drawdown': max_drawdown,
            'Win Rate': np.mean(strategy_returns > 0) * 100
        }

    def generate_evaluation_report(self, y_true: np.ndarray, y_pred: np.ndarray,
                                  model_name: str = 'Model') -> str:
        """
        Generate comprehensive evaluation report.
        
        Args:
            y_true: Actual values
            y_pred: Predicted values
            model_name: Name of the model
            
        Returns:
            Formatted report string
        """
        metrics = self.calculate_metrics(y_true, y_pred, model_name)
        
        report = f"""
        ╔════════════════════════════════════════════╗
        ║  {model_name} Evaluation Report  ║
        ╠════════════════════════════════════════════╣
        ║ MAE:                   {metrics['MAE']:.6f}
        ║ RMSE:                  {metrics['RMSE']:.6f}
        ║ MAPE:                  {metrics['MAPE']:.2f}%
        ║ R² Score:              {metrics['R2']:.4f}
        ║ Directional Accuracy:  {metrics['Directional Accuracy']:.2f}%
        ║ MASE:                  {metrics['MASE']:.4f}
        ╚════════════════════════════════════════════╝
        """
        
        return report

    def export_results(self, results: pd.DataFrame, filepath: str = 'results/evaluation.csv'):
        """
        Export evaluation results to file.
        
        Args:
            results: Results DataFrame
            filepath: Output file path
        """
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        results.to_csv(filepath, index=False)
        logger.info(f'Results exported to {filepath}')


if __name__ == '__main__':
    # Example usage
    evaluator = ModelEvaluator()
    
    # Generate sample predictions for demonstration
    y_true = np.array([5.0, 5.2, 5.1, 5.3, 5.4, 5.2, 5.1, 5.5])
    y_pred = np.array([5.1, 5.15, 5.2, 5.25, 5.3, 5.25, 5.15, 5.4])
    
    # Calculate metrics
    metrics = evaluator.calculate_metrics(y_true, y_pred, 'Test Model')
    print('\nMetrics:')
    for key, value in metrics.items():
        print(f'{key}: {value}')
    
    # Generate report
    report = evaluator.generate_evaluation_report(y_true, y_pred, 'Test Model')
    print(report)