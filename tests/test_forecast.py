"""
Unit tests for the forecasting module.

Run with: pytest tests/test_forecast.py -v
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from analysis.forecast import (
    load_sales,
    list_products,
    forecast_next_month,
    compare_models,
    get_portfolio_summary,
    detect_trend,
    detect_seasonality,
    detect_anomalies,
    linear_regression_forecast,
    exponential_smoothing_forecast,
    moving_average_forecast,
    ModelType,
)


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def sample_data():
    """Create sample sales data for testing."""
    data = {
        'month': pd.date_range('2025-01', periods=12, freq='MS'),
        'product': ['TestProduct'] * 12,
        'units_sold': [100, 110, 105, 115, 120, 125, 130, 128, 135, 140, 145, 150]
    }
    return pd.DataFrame(data)


@pytest.fixture
def csv_path(tmp_path, sample_data):
    """Create a temporary CSV file with sample data."""
    csv_file = tmp_path / "test_sales.csv"
    sample_data.to_csv(csv_file, index=False)
    return str(csv_file)


@pytest.fixture
def increasing_values():
    """Linear increasing values for trend testing."""
    return np.array([10, 20, 30, 40, 50, 60, 70, 80, 90, 100])


@pytest.fixture
def decreasing_values():
    """Linear decreasing values for trend testing."""
    return np.array([100, 90, 80, 70, 60, 50, 40, 30, 20, 10])


@pytest.fixture
def stable_values():
    """Stable values with small variation."""
    return np.array([100, 101, 99, 100, 102, 98, 100, 101, 99, 100])


# =============================================================================
# DATA LOADING TESTS
# =============================================================================

class TestDataLoading:
    def test_load_sales(self, csv_path):
        """Test that sales data loads correctly."""
        df = load_sales(csv_path)
        assert len(df) == 12
        assert 'month' in df.columns
        assert 'product' in df.columns
        assert 'units_sold' in df.columns
        assert pd.api.types.is_datetime64_any_dtype(df['month'])

    def test_list_products(self, csv_path):
        """Test product listing."""
        df = load_sales(csv_path)
        products = list_products(df)
        assert isinstance(products, list)
        assert 'TestProduct' in products


# =============================================================================
# PATTERN DETECTION TESTS
# =============================================================================

class TestPatternDetection:
    def test_detect_trend_up(self, increasing_values):
        """Test upward trend detection."""
        assert detect_trend(increasing_values) == 'up'

    def test_detect_trend_down(self, decreasing_values):
        """Test downward trend detection."""
        assert detect_trend(decreasing_values) == 'down'

    def test_detect_trend_stable(self, stable_values):
        """Test stable trend detection."""
        assert detect_trend(stable_values) == 'stable'

    def test_detect_trend_short_series(self):
        """Test trend detection with very short series."""
        assert detect_trend(np.array([1, 2])) == 'stable'

    def test_detect_seasonality_no_pattern(self):
        """Test seasonality detection with no seasonal pattern."""
        values = np.array([100, 110, 105, 115, 120, 125, 130, 128, 135, 140])
        assert detect_seasonality(values) == False

    def test_detect_anomalies(self):
        """Test anomaly detection."""
        values = np.array([100, 102, 98, 101, 99, 200, 100, 103, 97, 101])  # 200 is anomaly
        anomalies = detect_anomalies(values, threshold=2.0)
        assert 5 in anomalies  # Index of 200

    def test_detect_anomalies_no_outliers(self, stable_values):
        """Test anomaly detection with no outliers."""
        anomalies = detect_anomalies(stable_values)
        assert len(anomalies) == 0


# =============================================================================
# MODEL TESTS
# =============================================================================

class TestModels:
    def test_linear_regression_forecast(self, increasing_values):
        """Test linear regression produces reasonable forecast."""
        pred, lower, upper = linear_regression_forecast(increasing_values)
        assert pred > increasing_values[-1]  # Should continue trend
        assert lower < pred < upper  # CI should bracket prediction
        assert lower >= 0  # No negative forecasts

    def test_exponential_smoothing_forecast(self, increasing_values):
        """Test exponential smoothing forecast."""
        pred, lower, upper = exponential_smoothing_forecast(increasing_values)
        assert pred > 0
        assert lower < pred < upper

    def test_moving_average_forecast(self, stable_values):
        """Test moving average forecast."""
        pred, lower, upper = moving_average_forecast(stable_values, window=3)
        # Should be close to the mean of last 3 values
        expected = np.mean(stable_values[-3:])
        assert abs(pred - expected) < 1

    def test_forecast_non_negative(self):
        """Test that forecasts are never negative."""
        decreasing = np.array([100, 80, 60, 40, 20, 10, 5, 2, 1, 0])
        
        pred1, _, _ = linear_regression_forecast(decreasing)
        pred2, _, _ = exponential_smoothing_forecast(decreasing)
        pred3, _, _ = moving_average_forecast(decreasing)
        
        assert pred1 >= 0
        assert pred2 >= 0
        assert pred3 >= 0


# =============================================================================
# FORECAST API TESTS
# =============================================================================

class TestForecastAPI:
    def test_forecast_next_month(self, csv_path):
        """Test the main forecast function."""
        df = load_sales(csv_path)
        result = forecast_next_month(df, 'TestProduct', ModelType.LINEAR)
        
        assert 'product' in result
        assert 'next_month' in result
        assert 'predicted_units' in result
        assert 'lower_bound' in result
        assert 'upper_bound' in result
        assert 'suggested_stock' in result
        assert 'mae_last_2_months' in result
        assert 'trend' in result
        assert result['predicted_units'] >= 0

    def test_forecast_unknown_product(self, csv_path):
        """Test forecast with unknown product raises error."""
        df = load_sales(csv_path)
        with pytest.raises(ValueError, match="Unknown product"):
            forecast_next_month(df, 'NonexistentProduct', ModelType.LINEAR)

    def test_compare_models(self, csv_path):
        """Test model comparison."""
        df = load_sales(csv_path)
        result = compare_models(df, 'TestProduct')
        
        assert 'best_model' in result
        assert 'models' in result
        assert len(result['models']) == 3  # All three models

    def test_portfolio_summary(self, csv_path):
        """Test portfolio summary generation."""
        df = load_sales(csv_path)
        summary = get_portfolio_summary(df)
        
        assert 'total_products' in summary
        assert 'total_forecast_units' in summary
        assert 'trend_summary' in summary
        assert summary['total_products'] >= 1


# =============================================================================
# EDGE CASES
# =============================================================================

class TestEdgeCases:
    def test_single_value_forecast(self):
        """Test forecasting with minimal data."""
        values = np.array([100.0])
        pred, lower, upper = moving_average_forecast(values)
        assert pred == 100.0

    def test_constant_values(self):
        """Test with constant values (zero variance)."""
        values = np.array([50.0] * 10)
        pred, lower, upper = linear_regression_forecast(values)
        assert abs(pred - 50.0) < 1

    def test_high_variance_data(self):
        """Test with highly variable data."""
        np.random.seed(42)
        values = np.random.uniform(50, 150, size=12)
        
        # Should not crash
        pred, lower, upper = linear_regression_forecast(values)
        assert pred >= 0
        assert upper > lower


# =============================================================================
# INTEGRATION TESTS
# =============================================================================

class TestIntegration:
    def test_full_workflow(self, csv_path):
        """Test complete workflow from data load to forecast."""
        # Load data
        df = load_sales(csv_path)
        assert len(df) > 0
        
        # List products
        products = list_products(df)
        assert len(products) > 0
        
        # Forecast each product with each model
        for product in products:
            for model_type in ModelType:
                result = forecast_next_month(df, product, model_type)
                assert result['predicted_units'] >= 0
                assert result['suggested_stock'] >= result['predicted_units']


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
