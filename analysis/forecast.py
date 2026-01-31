"""
Retail Sales Forecasting Engine
===============================
Multi-model forecasting with confidence intervals, anomaly detection,
and comprehensive backtesting metrics.

Models:
- Linear Regression (trend-based)
- Exponential Smoothing (ETS)
- Moving Average (baseline)
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum


class ModelType(str, Enum):
    LINEAR = "linear_regression"
    ETS = "exponential_smoothing"
    MOVING_AVG = "moving_average"


@dataclass
class ForecastResult:
    """Structured forecast output with confidence intervals."""
    product: str
    model: str
    next_month: str
    predicted_units: float
    lower_bound: float
    upper_bound: float
    suggested_stock: int
    mae: float
    mape: float
    trend: str  # "up", "down", "stable"
    seasonality_detected: bool
    history: List[Dict]


def load_sales(csv_path: str) -> pd.DataFrame:
    """Load and preprocess sales data."""
    df = pd.read_csv(csv_path)
    df["month"] = pd.to_datetime(df["month"])
    df = df.sort_values(["product", "month"])
    return df


def list_products(df: pd.DataFrame) -> List[str]:
    """Get sorted list of unique products."""
    return sorted(df["product"].unique().tolist())


def detect_trend(values: np.ndarray) -> str:
    """Detect overall trend direction using linear regression slope."""
    if len(values) < 3:
        return "stable"
    
    X = np.arange(len(values)).reshape(-1, 1)
    model = LinearRegression()
    model.fit(X, values)
    slope = model.coef_[0]
    
    # Normalize slope by mean value to get percentage change
    mean_val = np.mean(values)
    if mean_val == 0:
        return "stable"
    
    pct_change = (slope / mean_val) * 100
    
    if pct_change > 2:
        return "up"
    elif pct_change < -2:
        return "down"
    return "stable"


def detect_seasonality(values: np.ndarray, threshold: float = 0.3) -> bool:
    """Simple seasonality detection using autocorrelation."""
    if len(values) < 6:
        return False
    
    # Check for patterns at common seasonal lags (3, 6 months)
    for lag in [3, 6]:
        if len(values) > lag * 2:
            corr = np.corrcoef(values[:-lag], values[lag:])[0, 1]
            if abs(corr) > threshold:
                return True
    return False


def detect_anomalies(values: np.ndarray, threshold: float = 2.0) -> List[int]:
    """Detect anomalies using z-score method."""
    if len(values) < 5:
        return []
    
    mean = np.mean(values)
    std = np.std(values)
    if std == 0:
        return []
    
    z_scores = np.abs((values - mean) / std)
    return list(np.where(z_scores > threshold)[0])


# =============================================================================
# FORECASTING MODELS
# =============================================================================

def linear_regression_forecast(
    values: np.ndarray, 
    confidence: float = 0.95
) -> Tuple[float, float, float]:
    """
    Linear Regression forecast with confidence interval.
    Returns: (prediction, lower_bound, upper_bound)
    """
    n = len(values)
    X = np.arange(n).reshape(-1, 1)
    y = values
    
    model = LinearRegression()
    model.fit(X, y)
    
    # Predict next time step
    next_t = np.array([[n]])
    prediction = float(model.predict(next_t)[0])
    
    # Calculate prediction interval
    y_pred = model.predict(X)
    residuals = y - y_pred
    std_error = np.std(residuals)
    
    # t-value for confidence interval (approximation for large n)
    from scipy import stats
    t_val = stats.t.ppf((1 + confidence) / 2, n - 2)
    
    margin = t_val * std_error * np.sqrt(1 + 1/n + (n - np.mean(X))**2 / np.sum((X - np.mean(X))**2))
    
    return (
        max(0, prediction),
        max(0, prediction - margin),
        prediction + margin
    )


def exponential_smoothing_forecast(
    values: np.ndarray,
    alpha: float = 0.3,
    beta: float = 0.1,
    confidence: float = 0.95
) -> Tuple[float, float, float]:
    """
    Double Exponential Smoothing (Holt's method) for trend forecasting.
    Returns: (prediction, lower_bound, upper_bound)
    """
    n = len(values)
    if n < 3:
        return (values[-1], values[-1] * 0.8, values[-1] * 1.2)
    
    # Initialize
    level = values[0]
    trend = values[1] - values[0] if n > 1 else 0
    
    # Smooth
    for i in range(1, n):
        prev_level = level
        level = alpha * values[i] + (1 - alpha) * (level + trend)
        trend = beta * (level - prev_level) + (1 - beta) * trend
    
    # Forecast
    prediction = level + trend
    
    # Simple confidence interval based on historical error
    fitted = []
    l, t = values[0], (values[1] - values[0]) if n > 1 else 0
    for i in range(1, n):
        fitted.append(l + t)
        prev_l = l
        l = alpha * values[i] + (1 - alpha) * (l + t)
        t = beta * (l - prev_l) + (1 - beta) * t
    
    errors = values[1:] - np.array(fitted)
    std_error = np.std(errors) if len(errors) > 0 else values.std() * 0.1
    
    from scipy import stats
    z_val = stats.norm.ppf((1 + confidence) / 2)
    margin = z_val * std_error
    
    return (
        max(0, prediction),
        max(0, prediction - margin),
        prediction + margin
    )


def moving_average_forecast(
    values: np.ndarray,
    window: int = 3,
    confidence: float = 0.95
) -> Tuple[float, float, float]:
    """
    Simple Moving Average forecast.
    Returns: (prediction, lower_bound, upper_bound)
    """
    if len(values) < window:
        window = len(values)
    
    prediction = np.mean(values[-window:])
    
    # Confidence interval from historical variation
    if len(values) >= window * 2:
        ma_values = np.convolve(values, np.ones(window)/window, mode='valid')
        actual = values[window-1:]
        errors = actual - ma_values
        std_error = np.std(errors)
    else:
        std_error = np.std(values) * 0.2
    
    from scipy import stats
    z_val = stats.norm.ppf((1 + confidence) / 2)
    margin = z_val * std_error
    
    return (
        max(0, prediction),
        max(0, prediction - margin),
        prediction + margin
    )


# =============================================================================
# BACKTESTING & METRICS
# =============================================================================

def calculate_metrics(
    actual: np.ndarray, 
    predicted: np.ndarray
) -> Tuple[float, float]:
    """Calculate MAE and MAPE."""
    mae = float(np.mean(np.abs(actual - predicted)))
    
    # MAPE with protection against division by zero
    mask = actual != 0
    if mask.sum() > 0:
        mape = float(np.mean(np.abs((actual[mask] - predicted[mask]) / actual[mask])) * 100)
    else:
        mape = 0.0
    
    return round(mae, 2), round(mape, 2)


def backtest_model(
    values: np.ndarray,
    model_func,
    test_size: int = 2
) -> Tuple[float, float]:
    """
    Backtest a model using walk-forward validation.
    Returns MAE and MAPE on test set.
    """
    if len(values) < test_size + 3:
        return float("nan"), float("nan")
    
    train = values[:-test_size]
    test = values[-test_size:]
    
    predictions = []
    for i in range(test_size):
        train_data = values[:len(train) + i]
        pred, _, _ = model_func(train_data)
        predictions.append(pred)
    
    return calculate_metrics(test, np.array(predictions))


# =============================================================================
# MAIN FORECAST API
# =============================================================================

def forecast_next_month(
    df: pd.DataFrame, 
    product: str,
    model: ModelType = ModelType.LINEAR
) -> Dict:
    """
    Generate forecast for a product using specified model.
    """
    product_df = df[df["product"] == product].copy()
    if product_df.empty:
        raise ValueError(f"Unknown product: {product}")
    
    values = product_df["units_sold"].values.astype(float)
    
    # Select model
    model_funcs = {
        ModelType.LINEAR: linear_regression_forecast,
        ModelType.ETS: exponential_smoothing_forecast,
        ModelType.MOVING_AVG: moving_average_forecast,
    }
    model_func = model_funcs[model]
    
    # Generate forecast
    prediction, lower, upper = model_func(values)
    
    # Backtest metrics
    mae, mape = backtest_model(values, model_func)
    
    # Detect patterns
    trend = detect_trend(values)
    seasonality = detect_seasonality(values)
    anomalies = detect_anomalies(values)
    
    # Next month calculation
    last_month = product_df["month"].max()
    next_month = (last_month + pd.offsets.MonthBegin(1)).to_period("M").to_timestamp()
    
    # Suggested stock with buffer based on model confidence
    buffer = 1.15 if mae < 10 else 1.20  # More buffer for less accurate models
    suggested_stock = int(round(prediction * buffer))
    
    return {
        "product": product,
        "model": model.value,
        "next_month": next_month.strftime("%Y-%m"),
        "predicted_units": round(prediction, 2),
        "lower_bound": round(lower, 2),
        "upper_bound": round(upper, 2),
        "suggested_stock": suggested_stock,
        "mae_last_2_months": mae,
        "mape_last_2_months": mape,
        "trend": trend,
        "seasonality_detected": seasonality,
        "anomaly_indices": anomalies,
        "history": [
            {"month": m.strftime("%Y-%m"), "units_sold": int(u)}
            for m, u in zip(product_df["month"], product_df["units_sold"])
        ],
    }


def compare_models(df: pd.DataFrame, product: str) -> Dict:
    """
    Compare all models for a given product.
    Returns forecasts and metrics for each model.
    """
    results = {}
    for model_type in ModelType:
        try:
            results[model_type.value] = forecast_next_month(df, product, model_type)
        except Exception as e:
            results[model_type.value] = {"error": str(e)}
    
    # Determine best model by MAE
    valid_results = {k: v for k, v in results.items() if "mae_last_2_months" in v and not np.isnan(v["mae_last_2_months"])}
    if valid_results:
        best_model = min(valid_results.keys(), key=lambda k: valid_results[k]["mae_last_2_months"])
    else:
        best_model = ModelType.LINEAR.value
    
    return {
        "product": product,
        "best_model": best_model,
        "models": results
    }


def get_portfolio_summary(df: pd.DataFrame) -> Dict:
    """
    Generate summary statistics across all products.
    """
    products = list_products(df)
    
    total_forecast = 0
    total_stock = 0
    trends = {"up": 0, "down": 0, "stable": 0}
    
    product_summaries = []
    for product in products:
        forecast = forecast_next_month(df, product, ModelType.LINEAR)
        total_forecast += forecast["predicted_units"]
        total_stock += forecast["suggested_stock"]
        trends[forecast["trend"]] += 1
        
        product_summaries.append({
            "product": product,
            "predicted_units": forecast["predicted_units"],
            "trend": forecast["trend"],
            "mae": forecast["mae_last_2_months"]
        })
    
    return {
        "total_products": len(products),
        "total_forecast_units": round(total_forecast, 2),
        "total_suggested_stock": total_stock,
        "trend_summary": trends,
        "products": product_summaries
    }


if __name__ == "__main__":
    # Test the module
    df = load_sales("data/sales.csv")
    print("=" * 60)
    print("RETAIL SALES FORECASTING ENGINE")
    print("=" * 60)
    print(f"\nLoaded {len(df)} records for {len(list_products(df))} products\n")
    
    for product in list_products(df):
        print(f"\n{'='*40}")
        print(f"Product: {product}")
        print("="*40)
        
        comparison = compare_models(df, product)
        print(f"Best Model: {comparison['best_model']}")
        
        for model_name, result in comparison["models"].items():
            if "error" not in result:
                print(f"\n  {model_name}:")
                print(f"    Prediction: {result['predicted_units']} units")
                print(f"    Confidence: [{result['lower_bound']}, {result['upper_bound']}]")
                print(f"    MAE: {result['mae_last_2_months']}, MAPE: {result['mape_last_2_months']}%")
                print(f"    Trend: {result['trend']}, Seasonality: {result['seasonality_detected']}")
