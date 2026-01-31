# 📊 Retail Sales Forecast Engine

A production-ready demand forecasting dashboard with multi-model comparison, confidence intervals, and intelligent inventory recommendations.

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat&logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.109-009688?style=flat&logo=fastapi&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green?style=flat)

<p align="center">
  <img src="docs/screenshot-light.png" alt="Dashboard Light Mode" width="48%">
  <img src="docs/screenshot-dark.png" alt="Dashboard Dark Mode" width="48%">
</p>

## ✨ Features

### Multi-Model Forecasting
- **Linear Regression** — Trend-based extrapolation for steady growth patterns
- **Exponential Smoothing (Holt)** — Adaptive smoothing for trending data
- **Moving Average** — Baseline comparison for stable demand

### Intelligent Analytics
- **95% Confidence Intervals** — Quantified prediction uncertainty
- **Automatic Model Selection** — Best model chosen by backtest MAE
- **Trend Detection** — Automatic up/down/stable classification
- **Seasonality Detection** — Autocorrelation-based pattern recognition
- **Anomaly Flagging** — Z-score outlier detection

### Inventory Optimization
- **Smart Stock Recommendations** — Forecast + adaptive safety buffer
- **Buffer scaling** — Larger buffer for less confident predictions

### Modern Dashboard
- **Dark/Light Theme** — System-aware with manual toggle
- **Real-time Updates** — Instant model switching
- **Responsive Design** — Works on desktop and mobile
- **Interactive Charts** — Hover tooltips, confidence bands

---

## 🚀 Quick Start

### Prerequisites
- Python 3.10+
- pip

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/retail-forecast-dashboard.git
cd retail-forecast-dashboard

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Run the API

```bash
python -m uvicorn app.main:app --reload --port 8000
```

### Open the Dashboard

Option A: Open `static/index.html` directly in your browser

Option B: Visit http://127.0.0.1:8000/dashboard

---

## 📁 Project Structure

```
retail-forecast-dashboard/
├── app/
│   └── main.py              # FastAPI application
├── analysis/
│   └── forecast.py          # Forecasting models & analytics
├── data/
│   └── sales.csv            # Sample retail data
├── static/
│   └── index.html           # Dashboard frontend
├── reports/                  # Generated PDF reports
├── tests/
│   └── test_forecast.py     # Unit tests
├── requirements.txt
└── README.md
```

---

## 🔌 API Reference

### Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/` | Health check |
| `GET` | `/products` | List all products |
| `GET` | `/sales?product={name}` | Historical sales data |
| `GET` | `/forecast?product={name}&model={type}` | Generate forecast |
| `GET` | `/compare?product={name}` | Compare all models |
| `GET` | `/summary` | Portfolio-wide analytics |
| `GET` | `/trends` | Trend analysis for all products |
| `GET` | `/docs` | Interactive API documentation |

### Model Types

| Model | API Value | Best For |
|-------|-----------|----------|
| Linear Regression | `linear_regression` | Steady trends |
| Exponential Smoothing | `exponential_smoothing` | Adaptive trends |
| Moving Average | `moving_average` | Stable demand |

### Example Response

```json
{
  "product": "Bread",
  "model": "linear_regression",
  "next_month": "2026-02",
  "predicted_units": 218.45,
  "lower_bound": 195.2,
  "upper_bound": 241.7,
  "suggested_stock": 251,
  "mae_last_2_months": 8.34,
  "mape_last_2_months": 4.12,
  "trend": "up",
  "seasonality_detected": false
}
```

---

## 📈 Model Details

### Linear Regression
Simple but effective for products with consistent growth. Fits a line through historical data and extrapolates.

$$\hat{y}_{t+1} = \beta_0 + \beta_1 \cdot t$$

### Exponential Smoothing (Holt's Method)
Double exponential smoothing that adapts to changing trends:

$$L_t = \alpha y_t + (1-\alpha)(L_{t-1} + T_{t-1})$$
$$T_t = \beta(L_t - L_{t-1}) + (1-\beta)T_{t-1}$$

### Backtesting
Models are validated using walk-forward validation:
- Train on first 10 months
- Predict months 11-12
- Calculate MAE and MAPE on held-out data

---

## 🛠️ Development

### Running Tests

```bash
pytest tests/ -v
```

### Adding New Models

1. Add forecast function in `analysis/forecast.py`
2. Register in `ModelType` enum
3. Add to `model_funcs` dictionary in `forecast_next_month()`
4. Add tab in dashboard HTML

### Customization

**Change confidence level:**
```python
# In forecast.py
linear_regression_forecast(values, confidence=0.90)  # 90% CI
```

**Adjust safety buffer:**
```python
# In forecast.py, forecast_next_month()
buffer = 1.20  # 20% safety stock
```

---

## 🎯 Roadmap

- [ ] ARIMA/SARIMA models
- [ ] Prophet integration
- [ ] Multi-step forecasting (3, 6, 12 months)
- [ ] CSV upload in dashboard
- [ ] PostgreSQL backend
- [ ] Docker deployment
- [ ] Automated PDF report generation

---

## 📜 License

MIT License — see [LICENSE](LICENSE) for details.

---

## 🤝 Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

<p align="center">
  <b>Built with ❤️ for data-driven inventory management</b>
</p>
