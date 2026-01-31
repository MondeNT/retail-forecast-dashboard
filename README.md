# Retail Sales Forecast Engine

A production-ready demand forecasting dashboard with multi-model comparison, confidence intervals, and intelligent inventory recommendations.

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat&logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.109-009688?style=flat&logo=fastapi&logoColor=white)
![Chart.js](https://img.shields.io/badge/Chart.js-4.x-FF6384?style=flat&logo=chartdotjs&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green?style=flat)

---

## Dashboard Preview

### Light Mode
Interactive demand forecast with hover tooltips, model switching tabs (Linear, Exp Smooth, Moving Avg), and a decision summary panel showing predicted units, suggested stock levels, confidence intervals, and model comparison metrics.

<img width="1482" height="821" alt="Image" src="https://github.com/user-attachments/assets/71201a29-4249-480d-b656-7b2784e0430d" />

### Dark Mode
Full dark theme with automatic system detection and manual toggle. The product dropdown lets you switch between all tracked items. All charts, KPIs, and confidence bands adapt to the selected theme.

![Dashboard - Dark Mode](docs/screenshot-dark.png)

---

## Features

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
- **Buffer Scaling** — Larger buffer applied when model confidence is lower

### Dashboard
- **Dark/Light Theme** — System-aware with manual toggle
- **Real-time Model Switching** — Instantly compare Linear, Exp Smoothing, and Moving Average
- **Responsive Layout** — Works on desktop, tablet, and mobile
- **Interactive Charts** — Hover tooltips, forecast line, and confidence bands

---

## Quick Start

### Prerequisites
- Python 3.10+
- pip

### Installation

```bash
git clone https://github.com/MondeNT/retail-forecast-dashboard.git
cd retail-forecast-dashboard

python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

pip install -r requirements.txt
```

### Run the API

```bash
python3 -m uvicorn app.main:app --reload --port 8000
```

### Open the Dashboard

Visit [http://127.0.0.1:8000/dashboard](http://127.0.0.1:8000/dashboard)

API documentation is available at [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

---

## Project Structure

```
retail-forecast-dashboard/
├── app/
│   └── main.py              # FastAPI application
├── analysis/
│   └── forecast.py          # Forecasting models and analytics
├── data/
│   └── sales.csv            # Sample retail sales data (5 products, 12 months)
├── static/
│   └── index.html           # Dashboard frontend (Chart.js)
├── tests/
│   └── test_forecast.py     # Unit and integration tests
├── reports/                  # Generated PDF reports
├── docs/                     # Screenshots and documentation assets
├── Dockerfile
├── docker-compose.yml
├── Makefile
├── pyproject.toml
├── requirements.txt
└── README.md
```

---

## API Reference

### Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/` | Health check and endpoint listing |
| `GET` | `/health` | Detailed health status |
| `GET` | `/products` | List all available products |
| `GET` | `/sales?product={name}` | Historical sales data for a product |
| `GET` | `/forecast?product={name}&model={type}` | Generate forecast with selected model |
| `GET` | `/compare?product={name}` | Compare all models for a product |
| `GET` | `/summary` | Portfolio-wide analytics |
| `GET` | `/trends` | Trend analysis across all products |
| `GET` | `/docs` | Interactive Swagger documentation |

### Model Types

| Model | API Value | Best For |
|-------|-----------|----------|
| Linear Regression | `linear_regression` | Steady, consistent growth |
| Exponential Smoothing | `exponential_smoothing` | Adaptive, changing trends |
| Moving Average | `moving_average` | Stable, low-variance demand |

### Example Response

```json
{
  "product": "Bread",
  "model": "linear_regression",
  "next_month": "2026-02",
  "predicted_units": 208.56,
  "lower_bound": 188.84,
  "upper_bound": 228.28,
  "suggested_stock": 240,
  "mae_last_2_months": 15.98,
  "mape_last_2_months": 7.85,
  "trend": "up",
  "seasonality_detected": false,
  "anomaly_indices": []
}
```

---

## Model Details

### Linear Regression
Fits a trend line through historical data and extrapolates to the next time step. Effective for products with consistent growth patterns.

### Exponential Smoothing (Holt's Method)
Double exponential smoothing that maintains a level and trend component, adapting to recent changes in the data through configurable smoothing parameters (alpha, beta).

### Moving Average
Averages the most recent observations (default window of 3 months) as a forecast. Serves as a simple baseline for comparison against more sophisticated models.

### Backtesting
All models are validated using walk-forward validation:
- Train on the first 10 months of data
- Predict months 11 and 12
- Calculate MAE (Mean Absolute Error) and MAPE (Mean Absolute Percentage Error) on the held-out set

---

## Development

### Running Tests

```bash
pytest tests/ -v
```

### Adding a New Model

1. Add the forecast function in `analysis/forecast.py`
2. Register it in the `ModelType` enum
3. Add it to the `model_funcs` dictionary in `forecast_next_month()`
4. Add a corresponding tab in the dashboard HTML

### Docker

```bash
docker build -t forecast-dashboard .
docker run -p 8000:8000 forecast-dashboard
```

Or with Docker Compose:

```bash
docker-compose up
```

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Backend | Python, FastAPI, Uvicorn |
| Forecasting | scikit-learn, SciPy, NumPy, Pandas |
| Frontend | HTML, CSS, JavaScript, Chart.js |
| Testing | pytest |
| Deployment | Docker, Docker Compose |

---

## Roadmap

- [ ] ARIMA / SARIMA models
- [ ] Prophet integration
- [ ] Multi-step forecasting (3, 6, 12 months)
- [ ] CSV upload via the dashboard
- [ ] PostgreSQL backend
- [ ] Automated PDF report generation
- [ ] CI/CD with GitHub Actions

---

## License

MIT License — see [LICENSE](LICENSE) for details.

---

## Author

**Monde Ntjatje** — [github.com/MondeNT](https://github.com/MondeNT)
