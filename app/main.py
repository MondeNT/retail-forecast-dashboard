"""
Retail Sales Forecasting API
============================
Production-ready FastAPI backend with multi-model forecasting,
model comparison, and portfolio analytics.
"""

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pathlib import Path
from enum import Enum
from typing import Optional
import logging

from analysis.forecast import (
    load_sales, 
    forecast_next_month, 
    list_products,
    compare_models,
    get_portfolio_summary,
    ModelType
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "sales.csv"
REPORTS_DIR = BASE_DIR / "reports"
STATIC_DIR = BASE_DIR / "static"

# Create app
app = FastAPI(
    title="Retail Sales Forecasting Engine",
    description="""
    Multi-model demand forecasting API with confidence intervals,
    trend analysis, and inventory recommendations.
    
    ## Features
    - **Multiple Models**: Linear Regression, Exponential Smoothing, Moving Average
    - **Confidence Intervals**: 95% prediction intervals
    - **Model Comparison**: Automatic best-model selection
    - **Portfolio Analytics**: Cross-product insights
    - **Backtesting Metrics**: MAE and MAPE validation
    """,
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS for local development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files if directory exists
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


# =============================================================================
# HEALTH & INFO
# =============================================================================

@app.get("/", tags=["Health"])
def root():
    """API health check and info."""
    return {
        "status": "ok",
        "message": "Retail Forecasting Engine API v2.0",
        "endpoints": {
            "products": "/products",
            "sales": "/sales?product={name}",
            "forecast": "/forecast?product={name}&model={model}",
            "compare": "/compare?product={name}",
            "summary": "/summary",
            "docs": "/docs"
        }
    }


@app.get("/health", tags=["Health"])
def health_check():
    """Detailed health check."""
    try:
        df = load_sales(str(DATA_PATH))
        return {
            "status": "healthy",
            "data_loaded": True,
            "products_count": len(list_products(df)),
            "records_count": len(df)
        }
    except Exception as e:
        return JSONResponse(
            status_code=503,
            content={"status": "unhealthy", "error": str(e)}
        )


# =============================================================================
# PRODUCTS & SALES
# =============================================================================

@app.get("/products", tags=["Data"])
def get_products():
    """List all available products."""
    try:
        df = load_sales(str(DATA_PATH))
        products = list_products(df)
        return {
            "count": len(products),
            "products": products
        }
    except Exception as e:
        logger.error(f"Error loading products: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/sales", tags=["Data"])
def get_sales(
    product: str = Query(..., description="Product name")
):
    """Get historical sales data for a product."""
    try:
        df = load_sales(str(DATA_PATH))
        product_df = df[df["product"] == product].copy()
        
        if product_df.empty:
            raise HTTPException(status_code=404, detail=f"Product not found: {product}")
        
        history = [
            {"month": m.strftime("%Y-%m"), "units_sold": int(u)}
            for m, u in zip(product_df["month"], product_df["units_sold"])
        ]
        
        return {
            "product": product,
            "record_count": len(history),
            "date_range": {
                "start": history[0]["month"],
                "end": history[-1]["month"]
            },
            "history": history
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching sales for {product}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# FORECASTING
# =============================================================================

@app.get("/forecast", tags=["Forecasting"])
def get_forecast(
    product: str = Query(..., description="Product name"),
    model: ModelType = Query(ModelType.LINEAR, description="Forecasting model to use")
):
    """
    Generate next-month forecast for a product.
    
    - **product**: Product name (required)
    - **model**: One of 'linear_regression', 'exponential_smoothing', 'moving_average'
    """
    try:
        df = load_sales(str(DATA_PATH))
        
        if product not in list_products(df):
            raise HTTPException(status_code=404, detail=f"Product not found: {product}")
        
        result = forecast_next_month(df, product, model)
        return result
    
    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error forecasting {product}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/compare", tags=["Forecasting"])
def compare_all_models(
    product: str = Query(..., description="Product name")
):
    """
    Compare all forecasting models for a product.
    Returns predictions and accuracy metrics for each model.
    """
    try:
        df = load_sales(str(DATA_PATH))
        
        if product not in list_products(df):
            raise HTTPException(status_code=404, detail=f"Product not found: {product}")
        
        return compare_models(df, product)
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error comparing models for {product}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# ANALYTICS
# =============================================================================

@app.get("/summary", tags=["Analytics"])
def portfolio_summary():
    """
    Get portfolio-wide summary with total forecasts and trend analysis.
    """
    try:
        df = load_sales(str(DATA_PATH))
        return get_portfolio_summary(df)
    except Exception as e:
        logger.error(f"Error generating summary: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/trends", tags=["Analytics"])
def get_trends():
    """
    Get trend analysis for all products.
    """
    try:
        df = load_sales(str(DATA_PATH))
        products = list_products(df)
        
        trends = []
        for product in products:
            forecast = forecast_next_month(df, product, ModelType.LINEAR)
            trends.append({
                "product": product,
                "trend": forecast["trend"],
                "seasonality_detected": forecast["seasonality_detected"],
                "predicted_change": round(
                    ((forecast["predicted_units"] - forecast["history"][-1]["units_sold"]) 
                     / forecast["history"][-1]["units_sold"]) * 100, 
                    1
                )
            })
        
        return {"trends": trends}
    
    except Exception as e:
        logger.error(f"Error analyzing trends: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# REPORTS
# =============================================================================

@app.get("/report/latest", tags=["Reports"])
def latest_report():
    """Download the latest PDF executive summary."""
    if not REPORTS_DIR.exists():
        raise HTTPException(status_code=404, detail="No reports directory found")
    
    pdfs = sorted(REPORTS_DIR.glob("executive_summary_*.pdf"))
    if not pdfs:
        raise HTTPException(status_code=404, detail="No reports generated yet")
    
    latest_pdf = pdfs[-1]
    return FileResponse(
        str(latest_pdf),
        media_type="application/pdf",
        filename=latest_pdf.name
    )


@app.get("/reports", tags=["Reports"])
def list_reports():
    """List all available reports."""
    if not REPORTS_DIR.exists():
        return {"reports": []}
    
    pdfs = sorted(REPORTS_DIR.glob("*.pdf"))
    return {
        "count": len(pdfs),
        "reports": [
            {
                "name": p.name,
                "url": f"/report/{p.stem}"
            }
            for p in pdfs
        ]
    }


# =============================================================================
# STATIC DASHBOARD
# =============================================================================

@app.get("/dashboard", tags=["Dashboard"])
def serve_dashboard():
    """Serve the dashboard HTML."""
    dashboard_path = STATIC_DIR / "index.html"
    if dashboard_path.exists():
        return FileResponse(str(dashboard_path))
    raise HTTPException(status_code=404, detail="Dashboard not found")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
