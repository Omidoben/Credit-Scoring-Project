"""
Credit Scoring API
FastAPI application for M-Pesa transaction-based credit scoring

Author: OmidoBenard
Version: 1.0.0
"""

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
import os
from pydantic import BaseModel, Field, validator
from typing import List, Optional, Dict, Any
from datetime import datetime
import uvicorn

from src.pipeline.predict_pipeline import PredictPipeline
from src.utils.logger import get_logger
from src.utils.exception import CreditScoringException
from src.utils.config_loader import ConfigLoader

logger = get_logger(__name__)

config = ConfigLoader()
api_config = config.get('api', {})

# Initialize FastAPI app
app = FastAPI(
    title=api_config.get("title", "Credit Scoring API"),
    description=api_config.get("description", "API for predicting loan default risk based on M-Pesa transaction data"),
    version=api_config.get("version", "1.0.0"),
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global prediction pipeline
prediction_pipeline: Optional[PredictPipeline] = None

@app.get("/", response_class=HTMLResponse, include_in_schema=False)
async def serve_frontend():
    """Serve the frontend HTML"""
    html_path = os.path.join("templates", "index.html")
    with open(html_path, 'r') as f:
        return HTMLResponse(content=f.read())

# PYDANTIC MODELS (Request/Response Schemas)

class Transaction(BaseModel):
    """Transaction data model"""
    date: str = Field(..., description="Transaction date (YYYY-MM-DD)")
    transaction_type: str = Field(..., description="Type of transaction (Income, Expense, etc.)")
    amount: float = Field(..., description="Transaction amount")
    balance: float = Field(..., description="Account balance after transaction")
    description: Optional[str] = Field(None, description="Transaction description")
    
    @validator('date')
    def validate_date(cls, v):
        try:
            datetime.strptime(v, '%Y-%m-%d')
            return v
        except ValueError:
            raise ValueError('Date must be in YYYY-MM-DD format')
    
    @validator('transaction_type')
    def validate_transaction_type(cls, v):
        valid_types = ['Income', 'Merchant Payment', 'Transfer', 'Withdrawal', 
                      'Airtime', 'Fuliza Credit', 'Loan Repayment', 'Savings In']
        if v not in valid_types:
            raise ValueError(f'Invalid transaction type. Must be one of: {valid_types}')
        return v


class Demographics(BaseModel):
    """Demographics data model"""
    age: int = Field(..., ge=18, le=70, description="Applicant age (18-70)")
    employment_type: str = Field(..., description="Employment type (Salaried, Self-employed, Informal, Student/Other)")
    years_employed: float = Field(..., ge=0, description="Years of employment")
    education_level: str = Field(..., description="Education level (Primary, Secondary, Tertiary)")
    
    @validator('employment_type')
    def validate_employment(cls, v):
        valid_types = ['Salaried', 'Self-employed', 'Informal', 'Student/Other']
        if v not in valid_types:
            raise ValueError(f'Invalid employment type. Must be one of: {valid_types}')
        return v
    
    @validator('education_level')
    def validate_education(cls, v):
        valid_levels = ['Primary', 'Secondary', 'Tertiary']
        if v not in valid_levels:
            raise ValueError(f'Invalid education level. Must be one of: {valid_levels}')
        return v


class FinancialHistory(BaseModel):
    """Financial history data model"""
    account_age_months: int = Field(..., ge=0, description="Account age in months")
    past_defaults_count: int = Field(..., ge=0, description="Number of past defaults")
    past_repaid_loans: int = Field(..., ge=0, description="Number of loans repaid")
    credit_score: Optional[int] = Field(None, ge=300, le=850, description="Credit score (300-850)")
    overdraft_history: int = Field(..., ge=0, description="Number of overdraft occurrences")


class PredictionRequest(BaseModel):
    """Request model for prediction from raw data"""
    demographics: Demographics
    financial_history: FinancialHistory
    transactions: List[Transaction] = Field(..., min_items=1, description="List of transactions (at least 1)")
    
    class Config:
        schema_extra = {
            "example": {
                "demographics": {
                    "age": 35,
                    "employment_type": "Salaried",
                    "years_employed": 8,
                    "education_level": "Secondary"
                },
                "financial_history": {
                    "account_age_months": 48,
                    "past_defaults_count": 0,
                    "past_repaid_loans": 3,
                    "credit_score": 680,
                    "overdraft_history": 2
                },
                "transactions": [
                    {
                        "date": "2024-01-01",
                        "transaction_type": "Income",
                        "amount": 50000,
                        "balance": 50000,
                        "description": "Salary deposit"
                    },
                    {
                        "date": "2024-01-02",
                        "transaction_type": "Merchant Payment",
                        "amount": -5000,
                        "balance": 45000,
                        "description": "Rent payment"
                    }
                ]
            }
        }


class FeaturePredictionRequest(BaseModel):
    """Request model for prediction from pre-computed features"""
    monthly_avg_income: float
    income_volatility_cv: float
    income_trend_6months: float
    num_income_sources: int
    monthly_net_cashflow: float
    months_negative_cashflow: int
    balance_min: float
    months_balance_below_500: int
    fuliza_frequency: float
    fuliza_repayment_rate: float
    debt_to_income_ratio: float
    avg_fuliza_amount: float
    transaction_frequency: float
    savings_rate: float
    expense_consistency_cv: float
    age: int
    years_employed: float
    education_level_encoded: int
    employment_Salaried: int
    employment_Self_employed: int = Field(alias='employment_Self-employed')
    employment_Student_Other: int = Field(alias='employment_Student/Other')
    account_age_months: int
    past_defaults_count: int
    past_repaid_loans: int
    credit_score: Optional[int] = None
    overdraft_history: int
    
    class Config:
        allow_population_by_field_name = True


class PredictionResponse(BaseModel):
    """Response model for predictions"""
    prediction: int = Field(..., description="Binary prediction (0=Default, 1=Repaid)")
    prediction_label: str = Field(..., description="Human-readable prediction")
    probability_default: float = Field(..., description="Probability of default (0-1)")
    probability_repay: float = Field(..., description="Probability of repayment (0-1)")
    risk_level: str = Field(..., description="Risk level (Low, Medium, High)")
    recommendation: str = Field(..., description="Loan recommendation")
    timestamp: str = Field(..., description="Prediction timestamp")


class HealthResponse(BaseModel):
    """Response model for health check"""
    status: str
    timestamp: str
    model_loaded: bool
    version: str


class ErrorResponse(BaseModel):
    """Response model for errors"""
    error: str
    detail: Optional[str] = None
    timestamp: str

# Startup & Shutdown events

@app.on_event("startup")
async def startup_event():
    """Load model and initialize pipeline on startup"""
    global prediction_pipeline
    
    try:
        logger.info("Starting Credit Scoring API")
        
        logger.info("Loading prediction pipeline...")
        prediction_pipeline = PredictPipeline()
        
        logger.info("API started successfully")
        logger.info(f"  - Model loaded: {prediction_pipeline.predict_config.model_path}")
        logger.info(f"  - Features: {len(prediction_pipeline.feature_columns)}")
        
    except Exception as e:
        logger.error(f"Failed to start API: {e}")
        raise


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    logger.info("Shutting down Credit Scoring API...")


# API ENDPOINTS

@app.get("/", tags=["General"])
async def root():
    """Root endpoint"""
    return {
        "message": "Credit Scoring API",
        "version": "1.0.0",
        "documentation": "/docs",
        "health": "/health"
    }


@app.get("/health", response_model=HealthResponse, tags=["General"])
async def health_check():
    """
    Health check endpoint
    
    Returns service status and model availability
    """
    return {
        "status": "healthy" if prediction_pipeline is not None else "unhealthy",
        "timestamp": datetime.now().isoformat(),
        "model_loaded": prediction_pipeline is not None,
        "version": "1.0.0"
    }


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
async def predict_from_transactions(request: PredictionRequest):
    """
    Make prediction from raw transaction data
    
    This endpoint accepts demographics, financial history, and transaction data,
    extracts features, and returns a credit score prediction.
    
    - **demographics**: Applicant demographics (age, employment, education)
    - **financial_history**: Past financial behavior
    - **transactions**: List of M-Pesa transactions (minimum 1 transaction)
    
    Returns prediction with probability scores and recommendation.
    """
    try:
        if prediction_pipeline is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Prediction pipeline not initialized"
            )
        
        logger.info("Received prediction request")
        
        # Convert request to dictionary
        applicant_data = {
            'demographics': request.demographics.dict(),
            'financial_history': request.financial_history.dict(),
            'transactions': [t.dict() for t in request.transactions]
        }
        
        # Make prediction
        result = prediction_pipeline.predict_from_raw_data(applicant_data)
        
        # Add timestamp
        result['timestamp'] = datetime.now().isoformat()
        
        logger.info(f"Prediction complete: {result['prediction_label']}")
        
        return result
    
    except CreditScoringException as e:
        logger.error(f"Prediction failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )
    
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction failed: {str(e)}"
        )


@app.post("/predict-features", response_model=PredictionResponse, tags=["Prediction"])
async def predict_from_features(request: FeaturePredictionRequest):
    """
    Make prediction from pre-computed features
    
    This endpoint accepts pre-computed features and returns a prediction.
    Use this if you've already extracted features from transaction data.
    
    Faster than /predict endpoint as it skips feature extraction.
    """
    try:
        if prediction_pipeline is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Prediction pipeline not initialized"
            )
        
        logger.info("Received feature-based prediction request")
        
        # Convert request to dictionary
        features = request.dict(by_alias=True)
        
        # Make prediction
        result = prediction_pipeline.predict_from_features(features)
        
        # Add timestamp
        result['timestamp'] = datetime.now().isoformat()
        
        logger.info(f"Prediction complete: {result['prediction_label']}")
        
        return result
    
    except CreditScoringException as e:
        logger.error(f"Prediction failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )
    
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction failed: {str(e)}"
        )


@app.get("/model-info", tags=["Model"])
async def get_model_info():
    """
    Get information about the loaded model
    
    Returns model metadata including performance metrics
    """
    try:
        if prediction_pipeline is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Prediction pipeline not initialized"
            )
        
        import json
        import os
        
        # Load model metadata
        metadata_path = os.path.join(
            prediction_pipeline.predict_config.model_path.replace('.pkl', '_metadata.json')
        )
        
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
        else:
            metadata = {}
        
        return {
            "model_path": prediction_pipeline.predict_config.model_path,
            "num_features": len(prediction_pipeline.feature_columns),
            "feature_names": prediction_pipeline.feature_columns,
            "metadata": metadata
        }
    
    except Exception as e:
        logger.error(f"Error fetching model info: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


# Error Handlers

@app.exception_handler(ValueError)
async def value_error_handler(request, exc):
    """Handle validation errors"""
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={
            "error": "Validation Error",
            "detail": str(exc),
            "timestamp": datetime.now().isoformat()
        }
    )


@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """Handle general exceptions"""
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "error": "Internal Server Error",
            "detail": str(exc),
            "timestamp": datetime.now().isoformat()
        }
    )


# ============================================================================
# RUN SERVER
# ============================================================================

if __name__ == "__main__":
    # Get host and port from config
    host = api_config.get('host', '0.0.0.0')
    port = api_config.get('port', 8000)
    
    logger.info(f"Starting server on {host}:{port}")
    
    uvicorn.run(
        "app:app",
        host=host,
        port=port,
        reload=True,  # Set to False in production
        log_level="info"
    )