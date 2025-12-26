from fastapi import FastAPI, Depends, HTTPException, UploadFile, File, status
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
from pydantic import BaseModel, field_validator
from contextlib import asynccontextmanager
import pandas as pd
import joblib
import uvicorn
from datetime import datetime
from typing import Optional
import io

from database import get_db, init_db, test_connection, User, Prediction
from auth import router as auth_router
from auth_utils import get_current_active_user


# LOAD MODEL ARTIFACTS (GLOBAL)

model = None
scaler = None
label_encoders = None
feature_names = None


# LIFESPAN EVENT HANDLER

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan event handler for startup and shutdown"""
    # Startup
    global model, scaler, label_encoders, feature_names
    
    try:
        model = joblib.load("model.pkl")
        scaler = joblib.load("scaler.pkl")
        label_encoders = joblib.load("label_encoders.pkl")
        feature_names = joblib.load("feature_names.pkl")
        print(" Model artifacts loaded successfully")
    except Exception as e:
        print(f" Error loading model artifacts: {e}")
        raise
    
    if test_connection():
        init_db()
        print(" Database initialized successfully")
    else:
        print(" Database connection failed")
    
    yield
    
    # Shutdown (cleanup if needed)
    print(" Application shutting down...")


# APP INITIALIZATION

app = FastAPI(
    title="Loan Prediction System",
    version="2.0.0",
    description="AI-powered loan approval prediction system",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(auth_router)


# VALUE NORMALIZATION

def normalize_value(col: str, value: str) -> str:
    """Normalize input values to match training data format"""
    mappings = {
        "education_level": {
            "Bachelor": "Bachelor's",
            "Master": "Master's",
        },
        "employment_status": {
            "Self-Employed": "Self-employed"
        },
        "loan_purpose": {
            "Auto": "Car",
            "Personal": "Other"
        }
    }
    return mappings.get(col, {}).get(value, value)


# VALIDATION HELPER

def validate_categorical_value(col: str, value: str) -> str:
    """Validate and normalize categorical values"""
    normalized = normalize_value(col, value)
    
    if normalized not in label_encoders[col].classes_:
        valid_values = list(label_encoders[col].classes_)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid value '{value}' for {col}. Valid values: {valid_values}"
        )
    
    return normalized


# SCHEMAS

class PredictionRequest(BaseModel):
    name: str
    annual_income: float
    debt_to_income_ratio: float
    credit_score: int
    loan_amount: float
    interest_rate: float
    gender: str
    marital_status: str
    education_level: str
    employment_status: str
    loan_purpose: str
    grade_subgrade: str

    @field_validator('annual_income', 'loan_amount')
    @classmethod
    def validate_positive_amounts(cls, v):
        if v <= 0:
            raise ValueError('Amount must be positive')
        return v

    @field_validator('debt_to_income_ratio')
    @classmethod
    def validate_ratio(cls, v):
        if v < 0 or v > 1:
            raise ValueError('Debt to income ratio must be between 0 and 1')
        return v

    @field_validator('credit_score')
    @classmethod
    def validate_credit_score(cls, v):
        if v < 300 or v > 850:
            raise ValueError('Credit score must be between 300 and 850')
        return v

    @field_validator('interest_rate')
    @classmethod
    def validate_interest_rate(cls, v):
        if v < 0 or v > 100:
            raise ValueError('Interest rate must be between 0 and 100')
        return v

class PredictionResponse(BaseModel):
    name: str
    prediction: str
    prediction_value: int
    approval_probability: float
    rejection_probability: float

class BatchSummary(BaseModel):
    total: int
    approved: int
    rejected: int
    approval_rate: float

class BatchResult(BaseModel):
    name: str
    prediction: str
    approval_probability: float

class BatchPredictionResponse(BaseModel):
    summary: BatchSummary
    results: list[BatchResult]

# =====================================================
# PREDICTION HELPER
# =====================================================
def process_prediction(data: dict, exclude_name: bool = True):
    """Process prediction for a single applicant"""
    # Create dataframe
    if exclude_name:
        prediction_data = {k: v for k, v in data.items() if k != 'name'}
    else:
        prediction_data = data.copy()
        if 'name' in prediction_data:
            del prediction_data['name']
    
    df = pd.DataFrame([prediction_data])
    
    # Define categorical columns
    categorical_cols = [
        "gender", "marital_status", "education_level",
        "employment_status", "loan_purpose", "grade_subgrade"
    ]
    
    # Validate and encode categorical features
    for col in categorical_cols:
        if col in df.columns:
            df[col] = validate_categorical_value(col, str(df[col].iloc[0]))
            df[col] = label_encoders[col].transform([df[col].iloc[0]])[0]
    
    # Ensure correct column order
    df = df[feature_names]
    
    # Scale features
    scaled = scaler.transform(df)
    
    # Make prediction
    pred = int(model.predict(scaled)[0])
    proba = model.predict_proba(scaled)[0]
    
    return pred, proba

# =====================================================
# HEALTH CHECK
# =====================================================
@app.get("/health")
def health_check():
    """Check system health status"""
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "database_connected": test_connection(),
        "version": "2.0.0",
        "timestamp": datetime.utcnow().isoformat()
    }

# =====================================================
# SINGLE PREDICTION
# =====================================================
@app.post("/predict", response_model=PredictionResponse)
def predict_single(
    req: PredictionRequest,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """
    Predict loan approval for a single applicant
    
    Returns:
        - prediction: APPROVED or REJECTED
        - approval_probability: Probability of approval (0-1)
        - rejection_probability: Probability of rejection (0-1)
    """
    try:
        # Process prediction
        pred, proba = process_prediction(req.model_dump())
        
        # Save to database
        record = Prediction(
            name=req.name,
            annual_income=req.annual_income,
            debt_to_income_ratio=req.debt_to_income_ratio,
            credit_score=req.credit_score,
            loan_amount=req.loan_amount,
            interest_rate=req.interest_rate,
            gender=req.gender,
            marital_status=req.marital_status,
            education_level=req.education_level,
            employment_status=req.employment_status,
            loan_purpose=req.loan_purpose,
            grade_subgrade=req.grade_subgrade,
            prediction=pred,
            probability=float(proba[1]),
            user_id=current_user.id,
            created_at=datetime.utcnow()
        )
        
        db.add(record)
        db.commit()
        db.refresh(record)
        
        return PredictionResponse(
            name=req.name,
            prediction="APPROVED" if pred == 1 else "REJECTED",
            prediction_value=pred,
            approval_probability=float(proba[1]),
            rejection_probability=float(proba[0])
        )
        
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction failed: {str(e)}"
        )

# =====================================================
# BATCH PREDICTION
# =====================================================
@app.post("/batch_predict", response_model=BatchPredictionResponse)
async def predict_batch(
    file: UploadFile = File(...),
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """
    Predict loan approval for multiple applicants from CSV file
    
    CSV must contain columns: name, annual_income, debt_to_income_ratio, 
    credit_score, loan_amount, interest_rate, gender, marital_status, 
    education_level, employment_status, loan_purpose, grade_subgrade
    """
    try:
        # Validate file type
        if not file.filename.endswith('.csv'):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="File must be a CSV"
            )
        
        # Read CSV
        contents = await file.read()
        df = pd.read_csv(io.BytesIO(contents))
        
        # Validate required columns
        required_cols = set(feature_names + ["name"])
        missing_cols = required_cols - set(df.columns)
        
        if missing_cols:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Missing required columns: {missing_cols}"
            )
        
        # Store original data
        original_df = df.copy()
        
        # Process predictions
        results = []
        approved = 0
        
        # Define categorical columns
        categorical_cols = [
            "gender", "marital_status", "education_level",
            "employment_status", "loan_purpose", "grade_subgrade"
        ]
        
        # Normalize and encode categorical features
        for col in categorical_cols:
            df[col] = df[col].apply(lambda x: normalize_value(col, str(x)))
            
            # Validate all values
            invalid_values = set(df[col]) - set(label_encoders[col].classes_)
            if invalid_values:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Invalid values in {col}: {invalid_values}"
                )
            
            df[col] = label_encoders[col].transform(df[col])
        
        # Ensure correct column order and scale
        df_features = df[feature_names]
        scaled = scaler.transform(df_features)
        
        # Make predictions
        preds = model.predict(scaled)
        probas = model.predict_proba(scaled)
        
        # Process results and save to database
        for i in range(len(df)):
            pred = int(preds[i])
            proba = probas[i]
            
            # Create database record
            record = Prediction(
                name=str(original_df.iloc[i]["name"]),
                annual_income=float(original_df.iloc[i]["annual_income"]),
                debt_to_income_ratio=float(original_df.iloc[i]["debt_to_income_ratio"]),
                credit_score=int(original_df.iloc[i]["credit_score"]),
                loan_amount=float(original_df.iloc[i]["loan_amount"]),
                interest_rate=float(original_df.iloc[i]["interest_rate"]),
                gender=str(original_df.iloc[i]["gender"]),
                marital_status=str(original_df.iloc[i]["marital_status"]),
                education_level=str(original_df.iloc[i]["education_level"]),
                employment_status=str(original_df.iloc[i]["employment_status"]),
                loan_purpose=str(original_df.iloc[i]["loan_purpose"]),
                grade_subgrade=str(original_df.iloc[i]["grade_subgrade"]),
                prediction=pred,
                probability=float(proba[1]),
                user_id=current_user.id,
                created_at=datetime.utcnow()
            )
            
            db.add(record)
            
            if pred == 1:
                approved += 1
            
            results.append(BatchResult(
                name=str(original_df.iloc[i]["name"]),
                prediction="APPROVED" if pred == 1 else "REJECTED",
                approval_probability=float(proba[1])
            ))
        
        # Commit all records
        db.commit()
        
        # Calculate summary
        total = len(df)
        rejected = total - approved
        approval_rate = round((approved / total) * 100, 2) if total > 0 else 0
        
        return BatchPredictionResponse(
            summary=BatchSummary(
                total=total,
                approved=approved,
                rejected=rejected,
                approval_rate=approval_rate
            ),
            results=results
        )
        
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Batch prediction failed: {str(e)}"
        )

# =====================================================
# GET CURRENT USER INFO
# =====================================================
@app.get("/auth/me")
def get_current_user_info(
    current_user: User = Depends(get_current_active_user)
):
    """Get current user information"""
    return {
        "id": current_user.id,
        "username": current_user.username,
        "email": current_user.email,
        "full_name": current_user.full_name,
        "is_active": current_user.is_active
    }

# =====================================================
# GET USER PREDICTIONS
# =====================================================
@app.get("/predictions")
def get_user_predictions(
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db),
    limit: int = 100,
    offset: int = 0
):
    """Get prediction history for current user"""
    predictions = db.query(Prediction).filter(
        Prediction.user_id == current_user.id
    ).order_by(
        Prediction.created_at.desc()
    ).limit(limit).offset(offset).all()
    
    return {
        "total": len(predictions),
        "predictions": [
            {
                "id": p.id,
                "name": p.name,
                "prediction": "APPROVED" if p.prediction == 1 else "REJECTED",
                "probability": p.probability,
                "created_at": p.created_at.isoformat()
            }
            for p in predictions
        ]
    }

# =====================================================
# RUN SERVER
# =====================================================
if __name__ == "__main__":
    uvicorn.run(
        "app:app",  # Changed from app to "app:app" string for reload support
        host="127.0.0.1",
        port=8000,
        reload=True,
        log_level="info"
    )