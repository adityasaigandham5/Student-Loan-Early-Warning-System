from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

import joblib
import json
import time

import pandas as pd
import numpy as np

# ---------------------------------------------------
# FastAPI App
# ---------------------------------------------------

app = FastAPI(
    title="Slice Student Loan Early Warning API",
    description="3-horizon default prediction: 30 / 60 / 90 day",
    version="1.0.0"
)

# ---------------------------------------------------
# CORS
# ---------------------------------------------------

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

# ---------------------------------------------------
# Load Models
# ---------------------------------------------------

model_30d = joblib.load(
    "../models/model_default_30d.pkl"
)

model_60d = joblib.load(
    "../models/model_default_60d.pkl"
)

model_90d = joblib.load(
    "../models/model_default_90d.pkl"
)

# ---------------------------------------------------
# Load Encoders
# ---------------------------------------------------

le_gender = joblib.load("../models/le_gender.pkl")
le_course = joblib.load("../models/le_course.pkl")
le_tier = joblib.load("../models/le_tier.pkl")
le_trend = joblib.load("../models/le_trend.pkl")

# ---------------------------------------------------
# Load Metrics
# ---------------------------------------------------

metrics = json.load(
    open("../models/model_metrics.json")
)

FEATURES = metrics["features"]

# ---------------------------------------------------
# Request Schema
# ---------------------------------------------------

class StudentLoan(BaseModel):

    age: int = Field(..., ge=18, le=35, example=24)

    gender: str = Field(
        "Male",
        example="Male"
    )

    course_type: str = Field(
        "Engineering",
        example="Engineering"
    )

    institute_tier: str = Field(
        "Tier2",
        example="Tier2"
    )

    gpa: float = Field(
        ...,
        ge=4.0,
        le=10.0,
        example=7.2
    )

    gpa_trend: str = Field(
        "Stable",
        example="Stable"
    )

    loan_amt: float = Field(
        ...,
        ge=50000,
        example=800000
    )

    interest_rate: float = Field(
        11.5,
        ge=8.0,
        le=16.0,
        example=11.5
    )

    tenure_mo: int = Field(
        84,
        ge=12,
        le=144,
        example=84
    )

    moratorium_mo: int = Field(
        12,
        ge=6,
        le=24,
        example=12
    )

    employed: int = Field(
        ...,
        ge=0,
        le=1,
        example=1
    )

    salary_mo: float = Field(
        45000,
        ge=0,
        example=45000
    )

    months_since_disb: int = Field(
        8,
        ge=1,
        le=60,
        example=8
    )

    missed_pmts_past: int = Field(
        0,
        ge=0,
        le=10,
        example=0
    )

    payment_delay_avg: float = Field(
        2.0,
        ge=0,
        le=30,
        example=2.0
    )

    auto_debit: int = Field(
        1,
        ge=0,
        le=1,
        example=1
    )

    co_borrower: int = Field(
        1,
        ge=0,
        le=1,
        example=1
    )

    family_income_mo: float = Field(
        60000,
        ge=0,
        example=60000
    )

# ---------------------------------------------------
# Feature Builder
# ---------------------------------------------------

def build_features(req: StudentLoan):

    emi_rate = req.interest_rate / (12 * 100)

    emi = (
        req.loan_amt
        * emi_rate
        * (1 + emi_rate) ** req.tenure_mo
        /
        (
            (1 + emi_rate) ** req.tenure_mo - 1
        )
    )

    emi_inc = emi / max(req.salary_mo, 1)

    try:

        g_enc = int(
            le_gender.transform([req.gender])[0]
        )

        c_enc = int(
            le_course.transform([req.course_type])[0]
        )

        t_enc = int(
            le_tier.transform([req.institute_tier])[0]
        )

        tr_enc = int(
            le_trend.transform([req.gpa_trend])[0]
        )

    except ValueError as e:

        raise HTTPException(
            400,
            detail=f"Invalid value: {e}"
        )

    return pd.DataFrame([{

        'age': req.age,
        'gender_enc': g_enc,
        'course_enc': c_enc,
        'tier_enc': t_enc,
        'gpa': req.gpa,
        'trend_enc': tr_enc,

        'loan_amt': req.loan_amt,
        'interest_rate': req.interest_rate,
        'tenure_mo': req.tenure_mo,
        'moratorium_mo': req.moratorium_mo,

        'emi': emi,

        'employed': req.employed,
        'salary_mo': req.salary_mo,

        'emi_income': emi_inc,

        'months_since_disb': req.months_since_disb,

        'missed_pmts_past': req.missed_pmts_past,

        'payment_delay_avg': req.payment_delay_avg,

        'auto_debit': req.auto_debit,

        'co_borrower': req.co_borrower,

        'family_income_mo': req.family_income_mo,

        'family_support':
            req.family_income_mo / (emi + 1),

        'loan_burden':
            req.loan_amt /
            (req.family_income_mo * 12 + 1),

        'is_early_vintage':
            int(req.months_since_disb <= 6),

        'payment_score':
            (
                (1 - req.missed_pmts_past / 10)
                *
                (1 - req.payment_delay_avg / 30)
            ),

        'academic_risk':
            int(
                req.gpa < 6.5
                or
                req.gpa_trend == 'Declining'
            ),

        'employment_stress':
            int(
                req.employed == 0
                and
                emi_inc > 0.45
            ),

        'support_score':
            req.co_borrower + req.auto_debit,

        'income_adequacy':
            req.salary_mo / (emi + 1)

    }])[FEATURES]

# ---------------------------------------------------
# Alert System
# ---------------------------------------------------

def get_alert(prob, horizon):

    if prob > 0.30:
        return (
            "HIGH",
            f"Immediate intervention needed - {horizon} default risk"
        )

    elif prob > 0.12:
        return (
            "MEDIUM",
            f"Proactive reach-out recommended - {horizon} risk elevated"
        )

    else:
        return (
            "LOW",
            "Normal - monitor regularly"
        )

# ---------------------------------------------------
# Root API
# ---------------------------------------------------

@app.get("/")
def root():

    return {
        "status": "running",
        "models": ["30d", "60d", "90d"],
        "auc_90d":
            metrics["horizons"]["default_90d"]["auc_roc"]
    }

# ---------------------------------------------------
# Prediction API
# ---------------------------------------------------

@app.post("/predict/early-warning")
def predict_all_horizons(req: StudentLoan):

    start = time.time()

    feats = build_features(req)

    p30 = float(
        model_30d.predict_proba(feats)[0][1]
    )

    p60 = float(
        model_60d.predict_proba(feats)[0][1]
    )

    p90 = float(
        model_90d.predict_proba(feats)[0][1]
    )

    a30, m30 = get_alert(p30, "30-day")
    a60, m60 = get_alert(p60, "60-day")
    a90, m90 = get_alert(p90, "90-day")

    # ---------------------------------------------------
    # Recommended Action
    # ---------------------------------------------------

    if p30 > 0.30:

        action = (
            "CALL NOW - immediate payment reminder"
        )

    elif p60 > 0.20:

        action = (
            "OFFER RESTRUCTURING - lower EMI for 3 months"
        )

    elif p90 > 0.15:

        action = (
            "ASSIGN RELATIONSHIP MANAGER - proactive counselling"
        )

    else:

        action = (
            "MONITOR - standard monthly check-in"
        )

    return {

        "student_id": "STUDENT",

        "30_day": {
            "probability": round(p30, 4),
            "alert": a30,
            "message": m30
        },

        "60_day": {
            "probability": round(p60, 4),
            "alert": a60,
            "message": m60
        },

        "90_day": {
            "probability": round(p90, 4),
            "alert": a90,
            "message": m90
        },

        "recommended_action": action,

        "latency_ms":
            round((time.time() - start) * 1000, 2)
    }