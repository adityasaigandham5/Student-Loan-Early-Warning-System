import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

import joblib
import json

# ---------------------------------------------------
# Page Config
# ---------------------------------------------------

st.set_page_config(
    page_title="Student Loan Early Warning",
    page_icon="🎓",
    layout="wide"
)

# ---------------------------------------------------
# Load Models
# ---------------------------------------------------

@st.cache_resource
def load():

    return {

        'm30': joblib.load(
            "../models/model_default_30d.pkl"
        ),

        'm60': joblib.load(
            "../models/model_default_60d.pkl"
        ),

        'm90': joblib.load(
            "../models/model_default_90d.pkl"
        ),

        'lg': joblib.load(
            "../models/le_gender.pkl"
        ),

        'lc': joblib.load(
            "../models/le_course.pkl"
        ),

        'lt': joblib.load(
            "../models/le_tier.pkl"
        ),

        'ltr': joblib.load(
            "../models/le_trend.pkl"
        ),

        'met': json.load(
            open("../models/model_metrics.json")
        )
    }

M = load()

FEATURES = M['met']['features']

# ---------------------------------------------------
# Header
# ---------------------------------------------------

st.markdown(
    """
    <div style="
        background:linear-gradient(135deg,#1E3A5F,#4C1D95);
        padding:22px;
        border-radius:12px;
        margin-bottom:18px">

        <h1 style="color:white;margin:0">
        Student Loan Early Warning System
        </h1>

        <p style="color:#C4B5FD;margin:4px 0 0 0">
        3-Horizon Default Prediction:
        30 / 60 / 90 Days
        </p>

    </div>
    """,
    unsafe_allow_html=True
)

# ---------------------------------------------------
# Metrics
# ---------------------------------------------------

h = M['met']['horizons']

c1, c2, c3, c4 = st.columns(4)

c1.metric(
    "30-Day AUC",
    h['default_30d']['auc_roc']
)

c2.metric(
    "60-Day AUC",
    h['default_60d']['auc_roc']
)

c3.metric(
    "90-Day AUC",
    h['default_90d']['auc_roc']
)

c4.metric(
    "90-Day Default Rate",
    f"{h['default_90d']['default_rate']*100:.1f}%"
)

st.markdown("---")

# ---------------------------------------------------
# Input Section
# ---------------------------------------------------

st.subheader("Student Loan Application")

c1, c2, c3 = st.columns(3)

# ---------------------------------------------------
# Column 1
# ---------------------------------------------------

with c1:

    course = st.selectbox(
        "Course",
        ["Engineering","MBA","Medical","Law","Others"]
    )

    tier = st.selectbox(
        "Institute Tier",
        ["Tier1","Tier2","Tier3"]
    )

    gpa = st.slider(
        "GPA",
        4.0,
        10.0,
        7.2,
        0.1
    )

    gpa_trend = st.selectbox(
        "GPA Trend",
        ["Improving","Stable","Declining"]
    )

# ---------------------------------------------------
# Column 2
# ---------------------------------------------------

with c2:

    employed = st.selectbox(
        "Employment Status",
        [("Employed",1),("Unemployed",0)],
        format_func=lambda x:x[0]
    )

    salary = st.number_input(
        "Monthly Salary (INR)",
        0,
        500000,
        45000,
        5000
    )

    loan_amt = st.number_input(
        "Loan Amount (INR)",
        50000,
        5000000,
        800000,
        10000
    )

    interest = st.slider(
        "Interest Rate (%)",
        8.0,
        16.0,
        11.5,
        0.1
    )

# ---------------------------------------------------
# Column 3
# ---------------------------------------------------

with c3:

    tenure = st.selectbox(
        "Tenure (months)",
        [60,84,120,144]
    )

    months_d = st.slider(
        "Months Since Disbursement",
        1,
        60,
        8
    )

    missed = st.number_input(
        "Missed Payments (Past)",
        0,
        10,
        0
    )

    auto_d = st.selectbox(
        "Auto Debit",
        [("Yes",1),("No",0)],
        format_func=lambda x:x[0]
    )

    co_b = st.selectbox(
        "Co-Borrower",
        [("Yes",1),("No",0)],
        format_func=lambda x:x[0]
    )

# ---------------------------------------------------
# Prediction Button
# ---------------------------------------------------

if st.button(
    "RUN EARLY WARNING CHECK",
    type="primary",
    use_container_width=True
):

    # ---------------------------------------------------
    # EMI Calculation
    # ---------------------------------------------------

    emi_rate = interest / (12 * 100)

    emi = (
        loan_amt
        * emi_rate
        * (1 + emi_rate) ** tenure
        /
        (
            (1 + emi_rate) ** tenure - 1
        )
    )

    emi_inc = emi / max(salary, 1)

    # ---------------------------------------------------
    # Encoding
    # ---------------------------------------------------

    g_enc = int(
        M['lg'].transform(["Male"])[0]
    )

    c_enc = int(
        M['lc'].transform([course])[0]
    )

    t_enc = int(
        M['lt'].transform([tier])[0]
    )

    tr_enc = int(
        M['ltr'].transform([gpa_trend])[0]
    )

    # ---------------------------------------------------
    # Build DataFrame
    # ---------------------------------------------------

    X = pd.DataFrame([{

        'age': 24,
        'gender_enc': g_enc,
        'course_enc': c_enc,
        'tier_enc': t_enc,

        'gpa': gpa,
        'trend_enc': tr_enc,

        'loan_amt': loan_amt,
        'interest_rate': interest,

        'tenure_mo': tenure,
        'moratorium_mo': 12,

        'emi': emi,

        'employed': employed[1],

        'salary_mo': salary,

        'emi_income': emi_inc,

        'months_since_disb': months_d,

        'missed_pmts_past': missed,

        'payment_delay_avg': 2.0,

        'auto_debit': auto_d[1],

        'co_borrower': co_b[1],

        'family_income_mo': 60000,

        'family_support':
            60000 / (emi + 1),

        'loan_burden':
            loan_amt / (60000 * 12 + 1),

        'is_early_vintage':
            int(months_d <= 6),

        'payment_score':
            (1 - missed/10) * (1 - 2.0/30),

        'academic_risk':
            int(
                gpa < 6.5
                or
                gpa_trend == 'Declining'
            ),

        'employment_stress':
            int(
                employed[1] == 0
                and
                emi_inc > 0.45
            ),

        'support_score':
            co_b[1] + auto_d[1],

        'income_adequacy':
            salary / (emi + 1)

    }])[FEATURES]

    # ---------------------------------------------------
    # Predictions
    # ---------------------------------------------------

    p30 = float(
        M['m30'].predict_proba(X)[0][1]
    )

    p60 = float(
        M['m60'].predict_proba(X)[0][1]
    )

    p90 = float(
        M['m90'].predict_proba(X)[0][1]
    )

    # ---------------------------------------------------
    # Layout
    # ---------------------------------------------------

    col1, col2 = st.columns(2)

    # ---------------------------------------------------
    # Left Side
    # ---------------------------------------------------

    with col1:

        for label, prob in [

            ("30-Day Risk", p30),

            ("60-Day Risk", p60),

            ("90-Day Risk", p90)

        ]:

            st.markdown(
                f"### {label}: {prob*100:.1f}%"
            )

        # ---------------------------------------------------
        # Action Recommendation
        # ---------------------------------------------------

        if p30 > 0.30:

            st.error(
                "ACTION: Call student immediately"
            )

        elif p60 > 0.20:

            st.warning(
                "ACTION: Offer EMI restructuring"
            )

        elif p90 > 0.15:

            st.info(
                "ACTION: Assign relationship manager"
            )

        else:

            st.success(
                "LOW RISK - Standard monitoring"
            )

    # ---------------------------------------------------
    # Right Side Chart
    # ---------------------------------------------------

    with col2:

        fig = go.Figure()

        fig.add_trace(

            go.Bar(

                name='Default Probability',

                x=[
                    '30-Day',
                    '60-Day',
                    '90-Day'
                ],

                y=[
                    p30 * 100,
                    p60 * 100,
                    p90 * 100
                ],

                marker_color=[
                    'green',
                    'orange',
                    'red'
                ],

                text=[
                    f'{p30*100:.1f}%',
                    f'{p60*100:.1f}%',
                    f'{p90*100:.1f}%'
                ],

                textposition='outside'
            )
        )

        fig.update_layout(
            title='Default Probability by Horizon',
            height=320,
            yaxis_title='Probability (%)',
            yaxis_range=[0, 100]
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )