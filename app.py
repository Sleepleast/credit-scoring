import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent))
from inference import CreditScoringPredictor, LABEL_COLORS, TEST_CASES
from pipeline import CreditDataPreprocessor
# PAGE CONFIG
st.set_page_config(
    page_title="Credit Scoring System",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded",
)

# LOAD MODEL (cached)
@st.cache_resource
def load_predictor():
    return CreditScoringPredictor()


try:
    predictor = load_predictor()
    model_ready = True
except Exception as e:
    model_ready = False
    model_error = str(e)

# SIDEBAR
with st.sidebar:
    st.title("💳 Credit Scoring")
    st.caption("Model Deployment")
    st.divider()
    page = st.radio(
        "Navigation",
        ["Single Prediction", "Batch Prediction", "Test Cases", "About"],
    )

# HELPERS
SCORE_CARD_CSS = {
    "Good":     "background:#d4edda;border-left:6px solid #28a745;padding:16px;border-radius:8px;color:#1a1a1a;",
    "Standard": "background:#fff3cd;border-left:6px solid #ffc107;padding:16px;border-radius:8px;color:#1a1a1a;",
    "Poor":     "background:#f8d7da;border-left:6px solid #dc3545;padding:16px;border-radius:8px;color:#1a1a1a;",
}


def build_gauge(label: str, proba_dict: dict):
    colors = {"Poor": "#dc3545", "Standard": "#ffc107", "Good": "#28a745"}

    # Composite credit score — weighted average that places each predicted
    # class inside its matching colour band:
    #   Poor     ->  0-40  (band centre ~20)
    #   Standard -> 40-70  (band centre ~55)
    #   Good     -> 70-100 (band centre ~85)
    score_val = (
        proba_dict.get("Poor",     0) * 20
        + proba_dict.get("Standard", 0) * 55
        + proba_dict.get("Good",     0) * 85
    )

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score_val,
        number={"valueformat": ".1f"},
        title={"text": "Credit Score", "font": {"size": 16}},
        gauge={
            "axis": {"range": [0, 100]},
            "bar": {"color": colors.get(label, "#999")},
            "steps": [
                {"range": [0,  40],  "color": "#f8d7da"},
                {"range": [40, 70],  "color": "#fff3cd"},
                {"range": [70, 100], "color": "#d4edda"},
            ],
            "threshold": {"line": {"color": "black", "width": 4}, "value": score_val},
        },
    ))
    fig.update_layout(height=260, margin=dict(t=30, b=0))
    return fig


def build_proba_bar(proba_dict: dict):
    labels = list(proba_dict.keys())
    values = [proba_dict[l] * 100 for l in labels]
    colors = {"Poor": "#dc3545", "Standard": "#ffc107", "Good": "#28a745"}
    fig = go.Figure(go.Bar(
        x=labels, y=values,
        marker_color=[colors.get(l, "#999") for l in labels],
        text=[f"{v:.1f}%" for v in values],
        textposition="outside",
    ))
    fig.update_layout(
        yaxis_title="Probability (%)", yaxis_range=[0, 110],
        height=280, margin=dict(t=20, b=20),
        plot_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def input_form(prefix=""):
    """Renders input widgets and returns a dict of raw values."""
    c1, c2, c3 = st.columns(3)
    with c1:
        st.subheader("Personal Info")
        age             = st.number_input("Age", 18, 100, 35, key=f"{prefix}age")
        occupation      = st.selectbox("Occupation",
            ["Scientist","Engineer","Accountant","Journalist","Lawyer",
             "Manager","Media_Manager","Mechanic","Musician","Teacher",
             "Writer","Entrepreneur","Developer","Doctor","Other"],
            key=f"{prefix}occ")

    with c2:
        st.subheader("Income & Loans")
        annual_income   = st.number_input("Annual Income (USD)", 5000, 500000, 60000, step=1000, key=f"{prefix}ai")
        monthly_salary  = st.number_input("Monthly In-hand Salary (USD)", 100, 50000, 4500, step=100, key=f"{prefix}ms")
        num_loans       = st.number_input("Number of Active Loans", 0, 20, 2, key=f"{prefix}nl")
        total_emi       = st.number_input("Total EMI / Month (USD)", 0, 10000, 150, key=f"{prefix}emi")
        outstanding     = st.number_input("Outstanding Debt (USD)", 0, 50000, 1200, key=f"{prefix}od")

    with c3:
        st.subheader("Credit Behaviour")
        num_accounts    = st.number_input("Bank Accounts", 0, 20, 3, key=f"{prefix}ba")
        num_cards       = st.number_input("Credit Cards", 0, 20, 3, key=f"{prefix}cc")
        interest_rate   = st.number_input("Interest Rate (%)", 1, 50, 12, key=f"{prefix}ir")
        delay_days      = st.number_input("Avg Payment Delay (days)", 0, 100, 10, key=f"{prefix}dd")
        delayed_pay     = st.number_input("# Delayed Payments", 0, 50, 3, key=f"{prefix}dp")
        credit_inquiries= st.number_input("Credit Inquiries", 0, 20, 3, key=f"{prefix}ci")
        credit_util     = st.number_input("Credit Utilization (%)", 0.0, 100.0, 30.0, key=f"{prefix}cu")
        changed_limit   = st.number_input("Changed Credit Limit (%)", 0.0, 50.0, 5.0, key=f"{prefix}cl")

    with c1:
        st.subheader("Credit Profile")
        credit_mix      = st.selectbox("Credit Mix", ["Good","Standard","Bad"], key=f"{prefix}cm")
        pay_min         = st.selectbox("Pay Minimum Amount", ["Yes","No","NM"], key=f"{prefix}pm")
        pay_behaviour   = st.selectbox("Payment Behaviour",
            ["High_spent_Large_value_payments","High_spent_Medium_value_payments",
             "High_spent_Small_value_payments","Low_spent_Large_value_payments",
             "Low_spent_Medium_value_payments","Low_spent_Small_value_payments"],
            key=f"{prefix}pb")
        history_yrs     = st.number_input("Credit History (years)", 0, 40, 8, key=f"{prefix}hy")
        history_mths    = st.number_input("Credit History (months)", 0, 11, 0, key=f"{prefix}hm")
        invested        = st.number_input("Invested Monthly (USD)", 0, 10000, 300, key=f"{prefix}im")
        balance         = st.number_input("Monthly Balance (USD)", 0, 50000, 500, key=f"{prefix}mb")

    credit_history_age = f"{history_yrs} Years and {history_mths} Months"

    return {
        "Age": age,
        "Annual_Income": str(annual_income),
        "Monthly_Inhand_Salary": monthly_salary,
        "Num_Bank_Accounts": num_accounts,
        "Num_Credit_Card": num_cards,
        "Interest_Rate": interest_rate,
        "Num_of_Loan": str(num_loans),
        "Delay_from_due_date": delay_days,
        "Num_of_Delayed_Payment": delayed_pay,
        "Changed_Credit_Limit": changed_limit,
        "Num_Credit_Inquiries": credit_inquiries,
        "Credit_Mix": credit_mix,
        "Outstanding_Debt": outstanding,
        "Credit_Utilization_Ratio": credit_util,
        "Credit_History_Age": credit_history_age,
        "Payment_of_Min_Amount": pay_min,
        "Total_EMI_per_month": total_emi,
        "Amount_invested_monthly": invested,
        "Payment_Behaviour": pay_behaviour,
        "Monthly_Balance": balance,
        "Occupation": occupation,
    }


def show_result(result: dict):
    label = result["label"]
    emoji = LABEL_COLORS.get(label, "")
    st.markdown(
        f"<div style='{SCORE_CARD_CSS[label]}'>"
        f"<h2 style='margin:0;color:#1a1a1a;'>{emoji} {label} Credit Score</h2>"
        f"</div>",
        unsafe_allow_html=True,
    )
    st.write("")

    if "probabilities" in result:
        col1, col2 = st.columns(2)
        with col1:
            st.plotly_chart(build_gauge(label, result["probabilities"]), use_container_width=True)
        with col2:
            st.plotly_chart(build_proba_bar(result["probabilities"]), use_container_width=True)

    with st.expander("Raw prediction details"):
        st.json(result)


# PAGES
if not model_ready:
    st.error(f"Model not loaded: {model_error}. Please run `pipeline.py` first.")
    st.stop()

#PAGE 1: Single Prediction
if page == "Single Prediction":
    st.title("Credit Score — Single Customer")
    st.caption("Fill in the form below and click **Predict**.")

    with st.form("single_form"):
        raw = input_form("sp_")
        submitted = st.form_submit_button("Predict Credit Score", type="primary", use_container_width=True)

    if submitted:
        with st.spinner("Analysing…"):
            result = predictor.predict_single(raw)
        st.divider()
        show_result(result)

#PAGE 2: Batch Prediction
elif page == "Batch Prediction":
    st.title("Batch Credit Score Prediction")
    uploaded = st.file_uploader("Upload a CSV file", type=["csv"])

    if uploaded:
        df_up = pd.read_csv(uploaded)
        st.write(f"Loaded **{len(df_up):,}** rows.")
        st.dataframe(df_up.head())

        if st.button("⚡ Run Batch Prediction", type="primary"):
            with st.spinner("Running predictions…"):
                result_df = predictor.predict_batch(df_up)
            st.success("Done!")
            st.dataframe(result_df[["Predicted_Credit_Score"] + list(df_up.columns[:5])])

            dist = result_df["Predicted_Credit_Score"].value_counts().reset_index()
            dist.columns = ["Credit Score", "Count"]
            fig = px.pie(dist, names="Credit Score", values="Count",
                         color="Credit Score",
                         color_discrete_map={"Good":"#28a745","Standard":"#ffc107","Poor":"#dc3545"})
            st.plotly_chart(fig, use_container_width=True)

            csv_out = result_df.to_csv(index=False).encode()
            st.download_button("Download results CSV", csv_out,
                               "predictions.csv", "text/csv")

#PAGE 3: Test Cases
elif page == "Test Cases":
    st.title("Deployment Test Cases")
    st.caption("Three test cases — one per credit score class — to validate deployment.")

    for i, tc in enumerate(TEST_CASES, 1):
        with st.expander(f"Test Case {i} — {tc['label']}", expanded=True):
            col1, col2 = st.columns([1, 1])
            with col1:
                st.subheader("Input Features")
                st.json(tc["input"])
            with col2:
                result = predictor.predict_single(tc["input"])
                st.subheader("Prediction")
                show_result(result)

#PAGE 4: About
elif page == "About":
    st.title("About This Project")
    st.markdown("""
## Credit Scoring System
**NIM:** 2802437291
**Nama:** Andrew Steven Castilani  
**Course:** Model Deployment  
**Dataset:** A  
**University:** BINUS University

### Problem Statement
As a data scientist at a financial institution, the goal is to assess the **credit score**
of each customer using a machine learning approach. The system classifies customers into
three categories:
- 🟢 **Good** — low credit risk
- 🟡 **Standard** — moderate credit risk
- 🔴 **Poor** — high credit risk

### Pipeline Architecture
```
Raw CSV → Preprocessing (CreditDataPreprocessor)
        → Feature Engineering
        → Model Training (LR / RF / GB)
        → MLflow Experiment Tracking
        → Best Model Serialisation (.pkl)
        → Streamlit Web Inference
```

### Model Performance
| Model                | Accuracy | F1 Macro |
|----------------------|----------|----------|
| Logistic Regression  | ~72%     | ~69%     |
| Random Forest        | ~80%     | ~77%     |
| Gradient Boosting    | ~82%     | ~79%     |

### Deployment Stack (Local)
- **Framework:** Streamlit
- **Tracking:** MLflow
- **Serialisation:** Joblib / Pickle
- **Language:** Python 3.10+

### AWS Cloud Architecture
```
S3 Bucket (data + model artefacts)
  └── SageMaker Training Job (pipeline.py)
        └── SageMaker Model Endpoint
              └── Streamlit on EC2 / App Runner
```
    """)
