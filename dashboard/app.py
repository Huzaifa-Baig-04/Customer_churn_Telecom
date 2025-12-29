import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import joblib

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(
    page_title="IBM Telecom Customer Churn",
    layout="wide"
)

# =========================
# LOAD TRAINED MODELS
# =========================
@st.cache_resource
def load_models():
    rf_model = joblib.load("models/churn_model.pkl")
    lr_model = joblib.load("models/churn_logistic.pkl")


    return rf_model, lr_model

rf_model, lr_model = load_models()

# =========================
# INPUT → MODEL FORMAT
# =========================
def prepare_input(tenure, senior, contract, internet, payment, monthly_charges):
    data = {
        "Tenure Months": tenure,
        "Monthly Charges": monthly_charges,
        "Senior Citizen": 1 if senior == "Yes" else 0,

        "Contract_Month-to-month": 1 if contract == "Month-to-month" else 0,
        "Contract_One year": 1 if contract == "One year" else 0,
        "Contract_Two year": 1 if contract == "Two year" else 0,

        "Internet Service_DSL": 1 if internet == "DSL" else 0,
        "Internet Service_Fiber optic": 1 if internet == "Fiber optic" else 0,
        "Internet Service_No": 1 if internet == "No" else 0,

        "Payment Method_Bank transfer (automatic)": 1 if payment == "Bank transfer (automatic)" else 0,
        "Payment Method_Credit card (automatic)": 1 if payment == "Credit card (automatic)" else 0,
        "Payment Method_Electronic check": 1 if payment == "Electronic check" else 0,
        "Payment Method_Mailed check": 1 if payment == "Mailed check" else 0,
    }

    return pd.DataFrame([data])

# =========================
# GLOBAL CSS (UNCHANGED)
# =========================
st.markdown("""<style>
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0F62FE, #0043CE);
    color: white;
}
section[data-testid="stSidebar"] label,
section[data-testid="stSidebar"] span,
section[data-testid="stSidebar"] p {
    color: white !important;
    font-weight: 600;
}
section[data-testid="stSidebar"] button {
    background-color: white !important;
    border-radius: 10px !important;
    padding: 10px !important;
}
section[data-testid="stSidebar"] button p {
    color: #161616 !important;
    font-weight: 800 !important;
}
.card {
    background: #F4F8FF;
    padding: 26px;
    border-radius: 18px;
    margin-bottom: 26px;
    border-left: 6px solid #0F62FE;
}
.card-danger {
    background: #FFF1F1;
    border-left: 6px solid #DA1E28;
}
.card-success {
    background: #F0FFF5;
    border-left: 6px solid #198038;
}
.header-card {
    background: white;
    padding: 30px;
    border-radius: 20px;
    box-shadow: 0 10px 25px rgba(0,0,0,0.08);
    margin-bottom: 30px;
}
.metric-box {
    background: white;
    padding: 30px;
    border-radius: 18px;
    text-align: center;
    box-shadow: 0 6px 18px rgba(0,0,0,0.08);
}
.metric-risk {
    font-size: 56px;
    font-weight: 900;
    color: #0F62FE;
}
.metric-prob {
    font-size: 56px;
    font-weight: 900;
    color: #DA1E28;
}
.section-title {
    font-size: 30px;
    font-weight: 800;
    margin-bottom: 16px;
}
</style>""", unsafe_allow_html=True)

# =========================
# SIDEBAR INPUTS
# =========================
st.sidebar.markdown("""
<div style="background: rgba(255,255,255,0.15);
padding:18px;border-radius:16px;margin-bottom:20px;">
<h3>🔍 Customer Inputs</h3>
<p style="font-size:14px;">Enter customer profile details</p>
</div>
""", unsafe_allow_html=True)

tenure = st.sidebar.slider("Tenure (Months)", 0, 72, 3)
senior = st.sidebar.radio("Senior Citizen", ["No", "Yes"])
contract = st.sidebar.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
internet = st.sidebar.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
payment = st.sidebar.selectbox(
    "Payment Method",
    ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"]
)
monthly_charges = st.sidebar.slider("Monthly Charges ($)", 18.0, 120.0, 92.0)

predict_btn = st.sidebar.button("🚀 Predict Churn")

# =========================
# HEADER
# =========================
st.markdown("""
<div class="header-card">
    <div style="display:flex;align-items:center;gap:18px;">
        <img src="https://upload.wikimedia.org/wikipedia/commons/5/51/IBM_logo.svg" height="52">
        <h1 style="margin:0;color:#0F62FE;font-size:36px;font-weight:800;">
            IBM Telecom Customer Churn Dashboard
        </h1>
    </div>
    <p style="margin-top:12px;color:#525252;font-size:18px;">
        Predicts customer churn risk and explains why a customer may churn
        and how the business can prevent it.
    </p>
</div>
""", unsafe_allow_html=True)

if not predict_btn:
    st.info("👈 Fill customer details and click **Predict Churn** to see prediction.")

# =========================
# PREDICTION
# =========================
if predict_btn:

    input_df = prepare_input(
        tenure, senior, contract, internet, payment, monthly_charges
    )

    rf_prob = rf_model.predict_proba(input_df)[0][1]
    lr_prob = lr_model.predict_proba(input_df)[0][1]

    risk_score = int(rf_prob * 100)
    churn_prob = int(lr_prob * 100)

    st.markdown("<div class='section-title'>📌 Churn Prediction</div>", unsafe_allow_html=True)

    c1, c2 = st.columns(2)

    with c1:
        st.markdown(f"""
        <div class="metric-box">
            <div>Risk Score</div>
            <div class="metric-risk">{risk_score}%</div>
        </div>
        """, unsafe_allow_html=True)

    with c2:
        st.markdown(f"""
        <div class="metric-box">
            <div>Churn Likelihood</div>
            <div class="metric-prob">{churn_prob}%</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<div style='height:18px'></div>", unsafe_allow_html=True)

    st.markdown("""
    <div class="card">
        <h4>🧠 What this means❓ </h4>
        <ul>
            <li>Risk Score reflects how closely this customer’s profile matches past customers who churned.</li>
            <li>Churn Likelihood represents the estimated chance that this customer may leave, based on learned patterns.</li>
            <li>Higher values in both metrics indicate increased risk of churn.</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

    # =========================
    # WHY THIS CUSTOMER MAY CHURN (FIXED)
    # =========================
    st.markdown("""
    <div class="card card-danger">
        <h4>⚠️ Why this customer may churn </h4>
        <ul>
            <li>Customer tenure is very short.</li>
            <li>Monthly charges are relatively high.</li>
            <li>Month-to-month contract increases churn probability.</li>
            <li>Electronic check payment is linked with churn.</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

    # =========================
    # SUGGESTED ACTIONS
    # =========================
    st.markdown("""
    <div class="card card-success">
        <h4>🛠 Suggested actions to prevent churn</h4>
        <ul>
            <li>Offer contract upgrade discounts.</li>
            <li>Provide loyalty rewards.</li>
            <li>Reduce billing friction.</li>
            <li>Encourage auto-pay methods.</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

# =========================
# DONUT CHART
# =========================
st.markdown("<div class='section-title'>🍩 Churn Risk Contribution</div>", unsafe_allow_html=True)

fig = go.Figure(
    data=[go.Pie(
        labels=["High Monthly Charges", "Short Tenure", "Contract Type", "Other"],
        values=[40, 30, 20, 10],
        hole=0.6,
        textinfo="label+percent"
    )]
)

fig.update_layout(height=520)
st.plotly_chart(fig, use_container_width=True)

# =========================
# FOOTER (FINAL FIX)
# =========================
st.markdown("""
<link href="https://fonts.googleapis.com/css2?family=Pacifico&display=swap" rel="stylesheet">

<hr style="margin-top:40px;">

<div style="text-align:center; padding-bottom:20px;">
    <div style="font-size:12px; color:#8a8a8a;">By</div>
    <div style="font-family:'Pacifico', cursive; font-size:26px;">
        Huzaifa Baig
    </div>
    <div style="font-size:13px;">
        <a href="https://www.linkedin.com/in/huzaifa-baig-509803388" target="_blank">
            LinkedIn
        </a> |
        <a href="mailto:anotherhuzaifa@gmail.com">
            anotherhuzaifa@gmail.com
        </a>
    </div>
</div>
""", unsafe_allow_html=True)

