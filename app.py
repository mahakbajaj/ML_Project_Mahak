import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

st.set_page_config(page_title="Women's Health Risk Detection", layout="wide")
st.title(" Women's Health - Drug Interaction Risk Assessment")

@st.cache_resource
def load_artifacts():
    return (
        joblib.load('random_forest_model.pkl'),
        joblib.load('xgboost_model.pkl'),
        joblib.load('scaler.pkl'),
        joblib.load('le_primary.pkl'),
        joblib.load('le_secondary.pkl'),
        joblib.load('feature_selector.pkl')
    )

rf_model, xgb_model, scaler, le_primary, le_secondary, selector = load_artifacts()

st.sidebar.header("Patient Input Parameters")
age = st.sidebar.slider("Age", 18, 80, 30)
bmi = st.sidebar.slider("BMI", 15.0, 45.0, 24.0)
is_pregnant = st.sidebar.selectbox("Pregnant?", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
is_lactating = st.sidebar.selectbox("Lactating?", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
primary_med = st.sidebar.selectbox("Primary Medication", le_primary.classes_)
secondary_med = st.sidebar.selectbox("Secondary Medication", le_secondary.classes_)
dosage_mg = st.sidebar.selectbox("Dosage (mg)", [25, 50, 100, 200, 500, 1000])
liver_imp = st.sidebar.selectbox("Liver Impairment?", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
kidney_imp = st.sidebar.selectbox("Kidney Impairment?", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
prior_reaction = st.sidebar.selectbox("Prior Adverse Reaction?", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")

if st.button("Evaluate Interaction Risk"):
    p_code = le_primary.transform([primary_med])[0]
    s_code = le_secondary.transform([secondary_med])[0]
    preg_risk_idx = is_pregnant * (p_code + 1) * (s_code + 1)
    organ_vulnerability = liver_imp + kidney_imp

    raw_input = pd.DataFrame([{
        'Age': age, 'Is_Pregnant': is_pregnant, 'Is_Lactating': is_lactating,
        'Dosage_mg': dosage_mg, 'BMI': bmi, 'Liver_Impairment': liver_imp,
        'Kidney_Impairment': kidney_imp, 'Prior_Adverse_Reaction': prior_reaction,
        'Primary_Med_Code': p_code, 'Secondary_Med_Code': s_code,
        'Pregnancy_Drug_Risk_Index': preg_risk_idx, 'Organ_Vulnerability': organ_vulnerability
    }])

    scaled_input = scaler.transform(selector.transform(raw_input))

    pred_rf = rf_model.predict(scaled_input)[0]
    prob_rf = rf_model.predict_proba(scaled_input)[0]
    pred_xgb = xgb_model.predict(scaled_input)[0]
    prob_xgb = xgb_model.predict_proba(scaled_input)[0]

    risk_map = {0: "Low Risk", 1: "Moderate Risk", 2: "High Risk"}

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Random Forest")
        st.info(f"**Prediction:** {risk_map[pred_rf]}")
        st.write(f"Confidence: **{np.max(prob_rf)*100:.1f}%**")
    with c2:
        st.subheader("XGBoost")
        st.success(f"**Prediction:** {risk_map[pred_xgb]}")
        st.write(f"Confidence: **{np.max(prob_xgb)*100:.1f}%**")

    st.subheader("Model Probability Comparison")
    labels = ['Low Risk', 'Moderate Risk', 'High Risk']
    fig, ax = plt.subplots(figsize=(7, 3))
    x = np.arange(len(labels))
    width = 0.35
    ax.bar(x - width/2, prob_rf, width, label='Random Forest', color='#3498db')
    ax.bar(x + width/2, prob_xgb, width, label='XGBoost', color='#2ecc71')
    ax.set_ylabel('Probability')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    st.pyplot(fig)
