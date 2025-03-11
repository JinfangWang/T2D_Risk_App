import streamlit as st
import pandas as pd
import os
import pickle
import numpy as np
import openai
from sklearn.preprocessing import StandardScaler
from sklearn.exceptions import NotFittedError

# Load pre-trained Logistic Regression model and scaler
try:
    with open("logistic_regression.pkl", "rb") as f:
        clf = pickle.load(f)
    
    with open("scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
except Exception as e:
    st.error(f"❌ Error loading model or scaler: {e}")
    st.stop()

# Debugging: Ensure classifier and scaler are loaded correctly
#st.write("🔍 Loaded Model:", clf)
if not hasattr(clf, "coef_"):
    st.error("❌ The loaded classifier is not trained! Please re-train and save it.")
    st.stop()

if not isinstance(scaler, StandardScaler):
    st.error("❌ The loaded scaler is not a StandardScaler instance! Check `scaler.pkl`.")
    st.stop()

# Custom CSS for improved visual appearance
st.markdown(
    """
    <style>
    .main {
        background-color: #f5f5f5;
        padding: 2rem;
        border-radius: 10px;
    }
    .header {
        font-size: 2.5rem;
        font-weight: bold;
        text-align: center;
        color: #2c3e50;
        margin-bottom: 1rem;
    }
    .description {
        font-size: 1.1rem;
        color: #555555;
        text-align: center;
        margin-bottom: 2rem;
    }
    </style>
    """, unsafe_allow_html=True
)

with st.container():
    st.markdown('<div class="header">Personal Health Advisor: Understand Your Diabetes Risk</div>', unsafe_allow_html=True)
    st.markdown(
        """
        <div class="description">
        Welcome! Enter your health details below to get a personalized risk classification for Type 2 Diabetes 
        and receive tailored health advice. Our system uses advanced machine learning to analyze your data.
        </div>
        """, unsafe_allow_html=True
    )

    # User Input Form for Health Metrics
    with st.form("user_input_form"):
        col1, col2, col3 = st.columns(3)
        with col1:
            fbg_input = st.text_input("Fasting Blood Glucose (mg/dL)", "100")
        with col2:
            hba1c_input = st.text_input("HbA1c (%)", "5.5")
        with col3:
            systolic_input = st.text_input("Systolic BP (mmHg)", "120")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            diastolic_input = st.text_input("Diastolic BP (mmHg)", "80")
        with col2:
            bmi_input = st.text_input("BMI", "25.0")
        with col3:
            triglycerides_input = st.text_input("Triglycerides (mg/dL)", "150")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            hdl_input = st.text_input("HDL Cholesterol (mg/dL)", "50")
        with col2:
            ldl_input = st.text_input("LDL Cholesterol (mg/dL)", "100")
        with col3:
            ast_input = st.text_input("AST (GOT) (U/L)", "30")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            alt_input = st.text_input("ALT (GPT) (U/L)", "30")
        with col2:
            gamma_input = st.text_input("Gamma-GTP (U/L)", "25")
        with col3:
            egfr_input = st.text_input("eGFR (mL/min/1.73m²)", "90")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            age_input = st.text_input("Age", "50")
        with col2:
            sex = st.selectbox("Sex", ["Male", "Female"])
        with col3:
            optional_metric = st.text_input("Optional Metric (if any)", "")
        
        submitted = st.form_submit_button("Submit")

    if submitted:
        try:
            user_data = pd.DataFrame({
                "Fasting_Blood_Glucose": [float(fbg_input)],
                "HbA1c": [float(hba1c_input)],
                "Systolic_BP": [float(systolic_input)],
                "Diastolic_BP": [float(diastolic_input)],
                "BMI": [float(bmi_input)],
                "Triglycerides": [float(triglycerides_input)],
                "HDL_Cholesterol": [float(hdl_input)],
                "LDL_Cholesterol": [float(ldl_input)],
                "AST(GOT)": [float(ast_input)],
                "ALT(GPT)": [float(alt_input)],
                "Gamma_GTP": [float(gamma_input)],
                "eGFR": [float(egfr_input)],
                "Age": [int(age_input)],
                "Sex": [1 if sex == "Male" else 0]
            })
        except ValueError:
            st.error("Please enter valid numeric values.")
            st.stop()

        X_user_scaled = scaler.transform(user_data)
        #st.write("🔍 Scaled user input shape:", X_user_scaled.shape)
        
        try:
            risk_probability = clf.predict_proba(X_user_scaled)[:, 1][0]
        except NotFittedError:
            st.error("Error: The classifier has not been fitted. Please re-train and save the model before running the app.")
            st.stop()
        
        st.subheader("Your Risk Profile")
        st.write(f"**Estimated Probability of Type 2 Diabetes:** {risk_probability:.2f}")
        
        # Generate personalized advice
        openai.api_key = os.getenv("OPENAI_API_KEY")
        if openai.api_key is None:
            st.error("OPENAI_API_KEY is not set in the environment!")
        else:
            prompt = f"""
            You are a medical data assistant. A user has an estimated T2D risk probability of {risk_probability:.2f}.
            Provide personalized advice on lifestyle modifications, diet, and medical follow-ups.
            """
            response = openai.chat.completions.create(
                model="gpt-4",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7
            )
            st.subheader("Personalized Health Advice")
            st.write(response.choices[0].message.content.strip())