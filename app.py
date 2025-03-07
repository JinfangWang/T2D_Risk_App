import streamlit as st
import pandas as pd
import os
import pickle
import numpy as np
import openai
from sklearn.preprocessing import StandardScaler
from sklearn.exceptions import NotFittedError
from scipy.spatial.distance import cdist
from PIL import Image

###################################
# 0) Setup session state
###################################
if 'language' not in st.session_state:
    st.session_state['language'] = None  # Start with no language chosen

###################################
# 1) Minimal text in each language
###################################
texts = {
    'English': {
        'title': "Understand Your Diabetes Risk",
        'description': (
            "Enter your health details to assess your Type 2 Diabetes risk and "
            "get personalized health advice based on advanced machine learning analysis."
        ),
        'button': "English"
    },
    'Japanese': {
        'title': "糖尿病リスクを理解する",
        'description': (
            "2型糖尿病のリスクを評価し、高度な機械学習分析に基づくパーソナライズされた "
            "健康アドバイスを取得するために、あなたの健康情報を入力してください。"
        ),
        'button': "日本語"
    },
    'Chinese': {
        'title': "了解您的糖尿病风险",
        'description': (
            "输入您的健康信息以评估2型糖尿病风险，并根据先进的机器学习分析 "
            "获得个性化的健康建议。"
        ),
        'button': "中文"
    }
}

###################################
# 2) Always Show Language Buttons at the Top
###################################
col1, col2 = st.columns([5, 1])  # Adjust proportions for right alignment

with col2:  # Right-aligned buttons
    if st.button(texts['English']['button']):
        st.session_state['language'] = 'English'
    if st.button(texts['Japanese']['button']):
        st.session_state['language'] = 'Japanese'
    if st.button(texts['Chinese']['button']):
        st.session_state['language'] = 'Chinese'

# Stop execution if language is not selected
if st.session_state['language'] is None:
    st.stop()

###################################
# 3) Display Title & Description in Selected Language
###################################
lang = st.session_state['language']
st.markdown(f"<h2 style='text-align: center;'>{texts[lang]['title']}</h2>", unsafe_allow_html=True)
st.markdown(f"<p style='text-align: center;'>{texts[lang]['description']}</p>", unsafe_allow_html=True)


###################################
# 4) Continue with the Normal App Logic
###################################

lang = st.session_state['language']
title = texts[lang]['title']
description = texts[lang]['description']

st.markdown(f"<h2 style='text-align: center;'>{title}</h2>", unsafe_allow_html=True)
st.markdown(f"<p style='text-align: center;'>{description}</p>", unsafe_allow_html=True)

###################################
# 4) Normal app logic below
###################################

###########################
# 2) DEFINE LABELS
###########################
labels = {
    'English': {
        'glucose': "Glucose (mg/dL)",
        'hba1c': "HbA1c (%)",
        'systolic': "Systolic (mmHg)",
        'diastolic': "Diastolic (mmHg)",
        'height': "Height (cm)",
        'weight': "Weight (kg)",
        'triglycerides': "Triglycerides (mg/dL)",
        'hdl': "HDL (mg/dL)",
        'ldl': "LDL (mg/dL)",
        'ast': "AST (GOT) (U/L)",
        'alt': "ALT (GPT) (U/L)",
        'gamma': "Gamma-GTP (U/L)",
        'egfr': "eGFR (mL/min/1.73m²)",
        'age': "Age",
        'sex': "Sex",
        'male': "Male",
        'female': "Female",
        'submit': "Submit",
        'risk_level': "Your Risk Level"
    },
    'Japanese': {
        'glucose': "血糖値 (mg/dL)",
        'hba1c': "HbA1c (%)",
        'systolic': "収縮期 (mmHg)",
        'diastolic': "拡張期 (mmHg)",
        'height': "身長 (cm)",
        'weight': "体重 (kg)",
        'triglycerides': "中性脂肪 (mg/dL)",
        'hdl': "HDL (mg/dL)",
        'ldl': "LDL (mg/dL)",
        'ast': "AST (GOT) (U/L)",
        'alt': "ALT (GPT) (U/L)",
        'gamma': "γ-GTP (U/L)",
        'egfr': "eGFR (mL/min/1.73m²)",
        'age': "年齢",
        'sex': "性別",
        'male': "男性",
        'female': "女性",
        'submit': "送信",
        'risk_level': "あなたのリスクレベル"
    },
    'Chinese': {
        'glucose': "血糖 (mg/dL)",
        'hba1c': "HbA1c (%)",
        'systolic': "收缩压 (mmHg)",
        'diastolic': "舒张压 (mmHg)",
        'height': "身高 (cm)",
        'weight': "体重 (kg)",
        'triglycerides': "甘油三酯 (mg/dL)",
        'hdl': "HDL (mg/dL)",
        'ldl': "LDL (mg/dL)",
        'ast': "AST (GOT) (U/L)",
        'alt': "ALT (GPT) (U/L)",
        'gamma': "γ-GTP (U/L)",
        'egfr': "eGFR (mL/min/1.73m²)",
        'age': "年龄",
        'sex': "性别",
        'male': "男",
        'female': "女",
        'submit': "提交",
        'risk_level': "您的风险等级"
    }
}

lab = labels[lang]

###########################
# 3) REMAINING APP LOGIC
###########################

# If not declared, do so here
def get_individual_cluster_mapping(lca_instance):
    """
    Prepares a DataFrame containing '加入者id', LIME-based cluster assignments,
    and all other relevant feature columns.
    """
    if 'Cluster_Original_Ordered' not in lca_instance.lime_importances_df.columns:
        print("Running `compute_and_order_cluster_risks()` to ensure LIME cluster assignments exist.")
        lca_instance.compute_and_order_cluster_risks()

    X_test_with_id = lca_instance.X_test.copy()
    X_test_with_id['加入者id'] = lca_instance.ID_test
    X_test_with_id['Cluster_LIME_Ordered'] = lca_instance.lime_importances_df['Cluster_Original_Ordered']

    feature_columns = lca_instance.X_train.columns.tolist()
    selected_columns = ['加入者id', 'Cluster_LIME_Ordered'] + feature_columns
    return X_test_with_id[selected_columns]

# Load data
try:
    df_clusters = pd.read_csv("cluster_mapping.csv", encoding="utf-8")
    with open("logistic_regression.pkl", "rb") as f:
        clf = pickle.load(f)
    with open("scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
except Exception as e:
    st.error(f"❌ Error loading files: {e}")
    st.stop()

if not hasattr(clf, "coef_"):
    st.error("❌ The loaded classifier is not trained! Please re-train and save it.")
    st.stop()

if not isinstance(scaler, StandardScaler):
    st.error("❌ The loaded scaler is not a StandardScaler instance! Check `scaler.pkl`.")
    st.stop()

# Possibly load an image
try:
    image = Image.open("predictive_clustering_with_diseases_20241226_ADA.jpg")
    image = image.resize((500, 500))
except:
    image = None

###############
# Form
###############
with st.form("user_input_form"):
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        fbg_input = st.text_input(lab['glucose'], "100")
    with col2:
        hba1c_input = st.text_input(lab['hba1c'], "5.4")
    with col3:
        systolic_input = st.text_input(lab['systolic'], "120")
    with col4:
        diastolic_input = st.text_input(lab['diastolic'], "80")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        height_input = st.text_input(lab['height'], "170")
    with col2:
        weight_input = st.text_input(lab['weight'], "70")
    with col3:
        triglycerides_input = st.text_input(lab['triglycerides'], "130")
    with col4:
        hdl_input = st.text_input(lab['hdl'], "55")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        ldl_input = st.text_input(lab['ldl'], "100")
    with col2:
        ast_input = st.text_input(lab['ast'], "30")
    with col3:
        alt_input = st.text_input(lab['alt'], "30")
    with col4:
        gamma_input = st.text_input(lab['gamma'], "25")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        egfr_input = st.text_input(lab['egfr'], "90")
    with col2:
        age_input = st.text_input(lab['age'], "50")
    with col3:
        sex_option = st.selectbox(lab['sex'], [lab['male'], lab['female']])
    with col4:
        pass

    submitted = st.form_submit_button(lab['submit'])

#############################
# On Submit
#############################
if submitted:
    # 1) Convert
    try:
        # BMI
        bmi_value = float(weight_input) / ((float(height_input) / 100) ** 2)
        # sex numeric
        sex_val = 1 if sex_option == lab['male'] else 0

        user_data = pd.DataFrame({
            "Systolic_BP": [float(systolic_input.strip())],
            "Diastolic_BP": [float(diastolic_input.strip())],
            "BMI": [bmi_value],
            "Triglycerides": [float(triglycerides_input.strip())],
            "HDL_Cholesterol": [float(hdl_input.strip())],
            "LDL_Cholesterol": [float(ldl_input.strip())],
            "AST(GOT)": [float(ast_input.strip())],
            "ALT(GPT)": [float(alt_input.strip())],
            "Gamma_GTP": [float(gamma_input.strip())],
            "eGFR": [float(egfr_input.strip())],
            "Age": [int(float(age_input.strip()))],
            "Sex": [sex_val]
        })
    except ValueError:
        st.error("🚨 Please enter only numeric values in all input fields.")
        st.stop()

    # 2) Standardize
    X_user_scaled = scaler.transform(user_data)

    # 3) Risk Probability
    try:
        risk_probability = clf.predict_proba(X_user_scaled)[:, 1][0]
    except NotFittedError:
        st.error("Error: The classifier has not been fitted. Please re-train and save the model before running the app.")
        st.stop()

    # 4) Clustering
    feature_columns = [col for col in df_clusters.columns if col not in ['加入者id', 'Cluster_LIME_Ordered']]
    expected_features = ['Systolic_BP', 'Diastolic_BP', 'BMI', 'Triglycerides', 'HDL_Cholesterol',
                         'LDL_Cholesterol', 'AST(GOT)', 'ALT(GPT)', 'Gamma_GTP', 'eGFR', 'Age', 'Sex']
    df_clusters_filtered = df_clusters[expected_features]
    X_scaled = scaler.transform(df_clusters_filtered)

    distances = cdist(X_user_scaled, X_scaled, metric='euclidean')
    closest_idx = np.argmin(distances)
    matched_individual = df_clusters.iloc[closest_idx]

    user_cluster = matched_individual['Cluster_LIME_Ordered']

    # 5) Risk color
    risk_color = "green" if risk_probability < 0.1 else "orange" if risk_probability < 0.3 else "red"
    st.markdown(
        f"""
        <div class="risk-box" style="background-color:{risk_color}; color:white;">
            🔥 Estimated Type 2 Diabetes Risk: <b>{risk_probability:.2f}</b>
        </div>
        """,
        unsafe_allow_html=True
    )

    # 6) Cluster Labels
    cluster_labels = [
        "Healthy", 
        "Mild dyslipidemia", 
        "Dyslipidemia", 
        "Hypertensive", 
        "Mild metabolic", 
        "Moderate metabolic", 
        "Severe metabolic"
    ]
    user_cluster_name = cluster_labels[int(user_cluster)]

    # Severity-based colors
    cluster_colors = [
        "#2ECC71",  # Healthy
        "#F1C40F",  # Mild dyslipidemia
        "#F39C12",  # Dyslipidemia
        "#D35400",  # Hypertensive
        "#E74C3C",  # Mild metabolic
        "#C0392B",  # Moderate metabolic
        "#900C3F"   # Severe metabolic
    ]
    # local heading
    if lang == 'English':
        heading_text = "#### 🏥 Your Risk Level"
    elif lang == 'Japanese':
        heading_text = "#### 🏥 あなたのリスクレベル"
    else:
        heading_text = "#### 🏥 您的风险等级"
    st.write(heading_text)

    for i, label in enumerate(cluster_labels):
        if i == int(user_cluster):
            st.markdown(
                f'<span style="background-color:{cluster_colors[i]}; '
                f'padding: 6px 12px; border-radius:6px; color:white; font-weight:bold;">'
                f'🏥 {label} (your level)</span>',
                unsafe_allow_html=True
            )
        else:
            st.markdown(f'<span style="color: grey;">◾ {label}</span>', unsafe_allow_html=True)

    # 7) LLM Advice
    if "OPENAI_API_KEY" not in st.secrets:
        st.error("🚨 OpenAI API Key is missing! Add it in Streamlit Secrets.")
    else:
        openai.api_key = st.secrets["OPENAI_API_KEY"]
        cluster_risks = {
            0: "Your metabolic health is currently in a good range, but maintaining a balanced diet and regular exercise will help sustain this condition.",
            1: "You have mild metabolic imbalances, especially in blood pressure and cholesterol. A focus on early lifestyle changes, such as improving diet quality and increasing physical activity, can prevent further risks.",
            2: "You show signs of metabolic stress, with elevated cholesterol and mild diabetic risk. Consider working on weight management and regular health monitoring to prevent progression.",
            3: "Hypertension and liver function issues are becoming significant. Reducing sodium intake, moderating alcohol consumption, and regular exercise are crucial for preventing cardiovascular complications.",
            4: "Obesity-related metabolic issues are evident, with increased risk of Type 2 Diabetes and heart disease. Prioritizing structured physical activity, fiber-rich diets, and weight management is necessary.",
            5: "Severe metabolic concerns, including liver dysfunction and diabetic complications, suggest a need for immediate intervention. Work closely with healthcare providers to manage blood sugar, liver health, and blood pressure.",
            6: "Your metabolic risk is at its highest, with very high chances of severe obesity-related complications. Intensive lifestyle changes and medical management are essential to prevent serious health outcomes."
        }

        user_risk_advice = cluster_risks[int(user_cluster)]

        # local advice heading
        if lang == 'English':
            advice_heading = "## 🩺 Personalized Health Advice"
        elif lang == 'Japanese':
            advice_heading = "## 🩺 個別の健康アドバイス"
        else:
            advice_heading = "## 🩺 个性化健康建议"
        st.write(advice_heading)

        prompt = f"""
        You are a medical expert specializing in diabetes prevention. A user has an estimated Type 2 Diabetes risk probability of {risk_probability:.2f}.
        They belong to **Cluster {user_cluster} - {user_cluster_name}**, which represents individuals with similar health characteristics.

        📌 **Health Summary**  
        - Risk Level: {user_cluster_name}  
        - Key Concerns: {user_risk_advice}  

        ⚡ **Quick Action Plan**  

        🥗 **Diet Tips**  
        ✅ Choose **fiber-rich foods** (vegetables, whole grains, legumes) to help blood sugar control.  
        ❌ Reduce **sugary drinks & processed snacks** to avoid insulin spikes.  
        🥑 Swap **bad fats** (fried foods) for **healthy fats** (avocados, nuts, fish).  

        🏃 **Exercise Tips**  
        🚶 Start with **daily 30-min walks** – even light activity helps!  
        💪 Add **2-3 days of strength training** for better metabolism.  
        🧘 Stay **consistent & active** – choose fun activities to keep motivated.  

        🏥 **Medical Check-ups**  
        📅 See a doctor **at least twice a year** for blood sugar monitoring.  
        💊 If needed, **consider medications** for better glucose control.  
        🧠 Mental well-being is key – **stress management & sleep** matter too!  

        🔹 **Every small step counts!** The goal is gradual improvement.  
        👨‍⚕️ **Consult a doctor before making major health changes.**  
        """

        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3
        )

        st.markdown(
            f"""
            <div class="advice-box">
                <b>📌 Lifestyle Recommendations for {user_cluster_name}:</b><br>
                {response.choices[0].message.content.strip()}
            </div>
            """,
            unsafe_allow_html=True
        )