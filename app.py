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
import base64

# Set page layout at the beginning (MUST be the first Streamlit command)
st.set_page_config(layout="centered")  # Or use "wide" if you prefer

# 1) Setup session state properly
###################################
if 'language' not in st.session_state:
    st.session_state['language'] = None  # Ensure language state is properly initialized

###################################
# 1) Language Buttons at Top-Right
###################################
col1, col2, col3, col4 = st.columns([5, 1, 1, 1])  # Adjust widths as needed

with col2:
    if st.button("English"):
        st.session_state['language'] = 'English'
with col3:
    if st.button("日本語"):
        st.session_state['language'] = 'Japanese'
with col4:
    if st.button("中文"):
        st.session_state['language'] = 'Chinese'

if st.session_state['language'] is None:
    st.markdown(
    """
    <div style="text-align:center; white-space:normal; word-wrap:break-word;">
        <h3>🩺 AI-powered Personalized Diabetes Risk Assessment</h3>
        <p>
            Enter your health metrics to get your 
            <strong>Type 2 Diabetes risk</strong> assessment
            and receive <strong>personalized health advices</strong>.
        </p>
        <p>
            健診データを入力し、<strong>2型糖尿病のリスク評価</strong> と 
            <strong>健康アドバイス</strong> を受け取ろう。
        </p>
        <p>
            输入您的健康数据，您将获得
            <strong>2型糖尿病的风险评估</strong>以及
            <strong>个性化的健康建议</strong>。
        </p>
        <br><br>
        <p style="color: blue; font-size: 25px; font-weight: bold;">
        <strong>Choose a language above to get started.</strong>
    </p>
    </div>
    """,
    unsafe_allow_html=True
    )
    st.stop()

###################################
# 1) Minimal text in each language
###################################
texts = {
    'English': {
        'title': "Understand Your Diabetes Risk",
        'description': (
            "Enter your health metrics to assess your Type 2 Diabetes risk and "
            "get personalized health advice based on advanced machine learning analysis."
        ),
        'button': "English"
    },
    'Japanese': {
        'title': "糖尿病リスクを理解する",
        'description': (
            "高度な機械学習・人工知能による、あなたの2型糖尿病のリスクを評価し、 "
            "健康アドバイスを得るために、あなたの健康情報を入力してください。"
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
# 3) Display Title & Description in Selected Language
###################################
lang = st.session_state['language']
st.markdown(f"<h2 style='text-align: center;'>{texts[lang]['title']}</h2>", unsafe_allow_html=True)
st.markdown(f"<p style='text-align: center;'>{texts[lang]['description']}</p>", unsafe_allow_html=True)


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

###############
# Form
###############
st.markdown(
    """
    <style>
    input[type="text"] {
        width: 80px !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)
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

        prompt_en = f"""
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
        
        prompt_jp = f"""
あなたは糖尿病予防の専門家です。ユーザーの推定2型糖尿病リスク確率は {risk_probability:.2f} です。
ユーザーは **クラスター {user_cluster} - {user_cluster_name}** に属しており、似たような健康特性を持つ人々を示します。

📌 **健康概要**  
- リスクレベル: {user_cluster_name}  
- 主な懸念事項: {user_risk_advice}  

⚡ **迅速なアクションプラン**  

🥗 **食事のヒント**  
✅ **食物繊維が豊富な食品**（野菜、全粒穀物、豆類）を選んで、血糖値管理をサポートしましょう。  
❌ **砂糖入り飲料や加工されたスナック菓子**を減らし、インスリンの急上昇を抑えましょう。  
🥑 **揚げ物などの悪い脂肪**を、**アボカド、ナッツ、魚**などの健康的な脂肪に置き換えましょう。  

🏃 **運動のヒント**  
🚶 **毎日30分のウォーキング**から始めるだけでも効果的です！  
💪 週に**2～3回の筋力トレーニング**を取り入れて、代謝を上げましょう。  
🧘 **継続してアクティブに**—楽しめるアクティビティを選び、モチベーションを維持してください。  

🏥 **医療チェックアップ**  
📅 **年に2回以上**は医師の診察を受け、血糖値を確認しましょう。  
💊 必要があれば、より良い血糖管理のために**薬の利用**を検討しましょう。  
🧠 メンタルヘルスも重要です—**ストレス管理や十分な睡眠**を心がけてください。  

🔹 **小さなステップが大切です！** 目標は徐々に改善していくこと。  
👨‍⚕️ **大きな健康の変化を始める前に医師に相談してください。**  
"""
        prompt_cn = f"""
您是一名专门研究糖尿病预防的医学专家。用户的2型糖尿病估计风险概率为 {risk_probability:.2f}。
他们属于 **聚类 {user_cluster} - {user_cluster_name}**，代表具有相似健康特征的人群。

📌 **健康摘要**  
- 风险等级：{user_cluster_name}  
- 主要关注点：{user_risk_advice}  

⚡ **快速行动计划**  

🥗 **饮食建议**  
✅ 选择 **富含纤维的食物**（如蔬菜、全谷物、豆类）帮助控制血糖。  
❌ 减少 **含糖饮料和加工零食**，避免胰岛素飙升。  
🥑 用 **健康脂肪**（鳄梨、坚果、鱼类）替代 **不健康脂肪**（油炸食品）。  

🏃 **运动建议**  
🚶 从 **每天30分钟的步行** 开始，即使轻度活动也能有益健康。  
💪 每周增加 **2-3次力量训练**，以提高新陈代谢。  
🧘 保持 **规律且活跃**——选择有趣的运动方式来保持动力。  

🏥 **医疗检查**  
📅 **每年至少进行两次**血糖监测及医生检查。  
💊 如有需要，**可考虑使用药物**以更好地控制血糖。  
🧠 心理健康同样重要——注意 **减压和充分睡眠**。  

🔹 **每一步都很关键！** 目标是逐渐改善。  
👨‍⚕️ **在进行重大健康调整之前，请咨询医生。**  
""" 
        
        if lang == 'English':
            prompt = prompt_en
        elif lang == 'Japanese':
            prompt = prompt_jp
        else:
            prompt = prompt_cn  

        response = openai.chat.completions.create(
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