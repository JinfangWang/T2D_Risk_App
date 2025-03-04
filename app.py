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

def get_individual_cluster_mapping(lca_instance):
    """
    Prepares a DataFrame containing '加入者id', LIME-based cluster assignments,
    and all other relevant feature columns.

    Parameters:
        - lca_instance (LimeClusteringAnalysis): An instance of LimeClusteringAnalysis.

    Returns:
        - pd.DataFrame: DataFrame with '加入者id', 'Cluster_LIME_Ordered', and all features.
    """
    # Ensure LIME cluster assignments exist
    if 'Cluster_Original_Ordered' not in lca_instance.lime_importances_df.columns:
        print("Running `compute_and_order_cluster_risks()` to ensure LIME cluster assignments exist.")
        lca_instance.compute_and_order_cluster_risks()

    # Restore individual IDs since they were removed from X_test
    X_test_with_id = lca_instance.X_test.copy()
    X_test_with_id['加入者id'] = lca_instance.ID_test

    # Merge LIME cluster assignments
    X_test_with_id['Cluster_LIME_Ordered'] = lca_instance.lime_importances_df['Cluster_Original_Ordered']

    # Select columns with IDs, cluster assignments, and features
    feature_columns = lca_instance.X_train.columns.tolist()
    selected_columns = ['加入者id', 'Cluster_LIME_Ordered'] + feature_columns

    return X_test_with_id[selected_columns]

# ✅ 1. Load Precomputed Data (Cluster Memberships and Features)
try:
    df_clusters = pd.read_csv("cluster_mapping.csv", encoding="utf-8")  # Ensure this file contains 'Cluster_LIME_Ordered'
    
    # Load pre-trained Logistic Regression model and scaler
    with open("logistic_regression.pkl", "rb") as f:
        clf = pickle.load(f)
    with open("scaler.pkl", "rb") as f:
        scaler = pickle.load(f)

except Exception as e:
    st.error(f"❌ Error loading files: {e}")
    st.stop()

# ✅ 2. Debugging: Ensure Model and Scaler Are Loaded Correctly
if not hasattr(clf, "coef_"):
    st.error("❌ The loaded classifier is not trained! Please re-train and save it.")
    st.stop()

if not isinstance(scaler, StandardScaler):
    st.error("❌ The loaded scaler is not a StandardScaler instance! Check `scaler.pkl`.")
    st.stop()

# ✅ 3. UI Styling
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

# ✅ 4. Load and Display the Image
image = Image.open("predictive_clustering_with_diseases_20241226_ADA.jpg")
image = image.resize((500, 500)) 


# Layout with header and image
col1, col2 = st.columns([3, 2])
with col1:
    st.markdown('<div class="header">Understand Your Diabetes Risk</div>', unsafe_allow_html=True)
    st.markdown(
        """
        <div class="description">
        Enter your health details to assess your Type 2 Diabetes risk and get personalized health advice based on advanced machine learning analysis.
        </div>
        """, unsafe_allow_html=True
    )
with col2:
    # Display a logo or image on the top right corner
    st.image(image, use_container_width=True)

# ✅ 5. User Input Form
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
    # ✅ 6. Convert User Input to DataFrame
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
    
    # ✅ 7. Standardize User Data
    X_user_scaled = scaler.transform(user_data)

    # ✅ 8. Compute Risk Probability
    try:
        risk_probability = clf.predict_proba(X_user_scaled)[:, 1][0]
    except NotFittedError:
        st.error("Error: The classifier has not been fitted. Please re-train and save the model before running the app.")
        st.stop()
    
    # ✅ 9. Find the Closest Match Using Euclidean Distance
    feature_columns = [col for col in df_clusters.columns if col not in ['加入者id', 'Cluster_LIME_Ordered']]
    X_scaled = scaler.transform(df_clusters.drop(columns=['加入者id', 'Cluster_LIME_Ordered']))
    distances = cdist(X_user_scaled, X_scaled, metric='euclidean')
    closest_idx = np.argmin(distances)
    matched_individual = df_clusters.iloc[closest_idx]

    # ✅ 10. Assign Cluster Membership
    user_cluster = matched_individual['Cluster_LIME_Ordered']

    # ✅ 11. Color-coded Risk Display
    risk_color = "green" if risk_probability < 0.1 else \
                 "orange" if risk_probability < 0.3 else "red"
    
    st.markdown(
        f"""
        <div class="risk-box" style="background-color:{risk_color}; color:white;">
            🔥 Estimated Type 2 Diabetes Risk: <b>{risk_probability:.2f}</b>
        </div>
        """,
        unsafe_allow_html=True
    )

    # Define cluster labels
    cluster_labels = ["Healthy", 
                      "Mild dyslipidemia", 
                      "Dyslipidemia", 
                      "Hypertensive", 
                      "Mild metabolic", 
                      "Moderate metabolic", 
                      "Severe metabolic"]

    cluster_colors = [
        "#2ECC71", "#F1C40F", "#F39C12", "#D35400",
        "#E74C3C", "#C0392B", "#900C3F"
    ]

    # Get the descriptive cluster name
    user_cluster_name = cluster_labels[int(user_cluster)]
    user_cluster_color = cluster_colors[int(user_cluster)]

    st.markdown(
        f"""
        <div style="background-color:{user_cluster_color}; padding:10px; border-radius:10px; text-align:center; color:white;">
            🏥 Your Health Group: <b>{user_cluster_name}</b>
        </div>
        """,
        unsafe_allow_html=True
    )

    # ✅ 11. Generate Personalized Advice Using OpenAI LLM
    #openai.api_key = os.getenv("OPENAI_API_KEY")
    openai.api_key = st.secrets["openai_api_key"]
    if openai.api_key is None:
        st.error("OPENAI_API_KEY is not set in the environment!")
    else:
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
            """, unsafe_allow_html=True
        )