import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier

# 1. APP CONFIGURATION
st.set_page_config(
    page_title="Heart Health AI",
    page_icon="❤️",
    layout="centered",
    initial_sidebar_state="expanded"
)

# 2. LOAD & TRAIN MODEL (Cached)
@st.cache_resource
def get_model():
    url = "https://raw.githubusercontent.com/sharmaroshan/Heart-UCI-Dataset/master/heart.csv"
    df = pd.read_csv(url)
    X = df.drop('target', axis=1)
    y = df['target']
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    return model, X.columns

model, feature_names = get_model()

# 3. UI: HEADER
st.title("❤️ Heart Disease Risk Calculator")
st.markdown("""
    <style>
    .big-font { font-size:20px !important; }
    </style>
    """, unsafe_allow_html=True)
st.markdown('<p class="big-font">Enter the patient data below to generate a risk assessment using the Random Forest model.</p>', unsafe_allow_html=True)

st.divider()

# 4. UI: FORM
st.subheader("📋 Patient Information")

# We use a form so the page doesn't reload with every single click
with st.form("patient_data"):
    col1, col2 = st.columns(2)
    
    with col1:
        age = st.number_input("Age", min_value=1, max_value=120, value=50, 
                              help="Age of the patient in years.")
        
        sex = st.selectbox("Sex", options=[1, 0], format_func=lambda x: "Male" if x == 1 else "Female",
                           help="Biological sex of the patient.")
        
        cp = st.selectbox("Chest Pain Type", options=[0, 1, 2, 3], 
                          format_func=lambda x: ["Typical Angina", "Atypical Angina", "Non-anginal Pain", "Asymptomatic"][x],
                          help="0: Typical Angina\n1: Atypical Angina\n2: Non-anginal Pain\n3: Asymptomatic")
        
        trestbps = st.number_input("Resting Blood Pressure (mm Hg)", min_value=50, max_value=250, value=120,
                                   help="Resting blood pressure (in mm Hg on admission to the hospital).")
        
        chol = st.number_input("Serum Cholesterol (mg/dl)", min_value=100, max_value=600, value=200,
                               help="Serum cholestoral in mg/dl.")
        
        fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", options=[0, 1], format_func=lambda x: "True" if x == 1 else "False",
                           help="Is fasting blood sugar > 120 mg/dl? (1 = true; 0 = false)")

    with col2:
        restecg = st.selectbox("Resting ECG Results", options=[0, 1, 2],
                               format_func=lambda x: ["Normal", "ST-T Wave Abnormality", "Left Ventricular Hypertrophy"][x],
                               help="0: Normal\n1: ST-T Wave Abnormality\n2: Left Ventricular Hypertrophy")
        
        thalach = st.number_input("Max Heart Rate Achieved", min_value=60, max_value=220, value=150,
                                  help="Maximum heart rate achieved during the test.")
        
        exang = st.selectbox("Exercise Induced Angina", options=[0, 1], format_func=lambda x: "Yes" if x == 1 else "No",
                             help="Did exercise induce angina (chest pain)?")
        
        oldpeak = st.number_input("ST Depression (Oldpeak)", min_value=0.0, max_value=10.0, value=1.0, step=0.1,
                                  help="ST depression induced by exercise relative to rest.")
        
        slope = st.selectbox("Slope of Peak Exercise ST", options=[0, 1, 2],
                             format_func=lambda x: ["Upsloping", "Flat", "Downsloping"][x],
                             help="The slope of the peak exercise ST segment.")
        
        ca = st.selectbox("Number of Major Vessels (0-3)", options=[0, 1, 2, 3],
                          help="Number of major vessels (0-3) colored by fluoroscopy.")
        
        thal = st.selectbox("Thalassemia", options=[0, 1, 2, 3],
                            format_func=lambda x: ["Unknown", "Normal", "Fixed Defect", "Reversable Defect"][x],
                            help="Blood disorder type.\n1: Normal\n2: Fixed Defect\n3: Reversable Defect")

    submit_btn = st.form_submit_button("Analyze Patient Status", type="primary")

# 5. PREDICTION & RESULTS
if submit_btn:
    # Prepare data
    data = {
        'age': age, 'sex': sex, 'cp': cp, 'trestbps': trestbps, 'chol': chol,
        'fbs': fbs, 'restecg': restecg, 'thalach': thalach, 'exang': exang,
        'oldpeak': oldpeak, 'slope': slope, 'ca': ca, 'thal': thal
    }
    input_df = pd.DataFrame([data], columns=feature_names)
    
    # Get Prediction
    prediction = model.predict(input_df)[0]
    prob = model.predict_proba(input_df)[0][1]

    st.divider()
    
    # Display Results with Metrics
    st.subheader("📊 Analysis Results")
    
    col_res1, col_res2 = st.columns([1, 2])
    
    with col_res1:
        if prediction == 1:
            st.metric(label="Risk Probability", value=f"{prob*100:.1f}%", delta="High Risk", delta_color="inverse")
        else:
            st.metric(label="Risk Probability", value=f"{prob*100:.1f}%", delta="Low Risk", delta_color="normal")

    with col_res2:
        if prediction == 1:
            st.error("#### 🚨 Prediction: Heart Disease Detected\nThe model suggests a high likelihood of heart disease. Please consult a cardiologist.")
        else:
            st.success("#### ✅ Prediction: Healthy\nThe model suggests a low likelihood of heart disease.")

# Sidebar Info
with st.sidebar:
    st.header("About")
    st.info("This app uses a Random Forest Classifier trained on the UCI Heart Disease dataset to predict patient risk.")

