import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier

# 1. SETUP: Page Title and Layout
st.set_page_config(page_title="Heart Disease Predictor", layout="wide")

st.write("""
# 🏥 Heart Disease Prediction System
This app uses a **Random Forest** machine learning model to predict the likelihood of heart disease based on medical attributes.
""")

# 2. LOAD & TRAIN (Runs once and caches the model for speed)
@st.cache_resource
def get_model():
    # Load the UCI Heart Disease Dataset
    url = "https://raw.githubusercontent.com/sharmaroshan/Heart-UCI-Dataset/master/heart.csv"
    df = pd.read_csv(url)
    
    # Simple preprocessing
    X = df.drop('target', axis=1)
    y = df['target']
    
    # Train Model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    
    return model, X.columns

model, feature_names = get_model()

# 3. SIDEBAR: User Inputs
st.sidebar.header('📝 Patient Data')

def user_input_features():
    # We use sliders and dropdowns for a better user experience
    age = st.sidebar.slider('Age', 20, 90, 50)
    sex = st.sidebar.selectbox('Sex', (1, 0), format_func=lambda x: 'Male' if x == 1 else 'Female')
    cp = st.sidebar.selectbox('Chest Pain Type', (0, 1, 2, 3), 
                              help="0: Typical Angina, 1: Atypical, 2: Non-anginal, 3: Asymptomatic")
    trestbps = st.sidebar.slider('Resting Blood Pressure', 90, 200, 120)
    chol = st.sidebar.slider('Cholesterol', 100, 600, 250)
    fbs = st.sidebar.selectbox('Fasting Blood Sugar > 120 mg/dl', (0, 1), format_func=lambda x: 'True' if x == 1 else 'False')
    restecg = st.sidebar.selectbox('Resting ECG Results', (0, 1, 2))
    thalach = st.sidebar.slider('Max Heart Rate', 60, 220, 150)
    exang = st.sidebar.selectbox('Exercise Induced Angina', (0, 1), format_func=lambda x: 'Yes' if x == 1 else 'No')
    oldpeak = st.sidebar.slider('ST Depression (Oldpeak)', 0.0, 6.2, 1.0)
    slope = st.sidebar.selectbox('Slope of Peak Exercise ST', (0, 1, 2))
    ca = st.sidebar.slider('Major Vessels Colored by Fluoroscopy', 0, 4, 0)
    thal = st.sidebar.selectbox('Thalassemia', (0, 1, 2, 3), format_func=lambda x: 'Normal' if x == 1 else 'Fixed Defect' if x == 2 else 'Reversable Defect')

    # Store in DataFrame matching training columns
    data = {
        'age': age, 'sex': sex, 'cp': cp, 'trestbps': trestbps, 'chol': chol,
        'fbs': fbs, 'restecg': restecg, 'thalach': thalach, 'exang': exang,
        'oldpeak': oldpeak, 'slope': slope, 'ca': ca, 'thal': thal
    }
    return pd.DataFrame(data, index=[0])

input_df = user_input_features()

# 4. MAIN PANEL: Display Results
st.subheader('Patient Summary')
st.write(input_df)

if st.button('🔎 Analyze Patient'):
    # Ensure columns match exact order
    input_df = input_df[feature_names]
    
    # Predict
    prediction = model.predict(input_df)
    prediction_proba = model.predict_proba(input_df)
    
    st.divider()
    
    col1, col2 = st.columns(2)
    
    with col1:
        if prediction[0] == 1:
            st.error("### 🚨 High Risk Detected")
            st.write("The model predicts this patient **has heart disease**.")
        else:
            st.success("### ✅ Low Risk Detected")
            st.write("The model predicts this patient is **healthy**.")

    with col2:
        st.write("### Confidence Score")
        prob_sick = prediction_proba[0][1]
        st.progress(prob_sick)
        st.write(f"Probability of Disease: **{prob_sick*100:.1f}%**")
