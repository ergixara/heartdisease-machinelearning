import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

# --- 1. PAGE CONFIGURATION ---
st.set_page_config(
    page_title="MediDash AI",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- 2. CUSTOM CSS (Replicating the Dark Dashboard Image) ---
st.markdown("""
    <style>
    /* MAIN BACKGROUND */
    .stApp {
        background-color: #0b0c0e;
        color: #e0e0e0;
        font-family: 'Inter', sans-serif;
    }
    
    /* SIDEBAR STYLING */
    [data-testid="stSidebar"] {
        background-color: #111317;
        border-right: 1px solid #2b2f36;
    }
    
    /* CARDS (DASHBOARD WIDGETS) */
    .css-card {
        background-color: #1a1d23;
        border-radius: 16px;
        padding: 20px;
        border: 1px solid #2b2f36;
        box-shadow: 0 4px 12px rgba(0,0,0,0.3);
        margin-bottom: 20px;
    }
    
    /* HEADERS */
    h1, h2, h3 {
        color: #ffffff !important;
        font-weight: 700;
    }
    
    /* INPUT FIELDS (Match the dark theme) */
    .stNumberInput input, .stSelectbox div[data-baseweb="select"] > div {
        background-color: #23272f !important;
        color: white !important;
        border-radius: 8px;
        border: 1px solid #3a3f4b;
    }
    
    /* TOOLTIP ICON COLOR */
    .stTooltipIcon {
        color: #a0aab5 !important;
    }

    /* CUSTOM METRIC STYLE */
    .metric-label {
        font-size: 12px;
        text-transform: uppercase;
        letter-spacing: 1px;
        color: #8b949e;
        margin-bottom: 5px;
    }
    .metric-value {
        font-size: 28px;
        font-weight: 700;
        color: #ffffff;
    }
    .status-pill {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 12px;
        font-weight: 600;
        margin-top: 5px;
    }
    .safe { background-color: rgba(76, 175, 80, 0.2); color: #4caf50; border: 1px solid #4caf50; }
    .risk { background-color: rgba(255, 82, 82, 0.2); color: #ff5252; border: 1px solid #ff5252; }
    
    /* BUTTON STYLING */
    div.stButton > button {
        background: linear-gradient(90deg, #6c5ce7, #a29bfe);
        color: white;
        border: none;
        padding: 12px 24px;
        border-radius: 12px;
        font-weight: 600;
        width: 100%;
        margin-top: 10px;
        transition: all 0.3s;
    }
    div.stButton > button:hover {
        opacity: 0.9;
        transform: translateY(-2px);
    }
    </style>
""", unsafe_allow_html=True)

# --- 3. MULTI-MODEL TRAINING ENGINE ---
@st.cache_resource
def train_models():
    # Load Data
    url = "https://raw.githubusercontent.com/sharmaroshan/Heart-UCI-Dataset/master/heart.csv"
    df = pd.read_csv(url)
    X = df.drop('target', axis=1)
    y = df['target']
    
    # 1. Random Forest (Robust, Non-linear)
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X, y)
    
    # 2. Logistic Regression (Linear, requires scaling)
    lr = make_pipeline(StandardScaler(), LogisticRegression())
    lr.fit(X, y)
    
    # 3. K-Nearest Neighbors (Distance-based, requires scaling)
    knn = make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=5))
    knn.fit(X, y)
    
    return {"Random Forest": rf, "Logistic Regression": lr, "KNN": knn}, X.columns

models, feature_names = train_models()

# --- 4. SIDEBAR: INPUTS (With Tooltips) ---
with st.sidebar:
    st.markdown("### 🧬 Patient Data")
    st.markdown("Input clinical parameters below.")
    
    with st.form("main_inputs"):
        # Section 1: Demographics & Vitals
        st.markdown("**1. Vitals**")
        age = st.number_input("Age", 20, 100, 50, help="Patient age in years.")
        sex = st.selectbox("Sex", [1, 0], format_func=lambda x: "Male" if x==1 else "Female", help="Biological sex (1=Male, 0=Female)")
        trestbps = st.number_input("Resting BP (mm Hg)", 90, 200, 120, help="Resting blood pressure on admission.")
        chol = st.number_input("Cholesterol (mg/dl)", 100, 600, 200, help="Serum cholestoral in mg/dl.")
        
        st.markdown("---")
        
        # Section 2: Cardiac Specifics
        st.markdown("**2. Cardiac Metrics**")
        cp = st.selectbox("Chest Pain Type", [0, 1, 2, 3], 
                          format_func=lambda x: ["Typical Angina", "Atypical Angina", "Non-anginal", "Asymptomatic"][x],
                          help="0: Typical Angina\n1: Atypical Angina\n2: Non-anginal Pain\n3: Asymptomatic")
        thalach = st.slider("Max Heart Rate", 60, 220, 150, help="Maximum heart rate achieved during stress test.")
        exang = st.selectbox("Exercise Angina?", [0, 1], format_func=lambda x: "No" if x==0 else "Yes", help="Chest pain induced by exercise?")
        
        with st.expander("Advanced Clinical Parameters"):
            oldpeak = st.slider("ST Depression", 0.0, 6.0, 1.0, step=0.1, help="ST depression induced by exercise relative to rest.")
            slope = st.selectbox("ST Slope", [0, 1, 2], help="The slope of the peak exercise ST segment (0=Upsloping, 1=Flat, 2=Downsloping)")
            ca = st.slider("Major Vessels (0-3)", 0, 3, 0, help="Number of major vessels colored by flourosopy.")
            thal = st.selectbox("Thalassemia", [0, 1, 2, 3], help="Blood disorder: 1=Normal, 2=Fixed Defect, 3=Reversable Defect")
            fbs = st.selectbox("Fasting Sugar > 120?", [0, 1], format_func=lambda x: "False" if x==0 else "True", help="Fasting blood sugar > 120 mg/dl")
            restecg = st.selectbox("ECG Results", [0, 1, 2], help="0: Normal, 1: ST-T Abnormality, 2: LV Hypertrophy")

        submit_btn = st.form_submit_button("RUN DIAGNOSTICS")

# --- 5. MAIN DASHBOARD ---
st.title("Cardio-Analysis Dashboard")
st.markdown("Real-time Multi-Model Consensus Engine")

if submit_btn:
    # Prepare Data
    input_data = pd.DataFrame([{
        'age': age, 'sex': sex, 'cp': cp, 'trestbps': trestbps, 'chol': chol,
        'fbs': fbs, 'restecg': restecg, 'thalach': thalach, 'exang': exang,
        'oldpeak': oldpeak, 'slope': slope, 'ca': ca, 'thal': thal
    }])
    input_data = input_data[feature_names]
    
    # Get Predictions from ALL models
    results = {}
    probabilities = {}
    
    for name, model in models.items():
        pred = model.predict(input_data)[0]
        prob = model.predict_proba(input_data)[0][1]
        results[name] = pred
        probabilities[name] = prob

    # Calculate Consensus (Average Probability)
    avg_risk = np.mean(list(probabilities.values())) * 100
    
    # --- DASHBOARD LAYOUT ---
    
    # ROW 1: SUMMARY CARDS
    c1, c2, c3 = st.columns(3)
    
    with c1:
        st.markdown(f"""
        <div class="css-card">
            <div class="metric-label">Overall Risk Score</div>
            <div class="metric-value">{avg_risk:.1f}%</div>
            <div class="status-pill {'risk' if avg_risk > 50 else 'safe'}">
                {'HIGH RISK' if avg_risk > 50 else 'LOW RISK'}
            </div>
        </div>
        """, unsafe_allow_html=True)
        
    with c2:
        agreement = sum(list(results.values()))
        st.markdown(f"""
        <div class="css-card">
            <div class="metric-label">Model Consensus</div>
            <div class="metric-value">{agreement}/3 Models</div>
            <div style="color: #8b949e; font-size: 14px; margin-top: 5px;">
                Agree on outcome
            </div>
        </div>
        """, unsafe_allow_html=True)

    with c3:
        primary_model = "Random Forest"
        rf_conf = probabilities[primary_model] * 100
        st.markdown(f"""
        <div class="css-card">
            <div class="metric-label">Primary Confidence</div>
            <div class="metric-value">{rf_conf:.1f}%</div>
            <div style="color: #8b949e; font-size: 14px; margin-top: 5px;">
                Based on Random Forest
            </div>
        </div>
        """, unsafe_allow_html=True)

    # ROW 2: VISUALIZATION (THE CHART)
    st.markdown("### 📊 Model Comparison")
    
    # Create a nice dark-themed bar chart
    fig = go.Figure()
    
    colors = ['#6c5ce7', '#00b894', '#0984e3'] # Purple, Green, Blue
    
    fig.add_trace(go.Bar(
        x=list(probabilities.values()),
        y=list(probabilities.keys()),
        orientation='h',
        marker=dict(color=list(probabilities.values()), colorscale='RdYlGn_r', cmin=0, cmax=1),
        text=[f"{p*100:.1f}%" for p in probabilities.values()],
        textposition='auto',
    ))

    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white'),
        xaxis=dict(range=[0, 1], showgrid=True, gridcolor='#333', title="Probability of Disease"),
        margin=dict(l=0, r=0, t=20, b=20),
        height=300
    )
    
    st.plotly_chart(fig, use_container_width=True)

    # ROW 3: DETAILED BREAKDOWN
    st.markdown("### 📋 Model Breakdown")
    cols = st.columns(3)
    
    for i, (name, prob) in enumerate(probabilities.items()):
        with cols[i]:
            status_color = "#ff5252" if prob > 0.5 else "#4caf50"
            status_text = "DETECTED" if prob > 0.5 else "SAFE"
            
            st.markdown(f"""
            <div class="css-card" style="text-align: center;">
                <h4 style="margin:0; color: #aaa;">{name}</h4>
                <h2 style="margin: 10px 0; color: {status_color};">{status_text}</h2>
                <p style="margin:0; font-size: 14px; color: #666;">Probability: {prob*100:.1f}%</p>
            </div>
            """, unsafe_allow_html=True)

else:
    # EMPTY STATE (Before User Clicks Button)
    st.info("👈 Enter patient data in the sidebar and click RUN DIAGNOSTICS to see the multi-model analysis.")
