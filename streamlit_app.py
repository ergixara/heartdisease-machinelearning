import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
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

# --- 2. "CREAMY" THEME CSS ---
st.markdown("""
    <style>
    /* GLOBAL FONTS & BACKGROUND */
    @import url('https://fonts.googleapis.com/css2?family=Nunito:wght@400;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Nunito', sans-serif;
    }
    
    /* Creamy/Warm Backgrounds */
    .stApp {
        background-color: #F9F7F2; /* Soft Cream */
        color: #4A403A; /* Coffee Brown Text */
    }
    
    /* SIDEBAR - Slightly Darker Beige */
    [data-testid="stSidebar"] {
        background-color: #F0EBE3;
        border-right: 1px solid #E6E0D4;
    }
    
    /* CARDS - White with Soft Warm Shadow */
    .css-card {
        background-color: #FFFFFF;
        border-radius: 20px;
        padding: 25px;
        border: 1px solid #EAEaea;
        box-shadow: 0 10px 25px rgba(200, 180, 160, 0.15); /* Warm Shadow */
        margin-bottom: 20px;
        transition: transform 0.2s ease;
    }
    .css-card:hover {
        transform: translateY(-3px);
    }
    
    /* HEADERS */
    h1, h2, h3 {
        color: #2C2420 !important;
        font-weight: 800;
    }
    
    /* INPUT FIELDS - Clean White */
    .stNumberInput input, .stSelectbox div[data-baseweb="select"] > div {
        background-color: #FFFFFF !important;
        color: #4A403A !important;
        border-radius: 12px;
        border: 1px solid #D6D0C4;
    }
    
    /* BUTTON STYLING - Soft Gradient */
    div.stButton > button {
        background: linear-gradient(135deg, #E6B89C 0%, #E29578 100%); /* Warm Peach/Terracotta */
        color: white;
        border: none;
        padding: 12px 24px;
        border-radius: 15px;
        font-weight: 700;
        letter-spacing: 0.5px;
        width: 100%;
        box-shadow: 0 4px 15px rgba(226, 149, 120, 0.4);
        transition: all 0.3s;
    }
    div.stButton > button:hover {
        transform: scale(1.02);
        box-shadow: 0 6px 20px rgba(226, 149, 120, 0.6);
    }
    
    /* CUSTOM METRICS */
    .metric-label {
        font-size: 13px;
        text-transform: uppercase;
        letter-spacing: 1.2px;
        color: #9A8C83; /* Muted Taupe */
        font-weight: 600;
        margin-bottom: 8px;
    }
    .metric-value {
        font-size: 32px;
        font-weight: 800;
        color: #2C2420;
    }
    
    /* PILLS */
    .status-pill {
        display: inline-block;
        padding: 6px 14px;
        border-radius: 20px;
        font-size: 12px;
        font-weight: 700;
        margin-top: 5px;
    }
    /* Soft Pastel Colors for Status */
    .safe { background-color: #E3F2E6; color: #2E7D32; } /* Pastel Green */
    .risk { background-color: #FFEBEE; color: #C62828; } /* Pastel Red */
    
    </style>
""", unsafe_allow_html=True)

# --- 3. MULTI-MODEL ENGINE (Same Logic) ---
@st.cache_resource
def train_models():
    # Load Data
    url = "https://raw.githubusercontent.com/sharmaroshan/Heart-UCI-Dataset/master/heart.csv"
    df = pd.read_csv(url)
    X = df.drop('target', axis=1)
    y = df['target']
    
    # 1. Random Forest
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X, y)
    
    # 2. Logistic Regression
    lr = make_pipeline(StandardScaler(), LogisticRegression())
    lr.fit(X, y)
    
    # 3. KNN
    knn = make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=5))
    knn.fit(X, y)
    
    return {"Random Forest": rf, "Logistic Regression": lr, "KNN": knn}, X.columns

models, feature_names = train_models()

# --- 4. SIDEBAR INPUTS ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2966/2966486.png", width=50) # Optional Icon
    st.title("MediDash")
    st.markdown("Enter patient metrics below.")
    
    with st.form("main_inputs"):
        st.markdown("### 👤 Vitals")
        age = st.number_input("Age", 20, 100, 50, help="Patient age in years.")
        sex = st.selectbox("Sex", [1, 0], format_func=lambda x: "Male" if x==1 else "Female", help="Biological sex.")
        trestbps = st.number_input("Resting BP", 90, 200, 120, help="Resting blood pressure (mm Hg).")
        chol = st.number_input("Cholesterol", 100, 600, 200, help="Serum cholestoral in mg/dl.")
        
        st.markdown("### ❤️ Heart Data")
        cp = st.selectbox("Chest Pain Type", [0, 1, 2, 3], 
                          format_func=lambda x: ["Typical Angina", "Atypical Angina", "Non-anginal", "Asymptomatic"][x])
        thalach = st.slider("Max Heart Rate", 60, 220, 150)
        exang = st.selectbox("Exercise Angina?", [0, 1], format_func=lambda x: "No" if x==0 else "Yes")
        
        with st.expander("Show Advanced Fields"):
            oldpeak = st.slider("ST Depression", 0.0, 6.0, 1.0, step=0.1)
            slope = st.selectbox("ST Slope", [0, 1, 2], help="0: Upsloping, 1: Flat, 2: Downsloping")
            ca = st.slider("Major Vessels", 0, 3, 0)
            thal = st.selectbox("Thalassemia", [0, 1, 2, 3], help="1=Normal, 2=Fixed, 3=Reversable")
            fbs = st.selectbox("Fasting Sugar > 120?", [0, 1], format_func=lambda x: "False" if x==0 else "True")
            restecg = st.selectbox("ECG Results", [0, 1, 2])

        st.markdown("<br>", unsafe_allow_html=True)
        submit_btn = st.form_submit_button("Analyze Patient")

# --- 5. MAIN DASHBOARD ---
st.markdown("## 📊 Patient Analysis Report")
st.markdown("Results generated by multi-model consensus engine.")

if submit_btn:
    # Prepare Data
    input_data = pd.DataFrame([{
        'age': age, 'sex': sex, 'cp': cp, 'trestbps': trestbps, 'chol': chol,
        'fbs': fbs, 'restecg': restecg, 'thalach': thalach, 'exang': exang,
        'oldpeak': oldpeak, 'slope': slope, 'ca': ca, 'thal': thal
    }])
    input_data = input_data[feature_names]
    
    # Get Predictions
    probabilities = {}
    predictions = {}
    for name, model in models.items():
        probabilities[name] = model.predict_proba(input_data)[0][1]
        predictions[name] = model.predict(input_data)[0]

    avg_risk = np.mean(list(probabilities.values())) * 100
    
    # --- ROW 1: CARDS ---
    c1, c2, c3 = st.columns(3)
    
    with c1:
        risk_class = "risk" if avg_risk > 50 else "safe"
        risk_label = "High Risk" if avg_risk > 50 else "Healthy"
        st.markdown(f"""
        <div class="css-card">
            <div class="metric-label">Composite Risk Score</div>
            <div class="metric-value">{avg_risk:.1f}%</div>
            <div class="status-pill {risk_class}">{risk_label}</div>
        </div>
        """, unsafe_allow_html=True)
        
    with c2:
        consensus_count = sum(list(predictions.values()))
        st.markdown(f"""
        <div class="css-card">
            <div class="metric-label">AI Consensus</div>
            <div class="metric-value">{consensus_count} / 3</div>
            <div style="font-size: 14px; color: #9A8C83; margin-top: 5px;">Models indicate disease</div>
        </div>
        """, unsafe_allow_html=True)

    with c3:
        # Confidence of the Random Forest model
        rf_conf = probabilities['Random Forest'] * 100
        st.markdown(f"""
        <div class="css-card">
            <div class="metric-label">Random Forest Conf.</div>
            <div class="metric-value">{rf_conf:.1f}%</div>
            <div style="font-size: 14px; color: #9A8C83; margin-top: 5px;">Primary Model Confidence</div>
        </div>
        """, unsafe_allow_html=True)

    # --- ROW 2: CHART ---
    st.markdown("### Model Agreement Analysis")
    
    fig = go.Figure()
    
    # Soft Pastel Colors for the Chart
    chart_colors = ['#A8DADC', '#457B9D', '#1D3557'] 
    
    fig.add_trace(go.Bar(
        x=list(probabilities.values()),
        y=list(probabilities.keys()),
        orientation='h',
        marker=dict(color=list(probabilities.values()), colorscale='Bluyl', cmin=0, cmax=1),
        text=[f"{p*100:.1f}%" for p in probabilities.values()],
        textposition='auto',
    ))

    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)', # Transparent to show cream bg
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#4A403A', family="Nunito"), # Dark text
        xaxis=dict(range=[0, 1], showgrid=False, title="Probability"),
        margin=dict(l=0, r=0, t=10, b=10),
        height=250
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # --- ROW 3: DETAILS ---
    st.markdown("### Individual Model Findings")
    col_d1, col_d2, col_d3 = st.columns(3)
    
    for i, (name, prob) in enumerate(probabilities.items()):
        status = "Detected" if prob > 0.5 else "Clear"
        color = "#C62828" if prob > 0.5 else "#2E7D32"
        
        with [col_d1, col_d2, col_d3][i]:
            st.markdown(f"""
            <div class="css-card" style="text-align: center; padding: 15px;">
                <h4 style="margin:0; color: #9A8C83; font-size: 14px;">{name}</h4>
                <h3 style="margin: 5px 0; color: {color};">{status}</h3>
                <p style="font-size: 13px; color: #9A8C83;">Risk: {prob*100:.1f}%</p>
            </div>
            """, unsafe_allow_html=True)

else:
    st.info("👈 Please input patient metrics in the sidebar to generate a report.")
