# 🏥 MediDash  
### Multi-Model Heart Disease Risk Prediction Dashboard

**MediDash** is an interactive **Streamlit-based web application** designed to assess heart disease risk using a **multi-model consensus approach**.  
Instead of relying on a single algorithm, MediDash combines predictions from multiple machine learning models to deliver a **more robust, transparent, and trustworthy risk assessment**.

---

## 🚀 Why MediDash?

Traditional predictors often depend on one model — which can be biased or unstable.  
**MediDash solves this by aggregating insights from multiple models**, allowing users to understand both the *final decision* and *how each model contributed to it*.

---

## ✨ Key Features

### 🔀 Multi-Model Consensus Engine
- Combines predictions from:
  - **Random Forest**
  - **Logistic Regression**
  - **K-Nearest Neighbors (KNN)**
- Reduces single-model bias and improves reliability

### 📊 Composite Risk Score
- Generates a **unified probability (%)**
- Based on model agreement and confidence levels

### 🧑‍⚕️ Interactive Patient Vitals
- Real-time input for:
  - Age  
  - Cholesterol  
  - Resting Blood Pressure  
  - Heart Rate  
  - Other clinical indicators

### 📈 Model Agreement Visualization
- Dynamic bar charts showing:
  - Individual model probabilities
  - How strongly models agree or disagree

### 🧠 Transparent Decision Logic
- Clearly displays whether:
  - **0/3**, **1/3**, **2/3**, or **3/3** models indicate risk

---

## 🛠️ Tech Stack

**Frontend**
- Streamlit

**Machine Learning**
- Scikit-Learn

**Data Processing**
- Pandas  
- NumPy  

**Visualization**
- Matplotlib  
- Plotly  

---

## ⚙️ How It Works

1. **User Input**  
   Patient data is entered through the Streamlit sidebar.

2. **Model Predictions**  
   Each pre-trained model independently predicts the probability of heart disease.

3. **Consensus Logic**  
   - The app evaluates model agreement  
   - Example: *“2 out of 3 models indicate elevated risk”*

4. **Final Assessment**  
   - Displays:
     - **Risk Status**: `Clear` or `Risk`
     - **Confidence Score**: Combined probability percentage

---

## 📌 Output Example

- ✅ **Clear** — Low predicted risk with strong model agreement  
- ⚠️ **Risk** — Elevated probability with multi-model confirmation  

Each result is accompanied by **visual explanations**, making the prediction easy to understand — even for non-technical users.

---

## 🎯 Project Goal

MediDash aims to demonstrate how **ensemble learning + explainability** can be applied to healthcare-related decision support systems, emphasizing **accuracy, transparency, and user trust**.
