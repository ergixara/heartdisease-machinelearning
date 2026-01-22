🏥 MediDash: Multi-Model Heart Disease Prediction

This Streamlit application provides an interactive platform for analyzing heart disease risk. Unlike standard single-model predictors, MediDash utilizes a consensus engine that aggregates insights from three distinct machine learning models to provide a more reliable and transparent risk assessment.

Key Features:
Multi-Model Consensus: Aggregates predictions from Random Forest, Logistic Regression, and K-Nearest Neighbors (KNN).

Composite Risk Score: Calculates a unified probability percentage based on model agreement.

Interactive Patient Vitals: Real-time input for metrics such as age, cholesterol, resting blood pressure, and heart rate.

Model Agreement Visualization: Dynamic bar charts showing the individual probability scores for each underlying algorithm.

Tech Stack:
Frontend: Streamlit

Machine Learning: Scikit-Learn

Data Manipulation: Pandas, NumPy

Visualization: Matplotlib / Plotly

The app takes patient data through the sidebar and passes it to three pre-trained models.

Individual Predictions: Each model calculates a probability of heart disease.

Consensus Logic: The app checks for agreement across models (e.g., "2/3 models indicate disease").

Final Output: The user receives a "Clear" or "Risk" status accompanied by a confidence interval.
