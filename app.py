import streamlit as st
import pandas as pd
import joblib
import numpy as np
import shap
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

model = joblib.load('random_forest_model.pkl')

# Mappings for dropdowns
reason_options = {1: "Certain infectious & parasitic diseases", 2: "Neoplasms", 3: "Blood & immune disorders", 4: "Endocrine/metabolic diseases", 5: "Mental/behavioral disorders", 6: "Nervous system diseases", 7: "Eye diseases", 8: "Ear/mastoid process diseases", 9: "Circulatory system diseases", 10: "Respiratory system diseases", 11: "Digestive system diseases", 12: "Skin/subcutaneous diseases", 13: "Musculoskeletal/connective diseases", 14: "Genitourinary diseases", 15: "Pregnancy/childbirth", 16: "Perinatal period conditions", 17: "Congenital/chromosomal abnormalities", 18: "Unclassified symptoms", 19: "Injury/poisoning/external causes", 20: "External morbidity/mortality causes", 21: "Health status/contact", 22: "Patient follow-up", 23: "Medical consultation", 24: "Blood donation", 25: "Lab examination", 26: "Unjustified absence", 27: "Physiotherapy", 28: "Dental consultation"}
month_options = {i: m for i, m in enumerate(["January", "February", "March", "April", "May", "June", "July","August", "September", "October", "November", "December"], 1)}
day_options = {2: "Monday", 3: "Tuesday", 4: "Wednesday", 5: "Thursday", 6: "Friday"}
season_options = {1: "Summer", 2: "Autumn", 3: "Winter", 4: "Spring"}
education_options = {1: "High School", 2: "Graduate", 3: "Postgraduate", 4: "Master/Doctor"}
binary_options = {0: "No", 1: "Yes"}

def load_test_data_for_fairness():
    df = pd.DataFrame({
        "Age": [25, 35, 45, 55, 30, 40, 50, 60],
        "Education": [1,2,3,4,2,1,4,3],
        "Risk_label": [0,0,1,1,0,1,1,0],
        "Risk_pred_before": [0,1,1,0,0,1,1,1],
        "Risk_pred_after": [0,0,1,1,0,1,1,1]
    })
    return df

st.set_page_config(page_title="💡 Absenteeism AI Predictor", layout="wide", page_icon="🤖")
st.title("🤖 AI-driven Absenteeism Prediction")

st.markdown("""
Welcome to your intelligent prediction system! Enter worker or department-level details below and get insights powered by transparent AI and fairness analytics.
""")

col1, col2 = st.columns(2)
user_inputs = {}

with st.form("input_form"):
    st.markdown("**worker or department attributes**")
    with col1:
        user_inputs['Reason for absence'] = st.selectbox("Reason for absence (ICD)", options=list(reason_options.keys()), format_func=lambda x: reason_options[x])
        user_inputs['Month of absence'] = st.selectbox("Month of absence", options=list(month_options.keys()), format_func=lambda x: month_options[x])
        user_inputs['Day of the week'] = st.selectbox("Day of week", options=list(day_options.keys()), format_func=lambda x: day_options[x])
        user_inputs['Seasons'] = st.selectbox("Season", options=list(season_options.keys()), format_func=lambda x: season_options[x])
        user_inputs['Transportation expense'] = st.number_input("Transportation expense", min_value=0)
        user_inputs['Distance from Residence to Work'] = st.number_input("Distance from Residence to Work (km)", min_value=0)
        user_inputs['Service time'] = st.number_input("Service time (years)", min_value=0)
        user_inputs['Age'] = st.number_input("Age", min_value=0)
        user_inputs['Work load Average/day '] = st.number_input("Work load Average/day", min_value=0.0)
    with col2:
        user_inputs['Hit target'] = st.number_input("Hit target (%)", min_value=0.0)
        user_inputs['Disciplinary failure'] = st.selectbox("Disciplinary failure", options=list(binary_options.keys()), format_func=lambda x: binary_options[x])
        user_inputs['Education'] = st.selectbox("Education", options=list(education_options.keys()), format_func=lambda x: education_options[x])
        user_inputs['Son'] = st.number_input("Number of children (son)", min_value=0)
        user_inputs['Social drinker'] = st.selectbox("Social drinker", options=list(binary_options.keys()), format_func=lambda x: binary_options[x])
        user_inputs['Social smoker'] = st.selectbox("Social smoker", options=list(binary_options.keys()), format_func=lambda x: binary_options[x])
        user_inputs['Pet'] = st.number_input("Number of pets", min_value=0)
        user_inputs['Weight'] = st.number_input("Weight (kg)", min_value=0)
        user_inputs['Height'] = st.number_input("Height (cm)", min_value=0)
        user_inputs['Body mass index'] = st.number_input("Body mass index", min_value=0.0)
    submitted = st.form_submit_button("Predict 🚀", type="primary")

# Threshold slider
threshold = st.sidebar.slider("Risk Threshold (%) to classify 'At Risk'", 0, 100, 50)

if submitted:
    input_df = pd.DataFrame([user_inputs])
    pred_hours = model.predict(input_df)[0]
    risk_score = min(max(pred_hours / 20, 0), 1)
    risk_percent = int(risk_score * 100)
    st.metric(label="Risk of excessive absence", value=f"{risk_percent}%")
    risk_label = "At risk" if risk_percent >= threshold else "Not at risk"
    st.markdown(f"### Risk classification (threshold = {threshold}%): **{risk_label}**")

    # Feature interpretability
    st.markdown("#### 🔎 Feature Influence (Interpretability)")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(input_df)
    features_list = list(user_inputs.keys())
    top_features = np.argsort(np.abs(shap_values[0]))[::-1][:3]
    for i in top_features:
        impact_type = 'High' if shap_values[0][i] > 0 else 'Low'
        st.info(f"{features_list[i]}: {impact_type} impact on prediction ({shap_values[0][i]:.2f})")

    # Fairness evaluation panel
    st.markdown("---")
    st.subheader("Fairness Evaluation")
    test_data = load_test_data_for_fairness()

    y_true = test_data['Risk_label']
    y_pred_before = test_data['Risk_pred_before']
    y_pred_after = test_data['Risk_pred_after']

    accuracy_before = accuracy_score(y_true, y_pred_before)
    accuracy_after = accuracy_score(y_true, y_pred_after)
    f1_before = f1_score(y_true, y_pred_before)
    f1_after = f1_score(y_true, y_pred_after)
    roc_auc_before = roc_auc_score(y_true, y_pred_before)
    roc_auc_after = roc_auc_score(y_true, y_pred_after)

    st.markdown("**Overall Performance Metrics:**")
    col3, col4 = st.columns(2)
    with col3:
        st.write("Before Mitigation")
        st.write(f"- Accuracy: {accuracy_before:.2f}")
        st.write(f"- F1 Score: {f1_before:.2f}")
        st.write(f"- ROC-AUC: {roc_auc_before:.2f}")
    with col4:
        st.write("After Mitigation")
        st.write(f"- Accuracy: {accuracy_after:.2f}")
        st.write(f"- F1 Score: {f1_after:.2f}")
        st.write(f"- ROC-AUC: {roc_auc_after:.2f}")

    # Fairness subgroup metrics
    st.markdown("**Fairness Indicators (Selection Rate & TPR):**")

    def selection_rate(df, group_col, group_val, pred_col):
        subgroup = df[df[group_col] == group_val]
        return subgroup[pred_col].mean()

    def true_positive_rate(df, group_col, group_val, pred_col):
        subgroup = df[df[group_col] == group_val]
        tp = ((subgroup['Risk_label'] == 1) & (subgroup[pred_col] == 1)).sum()
        actual_pos = (subgroup['Risk_label'] == 1).sum()
        return tp / actual_pos if actual_pos > 0 else 0

    education_levels = sorted(test_data['Education'].unique())
    age_bins = sorted(test_data['Age'].unique())

    st.write("**By Education Level:**")
    for level in education_levels:
        sr_before = selection_rate(test_data, 'Education', level, 'Risk_pred_before')
        sr_after = selection_rate(test_data, 'Education', level, 'Risk_pred_after')
        tpr_before = true_positive_rate(test_data, 'Education', level, 'Risk_pred_before')
        tpr_after = true_positive_rate(test_data, 'Education', level, 'Risk_pred_after')
        st.write(f"- {education_options[level]}: Selection Rate Before: {sr_before:.2f}, After: {sr_after:.2f}, TPR Before: {tpr_before:.2f}, After: {tpr_after:.2f}")

    st.write("**By Age Group:**")
    for age in age_bins:
        sr_before = selection_rate(test_data, 'Age', age, 'Risk_pred_before')
        sr_after = selection_rate(test_data, 'Age', age, 'Risk_pred_after')
        tpr_before = true_positive_rate(test_data, 'Age', age, 'Risk_pred_before')
        tpr_after = true_positive_rate(test_data, 'Age', age, 'Risk_pred_after')
        st.write(f"- Age {age}: Selection Rate Before: {sr_before:.2f}, After: {sr_after:.2f}, TPR Before: {tpr_before:.2f}, After: {tpr_after:.2f}")

    # Responsible AI Usage Section
    st.markdown("#### ℹ️ Responsible AI Usage & Model Scope")
    st.info("""
    - This system provides statistical predictions based solely on historical data and input features.
    - It cannot predict unforeseen or external factors affecting absenteeism.
    - Use model outputs responsibly: as decision support **not** as sole decision criterion.
    - Avoid overreliance or discriminatory use of the predictions.
    - Fairness metrics shown demonstrate mitigation benefits.
    """)

st.sidebar.image("https://images.unsplash.com/photo-1506744038136-46273834b3fb?auto=format&fit=crop&w=800&q=80", use_container_width=True)
st.sidebar.header("About")
st.sidebar.write("""
- RandomForestRegressor-based risk prediction
- Transparency with interpretability and fairness
- Supports responsible and ethical AI usage
""")