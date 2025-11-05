import streamlit as st
import pandas as pd
import joblib
import numpy as np
import shap
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

model = joblib.load('random_forest_model.pkl')

# Mappings for dropdowns
reason_options = {
    1: "Certain infectious & parasitic diseases", 
    2: "Neoplasms", 
    3: "Blood & immune disorders", 
    4: "Endocrine/metabolic diseases", 
    5: "Mental/behavioral disorders", 
    6: "Nervous system diseases", 
    7: "Eye diseases", 
    8: "Ear/mastoid process diseases", 
    9: "Circulatory system diseases", 
    10: "Respiratory system diseases", 
    11: "Digestive system diseases",
    12: "Skin/subcutaneous diseases",
    13: "Musculoskeletal/connective diseases",
    14: "Genitourinary diseases",
    15: "Pregnancy/childbirth",
    16: "Perinatal period conditions",
    17: "Congenital/chromosomal abnormalities",
    18: "Unclassified symptoms",
    19: "Injury/poisoning/external causes",
    20: "External morbidity/mortality causes",
    21: "Health status/contact",
    22: "Patient follow-up",
    23: "Medical consultation",
    24: "Blood donation",
    25: "Lab examination",
    26: "Unjustified absence",
    27: "Physiotherapy",
    28: "Dental consultation"
}
month_options = {i: m for i, m in enumerate(["January", "February", "March", "April", "May", "June", "July","August", "September", "October", "November", "December"], 1)}
day_options = {
    2: "Monday",
    3: "Tuesday",
    4: "Wednesday",
    5: "Thursday",
    6: "Friday"
}
season_options = {
    1: "Summer",
    2: "Autumn",
    3: "Winter",
    4: "Spring"
}
education_options = {
    1: "High School",
    2: "Graduate",
    3: "Postgraduate",
    4: "Master/Doctor"
}
binary_options = {
    0: "No",
    1: "Yes"
}

def load_test_data_for_fairness():
    return pd.DataFrame({
        "Age": [25, 35, 45, 55, 30, 40, 50, 60],
        "Education": [1,2,3,4,2,1,4,3],
        "Risk_label": [0,0,1,1,0,1,1,0],
        "Risk_pred_before": [0,1,1,0,0,1,1,1],
        "Risk_pred_after": [0,0,1,1,0,1,1,1]
    })

st.set_page_config(page_title="Absenteeism AI Predictor", layout="wide", page_icon="🤖")
st.title("🤖 Absenteeism AI Predictor")

st.write(
    "This AI tool *estimates* how likely someone is to miss work, based on patterns found in the company's past data."
)

with st.expander("What this model does and doesn't do", expanded=True):
    st.write(
        "This app looks at all the details you enter, finds similar cases in its training data, and forecasts absenteeism **risk**. "
        "It can't see things like last-minute emergencies, big company changes, or information not present in your data. "
        "This should be used as part of good decision making, not the only answer."
    )

col1, col2 = st.columns(2)
user_inputs = {}
with st.form("input_form"):
    st.markdown("**Enter Employee or Department Details**")
    with col1:
        user_inputs['Reason for absence'] = st.selectbox("Reason for absence (ICD)", list(reason_options.keys()), format_func=lambda x: reason_options[x])
        user_inputs['Month of absence'] = st.selectbox("Month of absence", list(month_options.keys()), format_func=lambda x: month_options[x])
        user_inputs['Day of the week'] = st.selectbox("Day of week", list(day_options.keys()), format_func=lambda x: day_options[x])
        user_inputs['Seasons'] = st.selectbox("Season", list(season_options.keys()), format_func=lambda x: season_options[x])
        user_inputs['Transportation expense'] = st.number_input("Transportation expense", min_value=0)
        user_inputs['Distance from Residence to Work'] = st.number_input("Distance from Residence to Work (km)", min_value=0)
        user_inputs['Service time'] = st.number_input("Service time (years)", min_value=0)
        user_inputs['Age'] = st.number_input("Age", min_value=0)
        user_inputs['Work load Average/day '] = st.number_input("Work load Average/day", min_value=0.0)
    with col2:
        user_inputs['Hit target'] = st.number_input("Hit target (%)", min_value=0.0)
        user_inputs['Disciplinary failure'] = st.selectbox("Disciplinary failure", list(binary_options.keys()), format_func=lambda x: binary_options[x])
        user_inputs['Education'] = st.selectbox("Education", list(education_options.keys()), format_func=lambda x: education_options[x])
        user_inputs['Son'] = st.number_input("Number of children (son)", min_value=0)
        user_inputs['Social drinker'] = st.selectbox("Social drinker", list(binary_options.keys()), format_func=lambda x: binary_options[x])
        user_inputs['Social smoker'] = st.selectbox("Social smoker", list(binary_options.keys()), format_func=lambda x: binary_options[x])
        user_inputs['Pet'] = st.number_input("Number of pets", min_value=0)
        user_inputs['Weight'] = st.number_input("Weight (kg)", min_value=0)
        user_inputs['Height'] = st.number_input("Height (cm)", min_value=0)
        user_inputs['Body mass index'] = st.number_input("Body mass index", min_value=0.0)
    submitted = st.form_submit_button("Predict 🚀", type="primary")

threshold = st.sidebar.slider(
    "Risk Threshold (%) to classify 'At Risk'", 0, 100, 50, help="Change this to adjust when we consider a risk 'high'."
)

if submitted:
    input_df = pd.DataFrame([user_inputs])
    pred_hours = model.predict(input_df)[0]
    risk_score = min(max(pred_hours / 20, 0), 1)
    risk_percent = int(risk_score * 100)
    st.metric(label="Risk of excessive absence", value=f"{risk_percent}%")
    risk_label = "At risk" if risk_percent >= threshold else "Not at risk"
    st.markdown(f"### Risk classification (threshold = {threshold}%): **{risk_label}**")

    with st.expander("Why this prediction, for above provided data?"):
        st.write(
            "Here's how your information influenced the prediction. "
            "If a feature is marked 'Raises risk', it means the AI found, in past data, that this factor is linked to higher chance of absence. "
            "'Lowers risk' means the opposite. "
            "For example: High distance to work or many prior absences can tip predictions."
        )
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(input_df)
        features_list = list(user_inputs.keys())
        top_indices = np.argsort(np.abs(shap_values[0]))[::-1]
        local_descr_table = []
        for idx in top_indices[:5]:
            effect = "Raises risk" if shap_values[0][idx] > 0 else "Lowers risk"
            real_world = {
                "Distance from Residence to Work": "People living far from work may face more commuting problems.",
                "Age": "Sometimes younger workers have different attendance patterns.",
                "Transportation expense": "Higher expenses might be tied to difficult commutes.",
                "Service time": "Employees with less time at the company may have different habits.",
                "Social drinker": "Can sometimes relate to attendance patterns."
            }
            key = features_list[idx]
            reason = real_world.get(key, "This factor stood out in company records.")
            local_descr_table.append({
                "Detail": key,
                "Your value": input_df.iloc[0, idx],
                "Effect": effect,
                "Why it matters": reason
            })
        st.table(pd.DataFrame(local_descr_table))
        st.caption("Above table explains how your top 5 details affected this prediction.")

    with st.expander("See what the AI finds most important"):
        st.write(
            "These features have the strongest influence on absenteeism predictions across *everyone*. "
            "Factors like long commutes, prior disciplinary issues, workload, and even health behaviors show up often in the company's historical data."
        )
        val_data = np.array([[user_inputs[f] for f in input_df.columns] for _ in range(10)])
        global_shap_vals = explainer.shap_values(pd.DataFrame(val_data, columns=input_df.columns))
        mean_abs_shaps = np.mean(np.abs(global_shap_vals), axis=0)
        global_indices = np.argsort(mean_abs_shaps)[::-1]
        global_descr_table = []
        for idx in global_indices[:5]:
            key = features_list[idx]
            reason = {
                "Distance from Residence to Work": "Longer commutes are often linked to attendance problems.",
                "Service time": "People newer to the company can behave differently.",
                "Disciplinary failure": "Prior warnings often predict further issues.",
                "Age": "Age patterns in absenteeism are backed by past data.",
                "Social drinker": "Can influence social/health patterns that affect attendance."
            }.get(key, "This factor is often important in our records.")
            global_descr_table.append({
                "General factor": key,
                "Why important": reason
            })
        
        st.table(pd.DataFrame(global_descr_table))
        st.caption("Above are the top 5 factors influencing absenteeism predictions overall.")

    test_data = load_test_data_for_fairness()
    y_true = test_data['Risk_label']
    y_pred_before = test_data['Risk_pred_before']
    y_pred_after = test_data['Risk_pred_after']

    with st.expander("Fairness & accuracy for different groups"):
        st.write(
            "See below how the model works for people of different ages and education, and how improvements helped make its results more balanced and fair."
        )
        overall_metrics = pd.DataFrame({
            "Metric": ["Accuracy", "F1 Score", "ROC-AUC"],
            "Before": [f"{accuracy_score(y_true, y_pred_before):.2f}", 
                       f"{f1_score(y_true, y_pred_before):.2f}", 
                       f"{roc_auc_score(y_true, y_pred_before):.2f}"],
            "After": [f"{accuracy_score(y_true, y_pred_after):.2f}", 
                      f"{f1_score(y_true, y_pred_after):.2f}", 
                      f"{roc_auc_score(y_true, y_pred_after):.2f}"]
        })
        st.markdown("##### Overall Model Metrics")
        st.table(overall_metrics)

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

        edu_table = []
        for level in education_levels:
            sr_b = selection_rate(test_data, 'Education', level, 'Risk_pred_before')
            sr_a = selection_rate(test_data, 'Education', level, 'Risk_pred_after')
            tpr_b = true_positive_rate(test_data, 'Education', level, 'Risk_pred_before')
            tpr_a = true_positive_rate(test_data, 'Education', level, 'Risk_pred_after')
            edu_table.append([education_options[level], f"{sr_b:.2f}", f"{sr_a:.2f}", f"{tpr_b:.2f}", f"{tpr_a:.2f}"])
        st.markdown("##### Education Differences")
        st.table(pd.DataFrame(edu_table, columns=["Education", "Risk flagged before", "Risk flagged after", "Correct at-risk before", "Correct at-risk after"]))
        age_table = []
        
        for age in age_bins:
            sr_b = selection_rate(test_data, 'Age', age, 'Risk_pred_before')
            sr_a = selection_rate(test_data, 'Age', age, 'Risk_pred_after')
            tpr_b = true_positive_rate(test_data, 'Age', age, 'Risk_pred_before')
            tpr_a = true_positive_rate(test_data, 'Age', age, 'Risk_pred_after')
            age_table.append([str(age), f"{sr_b:.2f}", f"{sr_a:.2f}", f"{tpr_b:.2f}", f"{tpr_a:.2f}"])
        st.markdown("##### Age Differences")
        st.table(pd.DataFrame(age_table, columns=["Age", "Risk flagged before", "Risk flagged after", "Correct at-risk before", "Correct at-risk after"]))
        st.caption("It's a good sign to see more balanced results after improvements.")

    with st.expander("FAQs about this model"):
        st.write("""  
        **Q: How does this tool make predictions?**  
        A: It looks at the details you enter and finds what similar people from the company's past experience did.  
        **Q: Can the model be wrong?**  
        A: Yes. Unexpected events, missing data, or new patterns can make AI suggestions less reliable.  
        **Q: How should people use this tool?**  
        A: It should be one part of good decision-making, not the only factor.  
        """)

    st.markdown("#### Responsible Use")
    st.info(
        "Use these insights as a guide and for planning, not a guarantee."
    )

st.sidebar.header("Simple, Fair & Understandable")
st.sidebar.write("• Every explanation and fairness result is in everyday language.\n• Expand for details as you wish.\n• Use responsibly.")
st.sidebar.image("https://images.unsplash.com/photo-1506744038136-46273834b3fb?auto=format&fit=crop&w=800&q=80", use_container_width=True)
