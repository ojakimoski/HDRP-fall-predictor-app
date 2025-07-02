import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Load models
fall_model = joblib.load('HDRP_fall_predictor.pkl')  # Fall risk predictor
hdrp_model = joblib.load('HDRP_adverse_event_predictor.joblib')  # HDRP model

# Feature names for consistency
fall_features = [
    'Heart Rate', 'Respiratory Rate', 'Gait Speed', 'Step Symmetry',
    'Postural Sway', 'Gait Variability Index', 'Sleep Interruption',
    'Reduced Deep Sleep', 'Restlessness', 'Missed Meal', 'Change in Medication'
]
hdrp_features = fall_features + ['Fall Probability']

# Set page config
st.set_page_config(page_title='Health Risk Predictor', layout='centered')

# App title
st.title('Health Deterioration Risk Predictor (HDRP)')

st.markdown("Enter patient data to predict **Fall Risk** and **Adverse Health Event Risk** (within 48–72 hours).")

# User input form
with st.form("input_form"):
    hr = st.number_input("Heart Rate (bpm)", min_value=30, max_value=180, value=70)
    rr = st.number_input("Respiratory Rate (breaths/min)", min_value=8, max_value=40, value=16)
    gait_speed = st.number_input("Gait Speed (m/s)", min_value=0.0, value=1.0, format="%.2f")
    step_sym = st.number_input("Step Symmetry", min_value=0.0, value=0.5, format="%.2f")
    sway = st.number_input("Postural Sway", min_value=0.0, value=0.1, format="%.2f")
    gait_var = st.number_input("Gait Variability Index", min_value=0.0, value=0.2, format="%.2f")
    sleep_int = st.number_input("Sleep Interruption (count)", min_value=0, value=0)
    deep_sleep = st.number_input("Reduced Deep Sleep (count)", min_value=0, value=0)
    restlessness = st.number_input("Restlessness (count)", min_value=0, value=0)
    missed_meal = st.number_input("Missed Meal (count)", min_value=0, value=0)
    med_change = st.number_input("Change in Medication (count)", min_value=0, value=0)

    submit = st.form_submit_button("Predict Risk")

if submit:
    # Step 1: Build input DataFrame for fall model
    input_dict = {
        'Heart Rate': hr,
        'Respiratory Rate': rr,
        'Gait Speed': gait_speed,
        'Step Symmetry': step_sym,
        'Postural Sway': sway,
        'Gait Variability Index': gait_var,
        'Sleep Interruption': sleep_int,
        'Reduced Deep Sleep': deep_sleep,
        'Restlessness': restlessness,
        'Missed Meal': missed_meal,
        'Change in Medication': med_change
    }
    
    fall_input_df = pd.DataFrame([input_dict])
    
    # Step 2: Predict fall probability
    fall_proba = fall_model.predict(fall_input_df)[0]  # regression output between 0 and 1

    # Step 3: Add predicted fall prob to input for HDRP model
    hdrp_input_df = fall_input_df.copy()
    hdrp_input_df['Fall Probability'] = fall_proba

    # Step 4: Predict adverse health event
    adverse_proba = hdrp_model.predict_proba(hdrp_input_df)[0][1]  # Get probability of class "1" (event)

    # Step 5: Get feature importances from HDRP model
    if hasattr(hdrp_model, "feature_importances_"):
        importances = pd.Series(hdrp_model.feature_importances_, index=hdrp_features)
        top_features = importances.sort_values(ascending=False).head(3)
        top_driver_str = ", ".join([f"{feat} ↑" if input_dict.get(feat, 0) > 0 else f"{feat} ↓" for feat in top_features.index])
    else:
        top_driver_str = "Not available"

    # Step 6: Categorise risk levels
    def categorize(prob):
        if prob >= 0.7:
            return "High", "🔴"
        elif prob >= 0.4:
            return "Moderate", "🟠"
        else:
            return "Low", "🟢"

    fall_risk_level, fall_icon = categorize(fall_proba)
    adverse_risk_level, adverse_icon = categorize(adverse_proba)

    # Step 7: Display results
    st.subheader("📊 Risk Predictions")
    st.markdown(f"**Fall Risk:** {fall_proba:.2f} ({fall_icon} {fall_risk_level})")
    st.markdown(f"**Adverse Health Event Risk:** {adverse_proba:.2f} ({adverse_icon} {adverse_risk_level})")
    st.markdown(f"**Top Risk Drivers:** {top_driver_str}")
