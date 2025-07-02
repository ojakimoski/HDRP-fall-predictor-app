import streamlit as st
import pandas as pd
import joblib

# Load the saved model
model = joblib.load('HDRP_fall_predictor.joblib')

st.title('HDRP Fall Predictor')

# Option to upload CSV file or manual input
option = st.radio('Choose input method:', ('Upload CSV', 'Manual input'))

if option == 'Upload CSV':
    uploaded_file = st.file_uploader('Upload your data CSV file', type=['csv'])
    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
        st.write('Data preview:')
        st.dataframe(data.head())
        
        # Make predictions if columns match
        feature_cols = ['Heart Rate', 'Respiratory Rate', 'Gait Speed', 'Step Symmetry',
                        'Postural Sway', 'Gait Variability Index', 'Sleep Interruption',
                        'Reduced Deep Sleep', 'Restlessness', 'Missed Meal',
                        'Change in Medication', 'Fall Probability']
        if all(col in data.columns for col in feature_cols):
            preds = model.predict(data[feature_cols])
            data['Predicted Adverse Health Event'] = preds
            st.write('Predictions:')
            st.dataframe(data[['Predicted Adverse Health Event']])
        else:
            st.error(f"CSV missing required columns: {feature_cols}")

else:  # Manual input
    st.write('Enter patient data manually:')
    input_data = {}
    input_data['Heart Rate'] = st.number_input('Heart Rate', value=70)
    input_data['Respiratory Rate'] = st.number_input('Respiratory Rate', value=16)
    input_data['Gait Speed'] = st.number_input('Gait Speed', value=1.0)
    input_data['Step Symmetry'] = st.number_input('Step Symmetry', value=0.5)
    input_data['Postural Sway'] = st.number_input('Postural Sway', value=0.1)
    input_data['Gait Variability Index'] = st.number_input('Gait Variability Index', value=0.2)
    input_data['Sleep Interruption'] = st.number_input('Sleep Interruption', value=0)
    input_data['Reduced Deep Sleep'] = st.number_input('Reduced Deep Sleep', value=0)
    input_data['Restlessness'] = st.number_input('Restlessness', value=0)
    input_data['Missed Meal'] = st.number_input('Missed Meal', value=0)
    input_data['Change in Medication'] = st.number_input('Change in Medication', value=0)
    input_data['Fall Probability'] = st.number_input('Fall Probability', value=0.2)

    if st.button('Predict'):
        input_df = pd.DataFrame([input_data])
        prediction = model.predict(input_df)[0]
        st.write(f'Predicted Adverse Health Event: {"Yes" if prediction == 1 else "No"}')

