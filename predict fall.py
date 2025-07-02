import joblib
import pandas as pd

# Load the saved model
model = joblib.load('HDRP_fall_predictor.pkl')

# Example: Load new data (replace with your input)
new_data = pd.DataFrame({
    'Heart Rate': [75],
    'Respiratory Rate': [18],
    'Gait Speed': [0.9],
    'Step Symmetry': [0.85],
    'Postural Sway': [0.12],
    'Gait Variability Index': [0.2],
    'Sleep Interruption': [0],
    'Reduced Deep Sleep': [0],
    'Restlessness': [0],
    'Missed Meal': [0],
    'Change in Medication': [0]
})

# Predict fall probability for new data
predicted_fall_prob = model.predict(new_data)
print(f"Predicted Fall Probability: {predicted_fall_prob[0]:.4f}")
