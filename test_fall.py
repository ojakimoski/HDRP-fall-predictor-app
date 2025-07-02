import joblib
import pandas as pd

models = joblib.load('HDRP_fall_predictor.joblib')
fall_model = models[0]

print("Fall model expects this number of features:")
print(fall_model.n_features_in_)

# You can also test with dummy input df to confirm order:
dummy_data = pd.DataFrame([[
    70, 16, 1.0, 0.5, 0.1, 0.2, 0, 0, 0, 0, 0
]], columns=[
    'Heart Rate', 'Respiratory Rate', 'Gait Speed', 'Step Symmetry',
    'Postural Sway', 'Gait Variability Index', 'Sleep Interruption',
    'Reduced Deep Sleep', 'Restlessness', 'Missed Meal', 'Change in Medication'
])

print("Dummy input columns:")
print(dummy_data.columns.tolist())
print("Number of columns in dummy input:", len(dummy_data.columns))
