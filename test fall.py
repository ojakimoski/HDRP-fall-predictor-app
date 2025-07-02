import joblib
import pandas as pd

models = joblib.load('HDRP_fall_predictor.joblib')
fall_model = models[0]

print("Fall model expects these features:")
print(fall_model.feature_names_in_)
