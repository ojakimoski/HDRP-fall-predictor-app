import joblib
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, precision_recall_fscore_support
import matplotlib.pyplot as plt
import numpy as np

# Load data
file_path = '/Users/oliviajakimoski/Documents/2025/cb/HDRP/Data for model/HDRP data.xlsx'
df = pd.read_excel(file_path, sheet_name='Sheet1', header=0)

# Clean column names (strip spaces and invisible chars)
df.columns = df.columns.astype(str).str.strip().str.replace('\u200b', '', regex=True)

# Define feature columns and target
feature_cols = ['Heart Rate', 'Respiratory Rate', 'Gait Speed', 'Step Symmetry',
                'Postural Sway', 'Gait Variability Index', 'Sleep Interruption',
                'Reduced Deep Sleep', 'Restlessness', 'Missed Meal',
                'Change in Medication', 'Fall Probability']

X = df[feature_cols]
y = df['Adverse Health Event']

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Hyperparameter tuning with GridSearchCV
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2]
}

grid_search = GridSearchCV(RandomForestClassifier(random_state=42),
                           param_grid, cv=5, n_jobs=-1, scoring='accuracy')
grid_search.fit(X_train, y_train)

print("Best hyperparameters:", grid_search.best_params_)

# Train model with best params
best_params = grid_search.best_params_
model = RandomForestClassifier(random_state=42, **best_params)
model.fit(X_train, y_train)

# Predict probabilities
y_proba = model.predict_proba(X_test)[:, 1]

# Threshold tuning
thresholds = np.arange(0.1, 1.0, 0.05)
print("\nThreshold tuning results:")
for thresh in thresholds:
    y_pred_thresh = (y_proba >= thresh).astype(int)
    precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred_thresh, average='binary', zero_division=0)
    print(f"Threshold: {thresh:.2f} | Precision: {precision:.3f} | Recall: {recall:.3f} | F1: {f1:.3f}")

# Choose threshold (set here based on your preference, e.g., 0.35)
chosen_threshold = 0.35
y_pred_final = (y_proba >= chosen_threshold).astype(int)

# Final evaluation
print(f"\nFinal evaluation at threshold {chosen_threshold}:")
print("Accuracy:", accuracy_score(y_test, y_pred_final))
print("Classification Report:\n", classification_report(y_test, y_pred_final))

# Feature importance
importances = pd.Series(model.feature_importances_, index=feature_cols).sort_values(ascending=False)
print("Feature importances:\n", importances)

# Plot feature importance
importances.plot(kind='bar')
plt.title('Feature Importance for HDRP Model')
plt.ylabel('Importance')
plt.tight_layout()
plt.show()

joblib.dump(model, 'HDRP_adverse_event_predictor.joblib')
