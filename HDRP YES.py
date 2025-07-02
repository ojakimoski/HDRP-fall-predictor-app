import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Load data
file_path = '/Users/oliviajakimoski/Documents/2025/cb/HDRP/Data for model/HDRP model test.xlsx'  # Update this path
df = pd.read_excel(file_path)

# Clean Fall Probability values to be between 0 and 1
df['Fall Probability'] = df['Fall Probability'].clip(0, 1)

# Prepare features and target
X = df.drop(columns=['Patient ID', 'Date & Time', 'Fall Probability'])
y = df['Fall Probability']

# Split dataset into train and test sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Random Forest Regressor
model = RandomForestRegressor(random_state=42)
model.fit(X_train, y_train)

# Predict on test data
y_pred = model.predict(X_test)

# Evaluate model performance
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error (MSE): {mse:.4f}")
print(f"R-squared (R2) Score: {r2:.4f}")

# Get feature importance from Random Forest
import matplotlib.pyplot as plt

feature_importances = model.feature_importances_
features = X.columns

# Combine feature names and their importance
feat_imp = pd.Series(feature_importances, index=features).sort_values(ascending=False)

print("Feature importances:")
print(feat_imp)

# Optional: plot feature importances
plt.figure(figsize=(10,6))
feat_imp.plot(kind='bar')
plt.title('Feature Importance - Random Forest Model')
plt.ylabel('Importance')
plt.xlabel('Feature')
plt.tight_layout()
plt.show()

import joblib

# Save model
joblib.dump(model, 'HDRP_fall_predictor.pkl')
print("Model saved as HDRP_fall_predictor.pkl")
