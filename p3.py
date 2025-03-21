import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import matplotlib.pyplot as plt

# Load the dataset (Replace 'enhanced_diabetes.csv' with your actual dataset file)
data = pd.read_csv("enhanced_diabetes1.csv")

# Handle missing values
data.fillna(data.mean(), inplace=True)

# Disease Statistics (Prevalence Ranges)
print("\nCalculating the prevalence of diabetes-related diseases...")

# Calculate the percentage of individuals with each disease
disease_columns = ['HeartDisease', 'KidneyDisease', 'Retinopathy', 'Neuropathy']
for disease in disease_columns:
    prevalence = (data[disease].sum() / len(data)) * 100
    print(f"Prevalence of {disease}: {prevalence:.2f}%")

# Prevention and Cure Methods
print("\nPrevention and Cure Methods:")
print("- HeartDisease: Maintain a healthy diet, exercise regularly, manage stress, and control blood sugar levels.")
print("- KidneyDisease: Stay hydrated, monitor blood pressure, and limit salt and protein intake.")
print("- Retinopathy: Schedule regular eye exams, control blood sugar and cholesterol levels.")
print("- Neuropathy: Avoid smoking, maintain good foot care, and use medications if prescribed.")

# Standardize features for consistent scaling
scaler = StandardScaler()
X = scaler.fit_transform(data.drop('Outcome', axis=1))
y = data['Outcome']  # Target variable

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Random Forest model
rf_model = RandomForestClassifier(n_estimators=200, random_state=42)
rf_model.fit(X_train, y_train)

# Make predictions
y_pred = rf_model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"\nModel Accuracy: {accuracy * 100:.2f}%")

# Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:")
print(conf_matrix)

# Visualize Confusion Matrix
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=["Non-Diabetic", "Diabetic"], yticklabels=["Non-Diabetic", "Diabetic"])
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# Classification Report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Feature Importance
feature_names = data.columns[:-1]  # Exclude 'Outcome' column
feature_importances = rf_model.feature_importances_

# Display Feature Importance Scores
print("\nFeature Importance Scores:")
for name, importance in zip(feature_names, feature_importances):
    print(f"{name}: {importance * 100:.2f}%")

# Visualize Feature Importance as a Bar Graph
plt.figure(figsize=(10, 6))
sns.barplot(x=feature_importances, y=feature_names)
plt.title("Feature Importance - Random Forest")
plt.xlabel("Importance Score")
plt.ylabel("Features")
plt.show()

# Individual Feature Contributions to Accuracy
print("\nAnalyzing individual feature contributions to accuracy:")
for i, feature_name in enumerate(feature_names):
    # Remove one feature at a time
    X_train_temp = np.delete(X_train, i, axis=1)
    X_test_temp = np.delete(X_test, i, axis=1)
    
    # Train a temporary model without the feature
    temp_model = RandomForestClassifier(n_estimators=100, random_state=42)
    temp_model.fit(X_train_temp, y_train)
    temp_pred = temp_model.predict(X_test_temp)
    temp_accuracy = accuracy_score(y_test, temp_pred)
    
    print(f"Accuracy without '{feature_name}': {temp_accuracy * 100:.2f}%")
