# Student Performance Prediction
# Using Random Forest Classifier (can be changed to other models)

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# ----------------------
# 1. Load the dataset
# ----------------------
# Replace 'student_data.csv' with your CSV file path
data = pd.read_csv('student_data.csv')

# Inspect dataset
print("First 5 rows:\n", data.head())
print("\nColumns:", data.columns)
print("\nMissing values:\n", data.isnull().sum())

# ----------------------
# 2. Preprocessing
# ----------------------
# Encode categorical variables
le = LabelEncoder()
for col in data.select_dtypes(include='object').columns:
    data[col] = le.fit_transform(data[col])

# Features and target
# Example features: age, studytime, failures, absences
# Example target: final grade 'G3' (you can convert to Pass/Fail)
X = data.drop('G3', axis=1)
y = data['G3']

# Optional: Convert G3 to Pass/Fail
# y = y.apply(lambda x: 1 if x >= 10 else 0)  # Pass if grade >=10

# Train-test split
X_train, X_test, y_trai_
