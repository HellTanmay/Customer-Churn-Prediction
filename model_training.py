import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from imblearn.over_sampling import SMOTE

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
data=pd.read_csv("Telco-Customer-Churn.csv")

# # Drop customerID if it exists
if 'customerID' in data.columns:
    data.drop('customerID', axis=1, inplace=True)

# Convert target variable
data['Churn'] = data['Churn'].apply(lambda x: 1 if x == 'Yes' else 0)

# Handle missing TotalCharges (convert to numeric)
data['TotalCharges'] = pd.to_numeric(data['TotalCharges'], errors='coerce')
data['TotalCharges'] = data['TotalCharges'].fillna(data['TotalCharges'].median())

# Encode categorical features
label_encoders = {}
for column in data.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    data[column] = le.fit_transform(data[column])
    label_encoders[column] = le

# Feature Scaling
scaler = StandardScaler()
scaled_features = scaler.fit_transform(data.drop('Churn', axis=1))

X_train,X_test,y_train,y_test=train_test_split(scaled_features, data['Churn'],test_size=0.2,random_state=42)


sm = SMOTE(random_state=42)
X_train, y_train = sm.fit_resample(X_train, y_train)



model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
rf_acc= accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

import joblib
joblib.dump(model, 'Output\RandomForest_Classifier_model.pkl')

joblib.dump(scaler, 'Output\scaler.pkl')

joblib.dump(label_encoders, 'Output\RandomForest_Classifier_model.pkl')



# model2=LogisticRegression(max_iter=1000)
# model2.fit(X_train,y_train)
# y_pred = model2.predict(X_test)
# lr_acc= accuracy_score(y_test, y_pred)

# print("Accuracy:", accuracy_score(y_test, y_pred))
# print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
# print("\nClassification Report:\n", classification_report(y_test, y_pred))