# Customer Churn Prediction 

This project addresses the critical business problem of predicting customer churn in the 
telecommunications industry, which enables companies to proactively retain customers and 
reduce revenue loss. The solution leverages the well-known Telco Customer Churn dataset 
from Kaggle, which contains customer demographics, service usage data, and account 
information relevant to the churn prediction scenario. For this analysis, a Machine Learning 
workflow was developed and implemented, applying the Random Forest algorithm to build 
and train a predictive model on a balanced dataset. After preprocessing, applying SMOTE to 
balance classes, and model training, the system achieved an accuracy of 78% when tested on 
unseen data, demonstrating the effectiveness of Random Forest in identifying customers 
likely to discontinue services. The final outcome provides actionable insights for targeted 
retention strategies and operational improvements. 

## Installation

To run this project install required dependencies
```
pip install scikit-learn imblearn pandas matplotlib joblib
```
Next Run the Project
```
python app.py
```
