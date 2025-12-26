
import joblib;
import pandas as pd;
from flask import Flask, request, jsonify,render_template
model2 = joblib.load('Output/RandomForest_Classifier_model.pkl')
scaler = joblib.load('Output/scaler.pkl')
le = joblib.load('Output/label_encoders.pkl')

app = Flask(__name__)
@app.route('/')
def home():
    return render_template('index.html')
@app.route('/predict', methods=['POST'])
def predict():
    new_customer = {
    "gender": "Female",
        "SeniorCitizen": 1,
        "Partner": "No",
        "Dependents": "No",
        "tenure": 1,
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "Fiber optic",
        "OnlineSecurity": "No",
        "OnlineBackup": "No",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "Yes",
        "StreamingMovies": "No",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 120,
        "TotalCharges": 120
    }
    input_data = request.json
    # Convert to DataFrame
    new_df = pd.DataFrame([input_data])

    # Encode and scale same as training
    for col in new_df.select_dtypes(include=['object']).columns:
        new_df[col] = le[col].transform(new_df[col])

    new_df = scaler.transform(new_df)

    # Predict churn
    prediction = model2.predict(new_df)

    return jsonify({
        "churn_prediction":int(prediction[0]),
    })
if __name__ == '__main__':
    app.run(debug=True)