import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from imblearn.over_sampling import SMOTE
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score, classification_report
from flask import Flask, request, jsonify

# Load dataset (Replace with your actual dataset path)
data = pd.read_csv("failure_prediction_dataset.csv")

# Data Preprocessing
data.fillna(method='ffill', inplace=True)  # Fill missing values
scaler = MinMaxScaler()
X = data.drop(columns=["failure"])
y = data["failure"]
X = scaler.fit_transform(X)

# Splitting dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Handle class imbalance
smote = SMOTE(sampling_strategy=0.5, random_state=42)
X_train_bal, y_train_bal = smote.fit_resample(X_train, y_train)

# Train LightGBM model
model = LGBMClassifier(n_estimators=50, learning_rate=0.1, random_state=42)
model.fit(X_train_bal, y_train_bal)

# Evaluate Model
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# Deploy model with Flask
app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    features = np.array(data['features']).reshape(1, -1)
    features = scaler.transform(features)
    prediction = model.predict(features)[0]
    return jsonify({"failure_prediction": int(prediction)})

if __name__ == '__main__':
    app.run(debug=True)