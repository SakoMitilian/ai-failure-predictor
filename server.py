from flask import Flask, request, jsonify
import pickle
import numpy as np
import pandas as pd

app = Flask(__name__)

# ✅ Load Trained Model & Scaler
model = pickle.load(open("model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

# ✅ Define the exact feature column order (MUST match training dataset)
FEATURE_COLUMNS = [
    "cpu_usage", "ram_usage", "disk_io", "temperature", "gpu_load",
    "network_latency", "error_logs", "disk_usage", "uptime"
]

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()

        # ✅ Convert input into a DataFrame with correct column names
        df = pd.DataFrame([data['features']], columns=FEATURE_COLUMNS)

        # ✅ Apply the same scaling used in training
        features_scaled = scaler.transform(df)

        # ✅ Make prediction
        probability = model.predict_proba(features_scaled)[0][1]
        print(f"Failure Probability: {probability}")

        # Extract key system metrics
        cpu_usage = df["cpu_usage"].values[0]
        temp = df["temperature"].values[0]
        error_logs = df["error_logs"].values[0]

        # **Dynamic Threshold Based on System Condition**
        if cpu_usage > 90 or temp > 85 or error_logs > 10:
            threshold = 0.15  # More sensitive for risky cases
        elif cpu_usage < 50 and temp < 60 and error_logs == 0:
            threshold = 0.35  # More strict for healthy systems
        else:
            threshold = 0.25  # Default case

        prediction = 1 if probability > threshold else 0

        return jsonify({"failure_prediction": prediction, "probability": probability})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=10000)
