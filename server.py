from flask import Flask, request, jsonify
import pickle
import numpy as np

app = Flask(__name__)

# Load Trained Model & Scaler
model = pickle.load(open("model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    features = np.array(data['features']).reshape(1, -1)
    features = scaler.transform(features)

    probability = model.predict_proba(features)[0][1]
    print(f"Failure Probability: {probability}")

    # Extract key system metrics
    cpu_usage = features[0][0]
    temp = features[0][3]
    error_logs = features[0][6]

    # **Dynamic Threshold Based on System Condition**
    if cpu_usage > 90 or temp > 85 or error_logs > 10:
        threshold = 0.15  # More sensitive for risky cases
    elif cpu_usage < 50 and temp < 60 and error_logs == 0:
        threshold = 0.35  # More strict for healthy systems
    else:
        threshold = 0.25  # Default case

    prediction = 1 if probability > threshold else 0
    return jsonify({"failure_prediction": prediction, "probability": probability})




if __name__ == '__main__':
    app.run(host='0.0.0.0', port=10000)
