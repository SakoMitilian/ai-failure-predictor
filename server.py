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

    # Adaptive Thresholding
    cpu_usage = features[0][0]
    error_logs = features[0][6]
    
    if cpu_usage > 90 or error_logs > 10:
        threshold = 0.1  # More aggressive for high-risk cases
    else:
        threshold = 0.3  # Less sensitive for normal systems

    prediction = 1 if probability > threshold else 0
    return jsonify({"failure_prediction": prediction, "probability": probability})



if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
