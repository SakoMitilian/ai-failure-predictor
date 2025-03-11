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

    # Get failure probability
    probability = model.predict_proba(features)[0][1]  # Probability of failure
    print(f"Failure Probability: {probability}")  # Print for debugging

    # Adjust failure threshold (default is 0.5, we will lower it)
    prediction = 1 if probability > 0.3 else 0  # More sensitive to failures

    return jsonify({"failure_prediction": prediction, "probability": probability})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
