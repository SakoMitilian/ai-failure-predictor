from flask import Flask, request, jsonify
import pickle
import numpy as np

app = Flask(__name__)

# Load Trained Model & Scaler
model = pickle.load(open("model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        features = np.array(data['features']).reshape(1, -1)
        features = scaler.transform(features)
        prediction = model.predict(features)[0]
        return jsonify({"failure_prediction": int(prediction)})
    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
