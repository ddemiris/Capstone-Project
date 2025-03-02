from flask import Flask, request, jsonify
import joblib
import pandas as pd
import numpy as np
import os
import logging
from datetime import datetime, timedelta

# Initialize Flask app
app = Flask(__name__)

# Setup logging
logging.basicConfig(filename='logs/api.log', level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

# Load pre-trained model
MODEL_PATH = "models/prophet_model.pkl"
if os.path.exists(MODEL_PATH):
    model = joblib.load(MODEL_PATH)
else:
    model = None


# Function to train model
def train_model():
    try:
        # Load data
        data = pd.read_csv("data/sales_data.csv", parse_dates=["transaction_date"], index_col="transaction_date")
        data = data.sort_index().reset_index().rename(columns={"transaction_date": "ds", "sales": "y"})

        # Train new model
        from fbprophet import Prophet
        new_model = Prophet()
        new_model.fit(data)

        # Save model
        joblib.dump(new_model, MODEL_PATH)
        return "Model trained and saved successfully."
    except Exception as e:
        logging.error(f"Training error: {e}")
        return f"Error during training: {e}"


# API Route: Train endpoint
@app.route('/train', methods=['POST'])
def train():
    response = train_model()
    return jsonify({"message": response})


# API Route: Predict endpoint
@app.route('/predict', methods=['GET'])
def predict():
    global model
    if model is None:
        return jsonify({"error": "Model is not loaded. Train the model first."})

    country = request.args.get("country", "global")
    target_date = request.args.get("date", (datetime.today() + timedelta(days=30)).strftime('%Y-%m-%d'))

    try:
        future = pd.DataFrame([target_date], columns=["ds"])
        forecast = model.predict(future)
        prediction = forecast["yhat"].values[0]

        logging.info(f"Prediction made for {country} on {target_date}: {prediction}")
        return jsonify({"country": country, "date": target_date, "prediction": prediction})
    except Exception as e:
        logging.error(f"Prediction error: {e}")
        return jsonify({"error": str(e)})


# API Route: Logfile endpoint
@app.route('/logs', methods=['GET'])
def get_logs():
    try:
        with open('logs/api.log', 'r') as log_file:
            logs = log_file.readlines()
        return jsonify({"logs": logs[-50:]})  # Return last 50 log entries
    except Exception as e:
        return jsonify({"error": str(e)})


# Run Flask app
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
