import os
import json
import numpy as np
import pandas as pd
import yfinance as yf
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
from flask import Flask, request, jsonify
from flask_cors import CORS
from flask_cors import CORS  # <-- Import this to handle CORS

app = Flask(__name__)
CORS(app)  # <-- Allow frontend to communicate with the backend

# Ensure the model directory exists
MODEL_WEIGHTS_PATH = "lstm_model_weights.h5"

def load_data(stock_symbol, look_back=60):
    try:
        # Fetch stock data
        df = yf.download(stock_symbol, period="1y", interval="1d")
        if df.empty:
            raise ValueError(f"No data found for stock symbol {stock_symbol}")

        data = df['Close'].values.reshape(-1, 1)

        # Scale the data
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(data)

        X, y = [], []
        for i in range(look_back, len(scaled_data)):
            X.append(scaled_data[i - look_back:i, 0])
            y.append(scaled_data[i, 0])

        return np.array(X), np.array(y), scaler

    except Exception as e:
        raise ValueError(f"Error loading data: {str(e)}")

# Build LSTM Model (Only if not already trained)
def build_model(input_shape):
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=input_shape),
        LSTM(50, return_sequences=False),
        Dense(25),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

# Load the model and weights if available
model = build_model((60, 1))

if os.path.exists(MODEL_WEIGHTS_PATH):
    model.load_weights(MODEL_WEIGHTS_PATH)
else:
    print("Warning: No pre-trained model weights found!")

@app.route("/", methods=["GET"])
def home():
    return jsonify({"message": "Stock Prediction API is running"}), 200

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        stock_symbol = data['symbol']
        
        # Load data
        look_back = 60
        X, y, scaler = load_data(stock_symbol, look_back)
        X = X.reshape(X.shape[0], X.shape[1], 1)
        
        # Train model
        model = build_model((X.shape[1], 1))
        model.fit(X, y, epochs=10, batch_size=16, verbose=0)
        
        # Predict next value
        last_sequence = X[-1].reshape(1, look_back, 1)
        predicted_price = model.predict(last_sequence)
        predicted_price = float(scaler.inverse_transform(predicted_price.reshape(-1, 1))[0, 0])  # Convert to float
        
        return jsonify({"symbol": stock_symbol, "predicted_price": predicted_price})
    
    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 10000))  # Ensure it binds to the correct port
    app.run(host='0.0.0.0', port=port)