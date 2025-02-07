import os
import numpy as np
import pandas as pd
import yfinance as yf
import tensorflow as tf
from flask import Flask, request, jsonify
from flask_cors import CORS
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

app = Flask(__name__)
CORS(app)

# Load stock data
def load_stock_data(ticker):
    data = yf.download(ticker, start="2020-01-01", end="2025-01-01")
    return data["Close"].values.reshape(-1, 1)

# Preprocess data
def preprocess_data(data):
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)
    return scaled_data, scaler

# Create sequences
def create_sequences(data, seq_length):
    sequences, labels = [], []
    for i in range(len(data) - seq_length - 1):
        sequences.append(data[i:i+seq_length])
        labels.append(data[i+seq_length])
    return np.array(sequences), np.array(labels)

# Build LSTM model
def build_model(input_shape):
    model = tf.keras.Sequential([
        tf.keras.layers.LSTM(50, return_sequences=True, input_shape=input_shape),
        tf.keras.layers.LSTM(50),
        tf.keras.layers.Dense(25),
        tf.keras.layers.Dense(1)
    ])
    model.compile(optimizer="adam", loss="mean_squared_error")
    return model

# Train the model
def train_model(ticker):
    data = load_stock_data(ticker)
    scaled_data, scaler = preprocess_data(data)
    
    seq_length = 60
    X, y = create_sequences(scaled_data, seq_length)
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))

    model = build_model((seq_length, 1))
    model.fit(X, y, epochs=10, batch_size=32, verbose=1)

    return model, scaler, data

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.json
        ticker = data.get("ticker", "AAPL")

        model, scaler, stock_data = train_model(ticker)

        last_60_days = stock_data[-60:].reshape(-1, 1)
        last_60_days_scaled = scaler.transform(last_60_days)
        X_test = np.array([last_60_days_scaled])
        X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

        predicted_price = model.predict(X_test)
        predicted_price = scaler.inverse_transform(predicted_price)

        return jsonify({"ticker": ticker, "predicted_price": predicted_price.tolist()})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))  # Default to port 5000 for Render
    app.run(host="0.0.0.0", port=port, debug=True)