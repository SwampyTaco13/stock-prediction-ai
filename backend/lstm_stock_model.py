import numpy as np
import pandas as pd
import yfinance as yf
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from flask import Flask, request, jsonify
from flask_cors import CORS

# Flask app setup
app = Flask(__name__)
CORS(app)

# Load and preprocess data
def load_data(stock_symbol, look_back=60):
    df = yf.download(stock_symbol, period="2y", interval="1d")
    data = df['Close'].values.reshape(-1, 1)
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)
    
    X, y = [], []
    for i in range(look_back, len(scaled_data)):
        X.append(scaled_data[i-look_back:i, 0])
        y.append(scaled_data[i, 0])
    
    return np.array(X), np.array(y), scaler

# Build LSTM model
def build_model(input_shape):
    model = Sequential([
        LSTM(units=50, return_sequences=True, input_shape=input_shape),
        Dropout(0.2),
        LSTM(units=50, return_sequences=False),
        Dropout(0.2),
        Dense(units=1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

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
        predicted_price = scaler.inverse_transform(predicted_price.reshape(-1, 1))[0, 0]
        
        return jsonify({"symbol": stock_symbol, "predicted_price": predicted_price})
    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=10000, debug=True)
