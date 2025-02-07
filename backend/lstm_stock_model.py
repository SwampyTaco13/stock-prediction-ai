import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, GRU
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import yfinance as yf
import time
import logging
from flask import Flask, jsonify, request
from flask_cors import CORS

logging.basicConfig(filename='stock_predictions.log', level=logging.INFO, format='%(asctime)s - %(message)s')

app = Flask(__name__)
CORS(app)

def fetch_stock_data(ticker='AAPL', start_date='2020-01-01', end_date='2025-01-01'):
    data = yf.download(ticker, start=start_date, end=end_date)
    data = data[['Close', 'Volume']]
    data.dropna(inplace=True)
    return data

def fetch_live_stock_data(ticker):
    data = yf.download(ticker, period='1d', interval='1m')
    return data[['Close', 'Volume']].iloc[-1]

tickers = ['AAPL', 'GOOGL', 'MSFT', 'TSLA']

historical_data = {ticker: fetch_stock_data(ticker) for ticker in tickers}

scalers = {}
for ticker, data in historical_data.items():
    scaler = MinMaxScaler(feature_range=(0, 1))
    data[['Scaled_Close', 'Scaled_Volume']] = scaler.fit_transform(data[['Close', 'Volume']])
    scalers[ticker] = scaler

def create_sequences(data, seq_length=50):
    X, y = [], []
    for i in range(seq_length, len(data)):
        X.append(data[i-seq_length:i])
        y.append(data[i, 0])
    return np.array(X), np.array(y)

seq_length = 50
training_data = {}
for ticker, data in historical_data.items():
    values = data[['Scaled_Close', 'Scaled_Volume']].values
    X, y = create_sequences(values, seq_length)
    training_data[ticker] = (X, y, values)

model = Sequential([
    GRU(units=64, return_sequences=True, input_shape=(seq_length, 2)),
    Dropout(0.2),
    GRU(units=64, return_sequences=False),
    Dropout(0.2),
    Dense(units=32),
    Dense(units=1)
])

model.compile(optimizer='adam', loss='mean_squared_error')

X, y, _ = training_data['AAPL']
model.fit(X, y, epochs=10, batch_size=32, verbose=1, validation_split=0.2)

def predict_next_point(model, latest_data, scaler):
    data_reshaped = np.reshape(latest_data, (1, len(latest_data), latest_data.shape[1]))
    scaled_prediction = model.predict(data_reshaped)
    return scaler.inverse_transform([[scaled_prediction[0, 0], 0]])[0, 0]

def generate_recommendation(predicted_price, current_price):
    threshold = 0.02
    change_percentage = (predicted_price - current_price) / current_price
    if change_percentage > threshold:
        return "Buy"
    elif change_percentage < -threshold:
        return "Sell"
    else:
        return "Hold"

@app.route('/predict', methods=['GET'])
def get_prediction():
    ticker = request.args.get('ticker', default='AAPL', type=str)
    if ticker not in tickers:
        return jsonify({"error": "Ticker not supported"}), 400
    
    live_data = fetch_live_stock_data(ticker)
    scaler = scalers[ticker]
    values = training_data[ticker][2]
    scaled_live_data = scaler.transform([[live_data['Close'], live_data['Volume']]])
    latest_data = np.vstack([values[-(seq_length - 1):], scaled_live_data])
    prediction = predict_next_point(model, latest_data, scaler)
    recommendation = generate_recommendation(prediction, live_data['Close'])
    
    response = {
        "ticker": ticker,
        "predicted_price": round(prediction, 2),
        "current_price": round(live_data['Close'], 2),
        "recommendation": recommendation
    }
    
    logging.info(f"API Prediction - {response}")
    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True, port=5000)