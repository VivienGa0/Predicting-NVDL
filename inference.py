#!/usr/bin/env python3
"""
NVDL Stock Price Prediction - Inference Script

This script loads a pre-trained model and scaler to predict the next
day's stock price for a given ticker.
"""
import argparse
import pandas as pd
import numpy as np
import yfinance as yf
import joblib
import torch
import torch.nn as nn
from statsmodels.tsa.arima.model import ARIMA


class Attention(nn.Module):
    """Attention mechanism layer."""
    def __init__(self, feature_dim, step_dim, bias=True, **kwargs):
        super(Attention, self).__init__(**kwargs)
        self.bias = bias
        self.feature_dim = feature_dim
        self.step_dim = step_dim
        
        weight = torch.zeros(feature_dim, 1)
        nn.init.xavier_uniform_(weight)
        self.weight = nn.Parameter(weight)
        if bias:
            self.b = nn.Parameter(torch.zeros(step_dim))
    
    def forward(self, x):
        eij = torch.mm(
            x.contiguous().view(-1, self.feature_dim), 
            self.weight
        ).view(-1, self.step_dim)
        
        if self.bias:
            eij = eij + self.b
            
        eij = torch.tanh(eij)
        a = torch.exp(eij)
        a = a / (torch.sum(a, 1, keepdim=True) + 1e-10)

        weighted_input = x * torch.unsqueeze(a, -1)
        return torch.sum(weighted_input, 1)


class LSTMAttentionModel(nn.Module):
    """PyTorch Bi-LSTM Model with Attention - must match the training script."""
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim, dropout_prob, step_dim):
        super(LSTMAttentionModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout_prob, bidirectional=True)
        self.attention = Attention(hidden_dim * 2, step_dim)
        self.fc = nn.Linear(hidden_dim * 2, output_dim)

    def forward(self, x):
        device = x.device
        h0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_dim, device=device).requires_grad_()
        c0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_dim, device=device).requires_grad_()
        
        out, (hn, cn) = self.lstm(x, (h0.detach(), c0.detach()))
        
        attn_out = self.attention(out)
        out = self.fc(attn_out)
        return out

def add_technical_indicators(data):
    """Helper function to compute technical indicators - must match train.py."""
    # RSI
    delta = data['Close'].diff(1)
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.ewm(com=13, min_periods=14).mean()
    avg_loss = loss.ewm(com=13, min_periods=14).mean()
    data['RSI'] = 100 - (100 / (1 + (avg_gain / avg_loss)))

    # MACD
    data['MACD'] = data['Close'].ewm(span=12, adjust=False).mean() - data['Close'].ewm(span=26, adjust=False).mean()

    # Bollinger Bands
    data['SMA_20'] = data['Close'].rolling(window=20).mean()
    data['Upper_Band'] = data['SMA_20'] + (data['Close'].rolling(window=20).std() * 2)
    data['Lower_Band'] = data['SMA_20'] - (data['Close'].rolling(window=20).std() * 2)

    # Stochastic Oscillator (%K)
    low_14 = data['Low'].rolling(window=14).min()
    high_14 = data['High'].rolling(window=14).max()
    data['%K'] = 100 * ((data['Close'] - low_14) / (high_14 - low_14))

    # Average True Range (ATR)
    high_low = data['High'] - data['Low']
    high_close = np.abs(data['High'] - data['Close'].shift())
    low_close = np.abs(data['Low'] - data['Close'].shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = ranges.max(axis=1)
    data['ATR'] = true_range.rolling(14).mean()
    return data

def predict_next_day(ticker, lstm_time_step):
    """
    Loads models and predicts the next day's closing price.
    """
    print(f"--- Running Inference for {ticker} ---")
    model_dir = f"models/{ticker}"

    # --- Load Models and Scaler ---
    try:
        num_features = 9 # Must match the number of features in train.py
        lstm_model = LSTMAttentionModel(input_dim=num_features, hidden_dim=128, num_layers=2, output_dim=1, dropout_prob=0.3, step_dim=lstm_time_step)
        lstm_model.load_state_dict(torch.load(f"{model_dir}/lstm_model.pth"))
        lstm_model.eval()

        arima_model = joblib.load(f"{model_dir}/arima_model.pkl")
        scaler = joblib.load(f"{model_dir}/scaler.pkl")
        print("Models and scaler loaded successfully.")
    except Exception as e:
        print(f"Error loading model files: {e}")
        print(f"Please run 'python train.py --ticker {ticker}' first to generate model files.")
        return

    # --- Fetch Latest Data ---
    # Fetch enough data for LSTM and all data for ARIMA refitting
    start_date = (pd.Timestamp.today() - pd.DateOffset(days=lstm_time_step + 50)).strftime('%Y-%m-%d')
    stock_data = yf.download(ticker, start=start_date, end=pd.Timestamp.today(), interval='1d')
    market_data = yf.download('^GSPC', start=start_date, end=pd.Timestamp.today(), interval='1d')
    
    data = stock_data.copy()
    data['SP500_Close'] = market_data['Close']

    all_data = yf.download(ticker, start="2020-01-01", end=pd.Timestamp.today(), interval='1d')
    last_actual_price = data['Close'].iloc[-1]
    print(f"Last available closing price ({data.index[-1].date()}): ${last_actual_price:.2f}")

    # --- Prepare Data for Prediction ---
    data = add_technical_indicators(data)
    features = ['Close', 'Volume', 'RSI', 'MACD', 'Upper_Band', 'Lower_Band', '%K', 'ATR', 'SP500_Close']
    data_features = data[features].dropna()

    # --- LSTM Prediction ---
    last_sequence = data_features.tail(lstm_time_step).values
    last_sequence_scaled = scaler.transform(last_sequence)
    input_tensor = torch.from_numpy(last_sequence_scaled).float().unsqueeze(0)
    with torch.no_grad():
        predicted_price_scaled = lstm_model(input_tensor).item()
    dummy_array = np.zeros((1, num_features)); dummy_array[0, 0] = predicted_price_scaled
    lstm_prediction = scaler.inverse_transform(dummy_array)[0, 0]

    # --- ARIMA Prediction ---
    print("Refitting ARIMA model on all available data...")
    best_order = arima_model.order # Get the best order from the saved model
    arima_model_refit = ARIMA(all_data['Close'], order=best_order).fit()
    arima_prediction = arima_model_refit.forecast(steps=1).iloc[0]
    print("ARIMA refit and forecast complete.")

    # --- Display Results ---
    print("\n--- Predictions for Next Trading Day ---")
    print(f"LSTM Predicted Close: ${lstm_prediction:.2f}")
    print(f"ARIMA Predicted Close: ${arima_prediction:.2f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict next day stock price.")
    parser.add_argument("--ticker", type=str, default="NVDA", help="Stock ticker symbol.")
    parser.add_argument("--lstm_steps", type=int, default=60, help="Time steps for LSTM (must match training).")
    
    args = parser.parse_args()
    predict_next_day(args.ticker, args.lstm_steps)