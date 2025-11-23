#!/usr/bin/env python3
"""
NVDL Stock Price Prediction - Training Script

This script fetches data, trains LSTM and ARIMA models, evaluates them,
and saves the trained models and scaler for later use in inference.
"""
import argparse
import os
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import joblib

# PyTorch for LSTM
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split

# Statsmodels for ARIMA
from statsmodels.tsa.arima.model import ARIMA
from pmdarima.arima import auto_arima


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
    """PyTorch Bi-LSTM Model with Attention for time series prediction."""
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim, dropout_prob, step_dim):
        super(LSTMAttentionModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout_prob, bidirectional=True)
        self.attention = Attention(hidden_dim * 2, step_dim)
        self.fc = nn.Linear(hidden_dim * 2, output_dim)

    def forward(self, x):
        # Initialize hidden and cell states
        device = x.device
        h0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_dim, device=device).requires_grad_()
        c0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_dim, device=device).requires_grad_()
        
        # We need to detach as we are not training the hidden state
        out, (hn, cn) = self.lstm(x, (h0.detach(), c0.detach()))
        
        attn_out = self.attention(out)
        out = self.fc(attn_out)
        return out

class ModelTrainer:
    """
    A class to fetch data, build, train, evaluate, and save stock prediction models.
    """
    def __init__(self, ticker, start_date, end_date, test_size, lstm_time_step, arima_order):
        self.ticker = ticker
        self.start_date = start_date
        self.end_date = end_date
        self.test_size = test_size
        self.lstm_time_step = lstm_time_step
        self.arima_order = arima_order
        self.feature_scaler = MinMaxScaler(feature_range=(0, 1))
        self.df_features = None
        self.train_df = None
        self.test_df = None
        self.results = {}
        self.model_dir = f"models/{self.ticker}"
        os.makedirs(self.model_dir, exist_ok=True)

    def fetch_data(self):
        print(f"Fetching historical data for {self.ticker}...")
        stock_data = yf.download(self.ticker, start=self.start_date, end=self.end_date, interval='1d')
        market_data = yf.download('^GSPC', start=self.start_date, end=self.end_date, interval='1d')

        if stock_data.empty:
            raise ValueError("No data returned from yfinance.")
        
        # Combine stock data with market data
        combined_data = stock_data.copy()
        combined_data['SP500_Close'] = market_data['Close']

        return combined_data

    def _add_technical_indicators(self, data):
        # RSI
        delta = data['Close'].diff(1)
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        avg_gain = gain.ewm(com=13, min_periods=14).mean()
        avg_loss = loss.ewm(com=13, min_periods=14).mean()
        data['RSI'] = 100 - (100 / (1 + (avg_gain / avg_loss)))

        # MACD
        exp1 = data['Close'].ewm(span=12, adjust=False).mean()
        exp2 = data['Close'].ewm(span=26, adjust=False).mean()
        data['MACD'] = exp1 - exp2

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

    def prepare_data(self):
        df_full = self.fetch_data()
        df_full = self._add_technical_indicators(df_full)
        self.feature_list = ['Close', 'Volume', 'RSI', 'MACD', 'Upper_Band', 'Lower_Band', '%K', 'ATR', 'SP500_Close']
        self.df_features = df_full[self.feature_list].copy()
        self.df_features.dropna(inplace=True)
        split_index = int(len(self.df_features) * (1 - self.test_size))
        self.train_df = self.df_features.iloc[:split_index]
        self.test_df = self.df_features.iloc[split_index:]
        print(f"\nTotal samples: {len(self.df_features)}, Training: {len(self.train_df)}, Testing: {len(self.test_df)}")

    def _create_lstm_dataset(self, data, target_col_index):
        X, y = [], []
        for i in range(len(data) - self.lstm_time_step):
            X.append(data[i:(i + self.lstm_time_step), :])
            y.append(data[i + self.lstm_time_step, target_col_index])
        return np.array(X), np.array(y)

    def train_and_save_lstm(self):
        print("\n--- Training and Saving PyTorch LSTM Model ---")
        train_scaled = self.feature_scaler.fit_transform(self.train_df)
        test_scaled = self.feature_scaler.transform(self.test_df)
        target_col_index = self.train_df.columns.get_loc('Close')

        X_train, y_train = self._create_lstm_dataset(train_scaled, target_col_index)
        X_test, y_test = self._create_lstm_dataset(test_scaled, target_col_index)
        
        # Convert to PyTorch tensors
        X_train_tensor = torch.from_numpy(X_train).float()
        y_train_tensor = torch.from_numpy(y_train).float().view(-1, 1)
        X_test_tensor = torch.from_numpy(X_test).float()
        
        # Split training data for validation (80% train, 20% validation)
        X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
            X_train_tensor, y_train_tensor, test_size=0.2, random_state=42
        )

        train_data = TensorDataset(X_train_split, y_train_split)
        val_data = TensorDataset(X_val_split, y_val_split)

        train_loader = DataLoader(train_data, shuffle=True, batch_size=64)
        val_loader = DataLoader(val_data, shuffle=False, batch_size=64)

        # Model parameters
        num_features = self.df_features.shape[1]
        lstm_model = LSTMAttentionModel(input_dim=num_features, hidden_dim=128, num_layers=2, output_dim=1, dropout_prob=0.3, step_dim=self.lstm_time_step)
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(lstm_model.parameters(), lr=0.001)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5, verbose=True)

        # Early stopping parameters
        epochs = 100
        patience = 10
        best_val_loss = float('inf')
        epochs_no_improve = 0
        best_model_state = lstm_model.state_dict() # Initialize with the starting model state

        # Training loop with validation and early stopping
        for epoch in range(epochs):
            lstm_model.train()
            for X_batch, y_batch in train_loader:
                optimizer.zero_grad()
                y_pred = lstm_model(X_batch)
                train_loss = criterion(y_pred, y_batch)
                train_loss.backward()
                optimizer.step()
            
            # Validation loop
            val_loss = 0
            lstm_model.eval()
            with torch.no_grad():
                for X_val, y_val in val_loader:
                    y_val_pred = lstm_model(X_val)
                    val_loss += criterion(y_val_pred, y_val).item()
            
            val_loss /= len(val_loader)

            # Step the scheduler
            scheduler.step(val_loss)

            if (epoch + 1) % 10 == 0:
                print(f'Epoch [{epoch+1}/{epochs}], Train Loss: {train_loss.item():.4f}, Val Loss: {val_loss:.4f}')

            # Early stopping check
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                epochs_no_improve = 0
                best_model_state = lstm_model.state_dict()
            else:
                epochs_no_improve += 1
            
            if epochs_no_improve == patience:
                print(f'Early stopping triggered after {epoch+1} epochs.')
                lstm_model.load_state_dict(best_model_state)
                break

        # If the loop completes without early stopping, ensure the best model state is loaded
        if epochs_no_improve < patience:
            lstm_model.load_state_dict(best_model_state)

        # Save model and scaler
        torch.save(best_model_state, f"{self.model_dir}/lstm_model.pth")
        joblib.dump(self.feature_scaler, f"{self.model_dir}/scaler.pkl")
        print(f"LSTM model and scaler saved to {self.model_dir}/")

        # Evaluate
        lstm_model.eval()
        with torch.no_grad():
            predictions_scaled_tensor = lstm_model(X_test_tensor)
        
        predictions_scaled = predictions_scaled_tensor.numpy()

        dummy_array = np.zeros((len(predictions_scaled), num_features))
        dummy_array[:, target_col_index] = predictions_scaled.flatten()
        predictions = self.feature_scaler.inverse_transform(dummy_array)[:, target_col_index]
        actual = self.test_df['Close'].values[self.lstm_time_step:]
        self.results['LSTM'] = {'actual': actual, 'predicted': predictions}

    def train_and_save_arima(self):
        print("\n--- Training and Saving ARIMA Model ---")
        train_close = self.train_df['Close']
        
        # Use auto_arima to find the best order
        print("Finding best ARIMA order with auto_arima...")
        auto_arima_model = auto_arima(train_close, start_p=1, start_q=1, max_p=5, max_q=5, m=1,
                                      d=1, seasonal=False, trace=True, error_action='ignore',
                                      suppress_warnings=True, stepwise=True)
        print(f"Best ARIMA order: {auto_arima_model.order}")
        arima_model = ARIMA(train_close, order=auto_arima_model.order).fit()
        joblib.dump(arima_model, f"{self.model_dir}/arima_model.pkl")
        print(f"ARIMA model saved to {self.model_dir}/")

        # Evaluate with rolling forecast (for metrics only)
        history = [x for x in train_close.values]
        test_close = self.test_df['Close'].values
        best_order = auto_arima_model.order
        predictions = []
        for t in range(len(test_close)):
            model = ARIMA(history, order=best_order)
            model_fit = model.fit()
            yhat = model_fit.forecast(steps=1)[0]
            predictions.append(yhat)
            history.append(test_close[t])
        
        self.results['ARIMA'] = {'actual': test_close, 'predicted': np.array(predictions)}

    def evaluate_models(self):
        print("\n--- Final Model Performance Summary (Test Set) ---")
        self.metrics = {}
        for model_name, result in self.results.items():
            df = pd.DataFrame(result).dropna()
            rmse = np.sqrt(mean_squared_error(df['actual'], df['predicted']))
            mae = mean_absolute_error(df['actual'], df['predicted'])
            mape = np.mean(np.abs((df['actual'] - df['predicted']) / df['actual'])) * 100
            self.metrics[model_name] = {'RMSE': rmse, 'MAE': mae, 'MAPE': mape}
            print(f"{model_name} - RMSE: ${rmse:.2f}, MAE: ${mae:.2f}, MAPE: {mape:.2f}%")

    def plot_results(self):
        print("\nGenerating evaluation plots...")
        # Price Prediction Plot
        plt.figure(figsize=(16, 8))
        test_index = self.test_df.index
        plt.plot(test_index, self.test_df['Close'], label='Actual Price', color='black')
        if 'ARIMA' in self.results:
            plt.plot(test_index, self.results['ARIMA']['predicted'], label='ARIMA Prediction', color='green', linestyle='--')
        if 'LSTM' in self.results:
            lstm_index = test_index[self.lstm_time_step:]
            plt.plot(lstm_index, self.results['LSTM']['predicted'], label='LSTM Prediction', color='red', linestyle=':')
        plt.title(f'{self.ticker} Price Prediction Comparison on Test Set')
        plt.legend()
        plt.grid(True)
        plt.savefig('docs/model_comparison_plot.png', dpi=300)
        plt.close()
        print("Comparison plot saved to docs/model_comparison_plot.png")

        # Diagnostic Plots
        fig, axes = plt.subplots(len(self.results), 2, figsize=(16, 6 * len(self.results)))
        fig.suptitle(f'{self.ticker} Model Diagnostic Plots', fontsize=16, y=1.02)

        for i, (model_name, result) in enumerate(self.results.items()):
            df = pd.DataFrame(result).dropna()
            errors = df['actual'] - df['predicted']

            # Scatter plot of Actual vs. Predicted
            ax_scatter = axes[i, 0]
            sns.scatterplot(x=df['actual'], y=df['predicted'], ax=ax_scatter, alpha=0.6)
            ax_scatter.plot([df['actual'].min(), df['actual'].max()], [df['actual'].min(), df['actual'].max()], 'r--', lw=2)
            ax_scatter.set_xlabel('Actual Prices ($)')
            ax_scatter.set_ylabel('Predicted Prices ($)')
            ax_scatter.set_title(f'{model_name}: Actual vs. Predicted')
            ax_scatter.grid(True)

            # Histogram of Errors
            ax_hist = axes[i, 1]
            sns.histplot(errors, kde=True, ax=ax_hist)
            ax_hist.axvline(0, color='r', linestyle='--')
            ax_hist.set_xlabel('Prediction Error ($)')
            ax_hist.set_title(f'{model_name}: Distribution of Errors')
            ax_hist.grid(True)

        plt.tight_layout()
        plt.savefig('docs/diagnostic_plots.png', dpi=300)
        plt.close()
        print("Diagnostic plots saved to docs/diagnostic_plots.png")

        # Accuracy Score Bar Plots
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        fig.suptitle(f'{self.ticker} Model Accuracy Score Comparison', fontsize=16, y=1.02)
        models = list(self.metrics.keys())
        colors = ['#FF6B6B', '#4ECDC4']

        metric_details = {
            'RMSE': ('RMSE ($)', '${:.2f}'),
            'MAE': ('MAE ($)', '${:.2f}'),
            'MAPE': ('MAPE (%)', '{:.2f}%')
        }

        for i, (metric_name, (ylabel, fmt)) in enumerate(metric_details.items()):
            ax = axes[i]
            values = [self.metrics[model][metric_name] for model in models]
            bars = ax.bar(models, values, color=colors[:len(models)], alpha=0.8)
            ax.set_ylabel(ylabel)
            ax.set_title(metric_name)
            ax.grid(axis='y', linestyle='--', alpha=0.7)
            for bar, val in zip(bars, values):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(), fmt.format(val), ha='center', va='bottom', fontweight='bold')
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig('docs/accuracy_comparison_plot.png', dpi=300)
        plt.close()
        print("Accuracy comparison plot saved to docs/accuracy_comparison_plot.png")

    def run(self):
        self.prepare_data()
        self.train_and_save_lstm()
        self.train_and_save_arima()
        self.evaluate_models()
        self.plot_results()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train stock prediction models.")
    parser.add_argument("--ticker", type=str, default="NVDA", help="Stock ticker symbol.")
    parser.add_argument("--start_date", type=str, default="2020-01-01", help="Start date for training data.")
    parser.add_argument("--test_size", type=float, default=0.2, help="Proportion of data for testing.")
    parser.add_argument("--lstm_steps", type=int, default=60, help="Time steps for LSTM.")
    
    args = parser.parse_args()

    config = {
        "ticker": args.ticker,
        "start_date": args.start_date,
        "end_date": pd.Timestamp.today().strftime('%Y-%m-%d'),
        "test_size": args.test_size,
        "lstm_time_step": args.lstm_steps,
        "arima_order": (5, 1, 0)
    }

    trainer = ModelTrainer(**config)
    trainer.run()
