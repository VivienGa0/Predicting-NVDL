#!/usr/bin/env python3
"""
NVDL Stock Price Prediction Model
Using LSTM and ARIMA for price prediction
"""

import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error

# TensorFlow/Keras for LSTM
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

# Statsmodels for ARIMA
from statsmodels.tsa.arima.model import ARIMA

# --- Configuration ---
TICKER = "NVDA"
START_DATE = "2020-01-01"
END_DATE = pd.Timestamp.today().strftime('%Y-%m-%d')
TEST_SIZE = 0.2 # Use the last 20% of data for testing
LSTM_TIME_STEP = 60 # Number of past days used to predict the next day
ARIMA_ORDER = (5, 1, 0) # (p, d, q) for ARIMA - simple baseline
RSI_PERIOD = 14 # Period for RSI calculation

# --- Utility Functions ---

def fetch_data(ticker, start, end):
    """
    Fetches historical data using yfinance.
    """
    print(f"Fetching historical data for {ticker}...")
    try:
        data = yf.download(ticker, start=start, end=end, interval='1d')
        if data.empty:
            raise ValueError("No data returned from yfinance.")
        return data.copy()
    except Exception as e:
        print(f"Error fetching data: {e}")
        return pd.DataFrame()

def calculate_rsi(data, period=14):
    """
    Calculates the Relative Strength Index (RSI) manually using exponential moving average.
    """
    delta = data.diff(1)
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)

    avg_gain = gain.ewm(com=period - 1, min_periods=period).mean()
    avg_loss = loss.ewm(com=period - 1, min_periods=period).mean()

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def create_lstm_dataset(data, time_step, target_col_index):
    """Converts a series into a dataset matrix for LSTM training."""
    X, y = [], []
    for i in range(len(data) - time_step):
        X.append(data[i:(i + time_step), :])
        y.append(data[i + time_step, target_col_index])
    return np.array(X), np.array(y)

# --- 1. Data Preparation ---

df_full = fetch_data(TICKER, START_DATE, END_DATE)
if df_full.empty:
    print("Exiting due to data error.")
    exit()

# Feature Engineering
df_full['RSI'] = calculate_rsi(df_full['Close'], period=RSI_PERIOD)
df_features = df_full[['Close', 'Volume', 'RSI']].copy()
df_features.dropna(inplace=True) # Drop rows with NaN values (from RSI calculation)

# The ARIMA model will still only use the 'Close' price
df_close = df_features['Close'].copy()

# Split the data into training and testing sets based on time
split_index = int(len(df_features) * (1 - TEST_SIZE))
train_df = df_features.iloc[:split_index]
test_df = df_features.iloc[split_index:]

# ARIMA data split
arima_split_index = int(len(df_close) * (1 - TEST_SIZE))
train_df_arima = df_close.iloc[:arima_split_index]
test_df_arima = df_close.iloc[arima_split_index:]

print(f"\nTotal samples: {len(df_features)}")
print(f"Training samples: {len(train_df)}")
print(f"Testing samples: {len(test_df)}")

# --- 2. LSTM Model Implementation (Long-Term, Multi-feature) ---

print("\n--- Starting LSTM Model Training ---")

# Data scaling for LSTM
scaler = MinMaxScaler(feature_range=(0, 1))
train_scaled = scaler.fit_transform(train_df)
test_scaled = scaler.transform(test_df)

# Find the index of the 'Close' column to use as the target
target_col_index = 0 # 'Close' is the first column

# Create sequence datasets
X_train_lstm, y_train_lstm = create_lstm_dataset(train_scaled, LSTM_TIME_STEP, target_col_index)
X_test_lstm, y_test_lstm = create_lstm_dataset(test_scaled, LSTM_TIME_STEP, target_col_index)

# Split training data for validation
from sklearn.model_selection import train_test_split
X_train_lstm, X_val_lstm, y_train_lstm, y_val_lstm = train_test_split(X_train_lstm, y_train_lstm, test_size=0.2, random_state=42)

# Reshape input to be [samples, time_steps, features]
num_features = train_df.shape[1]
X_train_lstm = X_train_lstm.reshape(X_train_lstm.shape[0], X_train_lstm.shape[1], num_features)
X_val_lstm = X_val_lstm.reshape(X_val_lstm.shape[0], X_val_lstm.shape[1], num_features)
X_test_lstm = X_test_lstm.reshape(X_test_lstm.shape[0], X_test_lstm.shape[1], num_features)

# Build and compile LSTM Model
lstm_model = Sequential()
lstm_model.add(LSTM(100, return_sequences=True, input_shape=(LSTM_TIME_STEP, num_features)))
lstm_model.add(Dropout(0.2))
lstm_model.add(LSTM(100, return_sequences=False))
lstm_model.add(Dropout(0.2))
lstm_model.add(Dense(1))

lstm_model.compile(optimizer='adam', loss='mean_squared_error')

# Add Early Stopping
from tensorflow.keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# Train LSTM
lstm_model.fit(
    X_train_lstm, y_train_lstm,
    epochs=200,
    batch_size=64,
    validation_data=(X_val_lstm, y_val_lstm),
    callbacks=[early_stopping],
    verbose=0
)

# Predict and Inverse Transform
lstm_test_predict_scaled = lstm_model.predict(X_test_lstm)

# To inverse transform, we create a dummy array with the same number of features
# and place the scaled predictions in the correct column.
dummy_array = np.zeros((len(lstm_test_predict_scaled), num_features))
dummy_array[:, target_col_index:target_col_index+1] = lstm_test_predict_scaled
lstm_test_predict = scaler.inverse_transform(dummy_array)[:, target_col_index]

# Inverse transform the actual values
dummy_array_actual = np.zeros((len(y_test_lstm), num_features))
dummy_array_actual[:, target_col_index:target_col_index+1] = y_test_lstm.reshape(-1, 1)
y_test_lstm_actual = scaler.inverse_transform(dummy_array_actual)[:, target_col_index]

# Calculate RMSE for LSTM
lstm_rmse = np.sqrt(mean_squared_error(y_test_lstm_actual, lstm_test_predict))
print(f"LSTM Test RMSE: ${lstm_rmse:.2f}")

# Prepare LSTM prediction for plotting
lstm_test_results = pd.DataFrame({
    'Actual': y_test_lstm_actual.flatten(),
    'LSTM_Predicted': lstm_test_predict.flatten()
}, index=test_df.index[LSTM_TIME_STEP:])

# --- 3. ARIMA Model Implementation (Short-Term) ---

print("\n--- Starting ARIMA Model Training ---")

history = [x for x in train_df_arima.values]
arima_predictions = []

for t in range(len(test_df_arima)):
    try:
        arima_model = ARIMA(history, order=ARIMA_ORDER)
        model_fit = arima_model.fit()
        yhat = model_fit.forecast(steps=1)[0]
        arima_predictions.append(yhat)
        obs = test_df_arima.iloc[t]
        history.append(obs)
    except Exception as e:
        arima_predictions.append(np.nan)
        history.append(test_df_arima.iloc[t])
        print(f"ARIMA fit error at index {t}: {e}")

print("ARIMA rolling forecast complete.")

arima_predictions_series = pd.Series(arima_predictions, index=test_df_arima.index).dropna()

# Ensure data is 1-dimensional for DataFrame creation
actual_values = test_df_arima.loc[arima_predictions_series.index].values.flatten()
predicted_values = arima_predictions_series.values.flatten()

comparison_df = pd.DataFrame({
    'Actual': actual_values,
    'ARIMA_Predicted': predicted_values
}).dropna()

arima_rmse = np.sqrt(mean_squared_error(comparison_df['Actual'], comparison_df['ARIMA_Predicted']))
print(f"ARIMA Test RMSE: ${arima_rmse:.2f}")

# --- 4. Comparative Visualization ---

final_results = pd.DataFrame({
    'Actual Price': test_df_arima.values.flatten(),
    'ARIMA Prediction': arima_predictions_series.values.flatten()
}, index=test_df_arima.index)

final_results = final_results.merge(
    lstm_test_results.drop(columns=['Actual']),
    left_index=True,
    right_index=True,
    how='left'
)

plt.figure(figsize=(16, 8))
plt.plot(final_results.index, final_results['Actual Price'], label='Actual Price', color='black', linewidth=2)
plt.plot(final_results.index, final_results['ARIMA Prediction'], label='ARIMA Prediction', color='green', linestyle='--')
plt.plot(final_results.index, final_results['LSTM_Predicted'], label='LSTM Prediction (Multi-feature)', color='red', linestyle=':')

plt.title(f'{TICKER} Price Prediction Comparison (LSTM vs. ARIMA) on Test Set')
plt.xlabel('Date')
plt.ylabel('Closing Price ($)')
plt.legend()
plt.grid(True, alpha=0.5)
plt.xticks(rotation=45)
plt.tight_layout()

plot_path = 'docs/model_comparison_plot.png'
plt.savefig(plot_path, dpi=300, bbox_inches='tight')
print(f"\nComparison chart saved to: {plot_path}")
plt.close()

# --- 5. Final Summary ---
print("\n--- Final Model Performance Summary (Test Set) ---")
print(f"LSTM Test RMSE: ${lstm_rmse:.2f}")
print(f"ARIMA Test RMSE: ${arima_rmse:.2f}")

# --- 6. Accuracy Metrics Calculation ---
lstm_mae = mean_absolute_error(y_test_lstm_actual, lstm_test_predict)
lstm_mape = np.mean(np.abs((y_test_lstm_actual - lstm_test_predict) / y_test_lstm_actual)) * 100

arima_mae = mean_absolute_error(comparison_df['Actual'], comparison_df['ARIMA_Predicted'])
arima_mape = np.mean(np.abs((comparison_df['Actual'] - comparison_df['ARIMA_Predicted']) / comparison_df['Actual'])) * 100

print(f"\n--- Detailed Accuracy Metrics ---")
print(f"LSTM - RMSE: ${lstm_rmse:.2f}, MAE: ${lstm_mae:.2f}, MAPE: {lstm_mape:.2f}%")
print(f"ARIMA - RMSE: ${arima_rmse:.2f}, MAE: ${arima_mae:.2f}, MAPE: {arima_mape:.2f}%")

# --- 7. Accuracy Comparison Visualization ---
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
models = ['LSTM', 'ARIMA']
colors = ['#FF6B6B', '#4ECDC4']

rmse_values = [lstm_rmse, arima_rmse]
bars1 = axes[0].bar(models, rmse_values, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
axes[0].set_ylabel('RMSE ($)', fontsize=12, fontweight='bold')
axes[0].set_title('Root Mean Squared Error (RMSE)', fontsize=14, fontweight='bold')
axes[0].grid(axis='y', alpha=0.3, linestyle='--')
axes[0].set_ylim(0, max(rmse_values) * 1.2)
for i, (bar, val) in enumerate(zip(bars1, rmse_values)):
    axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(rmse_values)*0.05, f'${val:.2f}', ha='center', va='bottom', fontweight='bold', fontsize=11)

mae_values = [lstm_mae, arima_mae]
bars2 = axes[1].bar(models, mae_values, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
axes[1].set_ylabel('MAE ($)', fontsize=12, fontweight='bold')
axes[1].set_title('Mean Absolute Error (MAE)', fontsize=14, fontweight='bold')
axes[1].grid(axis='y', alpha=0.3, linestyle='--')
axes[1].set_ylim(0, max(mae_values) * 1.2)
for i, (bar, val) in enumerate(zip(bars2, mae_values)):
    axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(mae_values)*0.05, f'${val:.2f}', ha='center', va='bottom', fontweight='bold', fontsize=11)

mape_values = [lstm_mape, arima_mape]
bars3 = axes[2].bar(models, mape_values, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
axes[2].set_ylabel('MAPE (%)', fontsize=12, fontweight='bold')
axes[2].set_title('Mean Absolute Percentage Error (MAPE)', fontsize=14, fontweight='bold')
axes[2].grid(axis='y', alpha=0.3, linestyle='--')
axes[2].set_ylim(0, max(mape_values) * 1.2)
for i, (bar, val) in enumerate(zip(bars3, mape_values)):
    axes[2].text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(mape_values)*0.05, f'{val:.2f}%', ha='center', va='bottom', fontweight='bold', fontsize=11)

fig.suptitle(f'{TICKER} Model Accuracy Comparison', fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()

accuracy_plot_path = 'docs/accuracy_comparison.png'
plt.savefig(accuracy_plot_path, dpi=300, bbox_inches='tight')
print(f"\nAccuracy comparison chart saved to: {accuracy_plot_path}")
plt.close()