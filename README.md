# Stock Price Prediction: LSTM vs ARIMA Model Comparison

A machine learning project comparing Long Short-Term Memory (LSTM) neural networks and ARIMA time series models for stock price prediction. This project demonstrates data science techniques including deep learning, time series analysis, and model evaluation.

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Technologies Used](#technologies-used)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Methodology](#methodology)
- [Results](#results)
- [Model Performance](#model-performance)
- [Visualizations](#visualizations)
- [Key Insights](#key-insights)
- [Future Improvements](#future-improvements)
- [Contributing](#contributing)
- [License](#license)

## Overview

This project implements and compares two approaches to stock price prediction:

1. **LSTM (Long Short-Term Memory)**: A deep learning approach using recurrent neural networks to capture long-term dependencies in stock price sequences
2. **ARIMA (AutoRegressive Integrated Moving Average)**: A classical time series forecasting method that models temporal dependencies through autoregression and moving averages

The project fetches real-time stock data, trains both models, and provides performance metrics and visualizations to compare their effectiveness.

## Features

- Real-time data fetching using Yahoo Finance API
- Dual model implementation: LSTM and ARIMA for comparison
- Evaluation metrics: RMSE, MAE, and MAPE
- Visual analytics: time series comparison plots and accuracy metric visualizations
- Data processing: normalization, scaling, time series sequence generation, and temporal train-test split
- Production-ready code structure with documentation

## Technologies Used

### Core Libraries
- Python 3.10+: Primary programming language
- Pandas: Data manipulation and analysis
- NumPy: Numerical computing
- Matplotlib: Data visualization

### Machine Learning and Deep Learning
- TensorFlow/Keras: LSTM neural network implementation
- scikit-learn: Data preprocessing and evaluation metrics
- statsmodels: ARIMA time series modeling

### Data Sources
- yfinance: Real-time stock market data retrieval

### Environment Management
- Conda: Environment and dependency management

## Project Structure

```
Predicting-NVDL/
│
├── predict.py                 # Main prediction script
├── README.md                  # Project documentation
├── docs/                      # Generated visualizations
│   ├── model_comparison_plot.png
│   └── accuracy_comparison.png
└── Predicting                 # Additional resources
```

## Installation

### Prerequisites

- Python 3.10 or higher
- Conda (recommended) or pip

### Setup Instructions

1. Clone the repository
   ```bash
   git clone <repository-url>
   cd Predicting-NVDL
   ```

2. Create and activate conda environment
   ```bash
   conda create -n data-science python=3.10
   conda activate data-science
   ```

3. Install required packages
   ```bash
   pip install pandas numpy yfinance matplotlib scikit-learn tensorflow statsmodels
   ```

   Or using conda:
   ```bash
   conda install pandas numpy matplotlib scikit-learn
   pip install yfinance tensorflow statsmodels
   ```

4. Create output directory
   ```bash
   mkdir -p docs
   ```

## Usage

### Basic Execution

Run the prediction script with default settings (NVDA stock, 2020-present):

```bash
conda activate data-science
python predict.py
```

### Configuration

Modify the configuration variables in `predict.py`:

```python
TICKER = "NVDA"              # Stock ticker symbol
START_DATE = "2020-01-01"    # Start date for historical data
TEST_SIZE = 0.2              # Test set proportion (20%)
LSTM_TIME_STEP = 60          # Number of days for LSTM sequence
ARIMA_ORDER = (5, 1, 0)      # ARIMA model parameters (p, d, q)
```

### Output

The script generates:
- Console output with model performance metrics
- `docs/model_comparison_plot.png`: Time series comparison visualization
- `docs/accuracy_comparison.png`: Accuracy metrics comparison chart

## Methodology

### Data Preparation

1. Data fetching: Historical stock price data retrieved from Yahoo Finance
2. Temporal split: Data split chronologically (80% train, 20% test) to prevent look-ahead bias
3. Normalization: MinMaxScaler applied for LSTM model (scales data to [0, 1] range)

### LSTM Model Architecture

- Input layer: 60-day sequences (time steps)
- LSTM layer 1: 50 units with return_sequences=True
- Dropout layer 1: 20% dropout for regularization
- LSTM layer 2: 50 units with return_sequences=False
- Dropout layer 2: 20% dropout
- Dense output layer: Single neuron for price prediction
- Optimizer: Adam
- Loss function: Mean Squared Error
- Training: 10 epochs with batch size of 64

### ARIMA Model

- Order: (5, 1, 0) - 5 autoregressive terms, 1st order differencing, no moving average
- Forecasting method: Rolling window approach
- Training: Model refitted at each time step using all historical data up to that point

### Evaluation Metrics

- RMSE (Root Mean Squared Error): Penalizes larger errors more heavily
- MAE (Mean Absolute Error): Average magnitude of errors
- MAPE (Mean Absolute Percentage Error): Percentage-based error metric

## Results

### Model Performance Summary

Based on test set evaluation (297 samples):

| Model | RMSE | MAE | MAPE |
|-------|------|-----|------|
| LSTM | $11.15 | $9.29 | 6.15% |
| ARIMA | $4.09 | $3.03 | 2.16% |

### Key Findings

- ARIMA outperforms LSTM across all metrics:
  - 63% lower RMSE
  - 67% lower MAE
  - 65% lower MAPE

- ARIMA advantages:
  - Better suited for short-term forecasting
  - More interpretable model parameters
  - Lower computational requirements
  - Faster training and prediction

- LSTM characteristics:
  - Better for capturing complex non-linear patterns
  - Potential for improvement with more training data and hyperparameter tuning
  - More suitable for longer-term predictions with additional features

## Visualizations

### Model Comparison Plot

Time series visualization showing:
- Actual stock prices (black line)
- ARIMA predictions (green dashed line)
- LSTM predictions (red dotted line)

### Accuracy Comparison Chart

Side-by-side bar charts comparing:
- RMSE values
- MAE values
- MAPE percentages

## Key Insights

1. Model selection: For this use case and time period, ARIMA demonstrates superior performance, suggesting that the stock price series exhibits characteristics well-captured by classical time series methods.

2. Data characteristics: The effectiveness of ARIMA indicates the presence of:
   - Stationary or near-stationary price movements
   - Clear autoregressive patterns
   - Limited need for complex non-linear modeling

3. Practical applications:
   - ARIMA: Recommended for short-term trading strategies and risk management
   - LSTM: Better suited for scenarios with more complex patterns, multiple features, or longer prediction horizons

4. Model limitations:
   - Both models assume historical patterns will continue
   - External factors (news, events) not captured
   - Market volatility can significantly impact accuracy

## Future Improvements

### Model Enhancements
- Hyperparameter tuning for LSTM (grid search, Bayesian optimization)
- ARIMA auto-selection of optimal (p, d, q) parameters
- Ensemble methods combining both models
- Transformer-based models (e.g., Time Series Transformer)

### Feature Engineering
- Technical indicators (RSI, MACD, Bollinger Bands)
- Volume data integration
- Market sentiment analysis
- External economic indicators

### Advanced Techniques
- Walk-forward validation
- Cross-validation for time series
- Model retraining pipeline
- Real-time prediction API

### Code Improvements
- Command-line argument parsing
- Configuration file support (YAML/JSON)
- Model persistence (save/load trained models)
- Unit tests and integration tests
- Logging framework
- Docker containerization

## Contributing

Contributions are welcome. Please submit a Pull Request. For major changes, open an issue first to discuss what you would like to change.

## License

This project is open source and available under the MIT License.


## Acknowledgments

- Yahoo Finance for providing free market data
- TensorFlow and Keras communities for deep learning frameworks
- Statsmodels team for time series analysis tools

---

**Note**: This project is for educational and research purposes. Stock market predictions are inherently uncertain, and past performance does not guarantee future results. Always conduct thorough research and consult with financial advisors before making investment decisions.
