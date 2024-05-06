# Stock Market Forecasting with LSTM

This repository contains a simple Long Short-Term Memory (LSTM) model for stock market forecasting using historical stock prices. The data is sourced from Yahoo Finance, and the model is built using TensorFlow Keras. The model aims to predict the closing prices for a specific stock based on historical data.

## Key Features

1. **Data Source**: Fetches historical stock price data from Yahoo Finance using the `yfinance` package.
2. **Model Type**: Utilizes an LSTM model for capturing temporal trends in the stock market data.
3. **Evaluation**: Visualizes predicted prices against actual prices for easy performance comparison.
4. **Compatibility**: Developed with Python and TensorFlow for seamless execution and reproducibility.

## How to Run

1. **Install Dependencies**: Ensure that Python 3.7+ is installed. Install required packages using:
    ```bash
    pip install -r requirements.txt
    ```

2. **Execute the Script**: Run the forecasting script:
    ```bash
    python stock_forecasting.py
    ```

3. **View Results**: The script will download data, build and train the LSTM model, and output a plot comparing actual vs. predicted stock prices.

## Required Libraries
- `tensorflow`
- `yfinance`
- `pandas`
- `numpy`
- `matplotlib`
- `scikit-learn`
