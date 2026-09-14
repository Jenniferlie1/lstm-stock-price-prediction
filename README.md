# LSTM Stock Price Prediction — AMZN & CSCO
Time series forecasting of daily closing stock prices using LSTM, comparing a baseline architecture against a modified/tuned architecture.
## Overview
- Univariate time series forecasting on AMZN and CSCO closing prices
- Chronological train/test split (last 1 year held out as test set) to avoid data leakage
- Sliding window preprocessing (window size = 5) with MinMaxScaler normalization
## Workflow
1. EDA & Preprocessing — missing value checks, feature selection, distribution/outlier analysis, chronological train/test split, MinMax scaling, sliding window dataset construction
2. Baseline Architecture — single LSTM layer (50 units, ReLU) + Dense output layer
3. Modified Architecture — increased LSTM units (50 → 100), added Dense(32, ReLU) layer, L2 regularization, tuned Adam learning rate (0.0005)
4. Evaluation — baseline vs modified model comparison using RMSE, MAE, MAPE
## Tech Stack
Python TensorFlow/Keras scikit-learn Pandas NumPy Matplotlib
## Data
AMZN.csv, CSCO.csv — historical daily closing prices
