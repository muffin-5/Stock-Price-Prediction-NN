# Stock Price Prediction & Trading Signal Generation

> An end-to-end deep learning project that predicts a stock's closing price 5 days ahead and converts that regression output into actionable **Buy / Hold / Sell** trading signals.

![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-Keras-FF6F00?logo=tensorflow&logoColor=white)
![scikit--learn](https://img.shields.io/badge/scikit--learn-F7931E?logo=scikit-learn&logoColor=white)
![yfinance](https://img.shields.io/badge/Data-yFinance-0088CC)
![License](https://img.shields.io/badge/License-MIT-green)

---

## Table of Contents

- [Overview](#overview)
- [Why this project](#why-this-project)
- [Key Features](#key-features)
- [Methodology](#methodology)
- [Model Architecture](#model-architecture)
- [Evaluation](#evaluation)
- [Tech Stack](#tech-stack)
- [Getting Started](#getting-started)
- [Project Structure](#project-structure)
- [Results & Visualisations](#results--visualisations)
- [Limitations & Future Work](#limitations--future-work)
- [Disclaimer](#disclaimer)

---

## Overview

Stock prices are driven by a mix of market sentiment, macroeconomic indicators and company performance. While Recurrent Neural Networks (RNNs) and LSTMs are the usual choice for sequential financial data, this project deliberately tackles the problem with a **Multilayer Perceptron (MLP)** built on economic/technical indicators, demonstrating that a well-feature-engineered feed-forward network can still capture the temporal relationships in time-series data.

The pipeline covers the complete machine learning lifecycle:

```
Data Ingestion → Feature Engineering → Preprocessing → Deep Learning → Regression
                                                             ↓
                   Evaluation (Confusion Matrix, Classification Report, ROC-AUC)
                                                             ↓
                                       Buy / Hold / Sell Signal Generation
```

---

## Why this project

- **Feature engineering over model hype** – instead of feeding raw prices into an LSTM, 14+ technical indicators are engineered so a simpler MLP can learn from them.
- **Regression → Classification bridge** – a continuous price prediction is converted into discrete, tradable signals with a threshold-based rule set, a pattern common in real quant workflows.
- **Rigorous evaluation** – the model is assessed on both fronts: price regression (MSE) and signal classification (precision / recall / F1, confusion matrix, one-vs-rest ROC-AUC).

---

## Key Features

| Feature | Details |
| --- | --- |
| **Data ingestion** | Automated historical data download via `yfinance` (2000–2024, NSE tickers such as `INFY.NS`, `RELIANCE.NS`, `HDFCBANK.NS`, `ICICIBANK.NS`, `TATAMOTORS.NS`) |
| **Technical indicators** | RSI, MACD + signal line, Bollinger Bands (upper/lower), ATR, Stochastic Oscillator, SMA/EMA |
| **Lagged returns** | 5-day, 10-day and 20-day returns to encode momentum |
| **Target** | Closing price 5 trading days ahead (`Close_5d`) |
| **Preprocessing** | Chronological 95/5 train-test split (no look-ahead leakage) + `StandardScaler` standardisation |
| **Model** | 4-layer Keras MLP (64 → 32 → 16 → 1) with ReLU activations, L2 regularisation, AdamW optimiser, MSE loss |
| **Signals** | Buy (> +2%), Sell (< −2%), Hold (±2%) based on predicted 5-day % change |
| **Evaluation** | Classification report, confusion matrix heatmap, multi-class ROC curves with AUC |
| **Visualisation** | Actual vs. predicted price curves and colour-coded Buy/Hold/Sell signal charts |

---

## Methodology

### 1. Data Collection
Historical OHLCV data is pulled with `yfinance` for a chosen ticker between `2000-01-01` and `2024-01-01`, keeping only `Open`, `High`, `Low`, `Close`, `Volume`.

### 2. Feature Engineering (`ta` library)
- **Bollinger Bands** – 20-day moving average based high/low bands for volatility context
- **ATR (14-day)** – average price swing / volatility
- **RSI (14-day)** – overbought / oversold momentum
- **MACD (12/26) + Signal (9)** – trend and momentum shifts
- **Lagged returns** – 5d, 10d, 20d percentage returns

### 3. Target Construction
`Close_5d = Close.shift(-5)` – the closing price five days in the future. A 5-day horizon was chosen because technical indicators are inherently lagging; matching the prediction window to their response delay lets the model align indicator values with near-term future prices.

### 4. Preprocessing
- Chronological **95/5 train-test split** (time-series aware – no shuffling across time)
- `StandardScaler` fitted on train data, applied to test data
- `Next_day_close` removed from features to prevent target leakage

### 5. Signal Generation
Regression output is converted to a percentage change over 5 days, then bucketed:

| Signal | Condition |
| --- | --- |
| **Buy (1)** | predicted 5-day change > **+2%** |
| **Hold (0)** | change between **−2% and +2%** |
| **Sell (2)** | predicted 5-day change < **−2%** |

---

## Model Architecture

```
Input (n features)
  → Dense(64, ReLU) + L2(0.001)
  → Dense(32, ReLU) + L2(0.001)
  → Dense(16, ReLU) + L2(0.001)
  → Dense(1)  # regression output

Optimizer : AdamW (lr = 1e-4)
Loss     : Mean Squared Error
Regular. : L2 = 0.001
Epochs   : 100 | Batch size : 64 | Seed : 42
```

Layer sizes, learning rate and regularisation were selected through systematic trial and error over multiple hyperparameter combinations.

---

## Evaluation

Both regression and classification quality are measured:

1. **Regression** – test-set MSE via `model.evaluate(X_test, y_test)`, plus a predicted-vs-actual price curve.
2. **Classification** – signals are compared against actual market moves using:
   - Precision / Recall / F1-score per class (Hold, Buy, Sell)
   - Confusion matrix heatmap
   - **One-vs-rest ROC curves** with AUC for each class, using sigmoid-sharpened class probabilities derived from the predicted percentage change.

---

## Tech Stack

| Layer | Tools |
| --- | --- |
| Language | Python 3 |
| Data | `yfinance`, `pandas`, `numpy` |
| Technical analysis | `ta` |
| Modelling | TensorFlow / Keras |
| Preprocessing & metrics | `scikit-learn` |
| Visualisation | `matplotlib`, `seaborn` |
| Environment | Jupyter Notebook / Google Colab |

---

## Getting Started

### Prerequisites
- Python 3.10+
- `pip`

### Installation

```bash
git clone https://github.com/muffin-5/Stock-Price-Prediction-NN.git
cd Stock-Price-Prediction-NN
pip install -r requirements.txt
```

### Run

```bash
jupyter notebook Stock_Market_Prediction.ipynb
```

Or open it directly in Google Colab – the notebook includes an optional Google Drive cell for caching trained `.keras` models.

> Change the `ticker`, `start_date` and `end_date` variables in the second cell to experiment with any other stock.

---

## Project Structure

```
Stock-Price-Prediction-NN/
├── Stock_Market_Prediction.ipynb   # Full pipeline: data → features → model → signals → evaluation
├── requirements.txt                # Python dependencies
├── README.md                       # Project documentation
└── .gitignore
```

---

## Results & Visualisations

The notebook produces four core outputs:

1. **Price forecast** – predicted 5-day-ahead closing price overlaid on actual prices.
2. **Actual signal chart** – stock price colour-coded by true Buy (green) / Hold (blue) / Sell (red) movements.
3. **Predicted signal chart** – the same colour-coding driven by the model's predictions.
4. **Evaluation dashboard** – confusion matrix heatmap + one-vs-rest ROC curves with AUC scores.

Run the notebook to regenerate all figures with live market data.

---

## Limitations & Future Work

- **Univariate macro scope** – only price-derived economic indicators are used; news sentiment, fundamentals and macro data are not incorporated.
- **MLP vs. sequence models** – an LSTM / Transformer / Temporal Fusion Transformer could model long-range dependencies explicitly.
- **Transaction costs** – signal thresholds do not account for brokerage, slippage or taxes.
- **Backtesting** – a proper event-driven backtest with Sharpe ratio, drawdown and CAGR would quantify real-world profitability.
- **Hyperparameter search** – move from manual trial-and-error to Optuna/KerasTuner.

---

## Disclaimer

This project is for **educational and research purposes only**. It is not financial advice, and no prediction produced here should be used to make real investment decisions. Stock markets are inherently unpredictable – past performance does not guarantee future results.

---

**Author:** [muffin-5](https://github.com/muffin-5)
