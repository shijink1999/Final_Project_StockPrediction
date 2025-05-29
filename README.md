# Final_Project_StockPrediction
# 📈 Stock Buy Signal Prediction using XGBoost (NIFTYBEES)

This project uses machine learning (XGBoost) to predict *buy signals* for the NIFTYBEES ETF based on technical indicators and historical stock data.

![Predicted vs Actual Buy Signals](figures/buy_signals_chart.png)

---

## 📌 Objective

The goal is to identify *profitable buy entry points* in the Indian stock market by training a supervised classifier using technical indicators as features and future returns as the target signal.
## 📊 Data

- *Source:* [Yahoo Finance](https://finance.yahoo.com)
- *Stock:* NIFTYBEES.NS (ETF tracking the NIFTY 50 index)
- *Timeframe:* 2018-01-01 to 2024-12-31
- *Collected using:* yfinance Python library

## 🔧 Features (Technical Indicators)

The following indicators were calculated using the ta library:

- *Momentum*: RSI, Stochastic Oscillator, Williams %R
- *Trend*: EMA, MACD, ADX, CCI
- *Volatility*: Bollinger Band Width
- *Volume*: Daily percentage volume change
- *Lag Features*: RSI, MACD, return values with 1 and 2-day lags
## 🎯 Target Variable

- *Buy (1)*: If the next day’s return > 0.7%
- *No Buy (0)*: Otherwise

```python
data['Target'] = (data['Close'].pct_change().shift(-1) > 0.007).astype(int)
