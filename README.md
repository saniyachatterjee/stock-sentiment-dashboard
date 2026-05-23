# 📈 StockLens — AI Stock Sentiment & Price Forecasting

![Python](https://img.shields.io/badge/Python-3.9-blue?style=flat-square)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0-orange?style=flat-square)
![Streamlit](https://img.shields.io/badge/Streamlit-deployed-red?style=flat-square)
![License](https://img.shields.io/badge/license-MIT-green?style=flat-square)

> A real-time financial analytics dashboard combining **PyTorch LSTM price forecasting** 
> with **sentiment analysis** across AAPL, MSFT, and TSLA — deployed as an interactive 
> Streamlit web application.

---

## 🎯 Problem Statement

Financial decisions are made on incomplete information. This project builds an end-to-end 
ML pipeline that extracts sentiment signals from market price behavior, combines them with 
technical indicators, and forecasts next-day stock prices using a deep learning model.

---

## 🏗️ System Architecture

```
yfinance (Price Data)
        ↓
Feature Engineering
(Returns · RSI · Moving Averages · Volatility)
        ↓
Proxy Sentiment Scoring
(tanh-normalized daily returns → bullish/bearish/neutral)
        ↓
2-Layer PyTorch LSTM
(seq_len=20 · hidden=64 · dropout=0.2)
        ↓
Streamlit Dashboard
(Candlestick · Predictions · Backtesting · Correlation)
```

---

## 📊 Model Results

| Ticker | MAE | RMSE | Directional Accuracy |
|--------|-----|------|----------------------|
| AAPL | $5.09 | $6.20 | **54.1%** |
| MSFT | $8.25 | $9.87 | 49.2% |
| TSLA | $9.72 | $12.18 | 49.2% |

> **Directional accuracy** measures whether the model correctly predicted the 
> direction of next-day price movement (up or down). In an efficient market, 
> anything above 50% is meaningful signal.

---

## ✨ Dashboard Features

- **Candlestick chart** with LSTM predictions overlaid as dashed line
- **MA5 and MA20** moving average overlays
- **Sentiment bars** — green/red by daily market sentiment score
- **RSI chart** with overbought/oversold zones
- **Rolling volatility** chart
- **Backtesting engine** — sentiment strategy vs buy & hold with live threshold slider
- **Alpha metric** — strategy outperformance vs buy & hold
- **Sentiment-price correlation** — rolling 30-day correlation heatmap
- **Live data refresh** — fetches latest prices on demand
- **Model performance table** — all 3 tickers with BUY/SELL signals

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|-----------|
| Deep Learning | PyTorch — 2-layer LSTM |
| Data Pipeline | yfinance, pandas, numpy |
| Feature Engineering | scikit-learn, MinMaxScaler |
| Visualization | Plotly, Streamlit |
| Deployment | Streamlit Cloud |

---

## 📁 Project Structure

```
stock-sentiment-dashboard/
├── app.py                        # Streamlit dashboard
├── requirements.txt
├── data/
│   ├── fetch_prices.py           # yfinance price pipeline
│   ├── fetch_news.py             # News/RSS fetching
│   └── processed/
│       └── master.parquet        # Feature-engineered dataset
├── models/
│   ├── sentiment.py              # Proxy sentiment + technical indicators
│   ├── train.py                  # LSTM training pipeline
│   └── saved/                    # Trained model weights + scalers
└── notebooks/
    └── verify_day2.py            # Data validation scripts
```

---

## 🚀 Run Locally

```bash
# Clone the repo
git clone https://github.com/saniyachatterjee/stock-sentiment-dashboard
cd stock-sentiment-dashboard

# Create virtual environment
python3 -m venv venv
source venv/bin/activate        # Mac/Linux
venv\Scripts\activate           # Windows

# Install dependencies
pip install -r requirements.txt

# Fetch latest data and train model
python data/fetch_prices.py
python models/sentiment.py
python models/train.py

# Launch dashboard
streamlit run app.py
```

---

## 💡 Key Findings & Insights

- **AAPL** showed the strongest directional signal (54.1%), consistent with it being 
  a sentiment-driven retail stock with high news coverage
- **TSLA's** high daily volatility (avg ~2.3% daily move) makes directional prediction 
  harder despite strong momentum signals
- The **backtesting engine** revealed that price-derived sentiment works better as a 
  confirmation signal than a standalone predictor — a finding that points toward 
  integrating real news sentiment as a leading indicator in future work
- **Sentiment-price correlation** shifts over time, showing that the relationship 
  between market mood and price direction is regime-dependent

---

## 🔮 Future Improvements

- [ ] Integrate real news sentiment via FinBERT on live headlines
- [ ] Add Temporal Fusion Transformer for multi-horizon forecasting
- [ ] Expand to 10+ tickers with sector-level sentiment aggregation
- [ ] Add portfolio optimization layer across multiple stocks
- [ ] Deploy FastAPI backend for production-grade serving

---

## 👩‍💻 Author

**Saniya Chatterjee**
B.Tech Electronics & Telecommunications, Symbiosis Institute of Technology, Pune

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue?style=flat-square&logo=linkedin)](https://www.linkedin.com/in/saniya-chatterjee-245471266/)
[![Email](https://img.shields.io/badge/Email-saniyaa0103@gmail.com-red?style=flat-square&logo=gmail)](mailto:saniyaa0103@gmail.com)

---

## ⚠️ Disclaimer

This project is for educational and research purposes only. Not financial advice.