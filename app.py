import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import torch
import torch.nn as nn
import pickle
from pathlib import Path

# ── page config ────────────────────────────────────────────
st.set_page_config(
    page_title="StockLens",
    page_icon="📈",
    layout="wide"
)

# ── LSTM definition (must match train.py exactly) ───────────
class StockLSTM(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=2, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size, hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            batch_first=True
        )
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])

# ── constants ──────────────────────────────────────────────
FEATURES = [
    "close", "open", "high", "low", "volume",
    "sentiment_score", "sentiment_3d_avg",
    "ma_5", "ma_20", "rsi", "momentum", "volatility",
    "daily_return", "return_5d",
    "bullish_count", "bearish_count"
]
MODEL_DIR  = Path("models/saved")
TICKERS    = ["AAPL", "MSFT", "TSLA"]
TICKER_NAMES = {
    "AAPL": "Apple Inc.",
    "MSFT": "Microsoft Corp.",
    "TSLA": "Tesla Inc."
}

# ── loaders ────────────────────────────────────────────────
@st.cache_data
def load_master():
    return pd.read_parquet("data/processed/master.parquet")

@st.cache_resource
def load_model(ticker):
    model = StockLSTM(input_size=len(FEATURES))
    state = torch.load(
        MODEL_DIR / f"{ticker}_lstm.pt",
        map_location="cpu",
        weights_only=True
    )
    model.load_state_dict(state)
    model.eval()
    return model

@st.cache_resource
def load_scalers(ticker):
    with open(MODEL_DIR / f"{ticker}_scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    with open(MODEL_DIR / f"{ticker}_target_scaler.pkl", "rb") as f:
        target_scaler = pickle.load(f)
    return scaler, target_scaler

@st.cache_data
def load_predictions(ticker):
    return pd.read_csv(MODEL_DIR / f"{ticker}_predictions.csv")

# ── prediction helper ──────────────────────────────────────
def predict_next_day(ticker, master):
    model = load_model(ticker)
    scaler, target_scaler = load_scalers(ticker)

    df_t = master[master["ticker"] == ticker].sort_values("date")
    df_t = df_t[FEATURES].dropna()

    last_20 = df_t.tail(20).values
    scaled  = scaler.transform(last_20)
    X       = torch.tensor(scaled[np.newaxis, :, :], dtype=torch.float32)

    with torch.no_grad():
        pred_scaled = model(X).numpy()

    pred_price = target_scaler.inverse_transform(pred_scaled)[0][0]
    return pred_price

# ── styling ────────────────────────────────────────────────
st.markdown("""
<style>
.metric-card {
    background: #1E1E2E;
    border-radius: 12px;
    padding: 20px;
    text-align: center;
}
.big-number {
    font-size: 2rem;
    font-weight: 700;
    color: #CDD6F4;
}
.label {
    font-size: 0.85rem;
    color: #A6ADC8;
    margin-bottom: 6px;
}
.positive { color: #A6E3A1; }
.negative { color: #F38BA8; }
</style>
""", unsafe_allow_html=True)

# ── sidebar ────────────────────────────────────────────────
with st.sidebar:
    st.title("📈 StockLens")
    st.caption("AI-powered stock sentiment & forecasting")
    st.divider()

    ticker = st.selectbox(
        "Select Stock",
        TICKERS,
        format_func=lambda x: f"{x} — {TICKER_NAMES[x]}"
    )

    st.divider()
    st.caption("Model info")
    results = {
        "AAPL": {"mae": 5.09, "rmse": 6.20, "dir_acc": 54.1},
        "MSFT": {"mae": 8.25, "rmse": 9.87, "dir_acc": 49.2},
        "TSLA": {"mae": 9.72, "rmse": 12.18,"dir_acc": 49.2},
    }
    r = results[ticker]
    st.metric("Directional Accuracy", f"{r['dir_acc']}%")
    st.metric("MAE",  f"${r['mae']}")
    st.metric("RMSE", f"${r['rmse']}")

# ── load data ──────────────────────────────────────────────
master = load_master()
df = master[master["ticker"] == ticker].sort_values("date").copy()
preds_df = load_predictions(ticker)

# current price info
current_price  = df["close"].iloc[-1]
prev_price     = df["close"].iloc[-2]
price_change   = current_price - prev_price
price_change_pct = (price_change / prev_price) * 100
next_day_pred  = predict_next_day(ticker, master)
pred_change    = next_day_pred - current_price
current_sentiment = df["sentiment_score"].iloc[-1]
current_rsi    = df["rsi"].iloc[-1]

# ── header ─────────────────────────────────────────────────
st.title(f"{ticker} — {TICKER_NAMES[ticker]}")
st.caption(f"Last updated: {df['date'].iloc[-1].strftime('%B %d, %Y')}")

# ── metric cards ───────────────────────────────────────────
c1, c2, c3, c4 = st.columns(4)

with c1:
    st.metric(
        "Current Price",
        f"${current_price:.2f}",
        f"{price_change_pct:+.2f}% today"
    )
with c2:
    st.metric(
        "Next Day Forecast",
        f"${next_day_pred:.2f}",
        f"{pred_change:+.2f} predicted move"
    )
with c3:
    sentiment_label = (
        "🟢 Bullish" if current_sentiment > 0.1
        else "🔴 Bearish" if current_sentiment < -0.1
        else "⚪ Neutral"
    )
    st.metric("Market Sentiment", sentiment_label, f"score: {current_sentiment:.3f}")
with c4:
    rsi_label = (
        "Overbought" if current_rsi > 70
        else "Oversold" if current_rsi < 30
        else "Normal"
    )
    st.metric("RSI", f"{current_rsi:.1f}", rsi_label)

st.divider()

# ── price + prediction chart ───────────────────────────────
st.subheader("Price History vs Model Predictions")

# align predictions with test period dates
test_size   = len(preds_df)
test_dates  = df["date"].iloc[-test_size:].values

fig = make_subplots(
    rows=2, cols=1,
    shared_xaxes=True,
    row_heights=[0.7, 0.3],
    vertical_spacing=0.05
)

# candlestick
fig.add_trace(go.Candlestick(
    x=df["date"],
    open=df["open"], high=df["high"],
    low=df["low"],   close=df["close"],
    name="Price",
    increasing_line_color="#A6E3A1",
    decreasing_line_color="#F38BA8"
), row=1, col=1)

# predicted line
fig.add_trace(go.Scatter(
    x=test_dates,
    y=preds_df["predicted"],
    name="LSTM Prediction",
    line=dict(color="#89B4FA", width=2, dash="dash")
), row=1, col=1)

# MA lines
fig.add_trace(go.Scatter(
    x=df["date"], y=df["ma_5"],
    name="MA5", line=dict(color="#FAB387", width=1)
), row=1, col=1)

fig.add_trace(go.Scatter(
    x=df["date"], y=df["ma_20"],
    name="MA20", line=dict(color="#CBA6F7", width=1)
), row=1, col=1)

# sentiment bars
colors = [
    "#A6E3A1" if s > 0.1 else "#F38BA8" if s < -0.1 else "#9399B2"
    for s in df["sentiment_score"]
]
fig.add_trace(go.Bar(
    x=df["date"],
    y=df["sentiment_score"],
    name="Sentiment",
    marker_color=colors,
    opacity=0.8
), row=2, col=1)

fig.update_layout(
    height=600,
    paper_bgcolor="#1E1E2E",
    plot_bgcolor="#1E1E2E",
    font_color="#CDD6F4",
    xaxis_rangeslider_visible=False,
    legend=dict(
        bgcolor="#313244",
        bordercolor="#45475A",
        borderwidth=1
    )
)
fig.update_xaxes(gridcolor="#313244")
fig.update_yaxes(gridcolor="#313244")
fig.update_yaxes(title_text="Price (USD)", row=1, col=1)
fig.update_yaxes(title_text="Sentiment", row=2, col=1)

st.plotly_chart(fig, use_container_width=True)

# ── technical indicators ───────────────────────────────────
st.subheader("Technical Indicators")

col1, col2 = st.columns(2)

with col1:
    fig_rsi = go.Figure()
    fig_rsi.add_trace(go.Scatter(
        x=df["date"], y=df["rsi"],
        fill="tozeroy",
        line=dict(color="#89B4FA"),
        name="RSI"
    ))
    fig_rsi.add_hline(y=70, line_dash="dash",
                      line_color="#F38BA8", annotation_text="Overbought")
    fig_rsi.add_hline(y=30, line_dash="dash",
                      line_color="#A6E3A1", annotation_text="Oversold")
    fig_rsi.update_layout(
        title="RSI (14-day)",
        height=300,
        paper_bgcolor="#1E1E2E",
        plot_bgcolor="#1E1E2E",
        font_color="#CDD6F4"
    )
    fig_rsi.update_xaxes(gridcolor="#313244")
    fig_rsi.update_yaxes(gridcolor="#313244", range=[0, 100])
    st.plotly_chart(fig_rsi, use_container_width=True)

with col2:
    fig_vol = go.Figure()
    fig_vol.add_trace(go.Scatter(
        x=df["date"], y=df["volatility"],
        fill="tozeroy",
        line=dict(color="#FAB387"),
        name="Volatility"
    ))
    fig_vol.update_layout(
        title="Rolling Volatility (10-day)",
        height=300,
        paper_bgcolor="#1E1E2E",
        plot_bgcolor="#1E1E2E",
        font_color="#CDD6F4"
    )
    fig_vol.update_xaxes(gridcolor="#313244")
    fig_vol.update_yaxes(gridcolor="#313244")
    st.plotly_chart(fig_vol, use_container_width=True)

# ── prediction accuracy table ──────────────────────────────
st.subheader("Model Performance — All Tickers")

perf_data = []
for t in TICKERS:
    r = results[t]
    signal = "BUY 🟢" if predict_next_day(t, master) > master[
        master["ticker"]==t]["close"].iloc[-1] else "SELL 🔴"
    perf_data.append({
        "Ticker": t,
        "Current Price": f"${master[master['ticker']==t]['close'].iloc[-1]:.2f}",
        "Next Day Forecast": f"${predict_next_day(t, master):.2f}",
        "Signal": signal,
        "Directional Acc": f"{r['dir_acc']}%",
        "MAE": f"${r['mae']}",
        "RMSE": f"${r['rmse']}"
    })

st.dataframe(
    pd.DataFrame(perf_data),
    use_container_width=True,
    hide_index=True
)


# ── WOW FEATURE 1: Sentiment vs Price Correlation Heatmap ──
st.subheader("Sentiment → Next Day Price Move Correlation")
st.caption("How strongly does today's sentiment predict tomorrow's price direction?")

corr_data = []
for t in TICKERS:
    df_t = master[master["ticker"] == t].sort_values("date").copy()
    df_t["next_day_return"] = df_t["close"].pct_change().shift(-1)
    df_t = df_t.dropna(subset=["sentiment_score", "next_day_return"])

    # rolling 30-day correlation
    rolling_corr = df_t["sentiment_score"].rolling(30).corr(df_t["next_day_return"])
    df_t["rolling_corr"] = rolling_corr
    df_t["ticker"] = t
    corr_data.append(df_t[["date", "ticker", "rolling_corr"]].dropna())

corr_df = pd.concat(corr_data)

fig_corr = go.Figure()
colors_map = {"AAPL": "#89B4FA", "MSFT": "#A6E3A1", "TSLA": "#FAB387"}

for t in TICKERS:
    df_t = corr_df[corr_df["ticker"] == t]
    fig_corr.add_trace(go.Scatter(
        x=df_t["date"],
        y=df_t["rolling_corr"],
        name=t,
        line=dict(color=colors_map[t], width=2),
        fill="tozeroy",
        opacity=0.6
    ))

fig_corr.add_hline(y=0, line_color="#CDD6F4", line_width=1)
fig_corr.add_hrect(y0=0.2,  y1=1,  fillcolor="#A6E3A1", opacity=0.05)
fig_corr.add_hrect(y0=-1,   y1=-0.2, fillcolor="#F38BA8", opacity=0.05)

fig_corr.update_layout(
    height=350,
    paper_bgcolor="#1E1E2E",
    plot_bgcolor="#1E1E2E",
    font_color="#CDD6F4",
    yaxis_title="Correlation coefficient",
    xaxis_title="Date",
    legend=dict(bgcolor="#313244")
)
fig_corr.update_xaxes(gridcolor="#313244")
fig_corr.update_yaxes(gridcolor="#313244", range=[-1, 1])

st.plotly_chart(fig_corr, use_container_width=True)

with st.expander("How to read this chart"):
    st.write("""
    - **Above 0 (green zone):** Positive sentiment predicted price rises that day
    - **Below 0 (red zone):** Sentiment was contrarian — positive news preceded drops
    - **Near 0:** Sentiment had little predictive power that month
    - This rolling correlation shifts over time, showing when sentiment matters more or less to the market
    """)

st.divider()

# ── WOW FEATURE 2: Backtesting ─────────────────────────────
st.subheader("Strategy Backtesting")
st.caption("Does following the model's signals beat simply holding the stock?")

col1, col2 = st.columns([1, 2])

with col1:
    threshold = st.slider(
        "Sentiment threshold for BUY signal",
        min_value=0.0,
        max_value=0.5,
        value=0.1,
        step=0.05,
        help="Only buy when sentiment score is above this value"
    )
    initial_cash = st.number_input(
        "Initial investment (USD)",
        min_value=1000,
        max_value=100000,
        value=10000,
        step=1000
    )

with col2:
    # run backtest on selected ticker
    df_bt = master[master["ticker"] == ticker].sort_values("date").copy()
    df_bt = df_bt.dropna(subset=["sentiment_score", "close"])

    # strategy: buy when sentiment > threshold, sell otherwise
    cash       = float(initial_cash)
    shares     = 0.0
    strategy   = []
    buyhold    = []
    bh_shares  = initial_cash / df_bt["close"].iloc[0]

    for _, row in df_bt.iterrows():
        price     = row["close"]
        sentiment = row["sentiment_score"]

        if sentiment > threshold and cash > 0:
            shares = cash / price
            cash   = 0.0
        elif sentiment < -threshold and shares > 0:
            cash   = shares * price
            shares = 0.0

        portfolio_val = cash + shares * price
        strategy.append(portfolio_val)
        buyhold.append(bh_shares * price)

    df_bt["strategy"]  = strategy
    df_bt["buy_hold"]  = buyhold

    strat_return  = (strategy[-1]  - initial_cash) / initial_cash * 100
    bh_return     = (buyhold[-1]   - initial_cash) / initial_cash * 100

    fig_bt = go.Figure()
    fig_bt.add_trace(go.Scatter(
        x=df_bt["date"], y=df_bt["strategy"],
        name="Sentiment Strategy",
        line=dict(color="#A6E3A1", width=2)
    ))
    fig_bt.add_trace(go.Scatter(
        x=df_bt["date"], y=df_bt["buy_hold"],
        name="Buy & Hold",
        line=dict(color="#89B4FA", width=2, dash="dash")
    ))
    fig_bt.add_hline(
        y=initial_cash,
        line_dash="dot",
        line_color="#9399B2",
        annotation_text="Starting value"
    )
    fig_bt.update_layout(
        height=350,
        paper_bgcolor="#1E1E2E",
        plot_bgcolor="#1E1E2E",
        font_color="#CDD6F4",
        yaxis_title="Portfolio Value (USD)",
        legend=dict(bgcolor="#313244")
    )
    fig_bt.update_xaxes(gridcolor="#313244")
    fig_bt.update_yaxes(gridcolor="#313244")
    st.plotly_chart(fig_bt, use_container_width=True)

# backtest summary metrics
m1, m2, m3 = st.columns(3)
m1.metric(
    "Strategy Return",
    f"{strat_return:+.1f}%",
    f"${strategy[-1]-initial_cash:+,.0f}"
)
m2.metric(
    "Buy & Hold Return",
    f"{bh_return:+.1f}%",
    f"${buyhold[-1]-initial_cash:+,.0f}"
)
m3.metric(
    "Alpha",
    f"{strat_return - bh_return:+.1f}%",
    "vs buy & hold"
)

with st.expander("How the strategy works"):
    st.write(f"""
    - **BUY:** When sentiment score > {threshold:.2f} (bullish signal), invest all cash
    - **SELL:** When sentiment score < -{threshold:.2f} (bearish signal), liquidate to cash
    - **Hold:** When sentiment is between -{threshold:.2f} and {threshold:.2f}, maintain position
    - Adjust the threshold slider to see how signal sensitivity affects returns
    """)

st.divider()

# ── WOW FEATURE 3: Live Refresh ────────────────────────────
st.subheader("Data Freshness")

last_date = master["date"].max()
days_old  = (pd.Timestamp.now() - last_date).days

col1, col2 = st.columns([1, 3])
with col1:
    st.metric("Last data point", last_date.strftime("%b %d, %Y"))
    st.metric("Data age", f"{days_old} days ago")

    if st.button("🔄 Refresh Price Data", type="primary"):
        with st.spinner("Fetching latest prices..."):
            import subprocess
            subprocess.run(["python", "data/fetch_prices.py"])
            subprocess.run(["python", "models/sentiment.py"])
            st.cache_data.clear()
            st.success("Data refreshed!")
            st.rerun()

with col2:
    # show recent data table
    recent = master.sort_values("date", ascending=False)
    recent = recent[recent["ticker"] == ticker].head(7)
    st.dataframe(
        recent[["date", "close", "daily_return",
                "sentiment_score", "rsi", "ma_5"]].round(3),
        use_container_width=True,
        hide_index=True
    )

st.caption("⚠️ This is a research project. Not financial advice.")