import os
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime

from ta.trend import EMAIndicator
from ta.momentum import RSIIndicator
from ta.volatility import AverageTrueRange

from alpaca.trading.client import TradingClient
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame

# ------------------ PAGE CONFIG ------------------
st.set_page_config(
    page_title="Stock Trading Bot",
    page_icon="📈",
    layout="wide"
)

# ------------------ GET API KEYS ------------------
def get_credentials():
    # First try Streamlit secrets (for deployment)
    try:
        key = st.secrets["APCA_API_KEY_ID"]
        secret = st.secrets["APCA_API_SECRET_KEY"]
        return key, secret
    except Exception:
        pass

    # Fallback to environment variables (local use)
    from dotenv import load_dotenv
    load_dotenv()
    return os.getenv("APCA_API_KEY_ID"), os.getenv("APCA_API_SECRET_KEY")

API_KEY, API_SECRET = get_credentials()

if not API_KEY or not API_SECRET:
    st.error("❌ API keys missing. Add them in Streamlit Secrets or .env file.")
    st.stop()

# ------------------ SIDEBAR ------------------
st.sidebar.title("⚙️ Settings")

paper_mode = st.sidebar.toggle("Paper Trading", value=True)

st.sidebar.markdown("---")
st.sidebar.subheader("Strategy")
fast_ema = st.sidebar.number_input("Fast EMA", 5, 50, 20)
slow_ema = st.sidebar.number_input("Slow EMA", 20, 200, 50)
rsi_period = st.sidebar.number_input("RSI Period", 5, 30, 14)
rsi_max_buy = st.sidebar.slider("RSI Max Buy", 50, 80, 65)
atr_period = st.sidebar.number_input("ATR Period", 5, 30, 14)
atr_multiplier = st.sidebar.slider("ATR Multiplier", 1.0, 4.0, 2.0, 0.1)
risk_per_trade = st.sidebar.slider("Risk per Trade %", 0.5, 5.0, 1.0) / 100

st.sidebar.markdown("---")
symbols_input = st.sidebar.text_area(
    "Watchlist",
    value="AAPL,MSFT,NVDA,AMZN,META,GOOGL,SPY,QQQ"
)
SYMBOLS = [s.strip().upper() for s in symbols_input.split(",") if s.strip()]

# ------------------ CLIENTS ------------------
@st.cache_resource
def get_clients(paper: bool):
    trading = TradingClient(API_KEY, API_SECRET, paper=paper)
    data = StockHistoricalDataClient(API_KEY, API_SECRET)
    return trading, data

trading_client, data_client = get_clients(paper_mode)

# ------------------ HELPERS ------------------
def get_daily_bars(symbol: str, limit: int = 150):
    request = StockBarsRequest(
        symbol_or_symbols=symbol,
        timeframe=TimeFrame.Day,
        limit=limit
    )
    bars = data_client.get_stock_bars(request)
    df = bars.df
    if isinstance(df.index, pd.MultiIndex):
        df = df.xs(symbol)
    df = df.reset_index()
    df.rename(columns={
        "timestamp": "date", "open": "Open", "high": "High",
        "low": "Low", "close": "Close", "volume": "Volume"
    }, inplace=True)
    df["date"] = pd.to_datetime(df["date"])
    df.set_index("date", inplace=True)
    return df

def add_indicators(df):
    df = df.copy()
    df["ema_fast"] = EMAIndicator(df["Close"], window=fast_ema).ema_indicator()
    df["ema_slow"] = EMAIndicator(df["Close"], window=slow_ema).ema_indicator()
    df["rsi"] = RSIIndicator(df["Close"], window=rsi_period).rsi()
    df["atr"] = AverageTrueRange(df["High"], df["Low"], df["Close"], window=atr_period).average_true_range()
    return df

def generate_signal(df):
    if len(df) < slow_ema + 5:
        return "HOLD"
    curr, prev = df.iloc[-1], df.iloc[-2]
    bullish = (prev["ema_fast"] <= prev["ema_slow"]) and (curr["ema_fast"] > curr["ema_slow"])
    rsi_ok = curr["rsi"] < rsi_max_buy
    bearish = (prev["ema_fast"] >= prev["ema_slow"]) and (curr["ema_fast"] < curr["ema_slow"])
    if bullish and rsi_ok:
        return "BUY"
    elif bearish:
        return "SELL"
    return "HOLD"

# ------------------ MAIN UI ------------------
st.title("📈 US Stock Trading Bot")

mode_text = "🟢 PAPER TRADING" if paper_mode else "🔴 LIVE TRADING"
st.subheader(mode_text)

if not paper_mode:
    st.error("⚠️ LIVE MODE – Real money at risk!")

# Account info
try:
    account = trading_client.get_account()
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Equity", f"${float(account.equity):,.2f}")
    col2.metric("Buying Power", f"${float(account.buying_power):,.2f}")
    col3.metric("Cash", f"${float(account.cash):,.2f}")
    positions = trading_client.get_all_positions()
    col4.metric("Open Positions", len(positions))
except Exception as e:
    st.error(f"Account error: {e}")
    st.stop()

st.divider()

# Positions
st.subheader("Open Positions")
if positions:
    pos_data = [{
        "Symbol": p.symbol,
        "Qty": float(p.qty),
        "Entry": float(p.avg_entry_price),
        "Current": float(p.current_price),
        "P/L $": float(p.unrealized_pl),
        "P/L %": float(p.unrealized_plpc) * 100
    } for p in positions]
    st.dataframe(pd.DataFrame(pos_data), use_container_width=True)
else:
    st.info("No open positions")

st.divider()

# Signals
st.subheader("Current Signals")
if st.button("🔄 Refresh Signals", use_container_width=True):
    st.rerun()

signal_rows = []
progress = st.progress(0)

for i, symbol in enumerate(SYMBOLS):
    try:
        df = get_daily_bars(symbol)
        df = add_indicators(df)
        signal = generate_signal(df)
        curr = df.iloc[-1]
        signal_rows.append({
            "Symbol": symbol,
            "Price": curr["Close"],
            "EMA Fast": curr["ema_fast"],
            "EMA Slow": curr["ema_slow"],
            "RSI": curr["rsi"],
            "Signal": signal
        })
    except Exception as e:
        signal_rows.append({"Symbol": symbol, "Signal": f"Error: {e}"})
    progress.progress((i + 1) / len(SYMBOLS))

progress.empty()
signals_df = pd.DataFrame(signal_rows)

def color_signal(val):
    if val == "BUY":
        return "background-color: #14532d; color: #4ade80"
    if val == "SELL":
        return "background-color: #7f1d1d; color: #fca5a5"
    return ""

st.dataframe(
    signals_df.style.applymap(color_signal, subset=["Signal"]),
    use_container_width=True
)

# Chart
st.divider()
st.subheader("Chart")
selected = st.selectbox("Choose symbol", SYMBOLS)

if selected:
    try:
        df = get_daily_bars(selected)
        df = add_indicators(df)

        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.7, 0.3],
                            vertical_spacing=0.05, subplot_titles=(f"{selected}", "RSI"))

        fig.add_trace(go.Candlestick(x=df.index, open=df["Open"], high=df["High"],
                                     low=df["Low"], close=df["Close"], name="Price"), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df["ema_fast"], name=f"EMA {fast_ema}", line=dict(color="cyan")), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df["ema_slow"], name=f"EMA {slow_ema}", line=dict(color="orange")), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df["rsi"], name="RSI", line=dict(color="purple")), row=2, col=1)
        fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)

        fig.update_layout(height=650, xaxis_rangeslider_visible=False, template="plotly_dark")
        st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.error(f"Chart error: {e}")

st.caption(f"Last update: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} • Not financial advice")