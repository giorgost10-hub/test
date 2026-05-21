"""
Stock Analysis Web App (Streamlit)
-----------------------------------
Web interface για τα 3 εργαλεία:
  - Single Stock Analysis (momentum + technical)
  - Backtester (validate strategy edge)
  - Screener (top candidates από S&P 500)

Εγκατάσταση:
    pip install streamlit yfinance pandas numpy plotly lxml

Τοπικά:
    streamlit run app.py

Θα ανοίξει αυτόματα στο browser στο http://localhost:8501
"""

import time
import warnings
warnings.filterwarnings('ignore')

from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
import yfinance as yf
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Stock Analyzer",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# INDICATORS (shared logic)
# ============================================================

def rsi(series, period=14):
    delta = series.diff()
    gain = delta.where(delta > 0, 0).rolling(period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))


def macd(series, fast=12, slow=26, signal=9):
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    return macd_line, signal_line, macd_line - signal_line


def bollinger(series, period=20, std=2):
    ma = series.rolling(period).mean()
    sd = series.rolling(period).std()
    return ma + sd*std, ma, ma - sd*std


def adx(df, period=14):
    high, low, close = df['High'], df['Low'], df['Close']
    plus_dm = high.diff().clip(lower=0)
    minus_dm = (-low.diff()).clip(lower=0)
    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low - close.shift()).abs()
    ], axis=1).max(axis=1)
    atr = tr.rolling(period).mean()
    plus_di = 100 * (plus_dm.rolling(period).mean() / atr)
    minus_di = 100 * (minus_dm.rolling(period).mean() / atr)
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di)
    return dx.rolling(period).mean()


# ============================================================
# DATA LOADING (cached για ταχύτητα)
# ============================================================

@st.cache_data(ttl=600)  # cache για 10 λεπτά
def load_data(ticker: str, period: str = "1y"):
    df = yf.download(ticker, period=period, progress=False, auto_adjust=True)
    if df.empty:
        return None
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df


@st.cache_data(ttl=3600)
def get_sp500_tickers():
    try:
        url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
        tables = pd.read_html(url)
        return [t.replace('.', '-') for t in tables[0]['Symbol'].tolist()]
    except Exception:
        return ['AAPL','MSFT','NVDA','GOOGL','AMZN','META','TSLA','BRK-B','LLY','AVGO',
                'JPM','WMT','V','XOM','UNH','MA','PG','JNJ','HD','COST']


@st.cache_data(ttl=3600)
def get_nasdaq100_tickers():
    try:
        tables = pd.read_html("https://en.wikipedia.org/wiki/Nasdaq-100")
        for tbl in tables:
            for col in ['Ticker', 'Symbol']:
                if col in tbl.columns:
                    return [t.replace('.', '-') for t in tbl[col].tolist()]
    except Exception:
        pass
    return ['AAPL','MSFT','NVDA','GOOGL','AMZN','META','TSLA','AVGO','COST','NFLX']


def get_dow_tickers():
    return ['AAPL','AMGN','AXP','BA','CAT','CRM','CSCO','CVX','DIS','GS',
            'HD','HON','IBM','JNJ','JPM','KO','MCD','MMM','MRK','MSFT',
            'NKE','PG','SHW','TRV','UNH','V','VZ','WMT','NVDA','AMZN']


# ============================================================
# CORE ANALYSIS (επιστρέφει dict με όλα τα δεδομένα)
# ============================================================

def full_analysis(df: pd.DataFrame) -> dict:
    close = df['Close']
    volume = df['Volume']
    current = close.iloc[-1]

    # === MOMENTUM ===
    timeframes = {'1w': 5, '1m': 21, '3m': 63, '6m': 126, '12m': 252}
    returns = {}
    for name, days in timeframes.items():
        if len(close) > days:
            returns[name] = ((current - close.iloc[-days-1]) / close.iloc[-days-1]) * 100

    weights = {'1w': 0.1, '1m': 0.2, '3m': 0.3, '6m': 0.2, '12m': 0.2}
    mom_score = sum(np.clip(returns.get(tf, 0) * 5, -100, 100) * w
                     for tf, w in weights.items() if tf in returns)

    # === TECHNICAL ===
    ma20 = close.rolling(20).mean().iloc[-1]
    ma50 = close.rolling(50).mean().iloc[-1]
    ma200 = close.rolling(200).mean().iloc[-1] if len(close) >= 200 else None
    rsi_val = rsi(close).iloc[-1]
    macd_line, signal_line, _ = macd(close)
    macd_bullish = macd_line.iloc[-1] > signal_line.iloc[-1]
    macd_cross = (macd_line.iloc[-2] <= signal_line.iloc[-2]) and macd_bullish
    bb_upper, bb_mid, bb_lower = bollinger(close)
    bb_pos = ((current - bb_lower.iloc[-1]) / (bb_upper.iloc[-1] - bb_lower.iloc[-1])) * 100
    adx_val = adx(df).iloc[-1]
    vol_ratio = volume.iloc[-1] / volume.tail(20).mean()

    # Signals
    signals = []
    bull = bear = 0

    def add_signal(text, direction, weight=1):
        nonlocal bull, bear
        signals.append((text, direction))
        if direction == 'bullish': bull += weight
        elif direction == 'bearish': bear += weight

    add_signal(f"Price ${current:.2f} vs MA20 ${ma20:.2f}",
               "bullish" if current > ma20 else "bearish")
    add_signal(f"Price vs MA50 ${ma50:.2f}",
               "bullish" if current > ma50 else "bearish")
    if ma200 is not None:
        add_signal(f"Price vs MA200 ${ma200:.2f}",
                   "bullish" if current > ma200 else "bearish")
        add_signal(f"MA50 vs MA200",
                   "bullish" if ma50 > ma200 else "bearish")

    if rsi_val > 70:
        add_signal(f"RSI {rsi_val:.1f} (overbought)", "bearish")
    elif rsi_val < 30:
        add_signal(f"RSI {rsi_val:.1f} (oversold)", "bullish")
    elif rsi_val > 50:
        add_signal(f"RSI {rsi_val:.1f} (bullish momentum)", "bullish")
    else:
        add_signal(f"RSI {rsi_val:.1f} (bearish momentum)", "bearish")

    if macd_cross:
        add_signal("MACD fresh bullish crossover 🔔", "bullish", weight=2)
    elif macd_bullish:
        add_signal("MACD above signal line", "bullish")
    else:
        add_signal("MACD below signal line", "bearish")

    add_signal(f"ADX {adx_val:.1f} ({'strong trend' if adx_val > 25 else 'weak trend'})",
               "neutral")
    if vol_ratio > 1.5:
        add_signal(f"Volume {vol_ratio:.1f}x avg", "bullish")
    elif vol_ratio < 0.5:
        add_signal(f"Volume {vol_ratio:.1f}x avg", "bearish")

    tech_score = ((bull - bear) / max(bull + bear, 1)) * 100
    combined = (mom_score + tech_score) / 2
    agreement = (mom_score > 0 and tech_score > 0) or (mom_score < 0 and tech_score < 0)

    if combined > 50 and agreement: verdict = "STRONG BUY"
    elif combined > 20 and agreement: verdict = "BUY"
    elif combined < -50 and agreement: verdict = "STRONG SELL"
    elif combined < -20 and agreement: verdict = "SELL"
    elif not agreement and abs(combined) > 10: verdict = "MIXED SIGNALS"
    else: verdict = "NEUTRAL"

    return {
        'current': current, 'returns': returns,
        'mom_score': mom_score, 'tech_score': tech_score, 'combined': combined,
        'verdict': verdict, 'agreement': agreement,
        'ma20': ma20, 'ma50': ma50, 'ma200': ma200,
        'rsi': rsi_val, 'adx': adx_val, 'bb_pos': bb_pos, 'vol_ratio': vol_ratio,
        'macd_cross': macd_cross, 'signals': signals,
        'bb_upper': bb_upper, 'bb_mid': bb_mid, 'bb_lower': bb_lower,
        'macd_line': macd_line, 'signal_line': signal_line,
    }


# ============================================================
# CHARTS
# ============================================================

def create_chart(df, analysis):
    fig = make_subplots(
        rows=3, cols=1, shared_xaxes=True,
        vertical_spacing=0.03, row_heights=[0.6, 0.2, 0.2],
        subplot_titles=("Price + MAs + Bollinger", "RSI", "MACD")
    )

    # Price
    fig.add_trace(go.Candlestick(
        x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'],
        name='Price', showlegend=False
    ), row=1, col=1)

    ma20 = df['Close'].rolling(20).mean()
    ma50 = df['Close'].rolling(50).mean()
    ma200 = df['Close'].rolling(200).mean()

    fig.add_trace(go.Scatter(x=df.index, y=ma20, name='MA20', line=dict(color='orange', width=1)), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=ma50, name='MA50', line=dict(color='blue', width=1)), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=ma200, name='MA200', line=dict(color='red', width=1.5)), row=1, col=1)

    fig.add_trace(go.Scatter(x=df.index, y=analysis['bb_upper'], name='BB Upper',
                              line=dict(color='gray', width=1, dash='dot'), showlegend=False), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=analysis['bb_lower'], name='BB Lower',
                              line=dict(color='gray', width=1, dash='dot'),
                              fill='tonexty', fillcolor='rgba(128,128,128,0.05)', showlegend=False), row=1, col=1)

    # RSI
    rsi_series = rsi(df['Close'])
    fig.add_trace(go.Scatter(x=df.index, y=rsi_series, name='RSI', line=dict(color='purple')), row=2, col=1)
    fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
    fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)

    # MACD
    fig.add_trace(go.Scatter(x=df.index, y=analysis['macd_line'], name='MACD', line=dict(color='blue')), row=3, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=analysis['signal_line'], name='Signal', line=dict(color='orange')), row=3, col=1)
    hist = analysis['macd_line'] - analysis['signal_line']
    colors = ['green' if v > 0 else 'red' for v in hist]
    fig.add_trace(go.Bar(x=df.index, y=hist, name='Histogram', marker_color=colors, showlegend=False), row=3, col=1)

    fig.update_layout(height=700, xaxis_rangeslider_visible=False, hovermode='x unified',
                       margin=dict(t=40, b=20))
    return fig


# ============================================================
# SCREENER (parallel)
# ============================================================

def screen_ticker_lite(ticker: str):
    """Lightweight analysis για screening (faster)."""
    try:
        df = yf.download(ticker, period="1y", progress=False, auto_adjust=True, threads=False)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        if df.empty or len(df) < 252:
            return None

        close = df['Close']
        volume = df['Volume']
        current = close.iloc[-1]
        if current < 5 or volume.tail(20).mean() < 100_000:
            return None

        a = full_analysis(df)
        return {
            'ticker': ticker, 'price': current,
            'verdict': a['verdict'], 'combined': a['combined'],
            'mom_score': a['mom_score'], 'tech_score': a['tech_score'],
            'rsi': a['rsi'], 'adx': a['adx'],
            'ret_1m': a['returns'].get('1m', 0),
            'ret_3m': a['returns'].get('3m', 0),
            'ret_12m': a['returns'].get('12m', 0),
            'macd_cross': a['macd_cross'],
            'above_ma200': current > a['ma200'] if a['ma200'] else False,
        }
    except Exception:
        return None


def run_screener(tickers, progress_callback=None, parallel=10):
    results = []
    with ThreadPoolExecutor(max_workers=parallel) as executor:
        futures = {executor.submit(screen_ticker_lite, t): t for t in tickers}
        done = 0
        for future in as_completed(futures):
            done += 1
            result = future.result()
            if result:
                results.append(result)
            if progress_callback:
                progress_callback(done / len(tickers))
    return results


# ============================================================
# BACKTESTING
# ============================================================

def compute_signals_vectorized(df):
    close = df['Close']
    volume = df['Volume']
    ma20 = close.rolling(20).mean()
    ma50 = close.rolling(50).mean()
    ma200 = close.rolling(200).mean()
    rsi_val = rsi(close)
    macd_line, signal_line, _ = macd(close)
    bb_upper, _, bb_lower = bollinger(close)
    bb_pos = ((close - bb_lower) / (bb_upper - bb_lower)) * 100
    vol_ratio = volume / volume.rolling(20).mean()

    ret_1w = (close / close.shift(5) - 1) * 100
    ret_1m = (close / close.shift(21) - 1) * 100
    ret_3m = (close / close.shift(63) - 1) * 100
    ret_6m = (close / close.shift(126) - 1) * 100
    ret_12m = (close / close.shift(252) - 1) * 100

    mom_score = (np.clip(ret_1w*5,-100,100)*0.1 + np.clip(ret_1m*5,-100,100)*0.2 +
                 np.clip(ret_3m*5,-100,100)*0.3 + np.clip(ret_6m*5,-100,100)*0.2 +
                 np.clip(ret_12m*5,-100,100)*0.2)

    bull = ((close > ma20).astype(int) + (close > ma50).astype(int) +
            (close > ma200).astype(int) + (ma50 > ma200).astype(int) +
            ((rsi_val > 50) & (rsi_val <= 70)).astype(int) + (rsi_val < 30).astype(int) +
            (macd_line > signal_line).astype(int) +
            ((macd_line > signal_line) & ~(macd_line.shift() > signal_line.shift())).astype(int) +
            (vol_ratio > 1.5).astype(int) + (bb_pos < 0).astype(int))

    bear = ((close <= ma20).astype(int) + (close <= ma50).astype(int) +
            (close <= ma200).astype(int) + (ma50 <= ma200).astype(int) +
            (rsi_val > 70).astype(int) + ((rsi_val <= 50) & (rsi_val >= 30)).astype(int) +
            (macd_line <= signal_line).astype(int) + (vol_ratio < 0.5).astype(int) +
            (bb_pos > 100).astype(int))

    tech_score = ((bull - bear) / (bull + bear).replace(0, 1)) * 100
    combined = (mom_score + tech_score) / 2
    agreement = ((mom_score > 0) & (tech_score > 0)) | ((mom_score < 0) & (tech_score < 0))

    signal = pd.Series('NEUTRAL', index=close.index)
    signal[(combined > 50) & agreement] = 'STRONG_BUY'
    signal[(combined > 20) & (combined <= 50) & agreement] = 'BUY'
    return pd.DataFrame({'close': close, 'combined': combined, 'signal': signal})


def run_backtest(df, target_pct=10, horizon=60):
    sig_df = compute_signals_vectorized(df)
    is_signal = sig_df['signal'].isin(['BUY', 'STRONG_BUY'])
    fresh = is_signal & ~is_signal.shift(1).fillna(False)
    signal_dates = sig_df.index[fresh]

    trades = []
    close = sig_df['close']
    for sig_date in signal_dates:
        idx = sig_df.index.get_loc(sig_date)
        if idx + 1 >= len(close): continue
        entry = close.iloc[idx + 1]
        end_idx = min(idx + 1 + horizon, len(close) - 1)
        future = close.iloc[idx + 1:end_idx + 1]
        max_gain = ((future.max() - entry) / entry) * 100
        final = ((future.iloc[-1] - entry) / entry) * 100
        trades.append({'date': sig_date, 'max_gain': max_gain, 'final_ret': final,
                        'hit_target': max_gain >= target_pct})

    if not trades:
        return None

    df_trades = pd.DataFrame(trades)
    # Baseline
    baseline_rets = []
    baseline_hits = 0
    for i in range(len(close) - horizon - 1):
        entry = close.iloc[i + 1]
        future = close.iloc[i+1:i+1+horizon]
        ret = (future.iloc[-1] - entry) / entry * 100
        max_g = (future.max() - entry) / entry * 100
        baseline_rets.append(ret)
        if max_g >= target_pct:
            baseline_hits += 1

    return {
        'n_trades': len(trades),
        'win_rate': df_trades['hit_target'].mean() * 100,
        'avg_return': df_trades['final_ret'].mean(),
        'baseline_win_rate': baseline_hits / len(baseline_rets) * 100 if baseline_rets else 0,
        'baseline_avg': np.mean(baseline_rets) if baseline_rets else 0,
        'trades': df_trades,
    }


# ============================================================
# UI
# ============================================================

st.title("📈 Stock Analysis Dashboard")
st.caption("Momentum + Technical Analysis | Educational tool, not financial advice")

tab1, tab2, tab3 = st.tabs(["🔍 Single Stock", "🎯 Screener", "📊 Backtest"])

# ============================================================
# TAB 1: SINGLE STOCK
# ============================================================
with tab1:
    col1, col2 = st.columns([1, 3])
    with col1:
        ticker_input = st.text_input("Ticker", value="AAPL", key="single_ticker").upper()
        period = st.selectbox("Period", ["6mo", "1y", "2y", "5y"], index=1)
        analyze_btn = st.button("Analyze", type="primary", use_container_width=True)

    if analyze_btn or ticker_input:
        with st.spinner(f"Loading {ticker_input}..."):
            df = load_data(ticker_input, period)

        if df is None or df.empty:
            st.error(f"Δεν βρέθηκαν δεδομένα για {ticker_input}")
        else:
            a = full_analysis(df)

            # Verdict banner
            verdict_color = {
                "STRONG BUY": "🟢", "BUY": "🟢", "NEUTRAL": "⚪",
                "MIXED SIGNALS": "🟡", "SELL": "🔴", "STRONG SELL": "🔴"
            }
            st.markdown(f"### {verdict_color.get(a['verdict'], '⚪')} **{a['verdict']}** — {ticker_input}")

            # Metrics
            c1, c2, c3, c4, c5 = st.columns(5)
            c1.metric("Price", f"${a['current']:.2f}")
            c2.metric("Combined Score", f"{a['combined']:+.1f}")
            c3.metric("Momentum", f"{a['mom_score']:+.1f}")
            c4.metric("Technical", f"{a['tech_score']:+.1f}")
            c5.metric("Agreement", "✅ YES" if a['agreement'] else "⚠️ NO")

            # Returns
            st.markdown("#### Returns")
            rcols = st.columns(len(a['returns']))
            for col, (tf, ret) in zip(rcols, a['returns'].items()):
                col.metric(tf, f"{ret:+.2f}%")

            # Chart
            st.plotly_chart(create_chart(df, a), use_container_width=True)

            # Signals
            st.markdown("#### Detailed Signals")
            sig_col1, sig_col2 = st.columns(2)
            for i, (text, direction) in enumerate(a['signals']):
                icon = {"bullish": "🟢", "bearish": "🔴", "neutral": "⚪"}[direction]
                target = sig_col1 if i % 2 == 0 else sig_col2
                target.write(f"{icon} {text}")


# ============================================================
# TAB 2: SCREENER
# ============================================================
with tab2:
    st.markdown("Σαρώνει ένα universe μετοχών και βρίσκει τις πιο bullish.")

    col1, col2, col3 = st.columns(3)
    universe = col1.selectbox("Universe", ["dow", "nasdaq100", "sp500"],
                                format_func=lambda x: {"dow": "Dow 30 (γρήγορο)",
                                                        "nasdaq100": "NASDAQ 100",
                                                        "sp500": "S&P 500 (αργό)"}[x])
    top_n = col2.number_input("Top N", min_value=5, max_value=50, value=20)
    min_score = col3.number_input("Min combined score", min_value=-100, max_value=100, value=20)

    if st.button("🚀 Run Screener", type="primary"):
        if universe == "sp500":
            tickers = get_sp500_tickers()
        elif universe == "nasdaq100":
            tickers = get_nasdaq100_tickers()
        else:
            tickers = get_dow_tickers()

        st.info(f"Σαρώνω {len(tickers)} μετοχές... ({30 if universe=='dow' else 90 if universe=='nasdaq100' else 180}s estimated)")
        progress = st.progress(0)
        status = st.empty()

        def update_progress(pct):
            progress.progress(pct)

        start = time.time()
        results = run_screener(tickers, progress_callback=update_progress, parallel=10)
        elapsed = time.time() - start

        progress.empty()
        status.success(f"✅ Σαρώθηκαν {len(results)} μετοχές σε {elapsed:.1f}s")

        if results:
            df_results = pd.DataFrame(results)
            df_filtered = df_results[df_results['combined'] >= min_score].sort_values('combined', ascending=False).head(top_n)

            if df_filtered.empty:
                st.warning(f"Καμία μετοχή με score >= {min_score}")
            else:
                # Format για display
                df_display = df_filtered.copy()
                df_display['price'] = df_display['price'].apply(lambda x: f"${x:.2f}")
                df_display['combined'] = df_display['combined'].apply(lambda x: f"{x:+.1f}")
                df_display['mom_score'] = df_display['mom_score'].apply(lambda x: f"{x:+.1f}")
                df_display['tech_score'] = df_display['tech_score'].apply(lambda x: f"{x:+.1f}")
                for col in ['ret_1m', 'ret_3m', 'ret_12m']:
                    df_display[col] = df_display[col].apply(lambda x: f"{x:+.1f}%")
                df_display['rsi'] = df_display['rsi'].apply(lambda x: f"{x:.1f}")
                df_display['macd_cross'] = df_display['macd_cross'].apply(lambda x: "🔔" if x else "")
                df_display['above_ma200'] = df_display['above_ma200'].apply(lambda x: "✓" if x else "")

                df_display = df_display[['ticker', 'price', 'verdict', 'combined',
                                          'mom_score', 'tech_score', 'rsi',
                                          'ret_1m', 'ret_3m', 'ret_12m',
                                          'macd_cross', 'above_ma200']]
                df_display.columns = ['Ticker', 'Price', 'Verdict', 'Combined',
                                       'Momentum', 'Technical', 'RSI',
                                       '1M %', '3M %', '12M %', 'MACD↑', 'MA200']

                st.dataframe(df_display, use_container_width=True, hide_index=True)

                # Distribution
                st.markdown("#### Universe Distribution")
                verdict_counts = df_results['verdict'].value_counts()
                cols = st.columns(len(verdict_counts))
                for col, (v, count) in zip(cols, verdict_counts.items()):
                    col.metric(v, count, f"{count/len(df_results)*100:.0f}%")

                # CSV export
                csv = df_filtered.to_csv(index=False)
                st.download_button("📥 Download CSV", csv, "screener_results.csv", "text/csv")


# ============================================================
# TAB 3: BACKTEST
# ============================================================
with tab3:
    st.markdown("Δοκίμασε αν η στρατηγική έχει edge πάνω από random entry.")

    col1, col2, col3, col4 = st.columns(4)
    bt_ticker = col1.text_input("Ticker", value="AAPL", key="bt_ticker").upper()
    bt_years = col2.selectbox("History", [5, 10, 15, 20], index=1)
    bt_target = col3.number_input("Target %", min_value=1, max_value=50, value=10)
    bt_horizon = col4.number_input("Horizon (days)", min_value=5, max_value=252, value=60)

    if st.button("📊 Run Backtest", type="primary"):
        with st.spinner(f"Backtesting {bt_ticker}..."):
            df = load_data(bt_ticker, period=f"{bt_years}y")
            if df is None or df.empty or len(df) < 300:
                st.error("Ανεπαρκή δεδομένα")
            else:
                result = run_backtest(df, target_pct=bt_target, horizon=bt_horizon)

                if result is None:
                    st.warning("Δεν εμφανίστηκαν σήματα στο διάστημα")
                else:
                    c1, c2, c3, c4 = st.columns(4)
                    c1.metric("Signals", result['n_trades'])
                    c2.metric("Win Rate", f"{result['win_rate']:.1f}%",
                              f"{result['win_rate'] - result['baseline_win_rate']:+.1f}pp vs random")
                    c3.metric("Avg Return", f"{result['avg_return']:+.2f}%",
                              f"{result['avg_return'] - result['baseline_avg']:+.2f}pp vs random")
                    c4.metric("Target Hit", f"{result['win_rate']:.0f}%")

                    edge_wr = result['win_rate'] - result['baseline_win_rate']
                    edge_ret = result['avg_return'] - result['baseline_avg']

                    if edge_wr > 5 and edge_ret > 1:
                        st.success("✅ Η στρατηγική έχει θετικό edge πάνω από random")
                    elif edge_wr > 0 and edge_ret > 0:
                        st.warning("🟡 Οριακό edge — πιθανότατα θόρυβος")
                    else:
                        st.error("❌ Δεν υπάρχει edge — η τυχαία είσοδος είναι το ίδιο/καλύτερη")

                    st.markdown("#### Comparison")
                    comp_df = pd.DataFrame({
                        'Metric': ['Win Rate', 'Avg Return'],
                        'Strategy': [f"{result['win_rate']:.1f}%", f"{result['avg_return']:+.2f}%"],
                        'Random Baseline': [f"{result['baseline_win_rate']:.1f}%",
                                            f"{result['baseline_avg']:+.2f}%"],
                    })
                    st.dataframe(comp_df, use_container_width=True, hide_index=True)


st.markdown("---")
st.caption("⚠️ Educational tool only. Past performance does not predict future results. "
            "Survivorship bias, no transaction costs included. Always do your own research.")
