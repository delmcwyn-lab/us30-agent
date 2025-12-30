import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from scipy import stats
from GoogleNews import GoogleNews
import google.generativeai as genai
import time

# --- PAGE SETUP ---
st.set_page_config(page_title="US30 Pro Agent", page_icon="💎", layout="centered")
st.title("💎 US30 Pro AI Agent")
st.caption("Upgrades: Analyst Ratings + ATR Position Sizing")

# --- SIDEBAR: SETTINGS ---
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # 1. API Key & Model
    API_KEY = st.text_input("Gemini API Key", type="password")
    
    valid_models = []
    if API_KEY:
        try:
            genai.configure(api_key=API_KEY)
            for m in genai.list_models():
                if 'generateContent' in m.supported_generation_methods:
                    valid_models.append(m.name)
        except: pass
    if not valid_models:
        valid_models = ["models/gemini-1.5-flash", "models/gemini-pro"]
    
    selected_model = st.selectbox("AI Model", valid_models, index=0)
    
    st.divider()
    
    # 2. Risk Management Inputs (Upgrade 3)
    st.subheader("💰 Risk Management")
    RISK_AMOUNT = st.number_input("Risk per Trade ($)", min_value=10.0, value=100.0, step=10.0)
    ATR_MULTIPLIER = st.slider("Stop Loss Width (x ATR)", 1.0, 3.0, 1.5, help="1.5x ATR is standard for intraday.")
    
    st.divider()
    
    # 3. Filters
    st.subheader("🔍 Filters")
    MIN_R2 = st.slider("Min Smoothness (R²)", 0.0, 1.0, 0.5)
    MIN_RVOL = st.slider("Min Relative Vol (RVOL)", 0.0, 3.0, 0.5)

# --- LOGIC ---
US30_TICKERS = [
    "MMM", "AXP", "AMGN", "AMZN", "AAPL", "BA", "CAT", "CVX", "CSCO", "KO",
    "DIS", "GS", "HD", "HON", "IBM", "JNJ", "JPM", "MCD", "MRK", "MSFT",
    "NKE", "NVDA", "PG", "CRM", "SHW", "TRV", "UNH", "VZ", "V", "WMT"
]

def get_top_performers(tickers, top_n=10):
    status_text = st.empty()
    status_text.text(f"Scanning US30 for Top {top_n} Performers...")
    performance = []
    progress_bar = st.progress(0)
    
    for i, ticker in enumerate(tickers):
        try:
            for _ in range(3):
                try:
                    df = yf.download(ticker, period="1y", progress=False, auto_adjust=True)
                    break
                except: time.sleep(1)
            
            if isinstance(df.columns, pd.MultiIndex):
                try: df = df.xs(ticker, level=1, axis=1)
                except: pass
            
            if len(df) > 200:
                close_col = df['Close'] if 'Close' in df.columns else df.iloc[:, 0]
                start_p = float(close_col.iloc[0])
                end_p = float(close_col.iloc[-1])
                perf_pct = ((end_p - start_p) / start_p) * 100
                performance.append((ticker, float(perf_pct)))
        except: continue
        progress_bar.progress((i + 1) / len(tickers))

    status_text.empty()
    progress_bar.empty()
    sorted_perf = sorted(performance, key=lambda x: x[1], reverse=True)
    return [x[0] for x in sorted_perf[:top_n]]

def get_advanced_technicals(ticker):
    """
    Fetches Price, R2, RVOL, Analyst Data, and Calculates ATR (12).
    """
    try:
        # 1. Analyst Data (Upgrade 1)
        t_obj = yf.Ticker(ticker)
        info = t_obj.info
        analyst_target = info.get('targetMeanPrice', 'N/A')
        analyst_rating = info.get('recommendationKey', 'N/A').upper().replace("_", " ")

        # 2. Intraday Data (1 Hour)
        df = yf.download(ticker, period="1mo", interval="1h", progress=False, auto_adjust=True)
        if not df.empty: df = df.iloc[:-1] # Drop incomplete candle
        if df.empty or len(df) < 20: return None
        
        if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
        
        # 3. Calculate ATR (12 Period) - Upgrade 3
        # True Range Calculation
        high = df['High']
        low = df['Low']
        close = df['Close']
        prev_close = close.shift(1)
        
        tr1 = high - low
        tr2 = (high - prev_close).abs()
        tr3 = (low - prev_close).abs()
        
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr_12 = tr.rolling(window=12).mean().iloc[-1]

        # 4. Standard Technicals (R2 & RVOL)
        recent = df.tail(14)
        y = recent['Close'].values.flatten()
        x = np.arange(len(y))
        slope, _, r_value, _, _ = stats.linregress(x, y)
        r_squared = r_value ** 2
        
        last_vol = df['Volume'].iloc[-1]
        sma_vol = df['Volume'].rolling(20).mean().iloc[-1]
        rvol = last_vol / sma_vol if sma_vol > 0 else 0.0
        
        return {
            "ticker": ticker,
            "price": round(float(y[-1]), 2),
            "trend": "UP" if slope > 0 else "DOWN",
            "r2": round(r_squared, 3),
            "rvol": round(float(rvol), 2),
            "analyst_rating": analyst_rating,
            "analyst_target": analyst_target,
            "atr": round(float(atr_12), 2)
        }
    except Exception as e:
        return None

def get_better_news(ticker):
    try:
        googlenews = GoogleNews(period='1d')
        googlenews.search(f"{ticker} stock news")
        results = googlenews.result()
        headlines = [f"- {item['title']}" for item in results[:3]]
        return "\n".join(headlines) if headlines else "No significant news."
    except: return "News fetch error."

def ask_gemini(shortlist_data, api_key, model_name, risk_amt, atr_mult):
    genai.configure(api_key=api_key)
    
    stocks_text = ""
    for s in shortlist_data:
        stocks_text += f"""
        TICKER: {s['ticker']} (Current Price: ${s['price']})
        - Trend: {s['trend']} | Smoothness(R2): {s['r2']} | RVOL: {s['rvol']}
        - ANALYST RATING: {s['analyst_rating']} | TARGET: ${s['analyst_target']}
        - ATR (VOLATILITY): ${s['atr']}
        - NEWS: {s['news']}
        ----------------------------------------------
        """

    prompt = f"""
    You are an Elite Risk-Managed Trading Algorithm.
    User Risk Budget: ${risk_amt} per trade.
    Stop Loss Rule: Must be exactly {atr_mult}x ATR away from Entry.
    
    MISSION:
    1. Select the SINGLE best stock based on Technical Trend + Analyst Confluence + News.
    2. Calculate the Position Size (Shares) using this formula:
       Shares = {risk_amt} / (Entry Price - Stop Loss Price)
    
    DATA:
    {stocks_text}
    
    OUTPUT FORMAT (Strictly follow):
    
    ### 🏆 SELECTED STOCK: [Ticker]
    **Rationale:** [Why Technicals + Analysts + News align]
    **Analyst Verdict:** [Rating & Target]
    
    ### 📉 TRADE PLAN (1-Hour Chart)
    * **Action:** [BUY / SELL]
    * **Entry:** [Current Price or slight pullback]
    * **Stop Loss:** [Calculate: Entry - ({atr_mult} * ATR)]
    * **Take Profit:** [Set at 1.5x or 2x reward relative to risk]
    
    ### ⚖️ POSITION SIZING (Risk: ${risk_amt})
    * **ATR Value:** $[Value]
    * **Stop Distance:** $[Value of {atr_mult} * ATR]
    * **QUANTITY:** [Calculated Number] Shares/Lots
    """
    
    try:
        model = genai.GenerativeModel(model_name)
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Error: {e}"

# --- MAIN APP UI ---
if st.button("🚀 Generate Pro Analysis"):
    if not API_KEY:
        st.error("Please enter your API Key in the sidebar first!")
    else:
        # 1. SCAN
        top_list = get_top_performers(US30_TICKERS)
        
        # 2. PROCESS & FILTER
        data_list = []
        status = st.empty()
        status.text("Calculating ATR, Analyst Ratings, and Technicals...")
        
        for t in top_list:
            d = get_advanced_technicals(t)
            if d: data_list.append(d)
        status.empty()
        
        # Show Dashboard
        if data_list:
            df_display = pd.DataFrame(data_list)[['ticker', 'rvol', 'r2', 'atr', 'analyst_rating']]
            st.subheader("📊 Market Data")
            st.dataframe(df_display.sort_values(by='rvol', ascending=False), use_container_width=True)

            # 3. SHORTLIST
            shortlist = []
            for d in data_list:
                if d['r2'] > MIN_R2 and d['rvol'] > MIN_RVOL:
                    d['news'] = get_better_news(d['ticker'])
                    shortlist.append(d)
            
            st.success(f"Found {len(shortlist)} stocks passing filters.")
            
            # 4. GEMINI DECISION
            if shortlist:
                st.subheader("🤖 AI Trade Recommendation")
                with st.spinner(f"Gemini is sizing your trade (Risk: ${RISK_AMOUNT})..."):
                    result = ask_gemini(shortlist, API_KEY, selected_model, RISK_AMOUNT, ATR_MULTIPLIER)
                    st.markdown(result)
            else:
                st.warning("No stocks met the strict criteria. Try lowering filters.")
