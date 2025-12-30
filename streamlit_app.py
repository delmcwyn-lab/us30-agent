import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from scipy import stats
from GoogleNews import GoogleNews
import google.generativeai as genai
import time

# --- PAGE SETUP ---
st.set_page_config(page_title="US30 AI Agent", page_icon="📈", layout="centered")
st.title("📈 US30 AI Trading Agent")

# --- SIDEBAR: SETTINGS ---
with st.sidebar:
    st.header("⚙️ Settings")
    API_KEY = st.text_input("Gemini API Key", type="password")
    
    # --- NEW: AUTO-DETECT MODELS ---
    valid_models = []
    if API_KEY:
        try:
            genai.configure(api_key=API_KEY)
            # List all models that support generating content
            for m in genai.list_models():
                if 'generateContent' in m.supported_generation_methods:
                    valid_models.append(m.name)
        except:
            pass
            
    # Default to a safe bet if list is empty, otherwise let user pick
    if not valid_models:
        valid_models = ["models/gemini-1.5-flash", "models/gemini-pro"]
        
    selected_model = st.selectbox("🤖 AI Model", valid_models, index=0)
    
    st.divider()
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
            # Retry logic
            for _ in range(3):
                try:
                    df = yf.download(ticker, period="1y", progress=False, auto_adjust=True)
                    break
                except:
                    time.sleep(1)
            
            # Multi-index fix
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

def check_hourly_technicals(ticker):
    try:
        df = yf.download(ticker, period="1mo", interval="1h", progress=False, auto_adjust=True)
        # Drop incomplete candle
        if not df.empty: df = df.iloc[:-1] 
        
        if df.empty or len(df) < 20: return None
        if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
        
        recent = df.tail(14)
        if 'Close' not in df.columns or 'Volume' not in df.columns: return None
        
        y = recent['Close'].values.flatten()
        x = np.arange(len(y))
        slope, _, r_value, _, _ = stats.linregress(x, y)
        r_squared = r_value ** 2
        
        last_vol = df['Volume'].iloc[-1]
        sma_vol = df['Volume'].rolling(20).mean().iloc[-1]
        rvol = last_vol / sma_vol if sma_vol > 0 else 0.0
        
        return {
            "ticker": ticker,
            "r2": round(r_squared, 3),
            "rvol": round(float(rvol), 2),
            "trend": "UP" if slope > 0 else "DOWN",
            "price": round(float(y[-1]), 2)
        }
    except: return None

def get_better_news(ticker):
    try:
        googlenews = GoogleNews(period='1d')
        googlenews.search(f"{ticker} stock news")
        results = googlenews.result()
        headlines = [f"- {item['title']}" for item in results[:3]]
        return "\n".join(headlines) if headlines else "No significant news."
    except: return "News fetch error."

def ask_gemini(shortlist_data, api_key, model_name):
    genai.configure(api_key=api_key)
    stocks_text = ""
    for s in shortlist_data:
        stocks_text += f"TICKER: {s['ticker']} (${s['price']}) | Trend: {s['trend']} | R2: {s['r2']} | RVOL: {s['rvol']}\nNEWS: {s['news']}\n---\n"

    prompt = f"""
    You are an Elite Intraday Trading Algorithm.
    Analyze these stocks and pick the SINGLE best setup for a 1-hour trade.
    
    DATA:
    {stocks_text}
    
    OUTPUT FORMAT:
    SELECTED STOCK: [Ticker]
    RATIONALE: [One sentence reason]
    SENTIMENT: [Bullish/Bearish]
    
    TRADE PLAN:
    - ACTION: [BUY / SELL]
    - ENTRY: [Price]
    - STOP LOSS: [Price ~0.5% away]
    - TAKE PROFIT: [Price ~1.0% away]
    """
    
    try:
        # Use the user-selected model
        model = genai.GenerativeModel(model_name)
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Error: {e}"

# --- MAIN APP UI ---
if st.button("🚀 Generate Analysis"):
    if not API_KEY:
        st.error("Please enter your API Key in the sidebar first!")
    else:
        # 1. SCAN
        top_list = get_top_performers(US30_TICKERS)
        
        # 2. FILTER
        data_list = []
        for t in top_list:
            d = check_hourly_technicals(t)
            if d: data_list.append(d)
        
        # Show Dashboard Table
        if data_list:
            df_display = pd.DataFrame(data_list)[['ticker', 'rvol', 'r2', 'trend']]
            st.subheader("📊 Live Market Dashboard")
            st.dataframe(df_display.sort_values(by='rvol', ascending=False), use_container_width=True)

            # 3. AI SHORTLIST
            shortlist = []
            for d in data_list:
                if d['r2'] > MIN_R2 and d['rvol'] > MIN_RVOL:
                    d['news'] = get_better_news(d['ticker'])
                    shortlist.append(d)
            
            st.success(f"Found {len(shortlist)} stocks passing filters.")
            
            # 4. GEMINI DECISION
            if shortlist:
                st.subheader("🤖 Gemini Recommendation")
                with st.spinner(f"Asking {selected_model}..."):
                    result = ask_gemini(shortlist, API_KEY, selected_model)
                    st.markdown(result)
            else:
                st.warning("No stocks met the criteria (Try lowering the settings in the sidebar).")
