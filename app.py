
import streamlit as st
import pandas as pd
import yfinance as yf
import json
import os
import plotly.graph_objects as go
from datetime import datetime
import google.generativeai as genai

# --- CONFIGURATION ---
st.set_page_config(page_title="Portfolio Health Dashboard", layout="wide", initial_sidebar_state="expanded")

# Initialize Gemini
api_key = os.environ.get("API_KEY") or st.secrets.get("API_KEY")

if api_key:
    genai.configure(api_key=api_key)
else:
    st.error("⚠️ API_KEY not found. Please add 'API_KEY' to your secrets.")
    st.stop()

# --- STYLING ---
st.markdown("""
    <style>
    .main { background-color: #0f172a; }
    .stMetric { background-color: #1e293b !important; padding: 15px; border-radius: 10px; border: 1px solid #334155; }
    div[data-testid="stExpander"] { border: none !important; background: #1e293b !important; border-radius: 10px; }
    .stDataFrame { background: #1e293b; border-radius: 10px; border: 1px solid #334155; }
    </style>
    """, unsafe_allow_html=True)

# --- DATA PERSISTENCE ---
TARGETS_FILE = "targets.json"

def load_targets():
    if os.path.exists(TARGETS_FILE):
        try:
            with open(TARGETS_FILE, "r") as f:
                return json.load(f)
        except:
            return {}
    return {}

def save_targets(targets):
    with open(TARGETS_FILE, "w") as f:
        json.dump(targets, f)

# --- TICKER RESOLUTION ---
@st.cache_data(ttl=86400) 
def resolve_ticker_logic(symbol, name):
    if not symbol or pd.isna(symbol):
        return None
        
    for suffix in ["", ".DE"]:
        test_ticker = f"{symbol}{suffix}"
        try:
            t = yf.Ticker(test_ticker)
            hist = t.history(period="1y")
            if len(hist) > 20:
                return test_ticker
        except:
            continue
    
    try:
        model = genai.GenerativeModel('gemini-3-flash-preview')
        prompt = f"Return ONLY the primary US ticker symbol for '{name}'. No text, just the symbol."
        response = model.generate_content(prompt)
        fallback = response.text.strip().upper().replace("$", "")
        if 0 < len(fallback) < 10:
            return fallback
    except:
        pass
    
    return f"{symbol}.DE"

@st.cache_data(ttl=3600)
def fetch_stock_data(ticker_symbol):
    try:
        ticker = yf.Ticker(ticker_symbol)
        df = ticker.history(period="2y")
        if df.empty: return None
        
        current_price = df['Close'].iloc[-1]
        df['SMA51'] = df['Close'].rolling(window=51).mean()
        df['SMA120'] = df['Close'].rolling(window=120).mean()
        df['SMA200'] = df['Close'].rolling(window=200).mean()
        df['SMA250'] = df['Close'].rolling(window=250).mean()
        
        info = ticker.info
        pe = info.get("trailingPE") or info.get("forwardPE")
            
        return {
            "df": df,
            "current_price": current_price,
            "pe": pe,
            "sma51": df['SMA51'].iloc[-1],
            "sma120": df['SMA120'].iloc[-1],
            "sma200": df['SMA200'].iloc[-1],
            "sma250": df['SMA250'].iloc[-1]
        }
    except:
        return None

def main():
    st.title("📈 Portfolio Health Dashboard")
    
    with st.sidebar:
        st.header("1. Data Ingestion")
        uploaded_file = st.file_uploader("Upload 'Portfolio Performance' CSV", type="csv")
        targets = load_targets()

    if uploaded_file:
        # Step 1: Delimiter detection
        content = uploaded_file.read().decode('utf-8-sig') # Handle potential BOM
        delimiter = ';' if content.count(';') > content.count(',') else ','
        uploaded_file.seek(0)
        
        df_raw = pd.read_csv(uploaded_file, sep=delimiter, engine='python')
        
        with st.sidebar.expander("🛠 Column Mapping", expanded=True):
            cols = df_raw.columns.tolist()
            def find_match(options, keywords):
                for k in keywords:
                    for opt in options:
                        if k.lower() in str(opt).lower(): return options.index(opt)
                return 0

            col_symbol = st.selectbox("Symbol Column", cols, index=find_match(cols, ["ticker", "symbol", "isin"]))
            col_shares = st.selectbox("Shares Column", cols, index=find_match(cols, ["shares", "stück", "anzahl"]))
            col_name = st.selectbox("Name Column", cols, index=find_match(cols, ["name", "bezeichnung", "security"]))
            col_type = st.selectbox("Type Column", cols, index=find_match(cols, ["type", "typ", "vorgang"]))

        def clean_num(val):
            if pd.isna(val): return 0.0
            if isinstance(val, (int, float)): return float(val)
            # String cleaning for German format: "1.234,56"
            s = str(val).strip().replace('\xa0', '').replace(' ', '')
            if not s: return 0.0
            
            # Logic: If both . and , exist, . is thousands. If only , exists, it's decimal.
            if ',' in s:
                if '.' in s: s = s.replace('.', '')
                s = s.replace(',', '.')
            
            try: return float(s)
            except: return 0.0

        df_raw[col_shares] = df_raw[col_shares].apply(clean_num)
        
        with st.expander("🔍 Mapping Debugger"):
            debug_df = df_raw[[col_name, col_symbol, col_shares, col_type]].copy()
            def debug_action(t_type):
                t = str(t_type).lower()
                if any(k in t for k in ["buy", "kauf", "einlieferung", "delivery", "inbound"]): return "BUY"
                if any(k in t for k in ["sell", "verkauf", "auslieferung", "outbound"]): return "SELL"
                return "IGNORE"
            debug_df["Action"] = debug_df[col_type].apply(debug_action)
            st.dataframe(debug_df, use_container_width=True)

        unique_assets = df_raw[col_symbol].dropna().unique()
        holdings_data = []
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for i, sym in enumerate(unique_assets):
            asset_rows = df_raw[df_raw[col_symbol] == sym]
            name = str(asset_rows[col_name].iloc[0])
            status_text.text(f"Processing: {name}")
            
            qty = 0
            for _, row in asset_rows.iterrows():
                t_type = str(row[col_type]).lower()
                s = abs(row[col_shares])
                is_buy = any(k in t_type for k in ["buy", "kauf", "einlieferung", "delivery", "inbound"])
                is_sell = any(k in t_type for k in ["sell", "verkauf", "auslieferung", "outbound"])
                if is_buy: qty += s
                elif is_sell: qty -= s
            
            if qty > 0.001:
                resolved = resolve_ticker_logic(sym, name)
                if resolved:
                    data = fetch_stock_data(resolved)
                    if data:
                        holdings_data.append({
                            "Symbol": sym, "Resolved": resolved, "Name": name, "Qty": qty,
                            "Price": data["current_price"], "Value": qty * data["current_price"],
                            "SMA200": data["sma200"], "SMA250": data["sma250"], 
                            "df": data["df"]
                        })
            progress_bar.progress((i + 1) / len(unique_assets))
        
        status_text.empty()
        progress_bar.empty()

        if not holdings_data:
            st.error("❌ No active holdings found. Check the 'Shares' values in the debugger above.")
            return

        portfolio_df = pd.DataFrame(holdings_data)
        total_value = portfolio_df["Value"].sum()
        
        # UI Metrics
        m1, m2, m3 = st.columns(3)
        m1.metric("Total Value", f"€{total_value:,.2f}")
        m2.metric("Bullish", len(portfolio_df[portfolio_df["Price"] > portfolio_df["SMA200"]]))
        m3.metric("Correction", len(portfolio_df[portfolio_df["Price"] < portfolio_df["SMA200"]]))

        # Main Table
        display_df = portfolio_df.copy()
        display_df["Weight %"] = (display_df["Value"] / total_value) * 100
        display_df["Target %"] = display_df["Symbol"].map(new_targets := targets).fillna(0)
        
        def calc_action(row):
            t_pct = row["Target %"]
            if t_pct <= 0: return "-"
            diff_val = ((t_pct / 100) * total_value) - row["Value"]
            shares = diff_val / row["Price"]
            return f"{'BUY' if shares > 0 else 'SELL'} {abs(shares):.1f}"

        display_df["Action"] = display_df.apply(calc_action, axis=1)
        st.dataframe(display_df[["Name", "Resolved", "Price", "Weight %", "Target %", "Action"]], use_container_width=True)

        st.subheader("Historical Analysis")
        choice = st.selectbox("Select Asset", portfolio_df["Name"])
        c_row = portfolio_df[portfolio_df["Name"] == choice].iloc[0]
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=c_row["df"].index, y=c_row["df"]['Close'], name='Price', line=dict(color='white')))
        fig.add_trace(go.Scatter(x=c_row["df"].index, y=c_row["df"]['SMA200'], name='SMA 200', line=dict(color='red', dash='dot')))
        fig.update_layout(template="plotly_dark", height=400)
        st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    main()
