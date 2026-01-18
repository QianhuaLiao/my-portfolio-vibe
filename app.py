
import streamlit as st
import pandas as pd
import yfinance as yf
import json
import os
import plotly.graph_objects as go
from datetime import datetime
import google.generativeai as genai
import re

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
def is_isin(s):
    return bool(re.match(r"^[A-Z]{2}[A-Z0-9]{9}[0-9]$", str(s).strip().upper()))

@st.cache_data(ttl=86400) 
def resolve_ticker_logic(symbol, name):
    if not symbol or pd.isna(symbol):
        return None
    
    clean_sym = str(symbol).strip().upper()
    
    # If it's an ISIN, yfinance won't work. We MUST convert it via AI or search.
    if is_isin(clean_sym):
        try:
            model = genai.GenerativeModel('gemini-3-flash-preview')
            prompt = f"Convert the ISIN '{clean_sym}' (Company: {name}) to its primary Yahoo Finance ticker symbol. Return ONLY the symbol (e.g. AAPL or BAS.DE). No extra text."
            response = model.generate_content(prompt)
            return response.text.strip().upper().replace("$", "")
        except:
            return None

    # Try standard suffixes for German markets if not an ISIN
    for suffix in ["", ".DE", ".F"]:
        test_ticker = f"{clean_sym}{suffix}"
        try:
            t = yf.Ticker(test_ticker)
            hist = t.history(period="5d")
            if not hist.empty:
                return test_ticker
        except:
            continue
    
    # AI Fallback for US Tickers if German ones fail
    try:
        model = genai.GenerativeModel('gemini-3-flash-preview')
        prompt = f"Find the primary Yahoo Finance ticker for '{name}'. Prefer US ticker if history is longer. Return ONLY the symbol. No text."
        response = model.generate_content(prompt)
        fallback = response.text.strip().upper().replace("$", "")
        if 0 < len(fallback) < 12:
            return fallback
    except:
        pass
    
    return f"{clean_sym}.DE"

@st.cache_data(ttl=3600)
def fetch_stock_data(ticker_symbol):
    try:
        ticker = yf.Ticker(ticker_symbol)
        df = ticker.history(period="2y")
        if df.empty: 
            return None
        
        current_price = df['Close'].iloc[-1]
        df['SMA200'] = df['Close'].rolling(window=200).mean()
        df['SMA250'] = df['Close'].rolling(window=250).mean()
        
        return {
            "df": df,
            "current_price": current_price,
            "sma200": df['SMA200'].iloc[-1],
            "sma250": df['SMA250'].iloc[-1]
        }
    except Exception as e:
        print(f"Error fetching {ticker_symbol}: {e}")
        return None

def main():
    st.title("📈 Portfolio Health Dashboard")
    
    with st.sidebar:
        st.header("1. Data Ingestion")
        uploaded_file = st.file_uploader("Upload 'Portfolio Performance' CSV", type="csv")
        targets = load_targets()

    if uploaded_file:
        content = uploaded_file.read().decode('utf-8-sig')
        delimiter = ';' if content.count(';') > content.count(',') else ','
        uploaded_file.seek(0)
        
        df_raw = pd.read_csv(uploaded_file, sep=delimiter, engine='python')
        
        with st.sidebar.expander("🛠 Column Mapping", expanded=False):
            cols = df_raw.columns.tolist()
            def find_match(options, keywords):
                for k in keywords:
                    for opt in options:
                        if k.lower() in str(opt).lower(): return options.index(opt)
                return 0

            col_symbol = st.selectbox("Symbol/ISIN Column", cols, index=find_match(cols, ["ticker", "symbol", "isin"]))
            col_shares = st.selectbox("Shares Column", cols, index=find_match(cols, ["shares", "stück", "anzahl"]))
            col_name = st.selectbox("Name Column", cols, index=find_match(cols, ["name", "bezeichnung", "security"]))
            col_type = st.selectbox("Type Column", cols, index=find_match(cols, ["type", "typ", "vorgang"]))

        def clean_num(val):
            if pd.isna(val): return 0.0
            if isinstance(val, (int, float)): return float(val)
            s = str(val).strip().replace('\xa0', '').replace(' ', '')
            if not s: return 0.0
            if ',' in s:
                if '.' in s: s = s.replace('.', '')
                s = s.replace(',', '.')
            try: return float(s)
            except: return 0.0

        df_raw[col_shares] = df_raw[col_shares].apply(clean_num)
        
        def interpret_action(t_type):
            t = str(t_type).strip().lower()
            if any(k in t for k in ["buy", "kauf", "einlieferung", "delivery", "inbound"]): return 1
            if any(k in t for k in ["sell", "verkauf", "auslieferung", "outbound"]): return -1
            return 0

        df_raw['_multiplier'] = df_raw[col_type].apply(interpret_action)
        df_raw['_net_shares'] = df_raw[col_shares] * df_raw['_multiplier']

        summary_calc = df_raw.groupby(col_symbol).agg({
            col_name: 'first',
            col_type: lambda x: list(set([str(i).strip() for i in x])),
            col_shares: 'sum',
            '_net_shares': 'sum'
        }).reset_index()
        summary_calc.columns = ['Symbol', 'Name', 'Types', 'Abs Shares', 'Net Position']

        holdings_to_process = summary_calc[summary_calc['Net Position'] > 0.001]
        
        if holdings_to_process.empty:
            st.error("❌ No active holdings found (Net Position ≤ 0).")
            st.dataframe(summary_calc)
            return

        holdings_data = []
        resolution_log = []
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for i, (idx, row) in enumerate(holdings_to_process.iterrows()):
            sym = row['Symbol']
            name = row['Name']
            qty = row['Net Position']
            
            status_text.text(f"Resolving: {name} ({sym})")
            resolved = resolve_ticker_logic(sym, name)
            
            fetch_status = "❌ Failed"
            if resolved:
                data = fetch_stock_data(resolved)
                if data:
                    fetch_status = "✅ Success"
                    holdings_data.append({
                        "Symbol": sym, "Resolved": resolved, "Name": name, "Qty": qty,
                        "Price": data["current_price"], "Value": qty * data["current_price"],
                        "SMA200": data["sma200"], "SMA250": data["sma250"], 
                        "df": data["df"]
                    })
            
            resolution_log.append({
                "Original": sym,
                "Resolved Ticker": resolved,
                "Name": name,
                "Status": fetch_status
            })
            progress_bar.progress((i + 1) / len(holdings_to_process))
        
        status_text.empty()
        progress_bar.empty()

        with st.expander("🔍 Ticker Resolution Debugger"):
            st.write("This log shows how your Symbols/ISINs were converted to Yahoo Tickers.")
            st.table(resolution_log)

        if not holdings_data:
            st.error("❌ Could not fetch market data for any resolved symbols. See the 'Ticker Resolution Debugger' above.")
            return

        portfolio_df = pd.DataFrame(holdings_data)
        total_value = portfolio_df["Value"].sum()
        
        m1, m2, m3 = st.columns(3)
        m1.metric("Total Portfolio Value", f"€{total_value:,.2f}")
        m2.metric("Bullish Assets", len(portfolio_df[portfolio_df["Price"] > portfolio_df["SMA200"]]))
        m3.metric("Bearish Assets", len(portfolio_df[portfolio_df["Price"] < portfolio_df["SMA200"]]))

        # Rebalancing Table
        display_df = portfolio_df.copy()
        display_df["Weight %"] = (display_df["Value"] / total_value) * 100
        display_df["Target %"] = display_df["Symbol"].map(targets).fillna(0)
        
        def calc_action(r):
            t_pct = r["Target %"]
            if t_pct <= 0: return "-"
            diff = ((t_pct / 100) * total_value) - r["Value"]
            shares = diff / r["Price"]
            return f"{'BUY' if shares > 0 else 'SELL'} {abs(shares):.1f}"

        display_df["Action Needed"] = display_df.apply(calc_action, axis=1)
        st.dataframe(display_df[["Name", "Resolved", "Price", "Weight %", "Target %", "Action Needed"]], use_container_width=True, hide_index=True)

        # Persistence for targets
        with st.sidebar:
            st.header("2. Set Targets (%)")
            new_targets = {}
            for _, row in portfolio_df.iterrows():
                new_targets[row['Symbol']] = st.number_input(f"{row['Name']}", 0, 100, int(targets.get(row['Symbol'], 0)))
            if st.button("Save Target Allocations"):
                save_targets(new_targets)
                st.success("Targets Saved!")
                st.rerun()

        st.subheader("Interactive Chart")
        asset_choice = st.selectbox("Select Asset to Visualize", portfolio_df["Name"])
        c_row = portfolio_df[portfolio_df["Name"] == asset_choice].iloc[0]
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=c_row["df"].index, y=c_row["df"]['Close'], name='Price', line=dict(color='#60a5fa')))
        fig.add_trace(go.Scatter(x=c_row["df"].index, y=c_row["df"]['SMA200'], name='SMA 200', line=dict(color='#f43f5e', dash='dot')))
        fig.update_layout(template="plotly_dark", height=450, margin=dict(l=0,r=0,t=20,b=0))
        st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    main()
