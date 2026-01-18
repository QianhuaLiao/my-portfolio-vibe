
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
            hist = t.history(period="1mo")
            if not hist.empty:
                return test_ticker
        except:
            continue
    
    try:
        model = genai.GenerativeModel('gemini-3-flash-preview')
        prompt = f"Return ONLY the primary US ticker symbol for '{name}'. No extra text."
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
        df['SMA200'] = df['Close'].rolling(window=200).mean()
        df['SMA250'] = df['Close'].rolling(window=250).mean()
        
        info = ticker.info
        pe = info.get("trailingPE") or info.get("forwardPE")
            
        return {
            "df": df,
            "current_price": current_price,
            "pe": pe,
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
        content = uploaded_file.read().decode('utf-8-sig')
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
            s = str(val).strip().replace('\xa0', '').replace(' ', '')
            if not s: return 0.0
            if ',' in s:
                if '.' in s: s = s.replace('.', '')
                s = s.replace(',', '.')
            try: return float(s)
            except: return 0.0

        df_raw[col_shares] = df_raw[col_shares].apply(clean_num)
        
        # Robust Action Detection with stripping
        def interpret_action(t_type):
            t = str(t_type).strip().lower()
            if any(k in t for k in ["buy", "kauf", "einlieferung", "delivery", "inbound"]): return 1
            if any(k in t for k in ["sell", "verkauf", "auslieferung", "outbound"]): return -1
            return 0

        df_raw['_multiplier'] = df_raw[col_type].apply(interpret_action)
        df_raw['_net_shares'] = df_raw[col_shares] * df_raw['_multiplier']

        # Summary for debugging
        summary_calc = df_raw.groupby(col_symbol).agg({
            col_name: 'first',
            col_type: lambda x: list(set([str(i).strip() for i in x])), # Show raw types seen
            col_shares: 'sum',
            '_net_shares': 'sum'
        }).reset_index()
        summary_calc.columns = ['Symbol', 'Name', 'Raw Types in CSV', 'Total Shares (Abs)', 'Calculated Net Position']

        with st.expander("🔍 Grouping & Net Position Debugger", expanded=True):
            st.write("If 'Calculated Net Position' is 0, verify that 'Raw Types in CSV' are being recognized as Buys/Sells.")
            st.dataframe(summary_calc, use_container_width=True)

        holdings_to_process = summary_calc[summary_calc['Calculated Net Position'] > 0.001]
        
        if holdings_to_process.empty:
            st.error("❌ No active holdings found (Net Position ≤ 0).")
            st.info("Check the 'Raw Types in CSV' column in the debugger above. Does it contain 'Buy' or 'Kauf'?")
            return

        holdings_data = []
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for i, (idx, row) in enumerate(holdings_to_process.iterrows()):
            sym = row['Symbol']
            name = row['Name']
            qty = row['Calculated Net Position']
            
            status_text.text(f"Fetching: {name}")
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
            progress_bar.progress((i + 1) / len(holdings_to_process))
        
        status_text.empty()
        progress_bar.empty()

        if not holdings_data:
            st.error("❌ Could not fetch market data for the resolved symbols.")
            return

        portfolio_df = pd.DataFrame(holdings_data)
        total_value = portfolio_df["Value"].sum()
        
        m1, m2, m3 = st.columns(3)
        m1.metric("Portfolio Value", f"€{total_value:,.2f}")
        m2.metric("Bullish (>200)", len(portfolio_df[portfolio_df["Price"] > portfolio_df["SMA200"]]))
        m3.metric("Bearish (<200)", len(portfolio_df[portfolio_df["Price"] < portfolio_df["SMA200"]]))

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

        st.subheader("Asset Chart")
        asset_choice = st.selectbox("Select Asset", portfolio_df["Name"])
        c_row = portfolio_df[portfolio_df["Name"] == asset_choice].iloc[0]
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=c_row["df"].index, y=c_row["df"]['Close'], name='Price', line=dict(color='#60a5fa')))
        fig.add_trace(go.Scatter(x=c_row["df"].index, y=c_row["df"]['SMA200'], name='SMA 200', line=dict(color='#f43f5e', dash='dot')))
        fig.update_layout(template="plotly_dark", height=450)
        st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    main()
