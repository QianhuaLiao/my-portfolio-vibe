
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
    st.error("⚠️ API_KEY not found. Please add 'API_KEY = \"your_key\"' to your Streamlit Secrets.")
    st.stop()

# --- STYLING ---
st.markdown("""
    <style>
    .main { background-color: #0f172a; }
    .stMetric { background-color: #1e293b !important; padding: 15px; border-radius: 10px; border: 1px solid #334155; }
    div[data-testid="stExpander"] { border: none !important; background: #1e293b !important; border-radius: 10px; }
    .stDataFrame { background: #1e293b; border-radius: 10px; }
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

# --- OPTIMIZED TICKER LOGIC ---
@st.cache_data(ttl=86400) # Cache for 24 hours to save API calls
def resolve_ticker_logic(symbol, name):
    # Step 1: Check if it's likely a German ISIN or Ticker
    for suffix in ["", ".DE"]:
        test_ticker = f"{symbol}{suffix}"
        try:
            # fetch only 1 day to see if ticker exists quickly
            t = yf.Ticker(test_ticker)
            if not t.history(period="1d").empty:
                # Check history length only if ticker is valid
                hist = t.history(period="5y")
                if len(hist) > 800:
                    return test_ticker
        except:
            continue
    
    # Step 2: AI Fallback (Gemini 3 Flash is significantly faster)
    try:
        model = genai.GenerativeModel('gemini-3-flash-preview')
        prompt = f"Act as a financial data expert. Find the primary US stock ticker (NYSE/NASDAQ) for '{name}'. Return ONLY the symbol, nothing else."
        response = model.generate_content(prompt)
        fallback = response.text.strip().upper().replace("$", "")
        if fallback and len(fallback) < 8:
            return fallback
    except:
        pass
    
    return f"{symbol}.DE"

@st.cache_data(ttl=3600)
def fetch_stock_data(ticker_symbol):
    try:
        ticker = yf.Ticker(ticker_symbol)
        df = ticker.history(period="5y")
        if df.empty: return None
        
        current_price = df['Close'].iloc[-1]
        df['SMA200'] = df['Close'].rolling(window=200).mean()
        df['SMA250'] = df['Close'].rolling(window=250).mean()
        df['SMA850'] = df['Close'].rolling(window=850).mean()
        
        info = ticker.info
        pe = info.get("trailingPE") or info.get("forwardPE")
            
        return {
            "df": df,
            "current_price": current_price,
            "pe": pe,
            "sma200": df['SMA200'].iloc[-1],
            "sma250": df['SMA250'].iloc[-1],
            "sma850": df['SMA850'].iloc[-1]
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
        # Detect delimiter and load
        df_raw = pd.read_csv(uploaded_file, sep=None, engine='python')
        
        with st.sidebar.expander("Column Mapping"):
            cols = df_raw.columns.tolist()
            col_symbol = st.selectbox("Symbol Column", cols, index=cols.index("Symbol") if "Symbol" in cols else 0)
            col_shares = st.selectbox("Shares Column", cols, index=cols.index("Shares") if "Shares" in cols else 0)
            col_name = st.selectbox("Name Column", cols, index=cols.index("Name") if "Name" in cols else 0)
            col_type = st.selectbox("Type Column", cols, index=cols.index("Type") if "Type" in cols else 0)

        def clean_num(val):
            if pd.isna(val): return 0.0
            if isinstance(val, str):
                val = val.replace('\xa0', '').replace(' ', '')
                if ',' in val and '.' in val: # e.g. 1.234,56
                    val = val.replace('.', '').replace(',', '.')
                elif ',' in val: # e.g. 1234,56
                    val = val.replace(',', '.')
            try: return float(val)
            except: return 0.0

        df_raw[col_shares] = df_raw[col_shares].apply(clean_num)
        
        # Aggregate holdings
        unique_assets = df_raw[col_symbol].dropna().unique()
        holdings_data = []
        
        # Use a container for progress to keep UI clean
        progress_container = st.container()
        with progress_container:
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            for i, sym in enumerate(unique_assets):
                asset_rows = df_raw[df_raw[col_symbol] == sym]
                name = str(asset_rows[col_name].iloc[0])
                status_text.text(f"Analyzing {name} ({i+1}/{len(unique_assets)})...")
                
                qty = 0
                for _, row in asset_rows.iterrows():
                    t_type = str(row[col_type]).lower()
                    s = row[col_shares]
                    if any(k in t_type for k in ["buy", "kauf", "einlieferung"]): qty += s
                    elif any(k in t_type for k in ["sell", "verkauf", "auslieferung"]): qty -= s
                
                if qty > 0.001:
                    # Logic is slow because of multiple YF calls; results are cached for next run
                    resolved = resolve_ticker_logic(sym, name)
                    data = fetch_stock_data(resolved)
                    if data:
                        holdings_data.append({
                            "Symbol": sym, "Resolved": resolved, "Name": name, "Qty": qty,
                            "Price": data["current_price"], "Value": qty * data["current_price"],
                            "P/E": data["pe"], "SMA200": data["sma200"], "SMA250": data["sma250"],
                            "SMA850": data["sma850"], "df": data["df"]
                        })
                progress_bar.progress((i + 1) / len(unique_assets))
        
        status_text.empty()
        progress_bar.empty()

        if not holdings_data:
            st.error("No active holdings found in this CSV.")
            return

        portfolio_df = pd.DataFrame(holdings_data)
        total_value = portfolio_df["Value"].sum()
        
        # Global Alert
        below_250 = portfolio_df[portfolio_df["Price"] < portfolio_df["SMA250"]]
        if not below_250.empty:
            st.warning(f"⚠️ {len(below_250)} assets are trading below their 250-day average.")

        with st.sidebar:
            st.subheader("Target Allocation")
            new_targets = {}
            for _, row in portfolio_df.iterrows():
                new_targets[row['Symbol']] = st.number_input(f"{row['Name']} (%)", 0, 100, targets.get(row['Symbol'], 0))
            if st.button("Save & Update"):
                save_targets(new_targets)
                st.rerun()

        # Metrics
        c1, c2, c3 = st.columns(3)
        c1.metric("Total Value", f"€{total_value:,.2f}")
        c2.metric("Positions", len(portfolio_df))
        c3.metric("Deep Value Opps", len(portfolio_df[portfolio_df["Price"] < portfolio_df["SMA850"]]))

        # Table
        display_df = portfolio_df.copy()
        display_df["Weight %"] = (display_df["Value"] / total_value) * 100
        display_df["Target %"] = display_df["Symbol"].map(new_targets).fillna(0)
        
        def calc_status(row):
            if row["Price"] < row["SMA850"]: return "🟢 Opportunity"
            if row["Price"] < row["SMA200"]: return "🔴 Bearish"
            return "✅ OK"

        def calc_action(row):
            t_pct = row["Target %"]
            if t_pct <= 0: return "-"
            target_val = (t_pct / 100) * total_value
            diff_eur = target_val - row["Value"]
            shares = diff_eur / row["Price"]
            if abs(shares) < 0.1: return "Hold"
            return f"{'BUY' if shares > 0 else 'SELL'} {abs(shares):.1f}"

        display_df["Status"] = display_df.apply(calc_status, axis=1)
        display_df["Action"] = display_df.apply(calc_action, axis=1)

        st.dataframe(
            display_df[["Name", "Resolved", "Price", "P/E", "Weight %", "Target %", "Status", "Action"]].sort_values("Value", ascending=False), 
            use_container_width=True, hide_index=True
        )

        # Chart
        st.divider()
        choice = st.selectbox("View Historical Trends", portfolio_df["Name"])
        chart_row = portfolio_df[portfolio_df["Name"] == choice].iloc[0]
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=chart_row["df"].index, y=chart_row["df"]['Close'], name='Price', line=dict(color='#10b981')))
        fig.add_trace(go.Scatter(x=chart_row["df"].index, y=chart_row["df"]['SMA200'], name='SMA 200', line=dict(color='#f43f5e', dash='dot')))
        fig.add_trace(go.Scatter(x=chart_row["df"].index, y=chart_row["df"]['SMA850'], name='SMA 850', line=dict(color='#f59e0b')))
        
        fig.update_layout(template="plotly_dark", height=450, margin=dict(l=0,r=0,b=0,t=30), hovermode="x unified")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("👋 Welcome! Please upload your Portfolio Performance CSV to begin.")

if __name__ == "__main__":
    main()
