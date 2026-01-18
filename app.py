
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
if "API_KEY" in os.environ:
    genai.configure(api_key=os.environ["API_KEY"])
else:
    st.error("API_KEY not found in environment variables. Please set it in Streamlit Secrets.")

# --- STYLING ---
st.markdown("""
    <style>
    .main { background-color: #0f172a; }
    .stMetric { background-color: #1e293b !important; padding: 15px; border-radius: 10px; border: 1px solid #334155; }
    div[data-testid="stExpander"] { border: none !important; background: #1e293b !important; border-radius: 10px; }
    footer {visibility: hidden;}
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

# --- SMART TICKER LOGIC ---
@st.cache_data(ttl=3600)
def resolve_ticker_logic(symbol, name):
    # Try direct and German suffixes
    for suffix in ["", ".DE"]:
        test_ticker = f"{symbol}{suffix}"
        try:
            t = yf.Ticker(test_ticker)
            hist = t.history(period="5y")
            if len(hist) > 850:
                return test_ticker
        except:
            continue
    
    # AI Fallback to US Markets
    try:
        model = genai.GenerativeModel('gemini-3-flash-preview')
        prompt = f"Find the primary US stock ticker (NYSE or NASDAQ) for the company '{name}'. Return ONLY the ticker symbol (e.g. AAPL)."
        response = model.generate_content(prompt)
        fallback = response.text.strip().upper()
        if fallback and len(fallback) < 10:
            return fallback
    except:
        pass
    
    return f"{symbol}.DE"

@st.cache_data(ttl=3600)
def fetch_stock_data(ticker_symbol):
    try:
        ticker = yf.Ticker(ticker_symbol)
        df = ticker.history(period="5y")
        if df.empty:
            return None
        
        current_price = df['Close'].iloc[-1]
        
        # SMAs
        df['SMA200'] = df['Close'].rolling(window=200).mean()
        df['SMA250'] = df['Close'].rolling(window=250).mean()
        df['SMA850'] = df['Close'].rolling(window=850).mean()
        
        try:
            pe = ticker.info.get("trailingPE", None)
        except:
            pe = None
            
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
        df_raw = pd.read_csv(uploaded_file, sep=None, engine='python')
        
        with st.sidebar.expander("Column Mapping"):
            cols = df_raw.columns.tolist()
            col_symbol = st.selectbox("Symbol/Ticker Column", cols, index=cols.index("Symbol") if "Symbol" in cols else 0)
            col_shares = st.selectbox("Shares Column", cols, index=cols.index("Shares") if "Shares" in cols else 0)
            col_name = st.selectbox("Name Column", cols, index=cols.index("Name") if "Name" in cols else 0)
            col_type = st.selectbox("Transaction Type Column", cols, index=cols.index("Type") if "Type" in cols else 0)

        def clean_num(val):
            if isinstance(val, str):
                return float(val.replace('.', '').replace(',', '.'))
            return float(val)

        df_raw[col_shares] = df_raw[col_shares].apply(clean_num)
        unique_assets = df_raw[col_symbol].dropna().unique()
        
        holdings_data = []
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for i, sym in enumerate(unique_assets):
            asset_rows = df_raw[df_raw[col_symbol] == sym]
            name = asset_rows[col_name].iloc[0]
            status_text.text(f"Fetching data for {name}...")
            
            qty = 0
            for _, row in asset_rows.iterrows():
                t_type = str(row[col_type]).lower()
                s = row[col_shares]
                if any(k in t_type for k in ["buy", "kauf", "einlieferung"]): qty += s
                elif any(k in t_type for k in ["sell", "verkauf", "auslieferung"]): qty -= s
            
            if qty > 0.01:
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
        
        progress_bar.empty()
        status_text.empty()

        if not holdings_data:
            st.error("No active holdings found.")
            return

        portfolio_df = pd.DataFrame(holdings_data)
        total_value = portfolio_df["Value"].sum()
        
        # Health Alert
        below_250 = portfolio_df[portfolio_df["Price"] < portfolio_df["SMA250"]]
        if not below_250.empty:
            st.warning(f"⚠️ {len(below_250)} assets are currently trading below their SMA 250 (Long-term Bearish).")

        with st.sidebar:
            st.subheader("Set Target Allocation (%)")
            new_targets = {}
            for _, row in portfolio_df.iterrows():
                new_targets[row['Symbol']] = st.number_input(row['Name'], 0, 100, targets.get(row['Symbol'], 0))
            if st.button("Save Targets"):
                save_targets(new_targets)
                st.success("Allocation saved to targets.json")

        # Top Level Metrics
        m1, m2, m3 = st.columns(3)
        m1.metric("Total Value", f"€{total_value:,.2f}")
        m2.metric("Assets", len(portfolio_df))
        m3.metric("Opps (Price < SMA850)", len(portfolio_df[portfolio_df["Price"] < portfolio_df["SMA850"]]))

        # Main Data Table
        display_df = portfolio_df.copy()
        display_df["Weight %"] = (display_df["Value"] / total_value) * 100
        display_df["Target %"] = display_df["Symbol"].map(new_targets).fillna(0)
        
        def get_status(row):
            if row["Price"] < row["SMA850"]: return "🟢 Deep Value"
            if row["Price"] < row["SMA200"]: return "🔴 Bearish"
            return "✅ Healthy"

        def get_action(row):
            target_val = (row["Target %"] / 100) * total_value
            diff = target_val - row["Value"]
            if row["Target %"] == 0: return "-"
            shares = diff / row["Price"]
            if abs(shares) < 0.1: return "Hold"
            return f"{'Buy' if shares > 0 else 'Sell'} {abs(shares):.1f}"

        display_df["Status"] = display_df.apply(get_status, axis=1)
        display_df["Action"] = display_df.apply(get_action, axis=1)

        st.subheader("Portfolio Health Breakdown")
        st.dataframe(
            display_df[["Name", "Resolved", "Price", "P/E", "Weight %", "Target %", "Status", "Action"]], 
            use_container_width=True, hide_index=True
        )

        # Detailed Charting
        st.divider()
        sel_name = st.selectbox("Select Asset to Visualize Trends", portfolio_df["Name"])
        row = portfolio_df[portfolio_df["Name"] == sel_name].iloc[0]
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=row["df"].index, y=row["df"]['Close'], name='Price', line=dict(color='#10b981', width=2)))
        fig.add_trace(go.Scatter(x=row["df"].index, y=row["df"]['SMA200'], name='SMA 200', line=dict(color='#f43f5e', width=1, dash='dash')))
        fig.add_trace(go.Scatter(x=row["df"].index, y=row["df"]['SMA850'], name='SMA 850', line=dict(color='#f59e0b', width=1.5)))
        
        fig.update_layout(
            title=f"{sel_name} ({row['Resolved']}) - Moving Average Analysis",
            template="plotly_dark", height=500, margin=dict(l=0, r=0, t=40, b=0)
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Upload your Portfolio Performance CSV to generate the dashboard.")

if __name__ == "__main__":
    main()
