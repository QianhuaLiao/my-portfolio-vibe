
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
    for suffix in ["", ".DE"]:
        test_ticker = f"{symbol}{suffix}"
        try:
            t = yf.Ticker(test_ticker)
            hist = t.history(period="2y")
            if len(hist) > 250:
                return test_ticker
        except:
            continue
    
    try:
        model = genai.GenerativeModel('gemini-3-flash-preview')
        prompt = f"Return ONLY the primary US ticker symbol for '{name}'. No text, just the symbol."
        response = model.generate_content(prompt)
        fallback = response.text.strip().upper().replace("$", "")
        if 0 < len(fallback) < 8:
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
        
        # SMAs Calculation
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
        # Load data once
        df_raw = pd.read_csv(uploaded_file, sep=None, engine='python')
        
        st.info("💡 **Step 1:** Verify or update column mappings in the sidebar if your data doesn't appear below.")
        
        with st.expander("🔍 View Raw CSV Data (First 5 rows)"):
            st.dataframe(df_raw.head(), use_container_width=True)

        with st.sidebar.expander("🛠 Column Mapping", expanded=True):
            cols = df_raw.columns.tolist()
            # Try to find best matches automatically
            def find_match(options, keywords):
                for k in keywords:
                    for opt in options:
                        if k.lower() in opt.lower(): return options.index(opt)
                return 0

            col_symbol = st.selectbox("Symbol Column", cols, index=find_match(cols, ["symbol", "ticker", "isin"]))
            col_shares = st.selectbox("Shares Column", cols, index=find_match(cols, ["shares", "stück", "anzahl"]))
            col_name = st.selectbox("Name Column", cols, index=find_match(cols, ["name", "bezeichnung", "security"]))
            col_type = st.selectbox("Type Column", cols, index=find_match(cols, ["type", "typ", "vorgang"]))

        def clean_num(val):
            if pd.isna(val): return 0.0
            if isinstance(val, str):
                val = val.replace('\xa0', '').replace(' ', '')
                if ',' in val and '.' in val:
                    val = val.replace('.', '').replace(',', '.')
                elif ',' in val:
                    val = val.replace(',', '.')
            try: return float(val)
            except: return 0.0

        # Process logic
        df_raw[col_shares] = df_raw[col_shares].apply(clean_num)
        unique_assets = df_raw[col_symbol].dropna().unique()
        holdings_data = []
        
        progress_container = st.container()
        with progress_container:
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            for i, sym in enumerate(unique_assets):
                asset_rows = df_raw[df_raw[col_symbol] == sym]
                name = str(asset_rows[col_name].iloc[0])
                status_text.text(f"Analyzing {name}...")
                
                qty = 0
                for _, row in asset_rows.iterrows():
                    t_type = str(row[col_type]).lower()
                    s = row[col_shares]
                    # Logic for Buy/Sell
                    if any(k in t_type for k in ["buy", "kauf", "einlieferung", "delivery"]): qty += s
                    elif any(k in t_type for k in ["sell", "verkauf", "auslieferung"]): qty -= s
                
                if qty > 0.001:
                    resolved = resolve_ticker_logic(sym, name)
                    data = fetch_stock_data(resolved)
                    if data:
                        holdings_data.append({
                            "Symbol": sym, "Resolved": resolved, "Name": name, "Qty": qty,
                            "Price": data["current_price"], "Value": qty * data["current_price"],
                            "P/E": data["pe"], 
                            "SMA51": data["sma51"], "SMA120": data["sma120"],
                            "SMA200": data["sma200"], "SMA250": data["sma250"], 
                            "df": data["df"]
                        })
                progress_bar.progress((i + 1) / len(unique_assets))
        
        status_text.empty()
        progress_bar.empty()

        if not holdings_data:
            st.error(f"❌ **No active holdings found.** Please check if your **Shares** column ({col_shares}) and **Type** column ({col_type}) are mapped correctly in the sidebar.")
            return

        portfolio_df = pd.DataFrame(holdings_data)
        total_value = portfolio_df["Value"].sum()
        
        # Dashboard Content
        below_250 = portfolio_df[portfolio_df["Price"] < portfolio_df["SMA250"]]
        if not below_250.empty:
            st.warning(f"🚨 {len(below_250)} assets are trading below their 250-day average.")

        with st.sidebar:
            st.subheader("2. Target Allocation")
            new_targets = {}
            for _, row in portfolio_df.iterrows():
                new_targets[row['Symbol']] = st.number_input(f"{row['Name']} (%)", 0, 100, targets.get(row['Symbol'], 0), key=f"t_{row['Symbol']}")
            if st.button("Save Targets"):
                save_targets(new_targets)
                st.success("Saved!")

        # Key Metrics
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Portfolio Value", f"€{total_value:,.2f}")
        m2.metric("Positions", len(portfolio_df))
        m3.metric("Bullish (>200d)", len(portfolio_df[portfolio_df["Price"] > portfolio_df["SMA200"]]))
        m4.metric("Correction (<120d)", len(portfolio_df[portfolio_df["Price"] < portfolio_df["SMA120"]]))

        # Main Table
        display_df = portfolio_df.copy()
        display_df["Weight %"] = (display_df["Value"] / total_value) * 100
        display_df["Target %"] = display_df["Symbol"].map(new_targets).fillna(0)
        
        def calc_status(row):
            if row["Price"] < row["SMA250"]: return "🔴 Bearish (250)"
            if row["Price"] < row["SMA200"]: return "🟠 Bearish (200)"
            if row["Price"] < row["SMA120"]: return "🟡 Correction"
            return "🟢 Bullish"

        def calc_action(row):
            t_pct = row["Target %"]
            if t_pct <= 0: return "Set Target"
            target_val = (t_pct / 100) * total_value
            diff_eur = target_val - row["Value"]
            shares = diff_eur / row["Price"]
            if abs(shares) < 0.1: return "Hold"
            return f"{'BUY' if shares > 0 else 'SELL'} {abs(shares):.1f}"

        display_df["Status"] = display_df.apply(calc_status, axis=1)
        display_df["Action"] = display_df.apply(calc_action, axis=1)

        st.subheader("Current Holdings Overview")
        st.dataframe(
            display_df[["Name", "Price", "P/E", "Weight %", "Target %", "Status", "Action"]].sort_values("Weight %", ascending=False), 
            use_container_width=True, hide_index=True
        )

        # Charts Section
        st.divider()
        st.subheader("Deep Dive Analysis")
        choice = st.selectbox("Select Asset to Visualize", portfolio_df["Name"])
        chart_row = portfolio_df[portfolio_df["Name"] == choice].iloc[0]
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=chart_row["df"].index, y=chart_row["df"]['Close'], name='Price', line=dict(color='#ffffff', width=2.5)))
        fig.add_trace(go.Scatter(x=chart_row["df"].index, y=chart_row["df"]['SMA51'], name='SMA 51', line=dict(color='#60a5fa', width=1)))
        fig.add_trace(go.Scatter(x=chart_row["df"].index, y=chart_row["df"]['SMA120'], name='SMA 120', line=dict(color='#c084fc', width=1)))
        fig.add_trace(go.Scatter(x=chart_row["df"].index, y=chart_row["df"]['SMA200'], name='SMA 200', line=dict(color='#f43f5e', width=1.5, dash='dot')))
        fig.add_trace(go.Scatter(x=chart_row["df"].index, y=chart_row["df"]['SMA250'], name='SMA 250', line=dict(color='#fbbf24', width=1.5)))
        
        fig.update_layout(
            template="plotly_dark", height=500, margin=dict(l=10,r=10,b=10,t=40),
            hovermode="x unified", legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("👋 **Welcome!** Please upload your Portfolio Performance CSV in the sidebar to analyze your health and rebalancing needs.")

if __name__ == "__main__":
    main()
