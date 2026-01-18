
import streamlit as st
import pandas as pd
import yfinance as yf
import json
import os
import plotly.graph_objects as go
from datetime import datetime, timedelta
import google.generativeai as genai
import re

# --- CONFIGURATION ---
st.set_page_config(page_title="Portfolio Health Dashboard", layout="wide", initial_sidebar_state="expanded")

# Initialize Gemini for Ticker Resolution
api_key = process.env.get("API_KEY")
if api_key:
    genai.configure(api_key=api_key)
else:
    st.error("⚠️ API_KEY environment variable not found.")
    st.stop()

# --- STYLING ---
st.markdown("""
    <style>
    .main { background-color: #0f172a; color: #f1f5f9; }
    .stMetric { background-color: #1e293b !important; padding: 20px; border-radius: 12px; border: 1px solid #334155; }
    div[data-testid="stExpander"] { border: none !important; background: #1e293b !important; border-radius: 12px; margin-bottom: 10px; }
    .stAlert { border-radius: 12px; }
    </style>
    """, unsafe_allow_html=True)

# --- DATA PERSISTENCE ---
TARGETS_FILE = "targets.json"

def load_targets():
    if os.path.exists(TARGETS_FILE):
        try:
            with open(TARGETS_FILE, "r") as f: return json.load(f)
        except: return {}
    return {}

def save_targets(targets):
    with open(TARGETS_FILE, "w") as f: json.dump(targets, f)

# --- SMART TICKER LOGIC ---
def is_isin(s):
    return bool(re.match(r"^[A-Z]{2}[A-Z0-9]{9}[0-9]$", str(s).strip().upper()))

@st.cache_data(ttl=86400)
def resolve_ticker(symbol, name):
    clean_sym = str(symbol).strip().upper()
    
    # Step 1: Try exact symbol
    try:
        t = yf.Ticker(clean_sym)
        hist = t.history(period="1mo")
        if not hist.empty: return clean_sym
    except: pass

    # Step 2: Try German Suffix
    if not is_isin(clean_sym):
        de_ticker = f"{clean_sym}.DE"
        try:
            t = yf.Ticker(de_ticker)
            hist = t.history(period="5y") # Check for 5y history
            if not hist.empty and len(hist) > 1000: # Approx 4+ years
                return de_ticker
        except: pass

    # Step 3: US Fallback / AI Search
    try:
        model = genai.GenerativeModel('gemini-3-pro-preview')
        prompt = f"Find the best Yahoo Finance ticker for '{name}' (Identifier: {symbol}). User is in Germany but wants a US-listed ticker (NYSE/NASDAQ) if the German one lacks 5+ years of history. Return ONLY the ticker symbol."
        response = model.generate_content(prompt)
        return response.text.strip().upper().replace("$", "")
    except:
        return f"{clean_sym}.DE" if not is_isin(clean_sym) else None

@st.cache_data(ttl=3600)
def get_market_data(ticker_symbol):
    try:
        t = yf.Ticker(ticker_symbol)
        # Fetch 5 years to support SMA 850 (approx 3.4 years of trading days)
        df = t.history(period="5y")
        if df.empty: return None
        
        info = t.info
        pe = info.get('trailingPE', 'N/A')
        
        # Calculate moving averages
        for window in [51, 120, 200, 250, 850]:
            df[f'SMA_{window}'] = df['Close'].rolling(window=window).mean()
            
        current_price = df['Close'].iloc[-1]
        
        return {
            "df": df,
            "price": current_price,
            "pe": pe,
            "sma200": df['SMA_200'].iloc[-1],
            "sma250": df['SMA_250'].iloc[-1],
            "sma850": df['SMA_850'].iloc[-1] if 'SMA_850' in df else None
        }
    except: return None

# --- APP LAYOUT ---
def main():
    st.title("🛡️ Portfolio Health Dashboard")
    targets = load_targets()

    with st.sidebar:
        st.header("Settings")
        uploaded_file = st.file_uploader("Upload Portfolio Performance CSV", type="csv")
        
    if not uploaded_file:
        st.info("Please upload your Portfolio Performance CSV to begin.")
        return

    # Ingestion
    content = uploaded_file.read().decode('utf-8-sig')
    delimiter = ';' if content.count(';') > content.count(',') else ','
    uploaded_file.seek(0)
    df_raw = pd.read_csv(uploaded_file, sep=delimiter)

    # Column Mapping
    cols = df_raw.columns.tolist()
    with st.sidebar.expander("Column Mapping"):
        c_sym = st.selectbox("Ticker/ISIN", cols, index=0)
        c_shares = st.selectbox("Shares", cols, index=min(4, len(cols)-1))
        c_name = st.selectbox("Name", cols, index=min(1, len(cols)-1))
        c_type = st.selectbox("Type", cols, index=min(3, len(cols)-1))

    # Parse Transactions
    def clean_val(v):
        if pd.isna(v): return 0.0
        s = str(v).replace('.', '').replace(',', '.')
        try: return float(s)
        except: return 0.0

    df_raw['Shares_Clean'] = df_raw[c_shares].apply(clean_val)
    df_raw['Multiplier'] = df_raw[c_type].apply(lambda x: -1 if 'sell' in str(x).lower() or 'verkauf' in str(x).lower() else 1)
    df_raw['Net_Shares'] = df_raw['Shares_Clean'] * df_raw['Multiplier']

    portfolio_sum = df_raw.groupby(c_sym).agg({c_name: 'first', 'Net_Shares': 'sum'}).reset_index()
    active_holdings = portfolio_sum[portfolio_sum['Net_Shares'] > 0.001]

    # Analysis
    holdings_data = []
    warnings_list = []
    
    progress = st.progress(0)
    for i, (idx, row) in enumerate(active_holdings.iterrows()):
        progress.progress((i + 1) / len(active_holdings))
        resolved = resolve_ticker(row[c_sym], row[c_name])
        m_data = get_market_data(resolved)
        
        if m_data:
            h_info = {
                "ID": row[c_sym],
                "Name": row[c_name],
                "Ticker": resolved,
                "Qty": row['Net_Shares'],
                "Price": m_data['price'],
                "Value": row['Net_Shares'] * m_data['price'],
                "PE": m_data['pe'],
                "SMA200": m_data['sma200'],
                "SMA250": m_data['sma250'],
                "SMA850": m_data['sma850'],
                "df": m_data['df']
            }
            holdings_data.append(h_info)
            
            # Warning System
            if m_data['price'] < m_data['sma250']:
                warnings_list.append(f"⚠️ **{row[c_name]}** breached SMA 250!")
            
    if not holdings_data:
        st.error("❌ Could not fetch market data for the resolved symbols.")
        return

    # Dashboard
    if warnings_list:
        for w in warnings_list: st.warning(w)

    df_final = pd.DataFrame(holdings_data)
    total_value = df_final['Value'].sum()

    st.subheader("Holdings Overview")
    
    # Target Inputs in Sidebar
    with st.sidebar:
        st.header("Allocation Targets")
        new_targets = {}
        for _, h in df_final.iterrows():
            new_targets[h['ID']] = st.number_input(f"{h['Name']} %", 0, 100, targets.get(h['ID'], 0))
        if st.button("Save Targets"):
            save_targets(new_targets)
            st.success("Targets updated!")

    # Display Table
    table_rows = []
    for _, h in df_final.iterrows():
        t_pct = new_targets.get(h['ID'], 0)
        curr_pct = (h['Value'] / total_value) * 100
        target_val = (t_pct / 100) * total_value
        diff = target_val - h['Value']
        
        status = "Normal"
        if h['Price'] < h['SMA200']: status = "Bearish Warning"
        if h['SMA850'] and h['Price'] < h['SMA850']: status = "Historical Opportunity"

        table_rows.append({
            "Name": h['Name'],
            "Ticker": h['Ticker'],
            "Price": f"€{h['Price']:.2f}",
            "P/E": h['PE'],
            "Weight %": f"{curr_pct:.1f}%",
            "Target %": f"{t_pct}%",
            "Status": status,
            "Action Needed": f"{'Buy' if diff > 0 else 'Trim'} €{abs(diff):,.0f}" if t_pct > 0 else "-"
        })

    st.table(pd.DataFrame(table_rows))

    # Visualization
    st.divider()
    selected_name = st.selectbox("Analyze History", df_final['Name'].tolist())
    s_h = df_final[df_final['Name'] == selected_name].iloc[0]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=s_h['df'].index, y=s_h['df']['Close'], name="Price", line=dict(color="#60a5fa", width=2)))
    fig.add_trace(go.Scatter(x=s_h['df'].index, y=s_h['df']['SMA_200'], name="SMA 200", line=dict(color="#f87171", dash="dash")))
    if s_h['SMA850']:
        fig.add_trace(go.Scatter(x=s_h['df'].index, y=s_h['df']['SMA_850'], name="SMA 850", line=dict(color="#fbbf24", width=2)))

    fig.update_layout(template="plotly_dark", height=500, margin=dict(l=0,r=0,b=0,t=40),
                      xaxis_title="Date", yaxis_title="Price (€)")
    st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    main()
