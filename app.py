
import streamlit as st
import pandas as pd
import yfinance as yf
import json
import os
import plotly.graph_objects as go
from datetime import datetime, timedelta
from @google/genai import GoogleGenAI

# --- CONFIGURATION ---
st.set_page_config(page_title="Portfolio Health Dashboard", layout="wide", initial_sidebar_state="expanded")

# Initialize Gemini for Ticker Resolution
ai = GoogleGenAI(apiKey=os.environ.get("API_KEY"))

# --- STYLING ---
st.markdown("""
    <style>
    .main { background-color: #0f172a; }
    .stMetric { background-color: #1e293b; padding: 15px; border-radius: 10px; border: 1px solid #334155; }
    div[data-testid="stExpander"] { border: none; background: #1e293b; border-radius: 10px; }
    </style>
    """, unsafe_allow_html=True)

# --- DATA PERSISTENCE ---
TARGETS_FILE = "targets.json"

def load_targets():
    if os.path.exists(TARGETS_FILE):
        with open(TARGETS_FILE, "r") as f:
            return json.load(f)
    return {}

def save_targets(targets):
    with open(TARGETS_FILE, "w") as f:
        json.dump(targets, f)

# --- SMART TICKER LOGIC ---
@st.cache_data(ttl=3600)
def resolve_ticker_logic(symbol, name):
    """
    Step 1: Try raw symbol
    Step 2: Try .DE suffix
    Step 3: Use Gemini to find US fallback if historical data is lacking
    """
    # Attempt German Ticker first
    for suffix in ["", ".DE"]:
        test_ticker = f"{symbol}{suffix}"
        t = yf.Ticker(test_ticker)
        hist = t.history(period="1mo")
        if not hist.empty:
            # Check for 5-year history requirement (approx 1260 trading days)
            long_hist = t.history(period="5y")
            if len(long_hist) > 850:
                return test_ticker
    
    # Step 3: US Fallback Search via Gemini
    try:
        prompt = f"Find the primary US stock ticker (NYSE or NASDAQ) for the company '{name}'. Return ONLY the ticker symbol."
        response = ai.models.generateContent(
            model='gemini-3-pro-preview',
            contents=prompt
        )
        fallback = response.text.strip()
        if fallback:
            return fallback
    except:
        pass
    
    return f"{symbol}.DE" # Final default

@st.cache_data(ttl=3600)
def fetch_stock_data(ticker_symbol):
    try:
        ticker = yf.Ticker(ticker_symbol)
        # Fetch 5 years to ensure we have enough data for SMA 850
        df = ticker.history(period="5y")
        if df.empty:
            return None
        
        info = ticker.info
        current_price = df['Close'].iloc[-1]
        
        # Calculate SMAs
        df['SMA51'] = df['Close'].rolling(window=51).mean()
        df['SMA120'] = df['Close'].rolling(window=120).mean()
        df['SMA200'] = df['Close'].rolling(window=200).mean()
        df['SMA250'] = df['Close'].rolling(window=250).mean()
        df['SMA850'] = df['Close'].rolling(window=850).mean()
        
        return {
            "df": df,
            "current_price": current_price,
            "pe": info.get("trailingPE", None),
            "sma200": df['SMA200'].iloc[-1],
            "sma250": df['SMA250'].iloc[-1],
            "sma850": df['SMA850'].iloc[-1]
        }
    except Exception as e:
        return None

# --- UI COMPONENTS ---
def main():
    st.title("📈 Portfolio Health Dashboard")
    
    # Sidebar: File Upload
    with st.sidebar:
        st.header("1. Data Ingestion")
        uploaded_file = st.file_uploader("Upload 'Portfolio Performance' CSV", type="csv")
        
        st.header("2. Target Allocation")
        targets = load_targets()

    if uploaded_file:
        df_raw = pd.read_csv(uploaded_file, sep=None, engine='python')
        
        # Column Mapping (Flexible)
        with st.sidebar.expander("Column Mapping"):
            col_symbol = st.selectbox("Symbol Column", df_raw.columns, index=df_raw.columns.get_loc("Symbol") if "Symbol" in df_raw.columns else 0)
            col_shares = st.selectbox("Shares Column", df_raw.columns, index=df_raw.columns.get_loc("Shares") if "Shares" in df_raw.columns else 0)
            col_name = st.selectbox("Name Column", df_raw.columns, index=df_raw.columns.get_loc("Name") if "Name" in df_raw.columns else 0)
            col_type = st.selectbox("Type Column", df_raw.columns, index=df_raw.columns.get_loc("Type") if "Type" in df_raw.columns else 0)

        # Process Transactions
        df_raw[col_shares] = pd.to_numeric(df_raw[col_shares].astype(str).str.replace(',', '.'), errors='coerce').fillna(0)
        
        # Calculate Quantities
        holdings_summary = []
        unique_assets = df_raw[col_symbol].unique()
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        holdings_data = []
        
        for i, sym in enumerate(unique_assets):
            if pd.isna(sym) or sym == "": continue
            
            asset_rows = df_raw[df_raw[col_symbol] == sym]
            name = asset_rows[col_name].iloc[0]
            
            # Simple Buy/Sell logic
            qty = 0
            for _, row in asset_rows.iterrows():
                t_type = str(row[col_type]).lower()
                if "buy" in t_type or "kauf" in t_type:
                    qty += row[col_shares]
                elif "sell" in t_type or "verkauf" in t_type:
                    qty -= row[col_shares]
            
            if qty > 0.001:
                status_text.text(f"Resolving {name}...")
                resolved = resolve_ticker_logic(sym, name)
                data = fetch_stock_data(resolved)
                
                if data:
                    holdings_data.append({
                        "Symbol": sym,
                        "Resolved": resolved,
                        "Name": name,
                        "Qty": qty,
                        "Price": data["current_price"],
                        "Value": qty * data["current_price"],
                        "P/E": data["pe"],
                        "SMA200": data["sma200"],
                        "SMA250": data["sma250"],
                        "SMA850": data["sma850"],
                        "df": data["df"]
                    })
            
            progress_bar.progress((i + 1) / len(unique_assets))
        
        status_text.empty()
        progress_bar.empty()
        
        if not holdings_data:
            st.error("No active holdings found in CSV.")
            return

        portfolio_df = pd.DataFrame(holdings_data)
        total_value = portfolio_df["Value"].sum()
        
        # Global Warnings
        below_250 = portfolio_df[portfolio_df["Price"] < portfolio_df["SMA250"]]
        if not below_250.empty:
            st.warning(f"⚠️ Warning: {len(below_250)} assets are currently trading below their SMA 250!")

        # Targets Sidebar Input
        with st.sidebar:
            st.subheader("Set Targets (%)")
            new_targets = {}
            for _, row in portfolio_df.iterrows():
                val = st.number_input(f"{row['Name']}", 0, 100, targets.get(row['Symbol'], 0), key=f"target_{row['Symbol']}")
                new_targets[row['Symbol']] = val
            if st.button("Save Targets"):
                save_targets(new_targets)
                st.success("Targets Saved!")

        # Main Table
        cols = st.columns(3)
        cols[0].metric("Total Portfolio Value", f"€{total_value:,.2f}")
        cols[1].metric("Positions", len(portfolio_df))
        cols[2].metric("Bearish Assets", len(below_250))

        # Analysis Table
        display_df = portfolio_df.copy()
        display_df["Weight %"] = (display_df["Value"] / total_value) * 100
        display_df["Target %"] = display_df["Symbol"].map(new_targets).fillna(0)
        
        def get_status(row):
            if row["Price"] < row["SMA850"]: return "🟢 Historical Opportunity"
            if row["Price"] < row["SMA200"]: return "🔴 Bearish Warning"
            return "✅ Healthy"

        display_df["Status"] = display_df.apply(get_status, axis=1)
        
        def get_action(row):
            target_val = (row["Target %"] / 100) * total_value
            diff = target_val - row["Value"]
            shares = diff / row["Price"]
            if abs(shares) < 0.1 or row["Target %"] == 0: return "Keep"
            return f"{'Buy' if shares > 0 else 'Sell'} {abs(shares):.2f} shares"

        display_df["Action Needed"] = display_df.apply(get_action, axis=1)

        st.subheader("Portfolio Summary")
        st.dataframe(
            display_df[["Name", "Resolved", "Price", "P/E", "Weight %", "Target %", "Status", "Action Needed"]],
            use_container_width=True,
            hide_index=True
        )

        # Detail View
        st.divider()
        selected_stock_name = st.selectbox("Select Asset for Detailed View", portfolio_df["Name"])
        stock_row = portfolio_df[portfolio_df["Name"] == selected_stock_name].iloc[0]
        
        # Charting
        df_plot = stock_row["df"]
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df_plot.index, y=df_plot['Close'], name='Price', line=dict(color='#10b981', width=2)))
        fig.add_trace(go.Scatter(x=df_plot.index, y=df_plot['SMA200'], name='SMA 200', line=dict(color='#f43f5e', width=1, dash='dash')))
        fig.add_trace(go.Scatter(x=df_plot.index, y=df_plot['SMA850'], name='SMA 850', line=dict(color='#f59e0b', width=1.5)))
        
        fig.update_layout(
            title=f"{stock_row['Name']} ({stock_row['Resolved']}) - 5 Year Analysis",
            template="plotly_dark",
            xaxis_title="Date",
            yaxis_title="Price (€)",
            height=500,
            margin=dict(l=0, r=0, t=40, b=0),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(fig, use_container_width=True)

    else:
        st.info("Please upload your Transactions CSV to begin.")
        st.image("https://images.unsplash.com/photo-1611974717482-5828aa9c4464?auto=format&fit=crop&q=80&w=2070", use_column_width=True)

if __name__ == "__main__":
    main()
