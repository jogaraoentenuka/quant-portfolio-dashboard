import yfinance as yf
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

# ---------------------------------------------------
# Page Config
# ---------------------------------------------------
st.set_page_config(page_title="Portfolio Analytics Dashboard", layout="wide")

st.title("📊 Level 2 Portfolio Analytics Dashboard")
st.markdown("Track performance, risk metrics, and drawdowns like a mini quant tool.")

# ---------------------------------------------------
# Sidebar Inputs
# ---------------------------------------------------
st.sidebar.header("📌 Portfolio Input")
ticker = st.sidebar.text_input("Stock Ticker", "AAPL").upper()
quantity = st.sidebar.number_input("Quantity", min_value=1, value=10)
start_date = st.sidebar.date_input("Start Date", pd.to_datetime("2018-01-01"))
end_date = st.sidebar.date_input("End Date", pd.to_datetime("today"))

st.sidebar.markdown("---")
st.sidebar.header("📊 Compare Stocks")
compare_tickers = st.sidebar.text_input(
    "Enter tickers separated by comma (e.g. AAPL,MSFT,TSLA)",
    "AAPL,MSFT"
)

st.sidebar.markdown("---")
st.sidebar.header("📊 Advanced Compare")

compare_input = st.sidebar.text_input(
    "Enter tickers (comma separated)",
    "AAPL,MSFT,TSLA"
)

benchmark = st.sidebar.text_input(
    "Benchmark (for Beta calculation)",
    "^GSPC"
)

rolling_window = st.sidebar.slider(
    "Rolling Window (days)",
    min_value=20,
    max_value=200,
    value=60
)

show_raw = st.sidebar.checkbox("Show Raw Prices Instead of Normalized")

# ---------------------------------------------------
# Fetch Data
# ---------------------------------------------------
data = yf.download(ticker, start=start_date, auto_adjust=True, progress=False)

if data.empty:
    st.error("No data found. Please check the ticker symbol.")
else:
    # Fix MultiIndex columns issue
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)

    # ---------------------------------------------------
    # Calculations
    # ---------------------------------------------------
    data["Daily Return"] = data["Close"].pct_change()
    data["Portfolio Value"] = data["Close"] * quantity

    current_value = data["Portfolio Value"].iloc[-1]
    initial_value = data["Portfolio Value"].iloc[0]
    total_return = (current_value / initial_value - 1)

    volatility = data["Daily Return"].std() * np.sqrt(252)

    sharpe_ratio = (
        (data["Daily Return"].mean() * 252) / volatility
        if volatility != 0 else 0
    )

    rolling_max = data["Portfolio Value"].cummax()
    drawdown = data["Portfolio Value"] / rolling_max - 1
    max_drawdown = drawdown.min()

    cagr = (
        (current_value / initial_value) ** (252 / len(data)) - 1
        if len(data) > 0 else 0
    )

    # ---------------------------------------------------
    # Metrics Section
    # ---------------------------------------------------
    st.subheader("📈 Performance Metrics")

    col1, col2, col3 = st.columns(3)
    col1.metric("Current Value", f"${current_value:,.2f}")
    col2.metric("Total Return", f"{total_return:.2%}")
    col3.metric("CAGR", f"{cagr:.2%}")

    col4, col5, col6 = st.columns(3)
    col4.metric("Annual Volatility", f"{volatility:.2%}")
    col5.metric("Sharpe Ratio", f"{sharpe_ratio:.2f}")
    col6.metric("Max Drawdown", f"{max_drawdown:.2%}")

    st.markdown("---")

    # ---------------------------------------------------
    # Portfolio Value Chart
    # ---------------------------------------------------
    st.subheader("📊 Portfolio Value Over Time")

    fig = px.line(
        data,
        x=data.index,
        y="Portfolio Value",
        title=f"{ticker} Portfolio Value",
    )

    fig.update_layout(
        xaxis_title="Date",
        yaxis_title="Portfolio Value ($)",
        template="plotly_white",
        plot_bgcolor="black",
        paper_bgcolor="black",
        margin=dict(l=40, r=40, t=60, b=40),
    )

    # Add black border
    fig.update_xaxes(showline=True, linewidth=2, linecolor='black', mirror=True)
    fig.update_yaxes(showline=True, linewidth=2, linecolor='black', mirror=True)

    st.plotly_chart(fig, use_container_width=True)

    # ---------------------------------------------------
    # Drawdown Chart
    # ---------------------------------------------------
    st.subheader("📉 Drawdown Analysis")

    fig2 = px.area(
        x=data.index,
        y=drawdown,
        labels={"y": "Drawdown"},
        title="Portfolio Drawdown"
    )

    fig2.update_layout(
        xaxis_title="Date",
        yaxis_title="Drawdown",
        template="plotly_white",
        plot_bgcolor="black",
        paper_bgcolor="black",
        margin=dict(l=40, r=40, t=60, b=40),
    )

    fig2.update_xaxes(showline=True, linewidth=2, linecolor='black', mirror=True)
    fig2.update_yaxes(showline=True, linewidth=2, linecolor='black', mirror=True)

    st.plotly_chart(fig2, use_container_width=True)
    
    st.markdown("---")
    st.subheader("📈 Comparative Growth Analysis")

    tickers_list = [t.strip().upper() for t in compare_tickers.split(",")]

    compare_data = yf.download(
        tickers_list,
        start=start_date,
        auto_adjust=True,
        progress=False
    )["Close"]

    # Handle single stock case
    if isinstance(compare_data, pd.Series):
        compare_data = compare_data.to_frame()

    # Normalize to 100
    normalized = compare_data / compare_data.iloc[0] * 100

    # Plot
    fig_compare = px.line(
        normalized,
        x=normalized.index,
        y=normalized.columns,
        title="Normalized Growth (Base = 100)"
    )

    fig_compare.update_layout(
        template="plotly_white",
        plot_bgcolor="black",
        paper_bgcolor="black",
        margin=dict(l=40, r=40, t=60, b=40),
    )

    fig_compare.update_xaxes(showline=True, linewidth=2, linecolor='black', mirror=True)
    fig_compare.update_yaxes(showline=True, linewidth=2, linecolor='black', mirror=True)

    st.plotly_chart(fig_compare, use_container_width=True)
    
    st.markdown("---")
    st.subheader("📈 Advanced Comparative Analytics")

    tickers_list = [t.strip().upper() for t in compare_input.split(",")]

    all_tickers = tickers_list + [benchmark]

    # Download with start + end date
    compare_data = yf.download(
        all_tickers,
        start=start_date,
        end=end_date,
        auto_adjust=True,
        progress=False
    )["Close"]

    # Handle single ticker case
    if isinstance(compare_data, pd.Series):
        compare_data = compare_data.to_frame()

    # Drop missing values for clean alignment
    compare_data = compare_data.dropna()

    # --------------------------------------------------
    # 1️⃣ Growth Comparison
    # --------------------------------------------------
    if not show_raw:
        normalized = compare_data / compare_data.iloc[0] * 100
        plot_data = normalized[tickers_list]
        title = "Normalized Growth (Base = 100)"
    else:
        plot_data = compare_data[tickers_list]
        title = "Raw Price Comparison"

    fig_growth = px.line(
        plot_data,
        title=title
    )

    fig_growth.update_layout(template="plotly_white")
    fig_growth.update_xaxes(showline=True, linewidth=2, linecolor='black', mirror=True)
    fig_growth.update_yaxes(showline=True, linewidth=2, linecolor='black', mirror=True)

    st.plotly_chart(fig_growth, use_container_width=True)

    # --------------------------------------------------
    # 2️⃣ Relative Strength (First two stocks)
    # --------------------------------------------------
    if len(tickers_list) >= 2:
        st.subheader("⚖ Relative Strength Ratio")

        rs_ratio = compare_data[tickers_list[0]] / compare_data[tickers_list[1]]
        rs_ratio = rs_ratio.dropna()

        fig_rs = px.line(
            rs_ratio,
            title=f"{tickers_list[0]} / {tickers_list[1]} Relative Strength"
        )

        fig_rs.update_layout(template="plotly_white")
        fig_rs.update_xaxes(showline=True, linewidth=2, linecolor='black', mirror=True)
        fig_rs.update_yaxes(showline=True, linewidth=2, linecolor='black', mirror=True)

        st.plotly_chart(fig_rs, use_container_width=True)

    # --------------------------------------------------
    # 3️⃣ Rolling Correlation (vs Benchmark)
    # --------------------------------------------------
    st.subheader("🔥 Rolling Correlation (vs Benchmark)")

    returns = compare_data.pct_change().dropna()

    rolling_corr = pd.DataFrame(index=returns.index)

    for stock in tickers_list:
        rolling_corr[stock] = (
            returns[stock]
            .rolling(rolling_window)
            .corr(returns[benchmark])
        )

    rolling_corr = rolling_corr.dropna()

    fig_corr = px.line(
        rolling_corr,
        title=f"Rolling {rolling_window}-Day Correlation with {benchmark}"
    )

    fig_corr.update_layout(template="plotly_white")
    fig_corr.update_xaxes(showline=True, linewidth=2, linecolor='black', mirror=True)
    fig_corr.update_yaxes(showline=True, linewidth=2, linecolor='black', mirror=True)

    st.plotly_chart(fig_corr, use_container_width=True)

    # --------------------------------------------------
    # 4️⃣ Rolling Volatility & Beta
    # --------------------------------------------------
    st.subheader("📉 Rolling Volatility & Beta")

    rolling_vol = returns[tickers_list].rolling(rolling_window).std() * np.sqrt(252)

    rolling_beta = pd.DataFrame(index=returns.index)

    for stock in tickers_list:
        cov = returns[stock].rolling(rolling_window).cov(returns[benchmark])
        var = returns[benchmark].rolling(rolling_window).var()
        rolling_beta[stock] = cov / var

    rolling_vol = rolling_vol.dropna()
    rolling_beta = rolling_beta.dropna()

    col1, col2 = st.columns(2)

    with col1:
        fig_vol = px.line(
            rolling_vol,
            title="Rolling Volatility (Annualized)"
        )
        fig_vol.update_layout(template="plotly_white")
        st.plotly_chart(fig_vol, use_container_width=True)

    with col2:
        fig_beta = px.line(
            rolling_beta,
            title=f"Rolling Beta vs {benchmark}"
        )
        fig_beta.update_layout(template="plotly_white")
        st.plotly_chart(fig_beta, use_container_width=True)

    # ---------------------------------------------------
    # Download Button
    # ---------------------------------------------------
    st.download_button(
        "⬇ Download Data as CSV",
        data.to_csv().encode("utf-8"),
        file_name=f"{ticker}_portfolio_data.csv",
        mime="text/csv"
    )