"""
Stock Screener Main Class
"""

## IMPORT UTILS FUNCTIONS ##
from screener import StockScreener, get_sp_tickers, get_sp500_stocks
import streamlit as st
import base64  # Import base64 for encoding

def to_csv_download_link(df, filename="filtered_stocks.csv"):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    return f'<a href="data:file/csv;base64,{b64}" download="{filename}">Download CSV file</a>'

if __name__ == '__main__':
    # Streamlit Config
    st.set_page_config(page_title="Stock Screener", page_icon=":chart_with_upwards_trend:")

    filters = []

    # Get sp500 tickers and sectors
    sp500 = get_sp_tickers()

    # Create Stock objects for all S&P 500 companies
    sp500_stocks = get_sp500_stocks(sp500[:10])

    # Screener
    screener = StockScreener(sp500_stocks, filters)

    # Create streamlit app
    screener.create_app()
