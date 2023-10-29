# -*- coding: utf-8 -*-
"""
StockScreener class and utils functions
"""
import requests
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import base64
import streamlit as st
import tensorflow as tf
from bs4 import BeautifulSoup
from sklearn.preprocessing import StandardScaler
from stock import (Stock, filter_sector, filter_price, filter_metric, filter_technical_indicator,
                   get_stock_price, get_historical)


## StockScreener Class ##
class StockScreener:
    def __init__(self, stocks, filters):
        self.stocks = stocks
        self.filters = filters
        self.models = {}
        self.scaler = StandardScaler()

    # Select stocks that pass all filters
    def apply_filters(self):
        filtered_stocks = []
        for stock in self.stocks:
            passed_all_filters = True
            for filter_func in self.filters:
                if not filter_func(stock):
                    passed_all_filters = False
                    break
            if passed_all_filters:
                filtered_stocks.append(stock)
        return filtered_stocks

    # Train deep learning models on selected stocks
    def train_models(self):
        training_models = st.empty()
        training_models.write("Training Model for each Ticker...")

        filtered_stocks = self.apply_filters()

        for stock in filtered_stocks:
            train_data = stock.technical_indicators
            train_labels = stock.label
            if len(train_data) == 0:
                continue

            train_data = self.scaler.fit_transform(train_data)
            train_labels = np.array(train_labels)

            model = create_model(train_data)
            history = model.fit(train_data, train_labels, epochs=100)
            self.models[stock.ticker] = model, history

        training_models.empty()
        return filtered_stocks

    # Make predictions for each stock using its corresponding model
    def predict_stocks(self, new_stocks):
        predicted_stocks = []
        for stock in new_stocks:
            if stock.ticker in self.models:
                model, _ = self.models[stock.ticker]
                new_features_aux = np.array(stock.today_technical_indicators).reshape(1, -1)
                new_stock_data = self.scaler.fit_transform(new_features_aux)
                prediction = model.predict(new_stock_data)
                stock.prediction = prediction
                if prediction > 0.5:
                    predicted_stocks.append(stock)
        return predicted_stocks

    # Create a web app for the stock screener
    def create_app(self):
        st.title(":green[Shares Screener]")

        # Create sidebar for filtering options
        sector_list = sorted(list(set(stock.sector for stock in self.stocks)))
        selected_sector = st.sidebar.selectbox("Sector", ["All"] + sector_list)

        min_price = st.sidebar.number_input("Min Price", value=10.0, step=0.01)
        max_price = st.sidebar.number_input("Max Price", value=1000.0, step=0.01)

        metric_list = sorted(list(set(metric for stock in self.stocks for metric in stock.metrics)))
        selected_metrics = st.sidebar.multiselect("Metrics", metric_list)

        metric_operator_list = [">", ">=", "<", "<=", "=="]
        selected_metric_operator = st.sidebar.selectbox("Metric Operator", metric_operator_list)

        metric_value = st.sidebar.text_input("Metric Value", "Enter value or the word price")
        try:
            metric_value = float(metric_value)
            print(metric_value)
        except:
            pass

        indicator_list = sorted(list(set(indicator for stock in self.stocks for indicator in stock.today_technical_indicators.keys())))
        selected_indicators = st.sidebar.multiselect("Indicators", indicator_list)

        indicator_operator_list = [">", ">=", "<", "<=", "=="]
        selected_indicator_operator = st.sidebar.selectbox("Indicator Operator", indicator_operator_list)

        indicator_value = st.sidebar.text_input("Indicator Value", "Enter value or the word price")
        try:
            indicator_value = float(indicator_value)
            print(indicator_value)
        except:
            pass

        # Update filters list with user inputs
        new_filters = []
        if selected_sector != "All":
            new_filters.append(lambda stock: filter_sector(stock, selected_sector))
        if selected_metrics:
            for selected_metric in selected_metrics:
                new_filters.append(lambda stock: filter_metric(stock, selected_metric, selected_metric_operator, metric_value))
        if selected_indicators:
            for selected_indicator in selected_indicators:
                new_filters.append(lambda stock: filter_technical_indicator(stock, selected_indicator, selected_indicator_operator, indicator_value))
        new_filters.append(lambda stock: filter_price(stock, min_price, max_price))
        self.filters = new_filters

        # Create "Apply Filters" button
        if st.sidebar.button("Apply Filters"):
            filtered_stocks = self.apply_filters()
            display_filtered_stocks(filtered_stocks, selected_metrics, selected_indicators)

        # Create "Train and Predict Models" button
        if st.sidebar.button("Train and Predict"):
            filtered_stocks = self.train_models()
            predicted_stocks = self.predict_stocks(filtered_stocks)
            display_filtered_stocks(predicted_stocks, selected_metrics, selected_indicators, self.models)


# Simple Dense model
def create_model(train_data):
    model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(64, input_shape=(train_data.shape[1],), activation='relu'),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model


def get_table_download_link(df):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="filtered_stocks.csv">Download CSV File</a>'
    return href



import matplotlib.pyplot as plt
import streamlit as st
import pandas as pd
import base64

def display_filtered_stocks(filtered_stocks, selected_metrics, selected_indicators, models=None):
    if len(filtered_stocks) == 0:
        st.write("No stocks match the specified criteria.")
    else:
        filtered_tickers = [stock.ticker for stock in filtered_stocks]
        tabs = st.tabs(filtered_tickers)
        
        for n in range(len(tabs)):
            stock = filtered_stocks[n]
            with tabs[n]:
                col1, col2, col3 = st.columns(3)
                for i, (metric, value) in enumerate(stock.metrics.items()):
                    col = [col1, col2, col3][i % 3]
                    col.metric(metric, value)
                
                # Display the candlestick chart with indicators
                st.plotly_chart(stock.plot_candlestick(selected_indicators))
                
                if selected_metrics:
                    st.write("### Metrics")
                    for metric in selected_metrics:
                        st.metric(metric, stock.metrics.get(metric, "N/A"))
                
                if selected_indicators:
                    st.write("### Technical Indicators")
                    for indicator in selected_indicators:
                        st.metric(indicator, stock.today_technical_indicators.get(indicator, "N/A"))
                
                if models and stock.ticker in models:
                    model, history = models[stock.ticker]
                    st.write("### Model Training Loss")
                    st.line_chart(pd.DataFrame(history.history['loss'], columns=['Loss']))
                    st.write("### Model Training Accuracy")
                    st.line_chart(pd.DataFrame(history.history['accuracy'], columns=['Accuracy']))
    
    table_data = []
    for stock in filtered_stocks:
        row = [stock.ticker, stock.sector, stock.price]
        for metric in selected_metrics:
            row.append(stock.metrics.get(metric, "N/A"))
        for indicator in selected_indicators:
            row.append(stock.today_technical_indicators.get(indicator, "N/A"))
        row.append(float(stock.prediction) if stock.prediction != 0 else "N/A")
        table_data.append(row)
    
    table_columns = ["Ticker", "Sector", "Price"]
    for metric in selected_metrics:
        table_columns.append(f"Metric: {metric}")
    for indicator in selected_indicators:
        table_columns.append(f"Indicator: {indicator}")
    table_columns.append("Prediction")
    
    if all(len(row) == len(table_columns) for row in table_data):
        df = pd.DataFrame(table_data, columns=table_columns)
        st.write(df)
 
        csv = df.to_csv(index=False)
        b64 = base64.b64encode(csv.encode()).decode()
        href = f'<a href="data:file/csv;base64,{b64}" download="filtered_stocks.csv">Download CSV File</a>'
        st.markdown(href, unsafe_allow_html=True)
    else:
        st.error("Error: The number of columns does not match the data provided.")



## GET SP 500 STOCK DATA ##

def get_sp_tickers():
    url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')

    table = soup.find('table', {'class': 'wikitable sortable'})
    rows = table.find_all('tr')[1:]  # skip the header row

    sp500 = []

    for row in rows:
        cells = row.find_all('td')
        ticker = cells[0].text.strip()
        company = cells[1].text.strip()
        sector = cells[3].text.strip()
        sp500.append({'ticker': ticker, 'company': company, 'sector': sector})

    return sp500


@st.cache_data(ttl=24 * 3600)
def get_sp500_stocks(sp500):
    sp500_stocks = []
    stock_download = st.empty()
    stock_issues = st.empty()
    for stock in sp500:
        stock_download.write(f"Downloading {stock['ticker']} Data")
        try:
            price = get_stock_price(stock['ticker'])
            data = get_historical(stock['ticker'])
            sp500_stocks.append(Stock(stock['ticker'], stock['sector'], price, data))
            stock_download.empty()
        except:
            stock_issues.write(f"There was an issue with {stock['ticker']}. ")

    stock_issues.empty()
    return sp500_stocks
