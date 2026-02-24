import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import pickle as pkl
import yfinance as yf
import warnings
warnings.filterwarnings('ignore')

# Load the model
model = pkl.load(open('model.pkl', 'rb'))

# Define the stock ticker and date range
start = '2010-01-01'
end = '2019-12-31'

# Define the Streamlit app
st.title('Stock Price Prediction App')

# Get user input for stock ticker
input = st.selectbox('Select Your Company', ['Apple', 'State Bank of India', 'Google', 'Microsoft', 'Amazon', 'Tesla'])
if input == 'Apple':
    ticker = 'AAPL'
elif input == 'State Bank of India':
    ticker = 'SBIN.NS'
elif input == 'Google':
    ticker = 'GOOGL'
elif input == 'Microsoft':
    ticker = 'MSFT'
elif input == 'Amazon':
    ticker = 'AMZN'
elif input == 'Tesla':
    ticker = 'TSLA'

# Fetch stock data
if ticker:
    df = yf.download(ticker, start=start, end=end)
    df = df.reset_index()
    df.columns = df.columns.droplevel(1)
    df = df[['Date', 'High', 'Low', 'Open', 'Close', 'Volume']]

    # Describe Data
    st.subheader(f"{input} Stock Price Data (2010-2019)")
    st.write(df.describe())
    
    # Visualizations
    st.subheader(f"{input} Closing Price vs Time Chart")
    fig = plt.figure(figsize = (12,6))
    plt.plot(df.Close)
    st.pyplot(fig)

    st.subheader(f"{input} Closing Price vs Time Chart with 100MA")
    ma100 = df.Close.rolling(100).mean()
    fig = plt.figure(figsize=(12, 6))
    plt.plot(ma100, 'r')
    plt.plot(df.Close)
    st.pyplot(fig)

    st.subheader(f"{input} Closing Price vs Time Chart with 200MA")
    ma100 = df.Close.rolling(100).mean()
    ma200 = df.Close.rolling(200).mean()
    fig = plt.figure(figsize=(12, 6))
    plt.plot(ma100, 'r')
    plt.plot(ma200, 'g')
    plt.plot(df.Close)
    st.pyplot(fig)

    # Splitting data into Training and Testing
    data_training = pd.DataFrame(df['Close'][0:int(len(df) * 0.70)])
    data_testing = pd.DataFrame(df['Close'][int(len(df) * 0.70):int(len(df))])

    # Scaling
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler(feature_range=(0, 1))

    data_training_array = scaler.fit_transform(data_training)

    # Splitting data into x_train and y_train
    x_train = []
    y_train = []

    for i in range(100, data_training_array.shape[0]):
        x_train.append(data_training_array[i - 100:i])
        y_train.append(data_training_array[i, 0])

    x_train, y_train = np.array(x_train), np.array(y_train)

    # Testing part
    past_100_days = data_training.tail(100)
    final_df = pd.concat([past_100_days, data_testing], ignore_index=True)
    input_data = scaler.transform(final_df)

    x_test = []
    y_test = []

    for i in range(100, input_data.shape[0]):
        x_test.append(input_data[i - 100:i])
        y_test.append(input_data[i, 0])

    x_test, y_test = np.array(x_test), np.array(y_test)
    y_pred = model.predict(x_test)

    scaler = scaler.scale_
    scale_factor = 1/scaler[0]
    y_pred = y_pred * scale_factor
    y_test = y_test * scale_factor

    # Final Graph
    st.subheader(f"{input} Predicted vs Actual Closing Price Data")
    plt.figure(figsize=(12, 6))
    plt.plot(y_test, 'b', label='Original Price')
    plt.plot(y_pred, 'r', label='Predicted Price')
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.legend()
    st.pyplot(fig)