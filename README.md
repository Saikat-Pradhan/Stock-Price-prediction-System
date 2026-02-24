# 📈 Stock Price Prediction System Web App
![Python](https://img.shields.io/badge/Python-3.12-blue?logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-WebApp-red?logo=streamlit)
![TensorFlow](https://img.shields.io/badge/TensorFlow-DeepLearning-orange?logo=tensorflow)
![LSTM](https://img.shields.io/badge/Model-LSTM-green)
![Machine Learning](https://img.shields.io/badge/Machine-Learning-yellow)
![Deep Learning](https://img.shields.io/badge/Deep-Learning-purple)

A smart Deep Learning–based Stock Market Prediction Web Application that forecasts future stock closing prices using historical market data.
Built using Python, LSTM Neural Networks, Streamlit, and deployed online for real-time stock analysis and prediction.

---

## 🔗 Live Demo

👉 Try the deployed web app here: https://stock-price-prediction-system-by-saikat-pradhan.streamlit.app/

---

## 🚀 Project Overview

This project demonstrates how Deep Learning (LSTM) can analyze historical stock price trends and predict future market behavior.

Users can select a company, choose a date range, and instantly visualize:

- Historical stock trends
- Moving averages
- Actual vs Predicted stock prices

The system fetches real-time historical data and generates intelligent predictions dynamically.

---

## Supported Companies

- 🍎 Apple
- 🏦 State Bank of India
- 🔍 Google
- 💻 Microsoft
- 📦 Amazon
- 🚗 Tesla

## 🎯 Application Features

- Interactive company selection
- Dynamic date range input
- Stock data visualization
- 100-Day Moving Average Analysis
- 200-Day Moving Average Analysis
- LSTM-based price prediction
- Actual vs Predicted comparison graph
- Real-time data fetching using Yahoo Finance

## 🧠 Technologies Used

- Python 🐍
- Streamlit 🌐
- TensorFlow / Keras 🤖
- LSTM Neural Network 🧠
- Pandas 📊
- NumPy 📐
- Matplotlib 📉
- Scikit-learn ⚙️
- yFinance 📈
- Pickle 📦

## 📊 Dataset

Stock data is dynamically fetched using Yahoo Finance API via yfinance.

The dataset includes:

- Date
- Open Price
- High Price
- Low Price
- Close Price
- Volume

This historical time-series data allows the LSTM model to learn market trends and patterns.

## 🏗️ Model Training

Model development is performed in: ``` 📓 Stock_Price_Prediction_Using_LSTM.ipynb ```

### Training Steps

- Data Collection
- Data Preprocessing
- Feature Scaling using MinMaxScaler
- Time-Series Window Creation
- LSTM Model Training
- Model Evaluation
- Saving trained model using Pickle
- Saved Model
- model.pkl → Trained LSTM Prediction Model

## 🧠 How the App Works

- User selects a company.
- Historical stock data is fetched automatically.
- Closing prices are scaled using MinMaxScaler.
- Previous 100-day sequences are created.
- LSTM model predicts future closing prices.
- Predicted prices are rescaled.
- Actual vs Predicted stock prices are displayed visually.

## 📁 Project Structure
```
Stock-Price-Prediction-System
│
├── app.py
├── model.pkl
├── requirements.txt
├── Stock_Price_Prediction_Using_LSTM.ipynb
└── README.md
```

## ⚙️ Setup Guide (Run Locally)
### 1️⃣ Clone the Repository
```
git clone https://github.com/Saikat-Pradhan/Stock-Price-Prediction-System.git
cd Stock-Price-Prediction-System
```

### 2️⃣ Install Dependencies
```
pip install -r requirements.txt
```

### 3️⃣ Run the Streamlit App
```
streamlit run app.py
```

### Then open your browser:
```
http://localhost:8501
```

## 📉 Visual Outputs

The application provides:

- Stock Closing Price Trend
- Moving Average Analysis
- Deep Learning Prediction Graph
- Actual vs Predicted Price Comparison

## 🌍 Deployment

✅ Successfully deployed using Streamlit Cloud

## ⭐ Support

If you like this project, please give it a star ⭐ on GitHub.

It motivates me to build more Machine Learning & AI projects 🚀

## 👨‍💻 Author

Saikat Pradhan

🔗 GitHub: https://github.com/Saikat-Pradhan
