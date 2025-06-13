import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sb
import yfinance as yf

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn import metrics

import warnings
warnings.filterwarnings('ignore')

# Streamlit Title
st.title("📈 Crypto Price Prediction App")
st.markdown("Predict the price movement of cryptocurrencies using Machine Learning.")
st.image("https://i.ibb.co/zVpz7DQG/Bitcoin-Untitled-design.png")

# User Input for Cryptocurrency Selection
crypto_symbol = st.text_input("Enter Cryptocurrency Symbol (e.g., BTC-USD, ETH-USD, DOGE-USD)", "BTC-USD")
start_date = st.date_input("Start Date", pd.to_datetime("2023-01-01"))
end_date = st.date_input("End Date", pd.to_datetime("today"))

if st.button("Fetch & Predict"):
    with st.spinner("Fetching data and training models..."):
        # Fetch Data from Yahoo Finance
        data = yf.download(crypto_symbol, start=start_date, end=end_date)
        data.reset_index(inplace=True)

        if data.empty:
            st.error("No data found for the given symbol and date range. Try another input.")
        else:
            # EDA: Plot Close Prices
            st.subheader(f"{crypto_symbol} Closing Price Trend")
            fig, ax = plt.subplots(figsize=(12, 5))
            ax.plot(data['Date'], data['Close'], color='blue', label='Close Price')
            ax.set_title(f"{crypto_symbol} Closing Price Trend")
            ax.set_xlabel("Date")
            ax.set_ylabel("Price (USD)")
            ax.legend()
            st.pyplot(fig)

            # Feature Engineering
            data['year'] = data['Date'].dt.year
            data['month'] = data['Date'].dt.month
            data['day'] = data['Date'].dt.day
            data['is_quarter_end'] = np.where(data['month'] % 3 == 0, 1, 0)

            data['open-close'] = data['Open'] - data['Close']
            data['low-high'] = data['Low'] - data['High']
            data['target'] = np.where(data['Close'].shift(-1) > data['Close'], 1, 0)

            # Drop unnecessary columns
            data = data.drop(['Date', 'Adj Close'], axis=1, errors='ignore')

            # Drop NaN from target
            data = data.dropna()

            # Selecting features
            features = data[['open-close', 'low-high', 'is_quarter_end']]
            target = data['target']

            # Feature Scaling
            scaler = StandardScaler()
            features_scaled = scaler.fit_transform(features)

            # Train-Test Split
            X_train, X_valid, Y_train, Y_valid = train_test_split(features_scaled, target, test_size=0.2, random_state=2022)

            # Model Training
            models = {
                "Logistic Regression": LogisticRegression(),
                "SVM (Poly Kernel)": SVC(kernel='poly', probability=True),
                "XGBoost": XGBClassifier()
            }

            st.subheader("🔍 Model Performance")
            for name, model in models.items():
                model.fit(X_train, Y_train)
                train_acc = metrics.roc_auc_score(Y_train, model.predict_proba(X_train)[:, 1])
                val_acc = metrics.roc_auc_score(Y_valid, model.predict_proba(X_valid)[:, 1])
                st.write(f"**{name}:**")
                st.write(f"✅ Training Accuracy: {train_acc:.3f}")
                st.write(f"✅ Validation Accuracy: {val_acc:.3f}")
                st.write("---")

            # Pie Chart for Class Distribution
            st.subheader("🔍 Target Variable Distribution")
            fig, ax = plt.subplots()
            ax.pie(data['target'].value_counts().values, labels=['Down (0)', 'Up (1)'], autopct='%1.1f%%', colors=['red', 'green'])
            st.pyplot(fig)

            # Correlation Heatmap
            st.subheader("🔍 Feature Correlation")
            fig, ax = plt.subplots(figsize=(8, 6))
            sb.heatmap(data.corr(), annot=True, cmap="coolwarm", ax=ax)
            st.pyplot(fig)

            st.success("✅ Prediction Completed!")
st.markdown("Made by AryanMandlik")

