##🎯 Project Overview
A production-ready Streamlit web application that predicts next-day cryptocurrency price movements using machine learning classification models. The application leverages real-time financial data from Yahoo Finance and employs three ensemble learning techniques to forecast whether a cryptocurrency's closing price will move up or down.
Key Highlights

##🚀 Real-time predictions for multiple cryptocurrencies (BTC, ETH, DOGE, etc.)
📈 3 ML models compared: Logistic Regression, SVM, XGBoost
📊 Interactive dashboard with EDA visualizations and correlation analysis
⚡ Feature engineering pipeline with technical indicators
📉 Performance metrics (ROC-AUC, Train/Validation accuracy)


##💡 Problem Statement & Motivation
Challenge: Cryptocurrency markets are highly volatile and difficult to predict. Traditional time-series analysis doesn't capture complex non-linear patterns in price movements.
Solution: Apply supervised learning classification to transform the continuous price prediction problem into a binary classification task: predict whether tomorrow's closing price will be higher (UP) or lower (DOWN) than today's closing price.
Why this approach?

Converts a regression problem into a classification problem (more suitable for market predictions)
Uses ROC-AUC metric, which handles class imbalance better
Enables ensemble methods to capture different decision boundaries
Provides probability scores for investment risk assessment


##✨ Features
###🔧 Core Functionality
Multi-cryptocurrency support: BTC-USD, ETH-USD, DOGE-USD, and any Yahoo Finance ticker
Customizable date ranges: Flexible backtesting and analysis periods
Real-time data fetching: Automatic price data retrieval from Yahoo Finance
Feature engineering: Automated technical indicator generation

###📊 Data Analysis & Visualization
Closing price trends: Interactive time-series visualization
Class distribution analysis: Pie chart showing UP/DOWN class balance
Feature correlation heatmap: Identify relationships between features
Model performance comparison: Side-by-side accuracy metrics

###🤖 Machine Learning Models
Logistic Regression – Linear baseline classifier
SVM (Polynomial Kernel) – Non-linear decision boundaries
XGBoost Classifier – Gradient boosting ensemble (best performer)

###📈 Performance Metrics
ROC-AUC Score (primary metric for imbalanced classification)
Training & Validation Accuracy
Model comparison dashboard

## 🛠️ Requirements
To run this project, ensure that the following Python libraries are installed in your environment:
- `streamlit`
- `numpy`
- `pandas`
- `matplotlib`
- `seaborn`
- `yfinance`
You can install all required dependencies using pip:
pip install streamlit numpy pandas matplotlib seaborn yfinance

## 💻 Running the Application
To launch the application, execute the following command in your command prompt or terminal:
streamlit run bitcoin_price_prediction.py

## 📈 Purpose
This project is designed for educational purposes, to demonstrate how financial data can be analyzed and visualized, and to introduce the concept of price prediction in the cryptocurrency domain using open-source tools.

## 📬 Contact
For any inquiries or suggestions regarding this project, please feel free to open an issue on the repository or contact the maintainer directly via GitHub.
