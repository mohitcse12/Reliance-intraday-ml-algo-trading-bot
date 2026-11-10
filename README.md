# 📈 Reliance Intraday AI/ML Algorithmic Trading Bot

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Machine Learning](https://img.shields.io/badge/Model-LightGBM-orange)
![Optimization](https://img.shields.io/badge/Tuning-Optuna-blueviolet)
![Broker API](https://img.shields.io/badge/Execution-Fyers%20API%20v3-green)
![Cloud](https://img.shields.io/badge/Deployment-AWS%20EC2-yellow)

An end-to-end, automated intraday algorithmic trading system engineered specifically for **Reliance Industries (NSE: RELIANCE-EQ)**. This project combines machine learning price-action predictions using **LightGBM** with low-latency order execution via the **Fyers API**, deployed headlessly on **AWS EC2** with real-time **Telegram** notifications and automated risk management.

## ✨ Key Features

* **AI/ML Signal Generation:** Uses a pre-trained **LightGBM** (Gradient Boosted Decision Tree) classifier to predict short-term intraday price directional movements based on technical indicators and historical OHLCV candle data.
* **Hyperparameter Optimization:** Model parameters are systematically tuned using **Optuna** to maximize out-of-sample precision and minimize overfitting on noisy market data.
* **Real-Time Broker Execution:** Integrated with **Fyers API v3** for live market data ingestion, authentication, and automated order placement during Indian market hours (9:15 AM – 3:30 PM IST).
* **Automated Risk Management:** Built-in capital protection featuring both a **Hard Stop-Loss** (to cap maximum downside per trade) and a dynamic **Trailing Stop-Loss** (to lock in profits as the price moves favorably).
* **Real-Time Telegram Alerts:** Pushes instant mobile notifications via the **Telegram Bot API** for bot startup, live trade entries, stop-loss updates, trade exits, and error logs.
* **24/7 Cloud Deployment:** Designed to run headlessly on an **AWS EC2 Linux** instance inside an isolated Python virtual environment (`venv`), managed via `tmux` / `screen` for uninterrupted execution.

## 🏗️ System Architecture & Workflow

1. **Data Ingestion:** Connects to the Fyers API to fetch real-time intraday candle data (OHLCV) for `NSE:RELIANCE-EQ`.
2. **Feature Engineering:** Computes momentum, trend, and volatility indicators on live incoming data streams.
3. **Model Inference:** Feeds processed features into the serialized LightGBM model (`.pkl`) to generate Buy (`1`), Sell (`-1`), or Hold (`0`) signals.
4. **Order Execution & Risk Control:** Fires market/bracket orders via Fyers API and continuously monitors open positions to adjust the trailing stop-loss dynamically.
5. **Logging & Notification:** Logs all system events locally (`live_ml_bot.log`) and dispatches real-time status alerts to Telegram.

## 🛠️ Tech Stack

* **Programming Language:** Python
* **Machine Learning & Data:** LightGBM, Optuna, Pandas, NumPy, Joblib
* **Broker & Execution API:** Fyers API v3 (`fyers-apiv3`)
* **Alerts & Networking:** Telegram Bot API, Requests
* **Infrastructure & Security:** AWS EC2 (Linux), Python `venv`, `tmux`/`screen`, `python-dotenv`

## 📂 Repository Structure

Reliance-intraday-ml-algo-trading-bot/
├── bot.py               # Main live trading execution, risk management & API engine
├── model_training.py    # LightGBM model training, feature engineering & Optuna tuning
├── requirements.txt     # Required Python libraries and dependencies
├── .env.example         # Template for environment variables & secret API keys
└── README.md            # Project documentation and setup guide

## 🚀 Installation & Setup Guide

> **Security Note:** Actual broker credentials and API keys are strictly excluded from this repository. Follow the steps below to configure your own environment variables locally or on a cloud server.

### 1. Clone the Repository
```bash
git clone https://github.com/mohitcse12/Reliance-intraday-ml-algo-trading-bot.git
cd Reliance-intraday-ml-algo-trading-bot
```

### 2. Create and Activate a Virtual Environment
```bash
python -m venv venv
# On Linux / AWS EC2 / macOS:
source venv/bin/activate
# On Windows:
venv\Scripts\activate
```

### 3. Train the ML Model (Optional)
If you have historical data and want to train the LightGBM model locally:
```bash
python model_training.py
```

### 4. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure Environment Variables
Create a `.env` file in the root directory by copying the example template:
```bash
cp .env.example .env
```
Open the `.env` file and insert your actual Fyers and Telegram credentials:
```env
FYERS_CLIENT_ID=your_fyers_app_id_here
FYERS_SECRET_KEY=your_fyers_secret_key_here
FYERS_ACCESS_TOKEN=your_fyers_access_token_here
TELEGRAM_BOT_TOKEN=your_telegram_bot_token_here
TELEGRAM_CHAT_ID=your_telegram_chat_id_here
```

### 5. Run the Trading Bot
```bash
python bot.py
```

### 6. Running Persistently on AWS EC2 (Optional)
To keep the bot running in the background after closing your SSH terminal:
```bash
tmux new -s reliance_bot
source venv/bin/activate
python bot.py
# Press Ctrl+B, then D to detach from the session safely
```

## ⚠️ Disclaimer

This project is developed strictly for **educational and research purposes**. Algorithmic trading in equity markets involves substantial financial risk. Past performance of machine learning models does not guarantee future results. The developer assumes no responsibility for any financial losses incurred by using this software.
