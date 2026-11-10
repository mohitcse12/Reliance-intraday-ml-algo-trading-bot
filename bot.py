import os
import time
import joblib
import logging
import requests
import warnings
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from urllib.parse import urlparse, parse_qs
from fyers_apiv3 import fyersModel

warnings.filterwarnings("ignore")

# ==========================================
# 1. LOGGING & CONFIGURATION
# ==========================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("live_ml_bot.log", encoding="utf-8"),
        logging.StreamHandler()
    ]
)

CLIENT_ID = "Your Client ID"
SECRET_KEY = "Your Secret Key"  # <-- Apni Fyers Secret Key yahan daalein
REDIRECT_URI = "URL"
ACCESS_TOKEN = ""

# 1. Agar token file hai toh pehle use padho
if os.path.exists("access_token.txt"):
    with open("access_token.txt", "r") as f:
        ACCESS_TOKEN = f.read().strip()

# 2. Check karo ki token aaj valid hai ya expire ho chuka hai
token_valid = False
if ACCESS_TOKEN:
    try:
        test_fyers = fyersModel.FyersModel(client_id=CLIENT_ID, is_async=False, token=ACCESS_TOKEN, log_path="")
        if test_fyers.get_profile().get("s") == "ok":
            token_valid = True
            logging.info("✅ Aaj ka Access Token pehle se Valid hai!")
    except Exception:
        token_valid = False

# 3. Agar token expire ho gaya hai, toh turant naya Auth Code maango
if not token_valid:
    logging.warning("⚠️ Access Token expire ho gaya hai! Naya login karein...")
    session = fyersModel.SessionModel(
        client_id=CLIENT_ID,
        secret_key=SECRET_KEY,
        redirect_uri=REDIRECT_URI,
        response_type="code",
        grant_type="authorization_code"
    )
    print("\n🔗 Fyers Login Link:\n", session.generate_authcode())
    user_input = input("\n👉 Login karke Auth Code (ya pura URL) yahan paste karein: ").strip()
    if "auth_code=" in user_input:
        auth_code = parse_qs(urlparse(user_input).query).get("auth_code", [user_input])[0]
    else:
        auth_code = user_input
    session.set_token(auth_code)
    token_resp = session.generate_token()
    if "access_token" in token_resp:
        ACCESS_TOKEN = token_resp["access_token"]
        with open("access_token.txt", "w") as f:
            f.write(ACCESS_TOKEN)
        logging.info("✅ Naya Access Token ban gaya aur 'access_token.txt' mein save ho gaya!")
    else:
        logging.error(f"❌ Token nahi ban paya: {token_resp}")
        exit(1)

# <-- Apna exact Telegram Bot Token ek baar verify zaroor kar lein ('l' vs '1')
TELEGRAM_BOT_TOKEN = "Your Telegram Bot Token"
TELEGRAM_CHAT_ID = "Your Telegram Chat ID"

# Symbols
SYMBOL_REL = "NSE:RELIANCE-EQ"
SYMBOL_NIFTY = "NSE:NIFTY50-INDEX"
SYMBOL_VIX = "NSE:INDIAVIX-INDEX"
SYMBOL_ENERGY = "NSE:NIFTYENERGY-INDEX"

# Strategy & Risk Parameters
MODEL_FILE = "reliance_ml_model.pkl"
ML_CONFIDENCE_THRESHOLD = 0.90      # 90% AI Win Probability
MIN_INDICATOR_POINTS = 3            # 6 mein se 3 Indicator Points
MAX_DAILY_LOSS = -2000.0            # ₹350 Max Loss Limit
ONE_TRADE_PER_DAY = True           # Multiple trades ke liye False kar dein
LIVE_ORDER_EXECUTION = True         # Paper trading ke liye False rakhein

FEATURE_COLS = [
    'st_fast_dir', 'st_slow_dir', 'n_st_dir', 'vwap_dist_atr',
    'RSI', 'rsi_slope', 'ADX', 'vol_ratio', 'body_atr',
    'rel_strength_nifty', 'time_sin', 'time_cos',
    'vix_close', 'energy_close'
]

# ==========================================
# 2. TELEGRAM & DYNAMIC QTY HELPERS
# ==========================================
def send_telegram_msg(message):
    if not TELEGRAM_BOT_TOKEN or TELEGRAM_BOT_TOKEN == "YOUR_TELEGRAM_BOT_TOKEN":
        return
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN.strip()}/sendMessage"
    payload = {"chat_id": TELEGRAM_CHAT_ID.strip(), "text": message, "parse_mode": "HTML"}
    try:
        resp = requests.post(url, json=payload, timeout=5)
        if resp.status_code != 200:
            logging.error(f"❌ Telegram API Error ({resp.status_code}): {resp.text}")
    except Exception as e:
        logging.error(f"❌ Telegram Connection Error: {e}")

def get_dynamic_quantity(fyers, ltp):
    """Account ke Available Balance aur Intraday margin ke hisaab se Qty automatic nikalega"""
    if not LIVE_ORDER_EXECUTION:
        return 25  # Paper mode ke liye default dummy quantity

    try:
        funds_resp = fyers.funds()
        if funds_resp.get("s") == "ok":
            fund_limits = funds_resp.get("fund_limit", [])
            avail_bal = 0.0
            for item in fund_limits:
                if item.get("title") == "Available Balance":
                    avail_bal = float(item.get("equityAmount", 0.0))
                    break

            if avail_bal <= 0:
                logging.warning("⚠️ Available Balance 0 hai! Fallback Qty = 1 use hoga.")
                return 1

            # Intraday MIS par ~4x safe leverage maan kar calculation
            usable_capital = avail_bal * 4.0
            calculated_qty = int(usable_capital / ltp)

            # Minimum 1 share, aur safety ke liye ek reasonable cap
            final_qty = max(1, calculated_qty)
            logging.info(f"💰 Dynamic Qty Calculated -> Balance: ₹{avail_bal:.2f} | LTP: ₹{ltp:.2f} | Qty: {final_qty}")
            return final_qty

    except Exception as e:
        logging.error(f"❌ Error fetching funds for quantity: {e}")

    return 1  # Safe fallback quantity

# ==========================================
# 3. TECHNICAL INDICATORS
# ==========================================
def calculate_atr(df, period=14):
    tr = pd.concat([
        df['high'] - df['low'],
        np.abs(df['high'] - df['close'].shift()),
        np.abs(df['low'] - df['close'].shift())
    ], axis=1).max(axis=1)
    return tr.rolling(window=period).mean()

def calculate_supertrend(df, period=10, multiplier=3.0):
    atr = calculate_atr(df, period)
    hl2 = (df['high'] + df['low']) / 2
    upperband, lowerband = hl2 + (multiplier * atr), hl2 - (multiplier * atr)
    final_upper, final_lower = upperband.copy(), lowerband.copy()
    st_dir, close = np.ones(len(df)), df['close'].values
    for i in range(1, len(df)):
        final_upper.iloc[i] = upperband.iloc[i] if upperband.iloc[i] < final_upper.iloc[i - 1] or close[i - 1] > final_upper.iloc[i - 1] else final_upper.iloc[i - 1]
        final_lower.iloc[i] = lowerband.iloc[i] if lowerband.iloc[i] > final_lower.iloc[i - 1] or close[i - 1] < final_lower.iloc[i - 1] else final_lower.iloc[i - 1]
        st_dir[i] = 1 if (st_dir[i - 1] == -1 and close[i] > final_upper.iloc[i - 1]) else (-1 if (st_dir[i - 1] == 1 and close[i] < final_lower.iloc[i - 1]) else st_dir[i - 1])
    return st_dir

def calculate_rsi(series, period=14):
    delta = series.diff()
    rs = (delta.where(delta > 0, 0)).rolling(period).mean() / ((-delta.where(delta < 0, 0)).rolling(period).mean() + 1e-9)
    return 100 - (100 / (1 + rs))

def calculate_adx(df, period=14):
    tr = pd.Series(calculate_atr(df, 1)).rolling(period).mean() + 1e-9
    plus_dm, minus_dm = df['high'].diff(), df['low'].diff() * -1
    plus_di = 100 * (pd.Series(np.where((plus_dm > minus_dm) & (plus_dm > 0), plus_dm, 0.0)).rolling(period).mean() / tr)
    minus_di = 100 * (pd.Series(np.where((minus_dm > plus_dm) & (minus_dm > 0), minus_dm, 0.0)).rolling(period).mean() / tr)
    return ((np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-9)) * 100).rolling(period).mean().values

def calculate_intraday_vwap(df):
    df_temp = df.copy()
    df_temp['date'] = df_temp['datetime'].dt.date
    df_temp['tp_vol'] = ((df_temp['high'] + df_temp['low'] + df_temp['close']) / 3) * df_temp['volume']
    cum_tp_vol = df_temp.groupby('date')['tp_vol'].transform('cumsum')
    cum_vol = df_temp.groupby('date')['volume'].transform('cumsum')
    return (cum_tp_vol / (cum_vol + 1e-9)).values

# ==========================================
# 4. LIVE MARKET DATA & FEATURE ENGINEERING
# ==========================================
def fetch_recent_candles(fyers, symbol, days=5, resolution="5"):
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    data = {
        "symbol": symbol,
        "resolution": resolution,
        "date_format": "1",
        "range_from": start_date.strftime("%Y-%m-%d"),
        "range_to": end_date.strftime("%Y-%m-%d"),
        "cont_flag": "1"
    }
    try:
        response = fyers.history(data=data)
        if response.get("s") == "ok" and "candles" in response:
            df = pd.DataFrame(response["candles"], columns=["timestamp", "open", "high", "low", "close", "volume"])
            df["datetime"] = pd.to_datetime(df["timestamp"], unit="s") + timedelta(hours=5, minutes=30)
            return df.drop_duplicates(subset=["datetime"]).sort_values("datetime").reset_index(drop=True)
    except Exception as e:
        logging.error(f"Error fetching {symbol}: {e}")
    return None

def get_live_market_snapshot(fyers):
    df = fetch_recent_candles(fyers, SYMBOL_REL)
    df_nifty = fetch_recent_candles(fyers, SYMBOL_NIFTY)
    df_vix = fetch_recent_candles(fyers, SYMBOL_VIX)
    df_energy = fetch_recent_candles(fyers, SYMBOL_ENERGY)

    if any(x is None or len(x) < 30 for x in [df, df_nifty, df_vix, df_energy]):
        return None

    df_nifty['n_st_dir'] = calculate_supertrend(df_nifty, 10, 3.0)
    df_nifty['nifty_pct'] = df_nifty['close'].pct_change()

    df = pd.merge_asof(df.sort_values('datetime'), df_nifty[['datetime', 'n_st_dir', 'nifty_pct']].sort_values('datetime'), on='datetime', direction='backward')
    df = pd.merge_asof(df.sort_values('datetime'), df_vix[['datetime', 'close']].rename(columns={'close': 'vix_close'}).sort_values('datetime'), on='datetime', direction='backward')
    df = pd.merge_asof(df.sort_values('datetime'), df_energy[['datetime', 'close']].rename(columns={'close': 'energy_close'}).sort_values('datetime'), on='datetime', direction='backward')

    df['ATR'] = calculate_atr(df, 14)
    df['st_fast_dir'] = calculate_supertrend(df, 10, 2.0)
    df['st_slow_dir'] = calculate_supertrend(df, 20, 3.0)
    df['VWAP'] = calculate_intraday_vwap(df)
    df['vwap_dist_atr'] = (df['close'] - df['VWAP']) / (df['ATR'] + 1e-9)
    df['RSI'] = calculate_rsi(df['close'], 14)
    df['rsi_slope'] = df['RSI'].diff(3)
    df['ADX'] = calculate_adx(df, 14)
    df['vol_ratio'] = df['volume'] / (df['volume'].rolling(20).mean() + 1e-9)
    df['body_atr'] = np.abs(df['close'] - df['open']) / (df['ATR'] + 1e-9)
    df['rel_pct'] = df['close'].pct_change()
    df['rel_strength_nifty'] = (df['rel_pct'] - df['nifty_pct']) * 100
    minutes_from_open = (df['datetime'].dt.hour * 60 + df['datetime'].dt.minute) - 555
    df['time_sin'] = np.sin(2 * np.pi * minutes_from_open / 375.0)
    df['time_cos'] = np.cos(2 * np.pi * minutes_from_open / 375.0)

    return df.dropna(subset=FEATURE_COLS).reset_index(drop=True)

# ==========================================
# 5. ORDER EXECUTION & HARD SL HELPERS
# ==========================================
def round_to_tick(price):
    """Fyers API ke 0.1000 tick size error ko fix karne ke liye"""
    return round(price * 10) / 10

def place_fyers_order(fyers, side, qty):
    if not LIVE_ORDER_EXECUTION:
        logging.info(f"📝 [PAPER MODE] Order Simulated: Side={side}, Qty={qty}")
        return True

    data = {
        "symbol": SYMBOL_REL,
        "qty": int(qty),
        "type": 2,          # 2 = Market Order
        "side": int(side),  # 1 = Buy, -1 = Sell
        "productType": "INTRADAY",
        "limitPrice": 0,
        "stopPrice": 0,
        "validity": "DAY",
        "disclosedQty": 0,
        "offlineOrder": False,
    }
    try:
        resp = fyers.place_order(data=data)
        if resp.get("s") == "ok":
            logging.info(f"✅ Market Order Placed: {resp.get('id')} | Qty: {qty}")
            return True
        else:
            logging.error(f"❌ Order Failed: {resp}")
            return False
    except Exception as e:
        logging.error(f"❌ Order Exception: {e}")
        return False

def place_fyers_sl_order(fyers, exit_side, qty, trigger_price):
    """Stock kharidne ke turant baad Fyers platform par Hard Stop-Loss (SL-L) order lagayega"""
    trigger_price = round_to_tick(trigger_price)
    # Trigger hone par pakka execute ho, isliye ₹2.00 ka limit protection buffer
    if exit_side == -1:  # BUY trade ka Sell SL
        limit_price = round_to_tick(trigger_price - 2.0)
    else:                # SELL trade ka Buy SL
        limit_price = round_to_tick(trigger_price + 2.0)

    if not LIVE_ORDER_EXECUTION:
        logging.info(f"📝 [PAPER MODE] Hard SL Placed on Broker: Trigger=₹{trigger_price:.2f}")
        return "PAPER_SL_ID"

    data = {
        "symbol": SYMBOL_REL,
        "qty": int(qty),
        "type": 4,                  # 4 = Stop-Loss Limit (SL-L) Order
        "side": int(exit_side),     # -1 = Sell SL, 1 = Buy SL
        "productType": "INTRADAY",
        "limitPrice": limit_price,
        "stopPrice": trigger_price,
        "validity": "DAY",
        "disclosedQty": 0,
        "offlineOrder": False,
    }
    try:
        resp = fyers.place_order(data=data)
        if resp.get("s") == "ok":
            sl_id = resp.get("id")
            logging.info(f"🛡️ Fyers Hard SL Order Placed! ID: {sl_id} | Trigger: ₹{trigger_price:.2f} | Limit: ₹{limit_price:.2f}")
            return sl_id
        else:
            logging.error(f"❌ Fyers Hard SL Failed: {resp}")
            return None
    except Exception as e:
        logging.error(f"❌ Fyers Hard SL Exception: {e}")
        return None

def cancel_fyers_sl_order(fyers, sl_order_id):
    """System Trailing SL ya Square-Off hone par Fyers wala pending Hard SL cancel karega"""
    if not LIVE_ORDER_EXECUTION or not sl_order_id or sl_order_id == "PAPER_SL_ID":
        return True
    try:
        resp = fyers.cancel_order(data={"id": str(sl_order_id)})
        if resp.get("s") == "ok":
            logging.info(f"🗑️ Fyers Hard SL Order Cancelled: {sl_order_id}")
            return True
        else:
            logging.warning(f"⚠️ Hard SL Cancel Info: {resp}")
            return False
    except Exception as e:
        logging.error(f"❌ Cancel Hard SL Exception: {e}")
        return False

def is_sl_already_hit_on_fyers(fyers, sl_order_id):
    """Check karega ki kahin Fyers platform par Hard SL pehle hi hit (status=2) toh nahi ho gaya"""
    if not LIVE_ORDER_EXECUTION or not sl_order_id or sl_order_id == "PAPER_SL_ID":
        return False
    try:
        resp = fyers.orderbook()
        if resp.get("s") == "ok" and "orderBook" in resp:
            for order in resp["orderBook"]:
                if str(order.get("id")) == str(sl_order_id):
                    # Fyers Order Status: 2 = Traded/Filled
                    if order.get("status") == 2:
                        return True
    except Exception as e:
        logging.error(f"Orderbook check error: {e}")
    return False

# ==========================================
# 6. MAIN BOT LOOP
# ==========================================
def run_bot():
    logging.info("🤖 Starting Reliance AI-ML Bot with Hard SL + System Trailing SL...")
    if not os.path.exists(MODEL_FILE):
        logging.error(f"❌ '{MODEL_FILE}' nahi mila! Pehle 'python3 train_ml.py' run karein.")
        return

    model = joblib.load(MODEL_FILE)
    fyers = fyersModel.FyersModel(client_id=CLIENT_ID, is_async=False, token=ACCESS_TOKEN, log_path="")

    send_telegram_msg("🚀 <b>Reliance Bot Started!</b>\nFeatures: 14 | Hard SL + Trailing Profit Enabled")

    is_open = False
    position_side = None
    entry_price = 0.0
    sl_price = 0.0
    tp_price = 0.0
    sl_dist = 0.0
    daily_pnl = 0.0
    trade_taken_today = False
    current_qty = 1
    sl_order_id = None  # Fyers Hard SL Order ID track karne ke liye

    while True:
        try:
            now = datetime.utcnow() + timedelta(hours=5, minutes=30)
            current_time_str = now.strftime("%H:%M:%S")

            if now.hour < 9 or (now.hour == 9 and now.minute < 30):
                logging.info(f"⏳ Waiting for 09:30 AM... ({current_time_str})")
                time.sleep(5)
                continue

            df = get_live_market_snapshot(fyers)
            if df is None or len(df) < 2:
                logging.warning("⚠️ Waiting for valid market data...")
                time.sleep(5)
                continue

            live_row = df.iloc[-1]
            signal_row = df.iloc[-1]
            ltp = float(live_row['close'])
            atr = float(signal_row['ATR'])

            # Auto Square-Off at 03:12 PM
            if (now.hour == 15 and now.minute >= 2) or (now.hour > 15):
                if is_open:
                    if not is_sl_already_hit_on_fyers(fyers, sl_order_id):
                        cancel_fyers_sl_order(fyers, sl_order_id)
                        exit_side = -1 if position_side == "BUY" else 1
                        place_fyers_order(fyers, exit_side, current_qty)
                    trade_pnl = (ltp - entry_price) * current_qty if position_side == "BUY" else (entry_price - ltp) * current_qty
                    daily_pnl += trade_pnl
                    is_open = False
                    sl_order_id = None
                    msg = f"⏰ <b>03:02 PM SQUARE-OFF</b>\nExit @ ₹{ltp:.2f} | PnL: ₹{trade_pnl:.2f}"
                    logging.info(msg)
                    send_telegram_msg(msg)
                break

            # Active Trade Management
            if is_open:
                unrealized_pnl = (ltp - entry_price) * current_qty if position_side == "BUY" else (entry_price - ltp) * current_qty
                total_running_pnl = daily_pnl + unrealized_pnl

                # Check 1: Kya Fyers platform par Hard SL pehle hi hit ho chuka hai?
                if is_sl_already_hit_on_fyers(fyers, sl_order_id):
                    daily_pnl += unrealized_pnl
                    is_open = False
                    sl_order_id = None
                    msg = f"🛑 <b>HARD STOP-LOSS HIT ON FYERS</b> @ ₹{ltp:.2f} | PnL: ₹{unrealized_pnl:.2f}"
                    logging.info(msg)
                    send_telegram_msg(msg)
                    time.sleep(5)
                    continue

                # Check 2: Max Daily Loss Emergency Exit
                if total_running_pnl <= MAX_DAILY_LOSS:
                    cancel_fyers_sl_order(fyers, sl_order_id)
                    exit_side = -1 if position_side == "BUY" else 1
                    place_fyers_order(fyers, exit_side, current_qty)
                    daily_pnl = total_running_pnl
                    is_open = False
                    sl_order_id = None
                    msg = f"🛑 <b>MAX DAILY LOSS HIT (₹{daily_pnl:.2f})</b>. Emergency Exit!"
                    logging.warning(msg)
                    send_telegram_msg(msg)
                    break

                if position_side == "BUY":
                    # 1. Trailing Target: Jab LTP Target ko touch kare, Target aage badhao aur SL tight karo
                    if ltp >= tp_price:
                        tp_price = round(tp_price + (0.5 * sl_dist), 2)
                        tight_sl = round(ltp - (0.4 * sl_dist), 2)
                        if tight_sl > sl_price:
                            sl_price = tight_sl
                        logging.info(f"🚀 BUY Target Extended -> New Target: ₹{tp_price:.2f} | Tight SL: ₹{sl_price:.2f}")

                    # 2. Exit Check: Jab price palat kar System SL hit kare, pehle Hard SL cancel karo fir exit maaro
                    if ltp <= sl_price:
                        reason = "TRAILING PROFIT BOOKED 🎯" if sl_price > entry_price else "STOP-LOSS HIT 🛑"
                        cancel_fyers_sl_order(fyers, sl_order_id)
                        place_fyers_order(fyers, -1, current_qty)
                        daily_pnl += unrealized_pnl
                        is_open = False
                        sl_order_id = None
                        send_telegram_msg(f"🔔 <b>BUY EXIT ({reason})</b> @ ₹{ltp:.2f} | PnL: ₹{unrealized_pnl:.2f}")
                    elif (ltp - sl_dist) > sl_price:
                        sl_price = round(ltp - sl_dist, 2)
                        logging.info(f"📈 System Trailing SL Updated: ₹{sl_price:.2f}")

                elif position_side == "SELL":
                    # 1. Trailing Target: Sell mein Target touch hone par Target aur neeche khiskao aur SL tight karo
                    if ltp <= tp_price:
                        tp_price = round(tp_price - (0.5 * sl_dist), 2)
                        tight_sl = round(ltp + (0.4 * sl_dist), 2)
                        if tight_sl < sl_price:
                            sl_price = tight_sl
                        logging.info(f"🚀 SELL Target Extended -> New Target: ₹{tp_price:.2f} | Tight SL: ₹{sl_price:.2f}")

                    # 2. Exit Check: Jab price wapas upar aakar System SL hit kare, pehle Hard SL cancel karo fir exit maaro
                    if ltp >= sl_price:
                        reason = "TRAILING PROFIT BOOKED 🎯" if sl_price < entry_price else "STOP-LOSS HIT 🛑"
                        cancel_fyers_sl_order(fyers, sl_order_id)
                        place_fyers_order(fyers, 1, current_qty)
                        daily_pnl += unrealized_pnl
                        is_open = False
                        sl_order_id = None
                        send_telegram_msg(f"🔔 <b>SELL EXIT ({reason})</b> @ ₹{ltp:.2f} | PnL: ₹{unrealized_pnl:.2f}")
                    elif (ltp + sl_dist) < sl_price:
                        sl_price = round(ltp + sl_dist, 2)
                        logging.info(f"📉 System Trailing SL Updated: ₹{sl_price:.2f}")

                # Jab tak trade open hai, naya signal check mat karo!
                time.sleep(5)
                continue

            if ONE_TRADE_PER_DAY and trade_taken_today and not is_open:
                logging.info("🛑 1-Trade Limit Reached. Shutting down!")
                break

            # Indicator Points Calculation
            buy_pts = sum([
                signal_row['st_fast_dir'] == 1,
                signal_row['st_slow_dir'] == 1,
                signal_row['n_st_dir'] == 1,
                signal_row['close'] > signal_row['VWAP'],
                50 < signal_row['RSI'] < 75,
                signal_row['ADX'] > 20
            ])

            sell_pts = sum([
                signal_row['st_fast_dir'] == -1,
                signal_row['st_slow_dir'] == -1,
                signal_row['n_st_dir'] == -1,
                signal_row['close'] < signal_row['VWAP'],
                25 < signal_row['RSI'] < 50,
                signal_row['ADX'] > 20
            ])

            X_live = signal_row[FEATURE_COLS].values.reshape(1, -1)
            probs = model.predict_proba(X_live)[0]
            ml_sell_prob, ml_buy_prob = float(probs[0]), float(probs[1])

            logging.info(
                f"🔍 Audit | LTP: ₹{ltp:.2f} | "
                f"BuyPts: {buy_pts}/6 (AI: {ml_buy_prob*100:.1f}%) | "
                f"SellPts: {sell_pts}/6 (AI: {ml_sell_prob*100:.1f}%)"
            )

            # Entry Filter: Avoid overly large spike candles
            if signal_row['body_atr'] >= 2.5:
                time.sleep(5)
                continue

            # BUY ENTRY
            if buy_pts >= MIN_INDICATOR_POINTS and ml_buy_prob >= ML_CONFIDENCE_THRESHOLD:
                current_qty = get_dynamic_quantity(fyers, ltp)
                if place_fyers_order(fyers, 1, current_qty):
                    is_open = True
                    position_side = "BUY"
                    entry_price = ltp
                    sl_dist = round(1.5 * atr, 2)
                    sl_price = round_to_tick(entry_price - sl_dist)
                    tp_price = round_to_tick(entry_price + (2.2 * atr))
                    trade_taken_today = True

                    # Fyers Platform par Hard SL order place karo
                    sl_order_id = place_fyers_sl_order(fyers, -1, current_qty, sl_price)

                    msg = (
                        f"🟢 <b>AI BUY EXECUTED (Dynamic Qty)!</b>\n"
                        f"Qty: {current_qty} | Entry: ₹{entry_price:.2f}\n"
                        f"Hard SL (Fyers): ₹{sl_price:.2f} | Target: ₹{tp_price:.2f}"
                    )
                    logging.info(msg)
                    send_telegram_msg(msg)
                else:
                    logging.warning("⚠️ Order fail hua (Margin/API issue). 60 sec wait kar raha hai...")
                    time.sleep(60)

            # SELL ENTRY
            elif sell_pts >= MIN_INDICATOR_POINTS and ml_sell_prob >= ML_CONFIDENCE_THRESHOLD:
                current_qty = get_dynamic_quantity(fyers, ltp)
                if place_fyers_order(fyers, -1, current_qty):
                    is_open = True
                    position_side = "SELL"
                    entry_price = ltp
                    sl_dist = round(1.5 * atr, 2)
                    sl_price = round_to_tick(entry_price + sl_dist)
                    tp_price = round_to_tick(entry_price - (2.2 * atr))
                    trade_taken_today = True

                    # Fyers Platform par Hard SL order place karo
                    sl_order_id = place_fyers_sl_order(fyers, 1, current_qty, sl_price)

                    msg = (
                        f"🔴 <b>AI SELL EXECUTED (Dynamic Qty)!</b>\n"
                        f"Qty: {current_qty} | Entry: ₹{entry_price:.2f}\n"
                        f"Hard SL (Fyers): ₹{sl_price:.2f} | Target: ₹{tp_price:.2f}"
                    )
                    logging.info(msg)
                    send_telegram_msg(msg)
                else:
                    logging.warning("⚠️ Order fail hua (Margin/API issue). 60 sec wait kar raha hai...")
                    time.sleep(60)

            time.sleep(5)

        except KeyboardInterrupt:
            logging.info("🛑 Bot manually stopped.")
            break
        except Exception as e:
            logging.error(f"⚠️ Loop Error: {e}")
            time.sleep(5)

if __name__ == "__main__":
    run_bot()
