import os
import time
import hmac
import hashlib
import json
import secrets
import threading
from datetime import datetime

import requests
import joblib
import numpy as np
import pandas as pd
from flask import Flask, jsonify, request
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import RobustScaler
from sklearn.pipeline import Pipeline
from apscheduler.schedulers.background import BackgroundScheduler

app = Flask(__name__)
try:
    current_ip = requests.get("https://api.ipify.org").text.strip()
    print(f"===> RENDER OUTGOING IP: {current_ip}")
except Exception as e:
    print(f"Could not fetch IP: {e}")

# =========================================================
# CONFIGURATION (LIVE SCALPING MODE)
# =========================================================
BASE_URL = os.environ.get(
    "DELTA_BASE_URL", "https://api.india.delta.exchange"
).strip().rstrip("/")
API_KEY = os.environ.get("DELTA_API_KEY", "").strip()
API_SECRET = os.environ.get("DELTA_API_SECRET", "").strip()

PAIRS = {
    "BTCUSD": {"model_file": "ai_brain_btc_v2.pkl"},
    "ETHUSD": {"model_file": "ai_brain_eth_v2.pkl"},
}

MEMORY_FILE = os.environ.get("MEMORY_FILE", "trade_memory.json")
BOT_TRIGGER_TOKEN = os.environ.get("BOT_TRIGGER_TOKEN", "")
MAX_BUDGET_INR = float(os.environ.get("MAX_BUDGET_INR", "1000"))
CONFIDENCE_BASE_THRESHOLD = 0.65
MAX_LEVERAGE = 3

FEATURES = [
    "return", "ma7", "ma25", "volume", "atr", "rsi",
    "lower_wick_ratio", "upper_wick_ratio", "vol_surge",
    "norm_atr", "wick_skew", "dist_ma25", "adx", "chop_index"
]

scheduler = BackgroundScheduler(timezone="Asia/Kolkata")
training_lock = threading.Lock()

# =========================================================
# AUTHENTICATION
# =========================================================
def generate_signature(secret, method, path, query="", payload=""):
    timestamp = str(int(time.time()))
    message = method.upper() + timestamp + path + query + payload
    signature = hmac.new(
        secret.encode("utf-8"),
        message.encode("utf-8"),
        hashlib.sha256
    ).hexdigest()
    return signature, timestamp


def get_headers(method, path, query="", payload=""):
    if not API_KEY or not API_SECRET:
        raise RuntimeError("Delta API credentials are missing.")
    signature, timestamp = generate_signature(
        API_SECRET, method, path, query, payload
    )
    return {
        "api-key": API_KEY,
        "signature": signature,
        "timestamp": timestamp,
        "Content-Type": "application/json",
        "Accept": "application/json",
        "User-Agent": "Crypto-AI-Trading-Bot/2.0",
    }


# =========================================================
# MEMORY MANAGEMENT
# =========================================================
def default_memory():
    return {
        "loss_penalty": 0.0,
        "total_trades": 0,
        "daily_trade_count": 0,
        "last_trade_date": "",
        "losses": 0,
        "wins": 0,
        "open_position": None,
    }


def load_memory():
    if not os.path.exists(MEMORY_FILE):
        return default_memory()
    try:
        with open(MEMORY_FILE, "r", encoding="utf-8") as f:
            memory = default_memory()
            loaded = json.load(f)
            if isinstance(loaded, dict):
                memory.update(loaded)
            return memory
    except Exception as exc:
        print(f"[MEMORY ERROR] {exc}")
        return default_memory()


def save_memory(data):
    try:
        temp_file = MEMORY_FILE + ".tmp"
        with open(temp_file, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4)
        os.replace(temp_file, MEMORY_FILE)
    except Exception as exc:
        print(f"[MEMORY SAVE ERROR] {exc}")


# =========================================================
# PRODUCT SPECS & BALANCES
# =========================================================
def get_product_specs(symbol):
    url = f"{BASE_URL}/v2/products/{symbol}"
    response = requests.get(url, timeout=10, headers={"Accept": "application/json"})
    response.raise_for_status()
    data = response.json()
    if not data.get("success") or not isinstance(data.get("result"), dict):
        raise RuntimeError(f"Could not retrieve Delta product metadata for {symbol}: {data}")
    product = data["result"]
    product_id = product.get("id")
    contract_value = product.get("contract_value")
    tick_size = product.get("tick_size")
    if product_id is None or contract_value in (None, "", 0, "0") or tick_size in (None, "", 0, "0"):
        raise RuntimeError(f"Incomplete product metadata for {symbol}; refusing to trade.")
    return {
        "product_id": int(product_id),
        "contract_value": float(contract_value),
        "tick_size": float(tick_size),
        "symbol": str(product.get("symbol", symbol)),
        "state": str(product.get("state", "")).lower(),
    }


def round_to_tick(price, tick_size):
    if tick_size <= 0:
        return float(price)
    return round(round(float(price) / tick_size) * tick_size, 8)


def get_available_balance():
    try:
        path = "/v2/wallet/balances"
        response = requests.get(BASE_URL + path, headers=get_headers("GET", path), timeout=10)
        response.raise_for_status()
        data = response.json()
        if not data.get("success"):
            print(f"[BALANCE API ERROR] {data}")
            return 0.0

        for item in data.get("result", []):
            bal = float(item.get("available_balance") or item.get("balance") or 0.0)
            if bal > 0:
                symbol = str(item.get("asset_symbol", "")).upper()
                if symbol in ["USD", "USDT"] or bal < 100.0:
                    return bal * 85.0
                return bal

        return 0.0
    except Exception as exc:
        print(f"[BALANCE ERROR] {exc}")
        return 0.0


def get_live_position():
    try:
        path = "/v2/positions/margined"
        response = requests.get(
            BASE_URL + path,
            headers=get_headers("GET", path),
            timeout=10,
        )
        response.raise_for_status()
        data = response.json()
        if data.get("success"):
            for position in data.get("result", []) or []:
                if abs(float(position.get("size", 0) or 0)) > 0:
                    return position
        return None
    except Exception as exc:
        print(f"[POSITION ERROR] {exc}")
        return None


# =========================================================
# MARKET DATA (1-MINUTE SCALPING RESOLUTION)
# =========================================================
def fetch_market_data(symbol, limit_candles=300):
    try:
        now = int(time.time())
        start_time = now - limit_candles * 60
        base_urls = list(dict.fromkeys([
            BASE_URL,
            "https://api.india.delta.exchange",
            "https://api.delta.exchange",
        ]))
        candles = None

        for base in base_urls:
            url = f"{base}/v2/history/candles"
            params = {
                "resolution": "1m",
                "symbol": symbol,
                "start": start_time,
                "end": now,
            }
            try:
                response = requests.get(
                    url,
                    params=params,
                    timeout=15,
                    headers={"Accept": "application/json", "User-Agent": "TradingBot/2.0"},
                )
                if response.status_code != 200:
                    continue
                data = response.json()
                if data.get("success") is False:
                    continue
                result = data.get("result")
                if isinstance(result, list) and result:
                    candles = result
                    break
            except Exception as exc:
                print(f"[API ATTEMPT ERROR {symbol}] {exc}")

        if not candles:
            return None

        df = pd.DataFrame(candles).rename(
            columns={"o": "open", "h": "high", "l": "low", "c": "close", "v": "volume"}
        )
        required = ["open", "high", "low", "close", "volume"]
        for column in required:
            if column not in df.columns:
                return None
            df[column] = pd.to_numeric(df[column], errors="coerce")

        df = df.dropna(subset=required).copy()
        if "time" in df.columns:
            df["time"] = pd.to_numeric(df["time"], errors="coerce")
            df = df.dropna(subset=["time"]).sort_values("time")
            df = df.drop_duplicates(subset=["time"], keep="last")
        return df.reset_index(drop=True)
    except Exception as exc:
        print(f"[FETCH ERROR {symbol}] {exc}")
        return None


def fetch_funding_rate(symbol):
    for base in dict.fromkeys([BASE_URL, "https://api.india.delta.exchange", "https://api.delta.exchange"]):
        try:
            response = requests.get(f"{base}/v2/tickers/{symbol}", timeout=5)
            data = response.json()
            if data.get("success"):
                result = data.get("result", {})
                return float(result.get("funding_rate", 0.0) or 0.0)
        except Exception:
            continue
    return 0.0


def fetch_order_book_metrics(symbol):
    for base in dict.fromkeys([BASE_URL, "https://api.india.delta.exchange", "https://api.delta.exchange"]):
        try:
            response = requests.get(f"{base}/v2/l2orderbook/{symbol}", timeout=5)
            data = response.json()
            if data.get("success"):
                book = data.get("result", {})
                bids = book.get("buy", [])
                asks = book.get("sell", [])
                if bids and asks:
                    bid_size = sum(float(item.get("size", 0) or 0) for item in bids[:10])
                    ask_size = sum(float(item.get("size", 0) or 0) for item in asks[:10])
                    total = bid_size + ask_size + 1e-9
                    return (bid_size - ask_size) / total, bid_size, ask_size
        except Exception:
            continue
    return 0.0, 0.0, 0.0


# =========================================================
# INDICATORS
# =========================================================
def add_indicators(df):
    df = df.copy()
    df["ma7"] = df["close"].rolling(7).mean()
    df["ma25"] = df["close"].rolling(25).mean()
    df["return"] = df["close"].pct_change()

    high_low = df["high"] - df["low"]
    high_close = (df["high"] - df["close"].shift()).abs()
    low_close = (df["low"] - df["close"].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df["atr"] = tr.rolling(14).mean()

    delta = df["close"].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / (loss + 1e-9)
    df["rsi"] = 100 - (100 / (1 + rs))

    candle_range = (df["high"] - df["low"]).replace(0, 1e-9)
    body_low = df[["open", "close"]].min(axis=1)
    body_high = df[["open", "close"]].max(axis=1)
    df["lower_wick_ratio"] = (body_low - df["low"]) / candle_range
    df["upper_wick_ratio"] = (df["high"] - body_high) / candle_range

    vol_ma20 = df["volume"].rolling(20).mean()
    df["vol_surge"] = df["volume"] / (vol_ma20 + 1e-9)
    df["norm_atr"] = df["atr"] / (df["close"] + 1e-9)
    df["wick_skew"] = df["lower_wick_ratio"] - df["upper_wick_ratio"]
    df["dist_ma25"] = (df["close"] - df["ma25"]) / (df["ma25"] + 1e-9)

    up = df["high"] - df["high"].shift(1)
    down = df["low"].shift(1) - df["low"]
    plus_dm = pd.Series(np.where((up > down) & (up > 0), up, 0.0), index=df.index)
    minus_dm = pd.Series(np.where((down > up) & (down > 0), down, 0.0), index=df.index)
    tr_smooth = tr.rolling(14).sum()
    plus_di = 100 * plus_dm.rolling(14).sum() / (tr_smooth + 1e-9)
    minus_di = 100 * minus_dm.rolling(14).sum() / (tr_smooth + 1e-9)
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di + 1e-9)
    df["adx"] = dx.rolling(14).mean()

    sum_tr = tr.rolling(14).sum()
    max_h = df["high"].rolling(14).max()
    min_l = df["low"].rolling(14).min()
    ratio = sum_tr / (max_h - min_l + 1e-9)
    df["chop_index"] = 100 * (np.log10(ratio.clip(lower=1e-9)) / np.log10(14))

    df["recent_low"] = df["low"].shift(1).rolling(20).min()
    df["recent_high"] = df["high"].shift(1).rolling(20).max()
    df["target"] = (df["close"].shift(-1) > df["close"]).astype(int)
    return df


# =========================================================
# WHALE-TRAP SCANNER
# =========================================================
def check_institutional_sweep(df, symbol):
    if len(df) < 30:
        return None
    latest = df.iloc[-1]
    atr = float(latest["atr"])
    if not np.isfinite(atr) or atr <= 0:
        return None

    obi_ratio, _, _ = fetch_order_book_metrics(symbol)
    funding_rate = fetch_funding_rate(symbol)
    tick_size = get_product_specs(symbol)["tick_size"]

    if (
        latest["low"] < latest["recent_low"]
        and latest["close"] > latest["recent_low"]
        and obi_ratio > 0.30
        and latest["lower_wick_ratio"] >= 0.35
        and latest["vol_surge"] >= 1.20
        and funding_rate < 0.04
    ):
        risk_dist = max(latest["close"] * 0.002, abs(latest["close"] - latest["low"]))
        sl = round_to_tick(latest["close"] - risk_dist, tick_size)
        tp = round_to_tick(latest["close"] + (risk_dist * 2.0), tick_size)
        return ("BUY", 0.90, sl, tp, "SCALP_SWEEP")

    if (
        latest["high"] > latest["recent_high"]
        and latest["close"] < latest["recent_high"]
        and obi_ratio < -0.30
        and latest["upper_wick_ratio"] >= 0.30
        and latest["vol_surge"] >= 1.20
        and funding_rate > -0.04
    ):
        risk_dist = max(latest["close"] * 0.002, abs(latest["high"] - latest["close"]))
        sl = round_to_tick(latest["close"] + risk_dist, tick_size)
        tp = round_to_tick(latest["close"] - (risk_dist * 2.0), tick_size)
        return ("SELL", 0.90, sl, tp, "SCALP_SWEEP")
        
    return None


# =========================================================
# AI TRAINING & MODEL LOADING
# =========================================================
def build_ai_pipeline():
    return Pipeline([
        ("scaler", RobustScaler()),
        ("model", GradientBoostingClassifier(
            n_estimators=150,
            learning_rate=0.06,
            max_depth=3,
            subsample=0.85,
            random_state=42,
        )),
    ])


def train_and_save_ai_brain_for_pair(symbol, model_file):
    if not training_lock.acquire(blocking=False):
        return None
    try:
        df = fetch_market_data(symbol, limit_candles=1000)
        if df is None or len(df) < 150:
            return None

        df = add_indicators(df)
        clean = df.dropna(subset=FEATURES + ["target"]).copy()
        if len(clean) < 80:
            return None

        X = clean[FEATURES].iloc[:-1]
        y = clean["target"].iloc[:-1]
        if y.nunique() < 2:
            return None

        pipeline = build_ai_pipeline()
        pipeline.fit(X, y)
        joblib.dump(pipeline, model_file)
        return pipeline
    except Exception as exc:
        print(f"[AI TRAIN ERROR {symbol}] {exc}")
        return None
    finally:
        training_lock.release()


def train_all_ai_brains():
    for symbol, config in PAIRS.items():
        train_and_save_ai_brain_for_pair(symbol, config["model_file"])


def get_or_load_ai_brain(symbol):
    model_file = PAIRS[symbol]["model_file"]
    if os.path.exists(model_file):
        try:
            return joblib.load(model_file)
        except Exception:
            pass
    return train_and_save_ai_brain_for_pair(symbol, model_file)
    # =========================================================
# POSITION SIZING
# =========================================================
def calculate_contracts(symbol, balance, leverage, entry_price):
    try:
        specs = get_product_specs(symbol)
        contract_value = float(specs.get("contract_value") or 0.0)
        if balance <= 0 or entry_price <= 0 or contract_value <= 0:
            return 0

        balance_usd = (balance / 85.0) if balance > 100.0 else balance
        usable_margin = balance_usd * 0.90
        eff_leverage = max(1, min(int(leverage or 3), int(MAX_LEVERAGE)))
        
        margin_per_contract = (entry_price * contract_value) / eff_leverage
        if margin_per_contract <= 0:
            return 0

        qty = int(usable_margin / margin_per_contract)
        if qty == 0 and usable_margin >= margin_per_contract * 0.70:
            qty = 1

        return max(0, qty)
    except Exception as exc:
        print(f"[SIZING ERROR {symbol}] {exc}")
        return 0


# =========================================================
# PREDICTION ENGINE (SCALPING LOGIC)
# =========================================================
def predict_signal(symbol, df):
    if df is None or len(df) < 30:
        return ("HOLD", 0.0, 0.0, 0.0, 0, "INSUFFICIENT_DATA")

    df = add_indicators(df)
    clean = df.dropna(subset=FEATURES).copy()
    if len(clean) < 20:
        return ("HOLD", 0.0, 0.0, 0.0, 0, "INSUFFICIENT_FEATURE_DATA")

    candle = clean.iloc[-2]
    chop = float(candle["chop_index"])
    adx = float(candle["adx"])
    if chop > 61.8 and adx < 20:
        return ("HOLD", 0.0, 0.0, 0.0, 0, "REGIME_FILTER_CHOPPY")

    institutional_signal = check_institutional_sweep(clean.iloc[:-1], symbol)
    if institutional_signal:
        action, confidence, sl, tp, tag = institutional_signal
        return (action, confidence, sl, tp, 3, tag)

    pipeline = get_or_load_ai_brain(symbol)
    if pipeline is None:
        return ("HOLD", 0.0, 0.0, 0.0, 0, "AI_TRAIN_FAIL")

    try:
        features = clean[FEATURES].iloc[[-2]]
        probabilities = pipeline.predict_proba(features)[0]
    except Exception as exc:
        print(f"[PREDICTION ERROR {symbol}] {exc}")
        return ("HOLD", 0.0, 0.0, 0.0, 0, "PREDICTION_ERROR")

    prob_up, prob_down = float(probabilities[1]), float(probabilities[0])
    atr = float(candle["atr"])
    close = float(candle["close"])

    memory = load_memory()
    threshold = min(0.85, CONFIDENCE_BASE_THRESHOLD + float(memory.get("loss_penalty", 0.0)))
    funding_rate = fetch_funding_rate(symbol)

    if prob_up >= threshold and funding_rate < 0.035:
        action, confidence = "BUY", prob_up
    elif prob_down >= threshold and funding_rate > -0.035:
        action, confidence = "SELL", prob_down
    else:
        return ("HOLD", max(prob_up, prob_down), 0.0, 0.0, 0, f"AI_WAIT (REQ:{threshold:.2f})")

    leverage = MAX_LEVERAGE
    tick_size = get_product_specs(symbol)["tick_size"]
    
    # Scalping targets
    sl_distance = atr * 0.75
    tp_distance = atr * 1.50
    
    if action == "BUY":
        sl = round_to_tick(close - sl_distance, tick_size)
        tp = round_to_tick(close + tp_distance, tick_size)
    else:
        sl = round_to_tick(close + sl_distance, tick_size)
        tp = round_to_tick(close - tp_distance, tick_size)

    # Fee buffer filter
    if abs(tp - close) / close < 0.0020:
        return ("HOLD", 0.0, 0.0, 0.0, 0, "SKIPPED_TARGET_TOO_SMALL_FOR_FEES")

    return (
        action, confidence, sl, tp, leverage,
        f"SCALP_AI_{symbol} (UP:{prob_up:.2f}, DOWN:{prob_down:.2f})"
    )


# =========================================================
# LIVE ORDER PLACEMENT (SERVER-SIDE TRAILING BRACKET)
# =========================================================
def place_order_with_brackets(symbol, action, size, stop_loss, take_profit):
    if size <= 0:
        return {"success": False, "error": "Invalid order size."}

    try:
        specs = get_product_specs(symbol)
        if specs["state"] not in ("", "live", "active", "open"):
            return {"success": False, "error": "Product not live"}

        path = "/v2/orders"
        tick_size = float(specs.get("tick_size", 0.1))

        tp_val = round_to_tick(float(take_profit), tick_size)
        trail_val = round_to_tick(abs(float(take_profit) - float(stop_loss)) * 0.45, tick_size)

        payload = {
            "product_id": specs["product_id"],
            "size": int(size),
            "side": "buy" if action == "BUY" else "sell",
            "order_type": "market_order",
            "bracket_take_profit_price": str(tp_val),
            "bracket_stop_trigger_method": "mark_price",
            "bracket_trail_amount": str(trail_val)
        }

        payload_str = json.dumps(payload, separators=(",", ":"))
        headers = get_headers("POST", path, payload=payload_str)

        response = requests.post(
            BASE_URL + path,
            headers=headers,
            data=payload_str,
            timeout=15
        )

        try:
            result = response.json()
        except ValueError:
            result = {
                "success": False,
                "error": f"Non-JSON response (HTTP {response.status_code})"
            }

        if response.status_code >= 400 or not result.get("success"):
            print(
                f"[LIVE ORDER API REJECTED {symbol}] "
                f"HTTP={response.status_code}, response={result}"
            )
            return {
                "success": False,
                "http_status": response.status_code,
                "exchange_response": result
            }

        return result

    except Exception as exc:
        print(f"[LIVE ORDER ERROR {symbol}] {exc}")
        return {"success": False, "error": str(exc)}


# =========================================================
# FLASK ROUTES
# =========================================================
@app.route("/", methods=["GET"])
@app.route("/health", methods=["GET"])
def health_check():
    return jsonify({
        "status": "ok",
        "mode": "LIVE_REAL_TRADING",
        "service": "Crypto Scalping AI Bot",
        "timeframe": "1m",
        "pairs": list(PAIRS.keys()),
    }), 200


@app.route("/execute-trade", methods=["GET"])
def execute_trade():
    if not BOT_TRIGGER_TOKEN:
        return jsonify({"status": "TRIGGER_NOT_CONFIGURED", "message": "Set BOT_TRIGGER_TOKEN."}), 503
    supplied_token = request.headers.get("X-Bot-Token", "")
    if not secrets.compare_digest(supplied_token, BOT_TRIGGER_TOKEN):
        return jsonify({"status": "UNAUTHORIZED"}), 401

    active_position = get_live_position()
    if active_position:
        symbol = active_position.get("product_symbol", active_position.get("symbol", "BTCUSD"))
        return jsonify({
            "status": "POSITION_ALREADY_OPEN",
            "active_symbol": symbol,
            "message": "Live position is open; Delta server-side trailing bracket is active.",
        }), 200

    balance = get_available_balance()
    if balance <= 0:
        return jsonify({
            "status": "FAILED_NO_BALANCE",
            "balance": balance,
            "message": "Delta wallet ma trading mate balance nathi."
        }), 200

    best_candidate = None

    for symbol in PAIRS:
        market_data = fetch_market_data(symbol, limit_candles=200)
        if market_data is None or market_data.empty:
            continue

        action, confidence, sl, tp, leverage, strategy = predict_signal(symbol, market_data)
        latest_close = float(market_data["close"].iloc[-1])

        if action in ("BUY", "SELL"):
            if best_candidate is None or confidence > best_candidate["confidence"]:
                best_candidate = {
                    "symbol": symbol,
                    "action": action,
                    "confidence": confidence,
                    "sl": sl,
                    "tp": tp,
                    "leverage": leverage,
                    "strategy": strategy,
                    "latest_close": latest_close,
                }

    if best_candidate:
        symbol = best_candidate["symbol"]
        action = best_candidate["action"]

        contracts = calculate_contracts(
            symbol, balance, best_candidate["leverage"], best_candidate["latest_close"]
        )
        if contracts <= 0:
            return jsonify({"status": "SKIPPED_BUDGET_TOO_SMALL", "symbol": symbol}), 200

        order_result = place_order_with_brackets(
            symbol, action, contracts, best_candidate["sl"], best_candidate["tp"]
        )

        if order_result.get("success") is not True:
            return jsonify({
                "status": "ORDER_FAILED",
                "symbol": symbol,
                "exchange_response": order_result,
            }), 502

        memory = load_memory()
        memory["open_position"] = {
            "symbol": symbol,
            "side": action,
            "entry": best_candidate["latest_close"],
            "sl": best_candidate["sl"],
            "tp": best_candidate["tp"],
            "contracts": contracts,
            "time": int(time.time()),
        }
        memory["total_trades"] = memory.get("total_trades", 0) + 1
        save_memory(memory)

        return jsonify({
            "status": "LIVE_ORDER_SUBMITTED",
            "symbol": symbol,
            "action": action,
            "confidence": f"{best_candidate['confidence'] * 100:.2f}%",
            "contracts": contracts,
            "leverage": f"{best_candidate['leverage']}x",
            "entry": best_candidate["latest_close"],
            "stop_loss": best_candidate["sl"],
            "take_profit": best_candidate["tp"],
            "strategy": best_candidate["strategy"],
            "exchange_response": order_result,
        }), 200

    return jsonify({
        "status": "WAIT_AND_SEE",
        "action": "HOLD",
        "scanned_pairs": list(PAIRS.keys()),
        "message": "No qualifying scalp signal found.",
        "balance": balance,
    }), 200


# =========================================================
# BACKGROUND INITIALIZATION
# =========================================================
def init_background_training():
    time.sleep(15)
    for symbol, config in PAIRS.items():
        if not os.path.exists(config["model_file"]):
            train_and_save_ai_brain_for_pair(symbol, config["model_file"])


def start_scheduler():
    if not scheduler.get_job("train_all_ai_brains"):
        scheduler.add_job(
            func=train_all_ai_brains,
            trigger="cron",
            day_of_week="sun",
            hour=23,
            minute=0,
            id="train_all_ai_brains",
            replace_existing=True,
            max_instances=1,
            coalesce=True,
        )
    if not scheduler.running:
        scheduler.start()


if os.environ.get("ENABLE_BACKGROUND_JOBS", "false").lower() == "true":
    start_scheduler()
    training_thread = threading.Thread(target=init_background_training, name="ai-brain-startup", daemon=True)
    training_thread.start()

if __name__ == "__main__":
    port = int(os.environ.get("PORT", "10000"))
    app.run(host="0.0.0.0", port=port)
        
