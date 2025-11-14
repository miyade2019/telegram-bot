# -- coding: utf-8 --
"""
بوت تداول إشارات + تيليغرام + رصد الحيتان + تعليم الأهداف + رسم بياني
نسخة معدلة لعدم حظر IP عبر تقليل الطلبات + كاش للتيكر + مهلات بين الأزواج
"""

import os, csv, time, html, traceback, requests
from datetime import datetime, timezone
import pandas as pd
import numpy as np
import ccxt

# ===== Matplotlib (للرسم بدون شاشة) =====
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

# ==================== إعدادات عامة ====================
API_KEY     = "8259936456:AAHYckhlv6swNQCPctBshVEI36qQw817zK4"
API_SECRET  = "-1002711156609"
EXCHANGE_ID = "binance"

# الأزواج الكاملة
SYMBOLS = [
    "SOL/USDT", "LINK/USDT", "JUP/USDT", "INJ/USDT", "TIA/USDT", "AVAX/USDT",
    "SOMI/USDT", "ADA/USDT", "FIL/USDT", "SUI/USDT", "WLD/USDT",
    "ARB/USDT", "CTSI/USDT", "TON/USDT", "PEPE/USDT", "TRUMP/USDT", "PYTH/USDT",
    "FET/USDT", "SEI/USDT", "DOGE/USDT", "GRT/USDT", "ETC/USDT","ONDO/USDT"
]

# الأزواج التي يتم فيها رصد الحيتان فقط (تخفيف)
WHALE_SYMBOLS = ["BTC/USDT", "ETH/USDT", "SOL/USDT", "BNB/USDT", "XRP/USDT"]

TIMEFRAME   = "5m"
FETCH_LIMIT = 300

DRY_RUN       = True
SLEEP_SECONDS = 120    # كان 60 → الآن 120 لتقليل الطلبات

# إدارة المخاطر
RISK_PER_TRADE       = 0.005
MAX_POSITION_PERCENT = 0.10
MIN_USDT_FOR_TRADE   = 10.0

# مؤشرات
ATR_PERIOD            = 14
SUPER_TREND_ATR_MULT  = 3.0
TRAIL_ATR_MULT        = 2.5
DONCHIAN_PERIOD       = 14
ADX_PERIOD            = 14
ADX_THRESHOLD         = 20
RSI_PERIOD            = 14
RSI_ENTRY_LOW         = 35
RSI_ENTRY_HIGH        = 75
EMA_FAST              = 20
EMA_MED               = 50
EMA_SLOW              = 200
MACD_FAST             = 12
MACD_SLOW             = 26
MACD_SIGNAL           = 9
STOCH_K               = 14
STOCH_D               = 3
VOL_MA                = 20
MIN_VOL_RATIO         = 1.00
TP_MULTS = [1.03, 1.06, 1.12]
COOLDOWN_MINUTES      = 45
ONE_SIGNAL_PER_BAR    = True
REQUIRE_DONCHIAN_BREAK = True
REQUIRE_VWAP           = True
USE_MACD_FILTER        = True

# ==================== رصد الحيتان ====================
WHALE_MIN_USDT     = 1_000_000
WHALE_LOOKBACK_MIN = 10   # كان 5 → الآن 10
WHALE_COOLDOWN_MIN = 10   # كان 15 → الآن 10

# ==================== بروكسي (اختياري) ====================
USE_PROXY  = False
HTTP_PROXY = "http://127.0.0.1:7890"
REQUESTS_PROXIES = {"http": HTTP_PROXY, "https": HTTP_PROXY} if USE_PROXY else None

# ==================== Telegram ====================
ENABLE_TELEGRAM = True
TELEGRAM_TOKEN  = "8259936456:AAHYckhlv6swNQCPctBshVEI36qQw817zK4"
TELEGRAM_CHAT_ID= "-1002711156609"

def _esc(s): return html.escape(str(s), quote=False)

def retry_call(fn, *args, max_tries=3, base_delay=1.0, exceptions=(Exception,), **kwargs):
    last_err = None
    for attempt in range(1, max_tries + 1):
        try:
            return fn(*args, **kwargs)
        except exceptions as e:
            last_err = e
            if attempt == max_tries: break
            time.sleep(base_delay * (2 ** (attempt - 1)))
    raise last_err

def tele_send(text, buttons=None):
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
        payload = {"chat_id": TELEGRAM_CHAT_ID, "text": text,
                   "parse_mode": "HTML", "disable_web_page_preview": True}
        if buttons:
            payload["reply_markup"] = {"inline_keyboard": buttons}
        r = retry_call(requests.post, url, json=payload, timeout=15,
                       exceptions=(requests.exceptions.RequestException,),
                       proxies=REQUESTS_PROXIES)
        return r.json().get("result", {}).get("message_id")
    except: return None

def tele_edit(mid, new_text, buttons=None):
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/editMessageText"
        payload = {"chat_id": TELEGRAM_CHAT_ID, "message_id": mid,
                   "text": new_text, "parse_mode": "HTML",
                   "disable_web_page_preview": True}
        if buttons:
            payload["reply_markup"] = {"inline_keyboard": buttons}
        retry_call(requests.post, url, json=payload, timeout=15,
                   exceptions=(requests.exceptions.RequestException,),
                   proxies=REQUESTS_PROXIES)
    except:
        pass

def tele_send_photo(path, caption=""):
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendPhoto"
        with open(path, "rb") as f:
            data = {"chat_id": TELEGRAM_CHAT_ID, "caption": caption, "parse_mode": "HTML"}
            retry_call(requests.post, url, data=data, files={"photo": f}, timeout=30,
                       exceptions=(requests.exceptions.RequestException,),
                       proxies=REQUESTS_PROXIES)
    except:
        pass

def binance_pair_link(symbol):
    return f"https://www.binance.com/en/trade/{symbol.replace('/','_')}"

# ==================== سجلات ====================
LOG_DIR, CHART_DIR = "bot_logs", os.path.join("bot_logs", "charts")
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(CHART_DIR, exist_ok=True)

# ……………………… (كل كود المؤشرات + الرسم كما هو دون أي تغيير) ………………………

# ==================== اتصال المنصة ====================
def ccxt_call_with_retry(fn, *args, max_tries=3, **kwargs):
    from ccxt.base.errors import NetworkError, ExchangeNotAvailable, RequestTimeout, DDoSProtection
    last_err = None
    for attempt in range(1, max_tries + 1):
        try:
            return fn(*args, **kwargs)
        except (NetworkError, ExchangeNotAvailable, RequestTimeout, DDoSProtection) as e:
            last_err = e
            if attempt == max_tries: break
            time.sleep(1.0 * attempt)
    raise last_err

def create_exchange():
    base = {
        "enableRateLimit": True,
        "timeout": 20000,
        "options": {"defaultType": "spot", "adjustForTimeDifference": True}
    }
    if USE_PROXY:
        base["proxies"] = {"http": HTTP_PROXY, "https": HTTP_PROXY}
    if not DRY_RUN:
        base.update({"apiKey": API_KEY, "secret": API_SECRET})

    hosts = ["api.binance.com", "api1.binance.com", "api2.binance.com"]
    for host in hosts:
        try:
            params = dict(base); params["hostname"] = host
            ex = getattr(ccxt, EXCHANGE_ID)(params)
            ccxt_call_with_retry(ex.publicGetTime)
            print(f"[create_exchange] Using host: {host}")
            return ex
        except Exception as e:
            print(f"[create_exchange] failed host {host}: {e}")
            time.sleep(1)

# ==================== رصد الحيتان ====================
_last_whale_alert = {}

def check_whales(exchange, symbol):
    if symbol not in WHALE_SYMBOLS:
        return
    
    now_ms = int(time.time() * 1000)
    since_ms = now_ms - WHALE_LOOKBACK_MIN * 60 * 1000

    last = _last_whale_alert.get(symbol)
    if last:
        mins = (datetime.now(timezone.utc) - last).total_seconds() / 60
        if mins < WHALE_COOLDOWN_MIN:
            return

    try:
        trades = ccxt_call_with_retry(exchange.fetch_trades, symbol, since=since_ms, limit=200)
    except:
        return
    
    if not trades: return

    for tr in trades:
        cost = tr.get("cost") or tr["price"] * tr["amount"]
        if cost >= WHALE_MIN_USDT:
            t_iso = datetime.now(timezone.utc).isoformat()
            tele_send(
                f"🐳 <b>صفقة حوت</b> على <b>{symbol}</b>\n"
                f"قيمة تقريبية: <code>{int(cost):,} USDT</code>\n"
                f"الوقت: {t_iso}\n"
                f"🔗 {binance_pair_link(symbol)}"
            )
            _last_whale_alert[symbol] = datetime.now(timezone.utc)
            return

# ==================== الحلقة الرئيسية ====================
_last_signal_time = {}
_last_bar_time    = {}
ticker_cache      = {}     # ← كاش للتيكر

def run_bot():
    exchange = create_exchange()
    tele_send("🚀 البوت يعمل الآن • DRY_RUN=" + str(DRY_RUN))

    open_positions = {}
    signal_msgs    = {}

    while True:
        try:
            for sym in SYMBOLS:

                time.sleep(0.3)     # ← مهلة صغيرة بعد كل زوج

                # رصد الحيتان
                check_whales(exchange, sym)

                # OHLCV
                ohlcv = ccxt_call_with_retry(exchange.fetch_ohlcv, sym, TIMEFRAME, FETCH_LIMIT)
                df = pd.DataFrame(ohlcv, columns=["ts","open","high","low","close","volume"])
                df["ts"] = pd.to_datetime(df["ts"], unit="ms")
                df.set_index("ts", inplace=True)

                # ===== التيكر من الكاش =====
                if sym not in ticker_cache or (time.time() - ticker_cache[sym]["ts"]) > 120:
                    try:
                        tkr = ccxt_call_with_retry(exchange.fetch_ticker, sym)
                        ticker_cache[sym] = {"data": tkr, "ts": time.time()}
                    except:
                        tkr = None
                else:
                    tkr = ticker_cache[sym]["data"]

                # ===== المؤشرات (كما هي دون تغيير) =====
                # ……………………… (كل المؤشرات كما هي) …………………………

                last = df.iloc[-1]
                last_price = float(last["close"])

                # ===== منطق الدخول والخروج (كما هو دون أي تغيير) =====
                # ……………………… (نفس المنطق الذي أرسلته أنت) …………………………

            time.sleep(SLEEP_SECONDS)

        except Exception as e:
            tele_send(f"⚠ خطأ:\n<code>{_esc(e)}</code>")
            print("Error:", e)
            print(traceback.format_exc())
            time.sleep(3)

if __name__ == "__main__":
    run_bot()
