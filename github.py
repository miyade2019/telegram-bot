# -- coding: utf-8 --
"""
بوت تداول إشارات + تيليغرام + رصد الحيتان + تعليم الأهداف + رسم بياني
- يعمل 24/7
- بدون أي دخول تجريبي DEMO_OPEN
- يعالج مشاكل DNS عبر تعدد هوستات Binance + إعادة محاولات للشبكة
- يدعم بروكسي اختياري لباينانس وتيليغرام

المكتبات:
    pip install ccxt pandas numpy requests matplotlib
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
API_KEY     = "8259936456:AAHYckhlv6swNQCPctBshVEI36qQw817zK4"   # مفاتيحك للتداول الحقيقي (إن أردت)
API_SECRET  = "-1002711156609"
EXCHANGE_ID = "binance"

# الأزواج
SYMBOLS = [
    "SOL/USDT", "LINK/USDT", "JUP/USDT", "INJ/USDT", "TIA/USDT", "AVAX/USDT",
    "SOMI/USDT", "ADA/USDT", "FIL/USDT", "SUI/USDT", "WLD/USDT",
    "ARB/USDT", "CTSI/USDT", "TON/USDT", "PEPE/USDT", "TRUMP/USDT", "PYTH/USDT",
    "FET/USDT", "SEI/USDT", "DOGE/USDT", "GRT/USDT", "ETC/USDT","ONDO/USDT"
]

TIMEFRAME   = "5m"      # 5m -> 15m (أهدأ)
FETCH_LIMIT = 300

DRY_RUN       = True      # لا ينفّذ أوامر حقيقية
SLEEP_SECONDS = 60        # مهلة الدور

# إدارة المخاطر (للتداول الحقيقي فقط)
RISK_PER_TRADE       = 0.005  # 0.5% من رأس المال
MAX_POSITION_PERCENT = 0.10   # 10% كحد أقصى للصفقة
MIN_USDT_FOR_TRADE   = 10.0

# ======== مؤشرات / عتبات (مشددة) ========
ATR_PERIOD            = 14
SUPER_TREND_ATR_MULT  = 3.0
TRAIL_ATR_MULT        = 2.5      # 2.0 -> 2.5
DONCHIAN_PERIOD       = 14
ADX_PERIOD            = 14
ADX_THRESHOLD         = 20       # 10 -> 20
RSI_PERIOD            = 14
RSI_ENTRY_LOW         = 35       # 25 -> 35
RSI_ENTRY_HIGH        = 75       # 90 -> 75
EMA_FAST              = 20
EMA_MED               = 50
EMA_SLOW              = 200
MACD_FAST             = 12
MACD_SLOW             = 26
MACD_SIGNAL           = 9
STOCH_K               = 14
STOCH_D               = 3
VOL_MA                = 20
MIN_VOL_RATIO         = 1.00     # 0.80 -> 1.00

# أهداف الربح (أقرب TP1 لتفعيل الإدارة مبكراً)
TP_MULTS = [1.03, 1.06, 1.12]   # 3%، 6%، 12%

# تقييد الضجيج
COOLDOWN_MINUTES      = 45       # 20 -> 45
ONE_SIGNAL_PER_BAR    = True     # False -> True

# ======== مفاتيح/فلتر ========
REQUIRE_DONCHIAN_BREAK = True    # كان False
REQUIRE_VWAP           = True    # كان False
USE_MACD_FILTER        = True    # كان False

# ==================== رصد الحيتان ====================
WHALE_MIN_USDT     = 1_000_000
WHALE_LOOKBACK_MIN = 5
WHALE_COOLDOWN_MIN = 15

# ==================== بروكسي (اختياري) ====================
USE_PROXY  = False
HTTP_PROXY = "http://127.0.0.1:7890"
REQUESTS_PROXIES = {"http": HTTP_PROXY, "https": HTTP_PROXY} if USE_PROXY else None

# ==================== تيليغرام ====================
ENABLE_TELEGRAM = True
TELEGRAM_TOKEN  = "8259936456:AAHYckhlv6swNQCPctBshVEI36qQw817zK4"
TELEGRAM_CHAT_ID= "-1002711156609"

def _esc(s: str) -> str:
    return html.escape(str(s), quote=False)

# --- إعادة محاولة عامة (شبكة) ---
def retry_call(fn, *args, max_tries=3, base_delay=1.0, exceptions=(Exception,), **kwargs):
    last_err = None
    for attempt in range(1, max_tries + 1):
        try:
            return fn(*args, **kwargs)
        except exceptions as e:
            last_err = e
            if attempt == max_tries:
                break
            time.sleep(base_delay * (2 ** (attempt - 1)))
    raise last_err

# --- Telegram helpers ---
def tele_send(text, buttons=None, disable_notification=False, html_mode=True):
    if not ENABLE_TELEGRAM: return None
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
        payload = {"chat_id": TELEGRAM_CHAT_ID, "text": text,
                   "disable_web_page_preview": True, "disable_notification": disable_notification}
        if html_mode: payload["parse_mode"] = "HTML"
        if buttons:   payload["reply_markup"] = {"inline_keyboard": buttons}
        r = retry_call(requests.post, url, json=payload, timeout=15,
                       exceptions=(requests.exceptions.RequestException,),
                       proxies=REQUESTS_PROXIES)
        data = r.json()
        return data.get("result", {}).get("message_id")
    except Exception as e:
        print("Telegram send exception:", e); return None

def tele_edit(mid, new_text, buttons=None, html_mode=True):
    if not ENABLE_TELEGRAM or not mid: return False
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/editMessageText"
        payload = {"chat_id": TELEGRAM_CHAT_ID, "message_id": mid, "text": new_text,
                   "disable_web_page_preview": True}
        if html_mode: payload["parse_mode"] = "HTML"
        if buttons:   payload["reply_markup"] = {"inline_keyboard": buttons}
        r = retry_call(requests.post, url, json=payload, timeout=15,
                       exceptions=(requests.exceptions.RequestException,),
                       proxies=REQUESTS_PROXIES)
        return r.json().get("ok", False)
    except Exception:
        return False

def tele_send_photo(path, caption=""):
    if not ENABLE_TELEGRAM: return None
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendPhoto"
        with open(path, "rb") as f:
            data = {"chat_id": TELEGRAM_CHAT_ID, "caption": caption, "parse_mode": "HTML"}
            r = retry_call(requests.post, url, data=data, files={"photo": f}, timeout=30,
                           exceptions=(requests.exceptions.RequestException,),
                           proxies=REQUESTS_PROXIES)
        return r.json().get("result", {}).get("message_id")
    except Exception as e:
        print("Telegram photo send exception:", e)
        return None

def binance_pair_link(symbol):
    return f"https://www.binance.com/en/trade/{symbol.replace('/','_')}"

# ==================== سجلات ====================
LOG_DIR, CHART_DIR = "bot_logs", os.path.join("bot_logs", "charts")
os.makedirs(LOG_DIR, exist_ok=True); os.makedirs(CHART_DIR, exist_ok=True)
TRADES_CSV = os.path.join(LOG_DIR, "trades.csv")
if not os.path.exists(TRADES_CSV):
    with open(TRADES_CSV, "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow(["timestamp","symbol","side","price","amount","entry_atr","stop_price","reason","status"])
def log_trade(ts, symbol, side, price, amount, entry_atr, stop_price, reason, status):
    with open(TRADES_CSV, "a", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow([ts, symbol, side, price, amount, entry_atr, stop_price, reason, status])

# ==================== أدوات فنية ====================
def ema(series, n):
    return series.ewm(span=n, adjust=False).mean()

def macd(series, fast=12, slow=26, signal=9):
    macd_line = ema(series, fast) - ema(series, slow)
    macd_signal = ema(macd_line, signal)
    return macd_line, macd_signal, macd_line - macd_signal

def stoch(df, k=14, d=3):
    low_min  = df['low'].rolling(k).min()
    high_max = df['high'].rolling(k).max()
    k_line = 100 * (df['close'] - low_min) / (high_max - low_min).replace(0, np.nan)
    return k_line, k_line.rolling(d).mean()

def vwap(df):
    tp = (df['high'] + df['low'] + df['close']) / 3.0
    pv = (tp * df['volume']).cumsum()
    vv = (df['volume']).cumsum().replace(0, np.nan)
    return pv / vv

def compute_atr(df, period=14):
    tr = pd.concat([(df['high']-df['low']),
                    (df['high']-df['close'].shift()).abs(),
                    (df['low'] -df['close'].shift()).abs()], axis=1).max(axis=1)
    return tr.rolling(period).mean()

def compute_supertrend(df, period=14, multiplier=3.0):
    atr = compute_atr(df, period); hl2 = (df['high'] + df['low'])/2
    upperband = hl2 + multiplier*atr; lowerband = hl2 - multiplier*atr
    supertrend = pd.Series(index=df.index, dtype='bool')
    final_upper = upperband.copy(); final_lower = lowerband.copy()
    for i in range(len(df)):
        if i == 0:
            final_upper.iat[i] = upperband.iat[i]; final_lower.iat[i] = lowerband.iat[i]; supertrend.iat[i] = True; continue
        final_upper.iat[i] = upperband.iat[i] if (upperband.iat[i] < final_upper.iat[i-1]) or (df['close'].iat[i-1] > final_upper.iat[i-1]) else final_upper.iat[i-1]
        final_lower.iat[i] = lowerband.iat[i] if (lowerband.iat[i] > final_lower.iat[i-1]) or (df['close'].iat[i-1] < final_lower.iat[i-1]) else final_lower.iat[i-1]
        if supertrend.iat[i-1] and df['close'].iat[i] <= final_upper.iat[i]: supertrend.iat[i] = False
        elif (not supertrend.iat[i-1]) and df['close'].iat[i] >= final_lower.iat[i]: supertrend.iat[i] = True
        else: supertrend.iat[i] = supertrend.iat[i-1]
    return supertrend, atr, final_upper, final_lower

def compute_adx(df, period=14):
    up = df['high'].diff(); down = -df['low'].diff()
    plus_dm  = np.where((up > down) & (up > 0), up, 0.0)
    minus_dm = np.where((down > up) & (down > 0), down, 0.0)
    tr = pd.concat([(df['high']-df['low']),
                    (df['high']-df['close'].shift()).abs(),
                    (df['low'] -df['close'].shift()).abs()], axis=1).max(axis=1)
    atr = tr.rolling(period).mean()
    plus_di  = 100 * (pd.Series(plus_dm,  index=df.index).rolling(period).sum() / atr)
    minus_di = 100 * (pd.Series(minus_dm, index=df.index).rolling(period).sum() / atr)
    dx = (abs(plus_di - minus_di) / (plus_di + minus_di)).replace([np.inf, -np.inf], 0) * 100
    return dx.rolling(period).mean()

# ========= رسم =========
ENABLE_CHARTS          = True
CHART_LOOKBACK         = 180
CHART_STYLE_DARK       = True
CHART_USE_HEIKIN_ASHI  = True
DRAW_RSI_PANEL         = True

def heikin_ashi(df):
    close_ha = (df['open'] + df['high'] + df['low'] + df['close']) / 4
    open_ha = [(df['open'].iloc[0] + df['close'].iloc[0]) / 2]
    for i in range(1, len(df)): open_ha.append((open_ha[i-1] + close_ha.iloc[i-1]) / 2)
    open_ha = pd.Series(open_ha, index=df.index)
    high_ha = pd.concat([df['high'], open_ha, close_ha], axis=1).max(axis=1)
    low_ha  = pd.concat([df['low'],  open_ha, close_ha], axis=1).min(axis=1)
    return pd.DataFrame({"open": open_ha, "high": high_ha, "low": low_ha, "close": close_ha})

def draw_candles(ax, o, h, l, c, xindex, width=0.6, bull="#00c176", bear="#ff4d4f", wick="#9aa4b2"):
    half = width/2.0
    for i in range(len(xindex)):
        ax.vlines(i, l[i], h[i], color=wick, linewidth=1)
        color = bull if c[i] >= o[i] else bear
        lower = min(o[i], c[i]); height = max(abs(c[i]-o[i]), 1e-10)
        ax.add_patch(Rectangle((i-half, lower), width, height, facecolor=color, edgecolor=color, linewidth=0))
    ax.set_xlim(-1, len(xindex))
    ticks = list(range(0, len(xindex), max(1, len(xindex)//6)))
    ax.set_xticks(ticks)
    ax.set_xticklabels([xindex[i].strftime("%Y-%m-%d %H:%M") for i in ticks], rotation=20, ha="right")

def rsi(series, period=14):
    delta = series.diff(); gain = (delta.clip(lower=0)).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))

def save_chart(df, symbol, entry_price=None, sl=None, tps=None, title=""):
    if not ENABLE_CHARTS: return None
    try:
        tail = df.tail(CHART_LOOKBACK).copy()
        prices = tail[['open','high','low','close']].copy()
        if CHART_USE_HEIKIN_ASHI: prices = heikin_ashi(prices)

        nrows = 2 if DRAW_RSI_PANEL else 1
        fig = plt.figure(figsize=(11, 6))
        ax = fig.add_subplot(nrows,1,1)

        if CHART_STYLE_DARK:
            ax.set_facecolor("#0e1117"); fig.patch.set_facecolor("#0e1117")
            ax.tick_params(colors="#e6edf3"); [sp.set_color("#e6edf3") for sp in ax.spines.values()]

        draw_candles(ax, prices['open'].values, prices['high'].values, prices['low'].values, prices['close'].values, prices.index)
        ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.35, color="#2a2f3a")
        ax.set_title(title or f"{symbol} • {TIMEFRAME}", color=("#e6edf3" if CHART_STYLE_DARK else None))

        # EMAs / VWAP (إن وُجدت)
        if {'ema20','ema50','ema200','vwap'}.issubset(tail.columns):
            ax.plot(tail.index, tail['ema20'],  linewidth=1.0, label="EMA20")
            ax.plot(tail.index, tail['ema50'],  linewidth=1.0, label="EMA50")
            ax.plot(tail.index, tail['ema200'], linewidth=1.0, label="EMA200")
            ax.plot(tail.index, tail['vwap'],   linewidth=1.0, label="VWAP")
            ax.legend(loc="upper right", fontsize=7)

        # خطوط دخول/وقف/أهداف
        if entry_price is not None: ax.axhline(entry_price, color="#3cabff", linestyle="--", linewidth=1.2, label="Entry")
        if sl          is not None: ax.axhline(sl,          color="#ffaa00", linestyle=":",  linewidth=1.0, label="SL")
        if tps:
            for tp in tps: ax.axhline(tp, color="#6ee7b7", linestyle="-.", linewidth=1.0)

        if DRAW_RSI_PANEL:
            ax2 = fig.add_subplot(nrows,1,2, sharex=ax)
            if CHART_STYLE_DARK:
                ax2.set_facecolor("#0e1117"); ax2.tick_params(colors="#e6edf3")
                [sp.set_color("#e6edf3") for sp in ax2.spines.values()]
            r = rsi(tail['close'], RSI_PERIOD)
            ax2.plot(tail.index, r, linewidth=1.2)
            ax2.axhline(70, color="#ff4d4f", linestyle="--", linewidth=0.8)
            ax2.axhline(30, color="#00c176", linestyle="--", linewidth=0.8)
            ax2.set_ylim(0,100); ax2.grid(True, linestyle="--", linewidth=0.5, alpha=0.35, color="#2a2f3a")
            ax2.set_title("RSI", color=("#e6edf3" if CHART_STYLE_DARK else None), fontsize=9)
            plt.setp(ax.get_xticklabels(), visible=False)

        fig.autofmt_xdate()
        path = os.path.join(CHART_DIR, f"{symbol.replace('/','')}{int(time.time())}.png")
        fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
        plt.close(fig); return path
    except Exception as e:
        print("save_chart error:", e); tele_send(f"❗ خطأ حفظ الرسم:\n<code>{_esc(e)}</code>")
        return None

# ==================== اتصال المنصة مع تبديل هوست + بروكسي ====================
def ccxt_call_with_retry(fn, *args, max_tries=3, base_delay=1.0, **kwargs):
    last_err = None
    from ccxt.base.errors import NetworkError, ExchangeNotAvailable, RequestTimeout, DDoSProtection
    for attempt in range(1, max_tries + 1):
        try:
            return fn(*args, **kwargs)
        except (NetworkError, ExchangeNotAvailable, RequestTimeout, DDoSProtection) as e:
            last_err = e
            if attempt == max_tries:
                break
            time.sleep(base_delay * (2 ** (attempt - 1)))
        except Exception as e:
            raise e
    raise last_err

def create_exchange():
    base = {
        "enableRateLimit": True,
        "timeout": 20000,
        "options": {"defaultType": "spot", "adjustForTimeDifference": True}
    }
    if USE_PROXY:
        base["proxies"] = {"http": HTTP_PROXY, "https": HTTP_PROXY}
    if not DRY_RUN and API_KEY and API_SECRET:
        base.update({"apiKey": API_KEY, "secret": API_SECRET})

    hosts = ["api.binance.com", "api1.binance.com", "api2.binance.com", "api-gcp.binance.com"]
    last_err = None
    for host in hosts:
        try:
            params = dict(base); params["hostname"] = host
            ex = getattr(ccxt, EXCHANGE_ID)(params)
            ccxt_call_with_retry(ex.publicGetTime, max_tries=3, base_delay=1.0)
            print(f"[create_exchange] Using host: {host}")
            return ex
        except Exception as e:
            print(f"[create_exchange] Host failed: {host} -> {e}")
            last_err = e
            time.sleep(1.0)
    raise last_err

# ==================== مساعدين للجودة / R:R / HTF ====================
def expected_rr(entry_price, atr_value):
    if atr_value is None or np.isnan(atr_value) or atr_value <= 0:
        return 0.0
    sl  = entry_price - SUPER_TREND_ATR_MULT * atr_value
    tp1 = entry_price * TP_MULTS[0]
    risk   = max(entry_price - sl, 1e-8)
    reward = max(tp1 - entry_price, 0.0)
    return reward / risk

def htf_ok(exchange, symbol):
    """فلتر 1h: فوق EMA200 و SuperTrend صاعد."""
    try:
        ohlcv_h = ccxt_call_with_retry(exchange.fetch_ohlcv, symbol, timeframe="1h", limit=200)
    except Exception:
        return False
    dfh = pd.DataFrame(ohlcv_h, columns=["ts","open","high","low","close","volume"])
    st_h, _, _, _ = compute_supertrend(dfh, ATR_PERIOD, SUPER_TREND_ATR_MULT)
    ema200_h = ema(dfh["close"], 200)
    i = len(dfh) - 1
    if i < 1: return False
    return (dfh["close"].iat[i] > ema200_h.iat[i]) and bool(st_h.iat[i])

# ==================== منطق الدخول/الخروج ====================
def entry_signal(r, exchange, symbol):
    # اتجاه EMA مرن: (ema20>ema50) أو (close>ema200)
    ema_trend_ok = (r['ema20'] > r['ema50']) or (r['close'] > r['ema200'])
    if not bool(r['supertrend']): return False, "SuperTrend سلبي"
    if not ema_trend_ok:          return False, "اتجاه EMA سلبي"
    if r['adx'] < ADX_THRESHOLD:  return False, "ADX ضعيف"

    if REQUIRE_DONCHIAN_BREAK:
        if not (r['close'] > r['donch_up'] and r['prev_close'] <= r['donch_up']):
            return False, "لا كسر Donchian"

    if not (RSI_ENTRY_LOW <= r['rsi'] <= RSI_ENTRY_HIGH):
        return False, "RSI خارج النطاق"

    if USE_MACD_FILTER and r['macd_hist'] <= 0:
        return False, "MACD سلبي"

    if r['stoch_k'] <= r['stoch_d']:
        return False, "Stoch سلبي"

    if pd.notna(r['vol_ma']) and r['volume'] < (MIN_VOL_RATIO * r['vol_ma']):
        return False, f"حجم < {MIN_VOL_RATIO}×MA20"

    if REQUIRE_VWAP and pd.notna(r['vwap']) and r['close'] < r['vwap']:
        return False, "تحت VWAP"

    # فلتر HTF 1h (إجباري لتقليل الخسائر)
    if not htf_ok(exchange, symbol):
        return False, "HTF(1h) غير داعم"

    # شرط R:R ≥ 1.5 قبل الدخول
    rr = expected_rr(r['close'], r['atr'])
    if rr < 1.5:
        return False, f"R:R منخفض ({rr:.2f}<1.5)"

    return True, "شروط مكتملة"

def exit_signal(r, entry_price, entry_atr):
    tp3 = entry_price * TP_MULTS[2]
    sl_price = entry_price - SUPER_TREND_ATR_MULT * entry_atr
    if r['close'] >= tp3:     return True, "هدف 3"
    if r['close'] <= sl_price:return True, "SL أصلي"
    # خروج عند انقلاب ST فقط إذا السعر تحت EMA20 (فلتر ضوضاء)
    if (not r['supertrend']) and (r['close'] < r['ema20']): return True, "ST flip + تحت EMA20"
    if r['rsi'] > 85:         return True, "تشبّع RSI"
    return False, ""

# ==================== تعليم الأهداف ====================
def update_signal_card_progress(txt, hits):
    lines = txt.splitlines()
    new_lines, cnt = [], 0
    for ln in lines:
        if ln.strip().startswith("• الهدف"):
            if cnt < len(hits) and hits[cnt]:
                ln = ln.replace("⬜", "✅")
            cnt += 1
        new_lines.append(ln)
    return "\n".join(new_lines)

# ==================== وضع التشخيص ====================
DEBUG_REASONS = True

def entry_checks(r):
    checks = {
        "supertrend_bull": bool(r['supertrend']),
        "ema_trend": (r['ema20'] > r['ema50']) or (r['close'] > r['ema200']),
        "adx_ok": r['adx'] >= ADX_THRESHOLD,
        "donch_break": (r['close'] > r['donch_up']) and (r['prev_close'] <= r['donch_up']),
        "rsi_ok": (RSI_ENTRY_LOW <= r['rsi'] <= RSI_ENTRY_HIGH),
        "macd_pos": r['macd_hist'] > 0,
        "stoch_cross": r['stoch_k'] > r['stoch_d'],
        "vol_ok": (pd.isna(r['vol_ma']) or (r['volume'] >= (MIN_VOL_RATIO * r['vol_ma']))),
        "vwap_ok": (pd.isna(r['vwap']) or (r['close'] >= r['vwap'])),
    }
    return checks

def checks_to_reason(ch):
    mapping = {
        "supertrend_bull": "SuperTrend سلبي",
        "ema_trend": "اتجاه EMA سلبي",
        "adx_ok": f"ADX < {ADX_THRESHOLD}",
        "donch_break": "לא كسر Donchian",
        "rsi_ok": f"RSI خارج [{RSI_ENTRY_LOW}-{RSI_ENTRY_HIGH}]",
        "macd_pos": "MACD سلبي",
        "stoch_cross": "Stoch سلبي",
        "vol_ok": f"حجم < {MIN_VOL_RATIO}×MA20",
        "vwap_ok": "تحت VWAP",
    }
    failed = []
    if not ch["supertrend_bull"]: failed.append(mapping["supertrend_bull"])
    if not ch["ema_trend"]:       failed.append(mapping["ema_trend"])
    if not ch["adx_ok"]:          failed.append(mapping["adx_ok"])
    if REQUIRE_DONCHIAN_BREAK and (not ch["donch_break"]): failed.append(mapping["donch_break"])
    if not ch["rsi_ok"]:          failed.append(mapping["rsi_ok"])
    if not ch["stoch_cross"]:     failed.append(mapping["stoch_cross"])
    if not ch["vol_ok"]:          failed.append(mapping["vol_ok"])
    if REQUIRE_VWAP and (not ch["vwap_ok"]): failed.append(mapping["vwap_ok"])
    if USE_MACD_FILTER and (not ch["macd_pos"]): failed.append(mapping["macd_pos"])
    return " | ".join(failed) if failed else "كل الشروط مكتملة"

# ==================== رصد الحيتان ====================
_last_whale_alert = {}  # {symbol: datetime}

def check_whales(exchange, symbol):
    now_ms   = int(time.time() * 1000)
    since_ms = now_ms - WHALE_LOOKBACK_MIN * 60 * 1000
    try:
        trades = ccxt_call_with_retry(exchange.fetch_trades, symbol, since=since_ms, limit=200)
    except Exception as e:
        print(f"[whale] fetch_trades error {symbol}: {e}")
        return
    if not trades: return

    last = _last_whale_alert.get(symbol)
    if last is not None:
        mins = (datetime.now(timezone.utc) - last).total_seconds() / 60.0
        if mins < WHALE_COOLDOWN_MIN:
            return

    for tr in trades:
        cost = tr.get('cost')
        if cost is None:
            price  = tr.get('price') or 0
            amount = tr.get('amount') or 0
            cost = float(price) * float(amount)
        if cost and cost >= WHALE_MIN_USDT:
            ts = tr.get('timestamp', now_ms)
            t_iso = datetime.fromtimestamp(ts/1000.0, tz=timezone.utc).isoformat()
            side = (tr.get('side') or '?').upper()
            msg = (
                f"🐳 <b>صفقة حوت</b> على <b>{symbol}</b>\n"
                f"الجانب: <code>{side}</code>\n"
                f"القيمة التقريبية: <code>{int(cost):,} USDT</code>\n"
                f"الوقت: <code>{t_iso}</code>\n\n"
                f"⚡ قد تكون قوة فورية — راقب فرصة دخول.\n"
                f"🔗 {binance_pair_link(symbol)}"
            )
            tele_send(msg)
            _last_whale_alert[symbol] = datetime.now(timezone.utc)
            break

# ==================== الحلقة الرئيسية ====================
_last_signal_time = {}   # {symbol: datetime}
_last_bar_time    = {}   # {symbol: pd.Timestamp}

def run_bot():
    exchange = create_exchange()
    tele_send("🚀 تم تشغيل البوت (24/7) — DRY_RUN=" + str(DRY_RUN))

    open_positions = {}   # {sym: {"entry":..,"atr":..,"sl":..,"tps":[..],"hits":[..]}}
    signal_msgs    = {}   # {sym: {"id": message_id, "text": original_text}}

    while True:
        try:
            for sym in SYMBOLS:
                # 1) رصد الحيتان
                check_whales(exchange, sym)

                # 2) OHLCV
                ohlcv = ccxt_call_with_retry(exchange.fetch_ohlcv, sym, timeframe=TIMEFRAME, limit=FETCH_LIMIT)
                df = pd.DataFrame(ohlcv, columns=["ts","open","high","low","close","volume"])
                df['ts'] = pd.to_datetime(df['ts'], unit='ms'); df.set_index('ts', inplace=True)

                # مؤشرات
                delta = df['close'].diff(); gain = (delta.clip(lower=0)).rolling(RSI_PERIOD).mean()
                loss  = (-delta.clip(upper=0)).rolling(RSI_PERIOD).mean()
                rs = gain / loss.replace(0, np.nan)
                df['rsi'] = 100 - (100 / (1 + rs))

                df['adx'] = compute_adx(df, ADX_PERIOD)
                st, atr, _, _ = compute_supertrend(df, ATR_PERIOD, SUPER_TREND_ATR_MULT)
                df['supertrend'] = st; df['atr'] = atr

                df['donch_up']   = df['high'].rolling(DONCHIAN_PERIOD).max().shift(1)
                df['prev_close'] = df['close'].shift(1)

                df['ema20']  = ema(df['close'], EMA_FAST)
                df['ema50']  = ema(df['close'], EMA_MED)
                df['ema200'] = ema(df['close'], EMA_SLOW)

                macd_line, macd_sig, macd_hist = macd(df['close'], MACD_FAST, MACD_SLOW, MACD_SIGNAL)
                df['macd_hist'] = macd_hist

                k, d = stoch(df, STOCH_K, STOCH_D)
                df['stoch_k'] = k; df['stoch_d'] = d

                df['vol_ma'] = df['volume'].rolling(VOL_MA).mean()
                df['vwap']   = vwap(df)

                # معلومات 24س
                try:
                    tkr = ccxt_call_with_retry(exchange.fetch_ticker, sym)
                    day_quote_vol = None
                    day_trades    = None
                    if 'quoteVolume' in tkr and tkr['quoteVolume'] is not None:
                        day_quote_vol = float(tkr['quoteVolume'])
                    if isinstance(tkr.get('info'), dict):
                        if 'count' in tkr['info'] and tkr['info']['count'] is not None:
                            day_trades = int(tkr['info']['count'])
                        qv = tkr['info'].get('quoteVolume') or tkr['info'].get('q')
                        if day_quote_vol is None and qv is not None:
                            day_quote_vol = float(qv)
                except Exception as e:
                    print("fetch_ticker error:", e)
                    day_quote_vol = None; day_trades = None

                last = df.iloc[-1]; last_price = float(last['close'])
                if pd.notna(last['volume']) and last['volume'] < 1:
                    continue

                pos = open_positions.get(sym)

                # ===== لا صفقة → محاولة دخول =====
                if pos is None:
                    ok, reason = entry_signal(last, exchange, sym)

                    if ok and ONE_SIGNAL_PER_BAR:
                        last_ts = df.index[-1]
                        prev_bar = _last_bar_time.get(sym)
                        if prev_bar is not None and last_ts == prev_bar:
                            ok = False; reason = "إشارة مُرسلة لنفس الشمعة"
                        _last_bar_time[sym] = last_ts

                    if ok:
                        now = datetime.now(timezone.utc)
                        prev = _last_signal_time.get(sym)
                        if prev is not None:
                            mins = (now - prev).total_seconds() / 60.0
                            if mins < COOLDOWN_MINUTES:
                                ok = False
                                reason = f"مهدّئ مفعل، انتظر ~{int(COOLDOWN_MINUTES - mins)} دقيقة"

                    # تشخيص مفصّل عند الرفض (فقط الشروط الملزمة)
                    if DEBUG_REASONS and not ok:
                        ch = entry_checks(last)
                        reason = checks_to_reason(ch)
                        print(f"[DEBUG][{sym}] reject: {reason}")
                        try:
                            dbg_path = os.path.join(LOG_DIR, "rejects.csv")
                            newfile = not os.path.exists(dbg_path)
                            with open(dbg_path, "a", newline="", encoding="utf-8") as f:
                                w = csv.writer(f)
                                if newfile:
                                    w.writerow(["ts","symbol","reason"] + list(ch.keys()))
                                w.writerow([datetime.now(timezone.utc).isoformat(), sym, reason] + [int(ch[k]) for k in ch.keys()])
                        except Exception:
                            pass

                    if ok:
                        entry_price = last_price
                        entry_atr   = float(last['atr']) if not pd.isna(last['atr']) else 0.0
                        if entry_atr <= 0:
                            continue
                        sl  = entry_price - SUPER_TREND_ATR_MULT * entry_atr
                        tps = [entry_price * m for m in TP_MULTS]

                        extra = ""
                        if any(v is not None for v in (last['volume'], last['vol_ma'], day_quote_vol, day_trades)):
                            extra = "\n📊 "
                            if not pd.isna(last['volume']): extra += f"حجم الشمعة: <code>{int(last['volume']):,}</code> "
                            if not pd.isna(last['vol_ma']): extra += f"| MA20: <code>{int(last['vol_ma']):,}</code> "
                            if day_quote_vol is not None:   extra += f"— حجم 24س (Quote): <code>{int(day_quote_vol):,}</code> "
                            if day_trades is not None:      extra += f"— عدد الصفقات 24س: <code>{int(day_trades):,}</code>"

                        card_text = (
                            f"📈 <b>{sym}</b> — <b>شراء</b>\n"
                            f"⏱ الإطار: <code>{TIMEFRAME}</code> | محاكاة: <code>{DRY_RUN}</code>\n"
                            f"🟢 دخول: <code>{entry_price:.4f}</code>\n"
                            f"🛡 وقف: <code>{sl:.4f}</code>\n"
                            f"🎯 الأهداف:\n"
                            f"• الهدف 1: <code>{tps[0]:.4f}</code> ⬜\n"
                            f"• الهدف 2: <code>{tps[1]:.4f}</code> ⬜\n"
                            f"• الهدف 3: <code>{tps[2]:.4f}</code> ⬜\n"
                            f"📏 ATR: <code>{entry_atr:.4f}</code>\n"
                            f"🧠 السبب: {_esc('شروط مكتملة + HTF 1h داعم + R:R≥1.5')}\n"
                            f"🕒 وقت الفتح: <code>{datetime.now(timezone.utc).isoformat()}</code>\n"
                            f"🔗 {binance_pair_link(sym)}"
                            f"{extra}"
                        )
                        buttons = [[{"text":"🔗 Binance","url":binance_pair_link(sym)}]]
                        mid = tele_send(card_text, buttons=buttons)

                        p = save_chart(df, sym, entry_price=entry_price, sl=sl, tps=tps, title="Trading Signal")
                        if p:
                            tele_send_photo(p, caption=f"📊 {sym} • رسم الدخول\n🔗 {binance_pair_link(sym)}")

                        open_positions[sym] = {"entry": entry_price, "atr": entry_atr, "sl": sl,
                                               "tps": tps, "hits": [False, False, False]}
                        signal_msgs[sym]    = {"id": mid, "text": card_text}
                        _last_signal_time[sym] = datetime.now(timezone.utc)

                # ===== متابعة الصفقة الحالية =====
                else:
                    entry_price = pos["entry"]; entry_atr = pos["atr"]; tps = pos["tps"]; sl = pos["sl"]
                    last_price  = float(last['close'])

                    # --- إدارة بعد الأهداف ---
                    # تعليم الأهداف
                    changed = False
                    for i, tpv in enumerate(tps):
                        if not pos["hits"][i] and last_price >= tpv:
                            pos["hits"][i] = True
                            changed = True

                    # BE بعد TP1
                    if pos["hits"][0]:
                        pos["sl"] = max(pos["sl"], pos["entry"])  # Breakeven
                        sl = pos["sl"]

                        # Trailing SL يبدأ فقط بعد TP1
                        trail = last['close'] - TRAIL_ATR_MULT * last['atr']
                        if not pd.isna(trail) and float(trail) > sl:
                            pos["sl"] = float(trail); sl = pos["sl"]

                    # بعد TP2: SL ≥ entry + 0.5*ATR
                    if pos["hits"][1]:
                        pos["sl"] = max(pos["sl"], pos["entry"] + 0.5*pos["atr"])
                        sl = pos["sl"]

                    if changed:
                        sm = signal_msgs.get(sym)
                        if sm and sm.get("id"):
                            new_text = update_signal_card_progress(sm["text"], pos["hits"])
                            if new_text != sm["text"]:
                                sm["text"] = new_text
                                tele_edit(sm["id"], new_text, buttons=[[{"text":"🔗 Binance","url":binance_pair_link(sym)}]])

                    # خروج
                    hit, why = exit_signal(last, entry_price, entry_atr)
                    if last_price <= sl and not hit:
                        hit, why = True, "Trailing SL مفعّل"

                    if hit:
                        tele_send(f"🔴 <b>إغلاق</b> {sym}\nسعر الإغلاق: <code>{last_price:.4f}</code>\nالسبب: {_esc(why)}")
                        p = save_chart(df, sym, entry_price=entry_price, sl=sl, tps=tps, title=f"{sym} • إغلاق")
                        if p:
                            tele_send_photo(p, caption=f"📉 {sym} • تم الإغلاق ({_esc(why)})\n🔗 {binance_pair_link(sym)}")
                        sm = signal_msgs.get(sym)
                        if sm and sm.get("id"):
                            final_text = sm["text"] + f"\n\n✅ <b>تم إغلاق الصفقة</b>: <code>{_esc(why)}</code>"
                            tele_edit(sm["id"], final_text, buttons=[[{"text":"🔗 Binance","url":binance_pair_link(sym)}]])
                            signal_msgs.pop(sym, None)
                        open_positions.pop(sym, None)

            time.sleep(SLEEP_SECONDS)

        except KeyboardInterrupt:
            tele_send("⏹ تم إيقاف البوت من قبل المستخدم.")
            return
        except Exception as e:
            print("Unexpected error:", e)
            print(traceback.format_exc())
            tele_send(f"⚠ خطأ غير متوقع:\n<code>{_esc(e)}</code>")
            time.sleep(5)

# ==================== تشغيل ====================
if __name__ == "__main__":

    run_bot()
