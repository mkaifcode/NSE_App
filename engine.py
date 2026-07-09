"""
NSE Trading App — Core Engine
Handles: data fetch, indicators, scoring, trade levels, NSE universe
"""

import yfinance as yf
import pandas as pd
import numpy as np
import ta
import requests
import warnings
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
warnings.filterwarnings("ignore")

# ── Scoring weights (total = 100) ──────────────────────────────
WEIGHTS = {
    "above_50dma":    15,
    "above_200dma":   15,
    "rsi_sweet_spot": 20,
    "macd_bullish":   15,
    "volume_surge":   15,
    "price_momentum": 10,
    "above_20dma":    10,
}
MIN_SCORE = 55

ALL_SECTORS = [
    "All Sectors",
    "IT / Tech",
    "Financial Services",
    "Pharma / Healthcare",
    "Auto / Consumer",
    "Capital Goods / Infra",
    "Metals / Materials",
    "Energy / Renewables",
    "FMCG / Consumer Staples",
    "Real Estate",
    "Telecom / Media",
    "Utilities",
]

# ── Curated 200-stock fallback universe ────────────────────────
FALLBACK_UNIVERSE = [t + ".NS" for t in [
    "RELIANCE","TCS","HDFCBANK","ICICIBANK","INFY","HINDUNILVR","ITC",
    "SBIN","BAJFINANCE","BHARTIARTL","KOTAKBANK","LT","HCLTECH","AXISBANK",
    "ASIANPAINT","MARUTI","TITAN","SUNPHARMA","ULTRACEMCO","NESTLEIND",
    "WIPRO","ONGC","NTPC","POWERGRID","TECHM","JSWSTEEL","TATASTEEL",
    "COALINDIA","TATAMOTORS","ADANIENT","ADANIPORTS","GRASIM","BPCL",
    "DIVISLAB","DRREDDY","CIPLA","EICHERMOT","BAJAJ-AUTO","HEROMOTOCO",
    "BRITANNIA","APOLLOHOSP","TATACONSUM","HINDALCO","VEDL","INDUSINDBK",
    "SHREECEM","M&M","BAJAJFINSV","PIDILITIND","BERGEPAINT","HAVELLS",
    "MUTHOOTFIN","MANAPPURAM","CHOLAFIN","PERSISTENT","COFORGE","MPHASIS",
    "LTTS","INOXWIND","SUZLON","TATAPOWER","CESC","NHPC","SJVN",
    "DABUR","MARICO","GODREJCP","EMAMILTD","COLPAL",
    "SSWL","BALKRISIND","SONACOMS","MOTHERSON","TIINDIA","BOSCHLTD",
    "JMFINANCIL","IIFL","ANGELONE","MOTILALOFS",
    "ALKEM","IPCALAB","GLENMARK","NATCOPHARM","TORNTPHARM","LUPIN",
    "NATIONALUM","HINDCOPPER","RATNAMANI","SAIL","WELCORP",
    "BHEL","THERMAX","GRINDWELL","CUMMINSIND","ABB","SIEMENS",
    "OBEROIRLTY","GODREJPROP","DLF","PRESTIGE","BRIGADE",
    "ZOMATO","NYKAA","DELHIVERY","IRCTC","CONCOR",
    "BANKBARODA","PNB","CANBK","UNIONBANK","FEDERALBNK","IDFCFIRSTB",
    "HDFCLIFE","SBILIFE","ICICIGI","STARHEALTH",
    "DIXON","AMBER","PGEL","KAYNES","AVALON",
    "DEEPAKNTR","AARTIIND","VINATIORG","NAVINFLUOR","CLEAN",
    "KPITTECH","TATAELXSI","ZENSARTECH","MASTEK","CYIENT",
    "KAJARIACER","CENTURYPLY","GREENPANEL","ASTRAL","FINOLEX",
    "ABCAPITAL","CANFINHOME","AAVAS","HOMEFIRST","APTUS",
]]


def get_nse_universe() -> list[str]:
    """Fetch full NSE equity list. Falls back to curated list."""
    try:
        url = "https://nsearchives.nseindia.com/content/equities/EQUITY_L.csv"
        headers = {"User-Agent": "Mozilla/5.0", "Accept-Language": "en-US,en;q=0.9"}
        resp = requests.get(url, headers=headers, timeout=15)
        resp.raise_for_status()
        df = pd.read_csv(pd.io.common.StringIO(resp.text))
        symbols = df["SYMBOL"].dropna().str.strip().tolist()
        return [s + ".NS" for s in symbols]
    except Exception:
        return FALLBACK_UNIVERSE


def fetch_ohlcv(ticker: str, period: str = "6mo") -> pd.DataFrame | None:
    try:
        df = yf.download(ticker, period=period, interval="1d",
                         progress=False, auto_adjust=True)
        if df is None or len(df) < 30:
            return None
        df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
        return df.dropna()
    except Exception:
        return None


def compute_indicators(df: pd.DataFrame) -> dict:
    close  = df["Close"]
    volume = df["Volume"]
    high   = df["High"]
    low    = df["Low"]

    sma20   = ta.trend.sma_indicator(close, 20)
    sma50   = ta.trend.sma_indicator(close, 50)
    sma200  = ta.trend.sma_indicator(close, 200)
    ema9    = ta.trend.ema_indicator(close, 9)
    ema21   = ta.trend.ema_indicator(close, 21)
    rsi     = ta.momentum.rsi(close, 14)
    macd_o  = ta.trend.MACD(close)
    macd_l  = macd_o.macd()
    macd_s  = macd_o.macd_signal()
    macd_h  = macd_o.macd_diff()
    bb      = ta.volatility.BollingerBands(close)
    atr     = ta.volatility.average_true_range(high, low, close, 14)
    avg_v20 = volume.rolling(20).mean()
    stoch   = ta.momentum.StochasticOscillator(high, low, close)

    p   = float(close.iloc[-1])
    p5  = float(close.iloc[-6])  if len(close) >= 6  else p
    p20 = float(close.iloc[-21]) if len(close) >= 21 else p

    return {
        "price":        p,
        "sma20":        float(sma20.iloc[-1]),
        "sma50":        float(sma50.iloc[-1]),
        "sma200":       float(sma200.iloc[-1]) if len(df) >= 200 else None,
        "ema9":         float(ema9.iloc[-1]),
        "ema21":        float(ema21.iloc[-1]),
        "rsi":          float(rsi.iloc[-1]),
        "macd":         float(macd_l.iloc[-1]),
        "macd_signal":  float(macd_s.iloc[-1]),
        "macd_hist":    float(macd_h.iloc[-1]),
        "bb_upper":     float(bb.bollinger_hband().iloc[-1]),
        "bb_lower":     float(bb.bollinger_lband().iloc[-1]),
        "bb_mid":       float(bb.bollinger_mavg().iloc[-1]),
        "atr":          float(atr.iloc[-1]),
        "stoch_k":      float(stoch.stoch().iloc[-1]),
        "stoch_d":      float(stoch.stoch_signal().iloc[-1]),
        "volume":       float(volume.iloc[-1]),
        "avg_vol_20":   float(avg_v20.iloc[-1]),
        "price_5d":     p5,
        "price_20d":    p20,
        "high_52w":     float(close.rolling(min(252,len(close))).max().iloc[-1]),
        "low_52w":      float(close.rolling(min(252,len(close))).min().iloc[-1]),
        # series for charts
        "_df":          df,
        "_close":       close,
        "_volume":      volume,
        "_sma20":       sma20,
        "_sma50":       sma50,
        "_sma200":      sma200,
        "_ema9":        ema9,
        "_macd_l":      macd_l,
        "_macd_s":      macd_s,
        "_macd_h":      macd_h,
        "_rsi":         rsi,
        "_bb_upper":    bb.bollinger_hband(),
        "_bb_lower":    bb.bollinger_lband(),
        "_bb_mid":      bb.bollinger_mavg(),
        "_avg_vol_20":  avg_v20,
    }


def score_stock(ind: dict) -> tuple[int, dict]:
    score = 0
    bd    = {}
    p     = ind["price"]

    v = p > ind["sma50"]
    bd["above_50dma"] = WEIGHTS["above_50dma"] if v else 0
    score += bd["above_50dma"]

    if ind["sma200"]:
        v = p > ind["sma200"]
        bd["above_200dma"] = WEIGHTS["above_200dma"] if v else 0
    else:
        bd["above_200dma"] = 0
    score += bd["above_200dma"]

    rsi = ind["rsi"]
    if   50 <= rsi <= 65: bd["rsi_sweet_spot"] = WEIGHTS["rsi_sweet_spot"]
    elif 45 <= rsi < 50 or 65 < rsi <= 72:
        bd["rsi_sweet_spot"] = int(WEIGHTS["rsi_sweet_spot"] * 0.5)
    else: bd["rsi_sweet_spot"] = 0
    score += bd["rsi_sweet_spot"]

    v = ind["macd"] > ind["macd_signal"]
    bd["macd_bullish"] = WEIGHTS["macd_bullish"] if v else 0
    score += bd["macd_bullish"]

    vr = ind["volume"] / ind["avg_vol_20"] if ind["avg_vol_20"] > 0 else 0
    if   vr >= 1.5: bd["volume_surge"] = WEIGHTS["volume_surge"]
    elif vr >= 1.2: bd["volume_surge"] = int(WEIGHTS["volume_surge"] * 0.5)
    else:           bd["volume_surge"] = 0
    score += bd["volume_surge"]

    chg5 = ((p - ind["price_5d"]) / ind["price_5d"]) * 100
    if   chg5 >= 3: bd["price_momentum"] = WEIGHTS["price_momentum"]
    elif chg5 >= 1: bd["price_momentum"] = int(WEIGHTS["price_momentum"] * 0.5)
    else:           bd["price_momentum"] = 0
    score += bd["price_momentum"]

    v = p > ind["sma20"]
    bd["above_20dma"] = WEIGHTS["above_20dma"] if v else 0
    score += bd["above_20dma"]

    return score, bd


def compute_levels(ind: dict) -> dict:
    p   = ind["price"]
    atr = max(ind["atr"], p * 0.01)
    sl  = round(max(p - 2.0 * atr, ind["sma50"] * 0.97), 1)
    return {
        "buy_low":    round(p - 0.3 * atr, 1),
        "buy_high":   round(p + 0.2 * atr, 1),
        "sl":         sl,
        "t1":         round(p + 1.5 * atr, 1),
        "t2":         round(p + 3.0 * atr, 1),
        "t3":         round(min(p + 5.0 * atr, ind["high_52w"] * 0.98), 1),
        "risk_pct":   round(((p - sl) / p) * 100, 1),
        "reward_pct": round(((p + 1.5 * atr - p) / p) * 100, 1),
        "rr":         round((1.5 * atr) / max(p - sl, 0.01), 2),
    }


def get_signal(score: int) -> str:
    if score >= 80: return "🔥 STRONG BUY"
    if score >= 65: return "✅ BUY"
    if score >= 55: return "⚡ WATCHLIST"
    return "❌ SKIP"


def get_signal_short(score: int) -> str:
    if score >= 80: return "STRONG BUY"
    if score >= 65: return "BUY"
    if score >= 55: return "WATCHLIST"
    return "SKIP"


def scan_one(ticker: str) -> dict | None:
    df = fetch_ohlcv(ticker)
    if df is None:
        return None
    try:
        ind       = compute_indicators(df)
        score, bd = score_stock(ind)
        lvl       = compute_levels(ind)
        p         = ind["price"]
        vr        = ind["volume"] / ind["avg_vol_20"] if ind["avg_vol_20"] > 0 else 0
        chg5      = ((p - ind["price_5d"])  / ind["price_5d"])  * 100
        chg20     = ((p - ind["price_20d"]) / ind["price_20d"]) * 100
        from52h   = ((p - ind["high_52w"])  / ind["high_52w"])  * 100

        return {
            "ticker":   ticker,
            "name":     ticker.replace(".NS",""),
            "price":    round(p, 2),
            "score":    score,
            "signal":   get_signal(score),
            "signal_s": get_signal_short(score),
            "rsi":      round(ind["rsi"], 1),
            "macd_h":   round(ind["macd_hist"], 3),
            "stoch_k":  round(ind["stoch_k"], 1),
            "vol_ratio":round(vr, 2),
            "chg_5d":   round(chg5, 1),
            "chg_20d":  round(chg20, 1),
            "from_52h": round(from52h, 1),
            "above_50": p > ind["sma50"],
            "above_200":bool(ind["sma200"] and p > ind["sma200"]),
            "atr":      round(ind["atr"], 2),
            "sl":       lvl["sl"],
            "buy_low":  lvl["buy_low"],
            "buy_high": lvl["buy_high"],
            "t1":       lvl["t1"],
            "t2":       lvl["t2"],
            "t3":       lvl["t3"],
            "risk_pct": lvl["risk_pct"],
            "rr":       lvl["rr"],
            "high_52w": round(ind["high_52w"], 2),
            "low_52w":  round(ind["low_52w"], 2),
            "_ind":     ind,
            "_bd":      bd,
        }
    except Exception:
        return None


def parallel_scan(tickers: list, max_workers: int = 20,
                  progress_cb=None) -> list:
    results = []
    done    = 0
    total   = len(tickers)
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {ex.submit(scan_one, t): t for t in tickers}
        for fut in as_completed(futures):
            done += 1
            r = fut.result()
            if r:
                results.append(r)
            if progress_cb:
                progress_cb(done, total)
    return sorted(results, key=lambda x: x["score"], reverse=True)


def analyse_holding(symbol: str, avg_price: float, qty: int) -> dict:
    """Analyse a single holding from demat — returns full analysis."""
    ticker = symbol + ".NS" if not symbol.endswith(".NS") else symbol
    df     = fetch_ohlcv(ticker, period="1y")
    if df is None:
        return {"error": True, "symbol": symbol}
    try:
        ind       = compute_indicators(df)
        score, bd = score_stock(ind)
        lvl       = compute_levels(ind)
        p         = ind["price"]
        invested  = avg_price * qty
        curr_val  = p * qty
        pnl       = curr_val - invested
        pnl_pct   = ((p - avg_price) / avg_price) * 100
        days_to_t1 = "~15–25 days" if score >= 65 else "~30–45 days"

        # Recommendation logic
        if p <= lvl["sl"]:
            action = "🔴 EXIT NOW — Stop Loss Hit"
            action_reason = "Price below algorithmic stop loss. Cut losses."
        elif score >= 65 and pnl_pct < 5:
            action = "🟢 HOLD — Target Not Reached"
            action_reason = f"Strong signal. T1 at ₹{lvl['t1']}. Stay invested."
        elif score >= 65 and pnl_pct >= 5 and pnl_pct < 10:
            action = "🟡 PARTIAL BOOK — Book 50%"
            action_reason = f"T1 nearly reached. Book 50% profit, let rest run to T2 ₹{lvl['t2']}."
        elif pnl_pct >= 10:
            action = "🟢 TRAIL SL — Strong Profit"
            action_reason = f"Excellent gain. Trail stop loss to ₹{round(p*0.95,1)} to protect profits."
        elif score < 55 and pnl_pct < -3:
            action = "🔴 CONSIDER EXIT — Weak Signal"
            action_reason = "Low score + price falling. Consider exiting to preserve capital."
        else:
            action = "🟡 HOLD & WATCH"
            action_reason = "Neutral signal. No urgent action. Monitor daily."

        return {
            "error":      False,
            "symbol":     symbol,
            "qty":        qty,
            "avg_price":  avg_price,
            "cmp":        round(p, 2),
            "invested":   round(invested, 0),
            "curr_val":   round(curr_val, 0),
            "pnl":        round(pnl, 0),
            "pnl_pct":    round(pnl_pct, 2),
            "score":      score,
            "signal":     get_signal(score),
            "rsi":        round(ind["rsi"], 1),
            "sl":         lvl["sl"],
            "t1":         lvl["t1"],
            "t2":         lvl["t2"],
            "action":     action,
            "reason":     action_reason,
            "days_t1":    days_to_t1,
            "above_50":   p > ind["sma50"],
            "above_200":  bool(ind["sma200"] and p > ind["sma200"]),
            "_ind":       ind,
            "_bd":        bd,
        }
    except Exception as e:
        return {"error": True, "symbol": symbol, "msg": str(e)}
