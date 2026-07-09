"""
NSE Trading App — Core Engine
Handles: data fetch, indicators, scoring, trade levels, NSE universe

IMPORTANT — READ BEFORE USING ANY OUTPUT OF THIS MODULE
=======================================================
The scoring model in `score_stock` has NOT been backtested, walk-forward
validated, or tested out-of-sample against a buy-and-hold benchmark. The
weights and the 55/65/80 signal thresholds are hand-chosen, not fitted.

All seven scoring factors (20/50/200 DMA, MACD, RSI band, 5-day momentum)
measure trend and momentum at different lags, so they are strongly
correlated. A high score is one bet — "this stock has been going up" —
not seven independent confirmations.

No transaction costs, slippage, market impact, circuit limits, or taxes
are modelled anywhere. Treat every number produced here as a starting
point for your own research, never as a recommendation.
"""

import warnings
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone
from io import StringIO

import numpy as np
import pandas as pd
import requests
import ta
import yfinance as yf

warnings.filterwarnings("ignore")

# ── Honesty flags ──────────────────────────────────────────────
MODEL_IS_BACKTESTED = False
MODEL_DISCLAIMER = (
    "Score is a hand-tuned heuristic. It has never been backtested or "
    "validated out-of-sample. Costs, slippage and liquidity are not modelled. "
    "Not investment advice."
)

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

# ── Risk / sizing / data-quality constants ─────────────────────
RISK_FRACTION      = 0.02    # 2% of capital risked per trade
MAX_RISK_PCT       = 12.0    # never quote a stop further than 12% below entry
MIN_ATR_PCT        = 0.005   # ATR floor = 0.5% of price
MIN_TURNOVER       = 2_00_00_000    # ₹2 crore avg daily traded value
SCAN_PERIOD        = "2y"    # ≥ 200 bars, so the 200-DMA actually exists
HOLDING_PERIOD     = "2y"    # same window → same score on every page
MIN_BARS           = 60
BARS_FOR_200DMA    = 200
BARS_FOR_52W       = 252

IST = timezone(timedelta(hours=5, minutes=30))
MARKET_CLOSE_IST   = (15, 40)   # bars stamped today are partial before this

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


class DataError(Exception):
    """Raised when a ticker cannot be turned into a usable OHLCV frame."""


# ══════════════════════════════════════════════════════════════
#  UNIVERSE
# ══════════════════════════════════════════════════════════════
def get_nse_universe() -> list[str]:
    """Fetch the NSE equity list, EQ series only. Falls back to curated list."""
    try:
        url = "https://nsearchives.nseindia.com/content/equities/EQUITY_L.csv"
        headers = {"User-Agent": "Mozilla/5.0", "Accept-Language": "en-US,en;q=0.9"}
        resp = requests.get(url, headers=headers, timeout=15)
        resp.raise_for_status()
        df = pd.read_csv(StringIO(resp.text))
        df.columns = [c.strip().upper() for c in df.columns]
        if "SERIES" in df.columns:
            # EQ only: drops BE (trade-to-trade), SM/ST (SME), and other
            # illiquid or restricted series that have no business in a screener.
            df = df[df["SERIES"].astype(str).str.strip().str.upper() == "EQ"]
        symbols = df["SYMBOL"].dropna().astype(str).str.strip()
        symbols = [s for s in symbols if s]
        if not symbols:
            raise DataError("empty symbol list")
        return [s + ".NS" for s in symbols]
    except Exception:
        return list(FALLBACK_UNIVERSE)


# ══════════════════════════════════════════════════════════════
#  DATA FETCH
# ══════════════════════════════════════════════════════════════
def _is_partial_last_bar(df: pd.DataFrame) -> bool:
    """True if the final daily bar is today's still-forming candle."""
    if df.empty:
        return False
    now_ist = datetime.now(IST)
    last = df.index[-1]
    last_date = last.date() if hasattr(last, "date") else None
    if last_date != now_ist.date():
        return False
    close_h, close_m = MARKET_CLOSE_IST
    return (now_ist.hour, now_ist.minute) < (close_h, close_m)


def fetch_ohlcv(ticker: str, period: str = SCAN_PERIOD,
                drop_partial: bool = True) -> pd.DataFrame | None:
    """Return a clean daily OHLCV frame, or None. See `fetch_ohlcv_checked`
    when you need the failure reason instead of a silent None."""
    try:
        return fetch_ohlcv_checked(ticker, period, drop_partial)
    except DataError:
        return None


def fetch_ohlcv_checked(ticker: str, period: str = SCAN_PERIOD,
                        drop_partial: bool = True) -> pd.DataFrame:
    """Same as fetch_ohlcv but raises DataError with a reason.

    Every failure mode gets a distinct reason string so a scan can report
    "fetched 340 of 2000" instead of silently showing a short table that
    looks identical to a complete one.
    """
    try:
        df = yf.download(ticker, period=period, interval="1d",
                         progress=False, auto_adjust=True, threads=False)
    except Exception as e:
        raise DataError(f"download failed: {type(e).__name__}") from e

    if df is None or df.empty:
        raise DataError("no data returned (delisted, wrong symbol, or rate limited)")

    df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
    df = df.dropna()

    if drop_partial and _is_partial_last_bar(df):
        # Today's bar is still forming: its volume is a fraction of the day's
        # total and its close is not a close. Scoring it corrupts the volume
        # ratio and every indicator that touches the last value.
        df = df.iloc[:-1]

    if len(df) < MIN_BARS:
        raise DataError(f"only {len(df)} usable bars (need {MIN_BARS})")

    return df


# ══════════════════════════════════════════════════════════════
#  INDICATORS
# ══════════════════════════════════════════════════════════════
def _last(series: pd.Series) -> float | None:
    """Last value as a float, or None if it is NaN / missing."""
    if series is None or len(series) == 0:
        return None
    v = series.iloc[-1]
    if pd.isna(v):
        return None
    return float(v)


def compute_indicators(df: pd.DataFrame) -> dict:
    close  = df["Close"]
    volume = df["Volume"]
    high   = df["High"]
    low    = df["Low"]
    n      = len(df)

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
    p5  = float(close.iloc[-6])  if n >= 6  else p
    p20 = float(close.iloc[-21]) if n >= 21 else p

    # 52-week extremes need 252 bars and must come from High/Low, not Close.
    # With fewer bars this is an N-day range and is labelled as such.
    win = min(BARS_FOR_52W, n)
    hi_series = high.rolling(win).max()
    lo_series = low.rolling(win).min()
    high_52w = _last(hi_series) or float(high.max())
    low_52w  = _last(lo_series) or float(low.min())

    avg_vol_20 = _last(avg_v20) or 0.0
    turnover   = avg_vol_20 * p

    sma200_val = _last(sma200) if n >= BARS_FOR_200DMA else None

    return {
        "price":        p,
        "bars":         n,
        "sma20":        _last(sma20),
        "sma50":        _last(sma50),
        "sma200":       sma200_val,
        "has_200dma":   sma200_val is not None,
        "ema9":         _last(ema9),
        "ema21":        _last(ema21),
        "rsi":          _last(rsi),
        "macd":         _last(macd_l),
        "macd_signal":  _last(macd_s),
        "macd_hist":    _last(macd_h),
        "bb_upper":     _last(bb.bollinger_hband()),
        "bb_lower":     _last(bb.bollinger_lband()),
        "bb_mid":       _last(bb.bollinger_mavg()),
        "atr":          _last(atr),
        "stoch_k":      _last(stoch.stoch()),
        "stoch_d":      _last(stoch.stoch_signal()),
        "volume":       float(volume.iloc[-1]),
        "avg_vol_20":   avg_vol_20,
        "turnover":     turnover,
        "liquid":       turnover >= MIN_TURNOVER,
        "price_5d":     p5,
        "price_20d":    p20,
        "high_52w":     high_52w,
        "low_52w":      low_52w,
        "range_window": win,          # 252 = a true 52-week range
        "full_52w":     win >= BARS_FOR_52W,
        "last_bar":     df.index[-1],
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


# ══════════════════════════════════════════════════════════════
#  SCORING
# ══════════════════════════════════════════════════════════════
def score_stock(ind: dict) -> tuple[int, dict]:
    """Return (score 0-100, per-factor breakdown).

    Factors whose inputs are unavailable are excluded from the denominator
    rather than silently scored zero — otherwise a missing 200-DMA would
    permanently cap the score at 85 and no stock could ever reach STRONG BUY.
    """
    bd        = {}
    earned    = 0
    available = 0
    p         = ind["price"]

    def award(key, points, applicable=True):
        nonlocal earned, available
        if not applicable:
            bd[key] = None          # not scored, not counted against the stock
            return
        bd[key] = points
        earned    += points
        available += WEIGHTS[key]

    sma50 = ind.get("sma50")
    award("above_50dma", WEIGHTS["above_50dma"] if (sma50 and p > sma50) else 0,
          applicable=sma50 is not None)

    sma200 = ind.get("sma200")
    award("above_200dma", WEIGHTS["above_200dma"] if (sma200 and p > sma200) else 0,
          applicable=sma200 is not None)

    rsi = ind.get("rsi")
    if rsi is None:
        award("rsi_sweet_spot", 0, applicable=False)
    else:
        if   50 <= rsi <= 65:                    pts = WEIGHTS["rsi_sweet_spot"]
        elif 45 <= rsi < 50 or 65 < rsi <= 72:   pts = int(WEIGHTS["rsi_sweet_spot"] * 0.5)
        else:                                    pts = 0
        award("rsi_sweet_spot", pts)

    macd, macd_sig = ind.get("macd"), ind.get("macd_signal")
    award("macd_bullish",
          WEIGHTS["macd_bullish"] if (macd is not None and macd_sig is not None
                                      and macd > macd_sig) else 0,
          applicable=macd is not None and macd_sig is not None)

    avg_v = ind.get("avg_vol_20") or 0
    if avg_v <= 0:
        award("volume_surge", 0, applicable=False)
    else:
        vr = ind["volume"] / avg_v
        if   vr >= 1.5: pts = WEIGHTS["volume_surge"]
        elif vr >= 1.2: pts = int(WEIGHTS["volume_surge"] * 0.5)
        else:           pts = 0
        award("volume_surge", pts)

    chg5 = ((p - ind["price_5d"]) / ind["price_5d"]) * 100 if ind["price_5d"] else 0
    if   chg5 >= 3: pts = WEIGHTS["price_momentum"]
    elif chg5 >= 1: pts = int(WEIGHTS["price_momentum"] * 0.5)
    else:           pts = 0
    award("price_momentum", pts)

    sma20 = ind.get("sma20")
    award("above_20dma", WEIGHTS["above_20dma"] if (sma20 and p > sma20) else 0,
          applicable=sma20 is not None)

    if available == 0:
        return 0, bd
    score = int(round(earned * 100 / available))
    return max(0, min(100, score)), bd


def get_signal(score: int, ind: dict | None = None) -> str:
    """Signal label. A stock without a full 200-DMA / 52-week history can
    never be promoted above WATCHLIST — its score rests on less evidence."""
    capped = bool(ind) and not ind.get("has_200dma", True)
    if capped:
        return "⚡ WATCHLIST" if score >= 55 else "❌ SKIP"
    if score >= 80: return "🔥 STRONG BUY"
    if score >= 65: return "✅ BUY"
    if score >= 55: return "⚡ WATCHLIST"
    return "❌ SKIP"


def get_signal_short(score: int, ind: dict | None = None) -> str:
    return get_signal(score, ind).split(" ", 1)[1]


# ══════════════════════════════════════════════════════════════
#  TRADE LEVELS
# ══════════════════════════════════════════════════════════════
def compute_levels(ind: dict) -> dict:
    """Entry/stop/target plan.

    Hard invariant: sl < price < t1 < t2 < t3. The old code took
    max(price - 2*ATR, sma50 * 0.97), which places the stop ABOVE the entry
    for any stock trading more than 3% under its 50-DMA — producing negative
    risk, absurd R/R, and a position size limited only by the 0.01 floor.
    """
    p   = ind["price"]
    atr = ind.get("atr") or 0.0
    atr = max(atr, p * MIN_ATR_PCT)

    # Stop: the tighter (higher) of a volatility stop and a structural stop,
    # but a structural stop is only a candidate when it actually sits below price.
    candidates = [p - 2.0 * atr]
    sma50 = ind.get("sma50")
    if sma50:
        struct = sma50 * 0.97
        if struct < p:
            candidates.append(struct)
    sl = max(candidates)

    sl = min(sl, p * 0.995)                    # always below entry
    sl = max(sl, p * (1 - MAX_RISK_PCT / 100)) # never risk more than MAX_RISK_PCT
    risk = p - sl                              # strictly positive by construction

    t1 = p + 1.5 * atr
    t2 = p + 3.0 * atr
    t3 = p + 5.0 * atr

    # Only let the 52-week high cap T3 when that cap leaves room above T2.
    # Capping unconditionally puts T3 *below* the entry on a fresh breakout.
    cap = ind["high_52w"] * 0.98
    if cap > t2:
        t3 = min(t3, cap)
    t3 = max(t3, t2 + 0.5 * atr)

    return {
        "buy_low":    round(p - 0.3 * atr, 1),
        "buy_high":   round(p + 0.2 * atr, 1),
        "sl":         round(sl, 1),
        "t1":         round(t1, 1),
        "t2":         round(t2, 1),
        "t3":         round(t3, 1),
        "risk_pct":   round((risk / p) * 100, 1),
        "reward_pct": round(((t1 - p) / p) * 100, 1),
        "rr":         round((t1 - p) / risk, 2),
        "capped_t3":  cap <= t2,
    }


def position_size(capital: float, price: float, sl: float,
                  risk_fraction: float = RISK_FRACTION) -> int:
    """Shares to buy under the fixed-fractional risk rule.

    Returns 0 when the stop is not below the entry (no definable risk) and
    caps the position at what the capital can actually buy — the old inline
    `int((capital*0.02) / max(price - sl, 0.5))` returned 4000 shares of a
    ₹100 stock on ₹1L of capital whenever the stop came out above price.
    """
    risk_per_share = price - sl
    if risk_per_share <= 0 or price <= 0 or capital <= 0:
        return 0
    by_risk    = int((capital * risk_fraction) // risk_per_share)
    affordable = int(capital // price)
    return max(0, min(by_risk, affordable))


def sessions_to_target(price: float, target: float, atr: float) -> int | None:
    """Rough ATR-based estimate of sessions to reach a target.

    This is arithmetic (distance / average daily range), not a forecast. The
    old code returned a hardcoded "~15–25 days" string keyed off the score,
    with no time model behind it at all.
    """
    if not atr or atr <= 0 or target <= price:
        return None
    return max(1, int(round((target - price) / atr)))


# ══════════════════════════════════════════════════════════════
#  SCAN
# ══════════════════════════════════════════════════════════════
def scan_one(ticker: str, require_liquid: bool = True) -> dict:
    """Scan a single ticker. Always returns a dict; check `["error"]`."""
    try:
        df = fetch_ohlcv_checked(ticker, period=SCAN_PERIOD)
    except DataError as e:
        return {"error": True, "ticker": ticker, "reason": str(e)}

    try:
        ind = compute_indicators(df)
        if ind["rsi"] is None or ind["atr"] is None:
            return {"error": True, "ticker": ticker, "reason": "indicators unavailable"}
        if require_liquid and not ind["liquid"]:
            return {"error": True, "ticker": ticker,
                    "reason": f"illiquid (₹{ind['turnover']/1e7:.2f} cr/day)"}

        score, bd = score_stock(ind)
        lvl       = compute_levels(ind)
        p         = ind["price"]
        vr        = ind["volume"] / ind["avg_vol_20"] if ind["avg_vol_20"] > 0 else 0
        chg5      = ((p - ind["price_5d"])  / ind["price_5d"])  * 100
        chg20     = ((p - ind["price_20d"]) / ind["price_20d"]) * 100
        from52h   = ((p - ind["high_52w"])  / ind["high_52w"])  * 100

        return {
            "error":     False,
            "ticker":    ticker,
            "name":      ticker.replace(".NS", ""),
            "price":     round(p, 2),
            "score":     score,
            "signal":    get_signal(score, ind),
            "signal_s":  get_signal_short(score, ind),
            "rsi":       round(ind["rsi"], 1),
            "macd_h":    round(ind["macd_hist"], 3) if ind["macd_hist"] is not None else 0.0,
            "stoch_k":   round(ind["stoch_k"], 1) if ind["stoch_k"] is not None else 0.0,
            "vol_ratio": round(vr, 2),
            "chg_5d":    round(chg5, 1),
            "chg_20d":   round(chg20, 1),
            "from_52h":  round(from52h, 1),
            "above_50":  bool(ind["sma50"] and p > ind["sma50"]),
            "above_200": bool(ind["sma200"] and p > ind["sma200"]),
            "has_200":   ind["has_200dma"],
            "turnover":  ind["turnover"],
            "turnover_cr": round(ind["turnover"] / 1e7, 2),
            "atr":       round(ind["atr"], 2),
            "sl":        lvl["sl"],
            "buy_low":   lvl["buy_low"],
            "buy_high":  lvl["buy_high"],
            "t1":        lvl["t1"],
            "t2":        lvl["t2"],
            "t3":        lvl["t3"],
            "risk_pct":  lvl["risk_pct"],
            "rr":        lvl["rr"],
            "high_52w":  round(ind["high_52w"], 2),
            "low_52w":   round(ind["low_52w"], 2),
            "full_52w":  ind["full_52w"],
            "_ind":      ind,
            "_bd":       bd,
        }
    except Exception as e:
        return {"error": True, "ticker": ticker,
                "reason": f"{type(e).__name__}: {e}"}


def parallel_scan(tickers: list, max_workers: int = 8,
                  progress_cb=None, require_liquid: bool = True) -> tuple[list, dict]:
    """Scan tickers concurrently.

    Returns (results, stats). `stats` exists because a scan where 1900 of 2000
    tickers were rate-limited used to render a clean table indistinguishable
    from a complete scan. Default workers dropped 20 → 8: Yahoo throttles
    aggressively and throttled requests come back as silent empty frames.
    """
    results  = []
    failures = Counter()
    done     = 0
    total    = len(tickers)

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {ex.submit(scan_one, t, require_liquid): t for t in tickers}
        for fut in as_completed(futures):
            done += 1
            try:
                r = fut.result()
            except Exception as e:
                r = {"error": True, "ticker": futures[fut], "reason": str(e)}
            if r.get("error"):
                reason = r.get("reason", "unknown")
                key = "illiquid" if reason.startswith("illiquid") else reason.split(":")[0]
                failures[key] += 1
            else:
                results.append(r)
            if progress_cb:
                progress_cb(done, total)

    results.sort(key=lambda x: x["score"], reverse=True)
    stats = {
        "requested": total,
        "succeeded": len(results),
        "failed":    total - len(results),
        "reasons":   dict(failures),
    }
    return results, stats


# ══════════════════════════════════════════════════════════════
#  HOLDINGS
# ══════════════════════════════════════════════════════════════
def analyse_holding(symbol: str, avg_price: float, qty: int) -> dict:
    """Analyse a single demat holding."""
    symbol = str(symbol).strip().upper()
    ticker = symbol if symbol.endswith(".NS") else symbol + ".NS"

    try:
        df = fetch_ohlcv_checked(ticker, period=HOLDING_PERIOD)
    except DataError as e:
        return {"error": True, "symbol": symbol, "msg": str(e)}

    try:
        ind = compute_indicators(df)
        if ind["rsi"] is None or ind["atr"] is None:
            return {"error": True, "symbol": symbol, "msg": "indicators unavailable"}

        score, bd = score_stock(ind)
        lvl       = compute_levels(ind)
        p         = ind["price"]
        invested  = avg_price * qty
        curr_val  = p * qty
        pnl       = curr_val - invested
        pnl_pct   = ((p - avg_price) / avg_price) * 100 if avg_price else 0.0

        sessions = sessions_to_target(p, lvl["t1"], ind["atr"])
        days_t1  = (f"≈{sessions} session{'s' if sessions != 1 else ''} at the current ATR "
                    f"(arithmetic, not a forecast)"
                    if sessions else "T1 already reached or ATR unavailable")

        # Recommendation. `sl` is now guaranteed below price, so "stop hit" means
        # an actual breakdown — previously this branch fired for every stock
        # merely trading 3% under its 50-DMA, regardless of the holder's basis.
        breakdown = p <= lvl["sl"]
        if breakdown:
            action = "🔴 EXIT NOW — Stop Loss Hit"
            action_reason = (f"Price ₹{p:.2f} is at or below the algorithmic stop "
                             f"₹{lvl['sl']}. Exit rules say cut it.")
        elif score < 40 and pnl_pct >= 10:
            action = "🟡 BOOK PROFIT — Signal Gone"
            action_reason = (f"Score is only {score}. The trend that earned this "
                             f"{pnl_pct:.1f}% gain is over. Consider booking.")
        elif score < 40:
            # A score this low never means "hold and watch", profit or not.
            action = "🔴 CONSIDER EXIT — Weak Signal"
            action_reason = f"Score {score} — trend, momentum and volume all absent."
        elif score >= 65 and pnl_pct >= 10:
            action = "🟢 TRAIL SL — Strong Profit"
            action_reason = (f"Trend intact with a big gain. Trail the stop to "
                             f"₹{round(max(lvl['sl'], p * 0.95), 1)}.")
        elif score >= 65 and 5 <= pnl_pct < 10:
            action = "🟡 PARTIAL BOOK — Book 50%"
            action_reason = f"Near T1 ₹{lvl['t1']}. Book half, let the rest run to T2 ₹{lvl['t2']}."
        elif score >= 65:
            action = "🟢 HOLD — Target Not Reached"
            action_reason = f"Signal still strong. T1 at ₹{lvl['t1']}. {days_t1}."
        elif pnl_pct >= 10:
            action = "🟡 BOOK PROFIT — Signal Fading"
            action_reason = (f"Good gain but score is only {score}. Trend no longer "
                             f"supports the position; consider booking.")
        elif score < 55 and pnl_pct < -3:
            action = "🔴 CONSIDER EXIT — Weak Signal"
            action_reason = "Low score and price falling. Consider exiting to preserve capital."
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
            "signal":     get_signal(score, ind),
            "rsi":        round(ind["rsi"], 1),
            "sl":         lvl["sl"],
            "t1":         lvl["t1"],
            "t2":         lvl["t2"],
            "rr":         lvl["rr"],
            "risk_pct":   lvl["risk_pct"],
            "action":     action,
            "reason":     action_reason,
            "days_t1":    days_t1,
            "above_50":   bool(ind["sma50"] and p > ind["sma50"]),
            "above_200":  bool(ind["sma200"] and p > ind["sma200"]),
            "has_200":    ind["has_200dma"],
            "liquid":     ind["liquid"],
            "turnover_cr": round(ind["turnover"] / 1e7, 2),
            "_ind":       ind,
            "_bd":        bd,
        }
    except Exception as e:
        return {"error": True, "symbol": symbol, "msg": f"{type(e).__name__}: {e}"}
