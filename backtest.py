"""
Backtest harness for the NSE Trader Pro scoring model.

Answers the only question that matters: does the score have positive
expectancy, net of costs, out-of-sample, versus simply holding the Nifty?

Design notes — the things that make a backtest honest:

  * NO LOOK-AHEAD. Indicators at bar i use only bars <= i. Decisions are made
    on bar i's close and executed at bar i+1's OPEN. A backtest that decides
    and fills on the same bar's close is reading tomorrow's newspaper.

  * INTRABAR AMBIGUITY RESOLVED PESSIMISTICALLY. When a bar's low touches the
    stop and its high touches the target, we assume the STOP filled first.
    Daily bars cannot tell us the order, so we assume the worse one.

  * COSTS ARE REAL. STT, exchange charges, stamp duty, GST and slippage are
    charged on every entry and exit. Slippage is the number most backtests
    quietly set to zero; here it defaults to 10 bps per side.

  * WALK-FORWARD. Parameters are chosen on a training window and evaluated on
    a later window the optimiser never saw. In-sample results are worthless.

KNOWN BIAS WE CANNOT FULLY REMOVE
---------------------------------
SURVIVORSHIP. FALLBACK_UNIVERSE is a list of companies that are healthy and
listed TODAY. Backtesting it over 2010-2025 implicitly assumes we knew in 2010
which firms would survive. This inflates returns. Real delisted names (and the
losers we never added) are absent. Treat every return below as an OPTIMISTIC
upper bound, not an expectation. The Nifty benchmark is computed on the index,
which has its own (smaller, rebalancing-driven) survivorship characteristics.

Usage:
    python backtest.py                 # baseline + per-year + walk-forward
    python backtest.py --quick         # curated universe, 2018+, faster
    python backtest.py --sweep         # parameter sweep (train) -> test
"""

from __future__ import annotations

import argparse
import os
import sys
import warnings
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import ta
import yfinance as yf

from engine import FALLBACK_UNIVERSE, WEIGHTS, MIN_ATR_PCT, MAX_RISK_PCT

warnings.filterwarnings("ignore")

CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".cache")
BENCHMARK = "^NSEI"
TRADING_DAYS = 252


# ══════════════════════════════════════════════════════════════
#  COSTS  (Indian equity delivery, retail, discount broker)
# ══════════════════════════════════════════════════════════════
@dataclass(frozen=True)
class Costs:
    """Round-trip cost model. Defaults are deliberately not optimistic."""
    brokerage_pct:   float = 0.0003   # 0.03%, capped at ₹20 by most brokers
    brokerage_cap:   float = 20.0
    stt_buy_pct:     float = 0.001    # 0.1% delivery
    stt_sell_pct:    float = 0.001
    exchange_pct:    float = 0.0000325
    sebi_pct:        float = 0.000001
    stamp_buy_pct:   float = 0.00015
    gst_pct:         float = 0.18     # on brokerage + exchange charges
    slippage_pct:    float = 0.0010   # 10 bps per side — the number most
                                      # backtests silently set to zero

    def entry_cost(self, value: float) -> float:
        brok = min(value * self.brokerage_pct, self.brokerage_cap)
        exch = value * self.exchange_pct
        return (brok + exch + (brok + exch) * self.gst_pct
                + value * self.stt_buy_pct
                + value * self.sebi_pct
                + value * self.stamp_buy_pct)

    def exit_cost(self, value: float) -> float:
        brok = min(value * self.brokerage_pct, self.brokerage_cap)
        exch = value * self.exchange_pct
        return (brok + exch + (brok + exch) * self.gst_pct
                + value * self.stt_sell_pct
                + value * self.sebi_pct)

    def buy_fill(self, px: float) -> float:
        return px * (1 + self.slippage_pct)

    def sell_fill(self, px: float) -> float:
        return px * (1 - self.slippage_pct)


# ══════════════════════════════════════════════════════════════
#  STRATEGY PARAMETERS
# ══════════════════════════════════════════════════════════════
@dataclass(frozen=True)
class Params:
    min_score:      int   = 55     # app default MIN_SCORE
    t1_atr:         float = 1.5    # app default
    t2_atr:         float = 3.0
    stop_atr:       float = 2.0    # app default -> R/R of t1/stop = 0.75
    max_positions:  int   = 3      # app rule 5: "max 2-3 positions"
    risk_fraction:  float = 0.02   # app rule: 2% risk
    max_hold_days:  int   = 30     # swing horizon; no exit signal otherwise
    book_half_at_t1: bool = True   # app rule 3: "book 50% at T1"
    require_200dma: bool  = False  # only trade above the 200-DMA
    rebalance_dow:  int   = 0      # 0 = Monday (app rule 6)


# ══════════════════════════════════════════════════════════════
#  DATA
# ══════════════════════════════════════════════════════════════
def _cache_path(ticker: str) -> str:
    return os.path.join(CACHE_DIR, ticker.replace("/", "_") + ".pkl")


def load_history(ticker: str, start: str = "2010-01-01") -> pd.DataFrame | None:
    os.makedirs(CACHE_DIR, exist_ok=True)
    path = _cache_path(ticker)
    if os.path.exists(path):
        try:
            df = pd.read_pickle(path)
            if len(df) > 0:
                return df
        except Exception:
            pass
    try:
        df = yf.download(ticker, start=start, progress=False,
                         auto_adjust=True, threads=False)
    except Exception:
        return None
    if df is None or df.empty:
        return None
    df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
    df = df.dropna()
    df.index = pd.to_datetime(df.index).tz_localize(None)
    df.to_pickle(path)
    return df


def load_universe(tickers: list[str], start: str) -> dict[str, pd.DataFrame]:
    data, missing = {}, []
    for i, t in enumerate(tickers, 1):
        df = load_history(t, start)
        if df is None or len(df) < 260:
            missing.append(t)
        else:
            data[t] = df
        if i % 25 == 0:
            print(f"  loaded {i}/{len(tickers)}...", flush=True)
    if missing:
        print(f"  skipped {len(missing)} tickers with no/short history "
              f"(e.g. {', '.join(missing[:5])})")
    return data


# ══════════════════════════════════════════════════════════════
#  VECTORISED INDICATORS + SCORE  (no look-ahead: row i uses bars <= i)
# ══════════════════════════════════════════════════════════════
def build_features(df: pd.DataFrame, p: Params) -> pd.DataFrame:
    close, high, low, vol = df["Close"], df["High"], df["Low"], df["Volume"]

    f = pd.DataFrame(index=df.index)
    f["close"]  = close
    f["open"]   = df["Open"]
    f["high"]   = high
    f["low"]    = low

    f["sma20"]  = ta.trend.sma_indicator(close, 20)
    f["sma50"]  = ta.trend.sma_indicator(close, 50)
    f["sma200"] = ta.trend.sma_indicator(close, 200)
    f["rsi"]    = ta.momentum.rsi(close, 14)
    macd        = ta.trend.MACD(close)
    f["macd"]   = macd.macd()
    f["macd_s"] = macd.macd_signal()
    f["atr"]    = ta.volatility.average_true_range(high, low, close, 14)
    f["avgvol"] = vol.rolling(20).mean()
    f["volrat"] = vol / f["avgvol"]
    f["chg5"]   = close.pct_change(5) * 100
    f["hi252"]  = high.rolling(252).max()
    f["turnover"] = f["avgvol"] * close

    # ── score, mirroring engine.score_stock exactly ──
    earned    = pd.Series(0.0, index=df.index)
    available = pd.Series(0.0, index=df.index)

    def add(mask_points: pd.Series, applicable: pd.Series, weight: float):
        nonlocal earned, available
        earned    = earned.add(mask_points.where(applicable, 0.0), fill_value=0)
        available = available.add(applicable.astype(float) * weight, fill_value=0)

    ok50 = f["sma50"].notna()
    add((close > f["sma50"]).astype(float) * WEIGHTS["above_50dma"], ok50, WEIGHTS["above_50dma"])

    ok200 = f["sma200"].notna()
    add((close > f["sma200"]).astype(float) * WEIGHTS["above_200dma"], ok200, WEIGHTS["above_200dma"])

    rsi = f["rsi"]
    rsi_pts = pd.Series(0.0, index=df.index)
    rsi_pts[(rsi >= 50) & (rsi <= 65)] = WEIGHTS["rsi_sweet_spot"]
    rsi_pts[((rsi >= 45) & (rsi < 50)) | ((rsi > 65) & (rsi <= 72))] = int(WEIGHTS["rsi_sweet_spot"] * 0.5)
    add(rsi_pts, rsi.notna(), WEIGHTS["rsi_sweet_spot"])

    okmacd = f["macd"].notna() & f["macd_s"].notna()
    add((f["macd"] > f["macd_s"]).astype(float) * WEIGHTS["macd_bullish"], okmacd, WEIGHTS["macd_bullish"])

    vr = f["volrat"]
    v_pts = pd.Series(0.0, index=df.index)
    v_pts[vr >= 1.2] = int(WEIGHTS["volume_surge"] * 0.5)
    v_pts[vr >= 1.5] = WEIGHTS["volume_surge"]
    add(v_pts, f["avgvol"].notna() & (f["avgvol"] > 0), WEIGHTS["volume_surge"])

    c_pts = pd.Series(0.0, index=df.index)
    c_pts[f["chg5"] >= 1] = int(WEIGHTS["price_momentum"] * 0.5)
    c_pts[f["chg5"] >= 3] = WEIGHTS["price_momentum"]
    add(c_pts, f["chg5"].notna(), WEIGHTS["price_momentum"])

    ok20 = f["sma20"].notna()
    add((close > f["sma20"]).astype(float) * WEIGHTS["above_20dma"], ok20, WEIGHTS["above_20dma"])

    f["score"] = np.where(available > 0, (earned / available * 100).round(), 0)

    # ── trade levels, mirroring engine.compute_levels invariants ──
    atr = f["atr"].clip(lower=close * MIN_ATR_PCT)
    f["atr_eff"] = atr

    struct = f["sma50"] * 0.97
    vol_stop = close - p.stop_atr * atr
    sl = np.where(struct.notna() & (struct < close),
                  np.maximum(vol_stop, struct), vol_stop)
    sl = np.minimum(sl, close * 0.995)
    sl = np.maximum(sl, close * (1 - MAX_RISK_PCT / 100))
    f["sl"] = sl

    f["t1"] = close + p.t1_atr * atr
    f["t2"] = close + p.t2_atr * atr

    return f


# ══════════════════════════════════════════════════════════════
#  PORTFOLIO BACKTEST
# ══════════════════════════════════════════════════════════════
@dataclass
class Position:
    ticker: str
    entry_date: pd.Timestamp
    entry_px: float
    qty: int
    sl: float
    t1: float
    t2: float
    orig_qty: int = 0          # qty before any partial book — odd lots make
    entry_total: float = 0.0   # qty*2 a wrong cost basis after booking half
    half_booked: bool = False
    realised: float = 0.0


@dataclass
class Trade:
    ticker: str
    entry_date: pd.Timestamp
    exit_date: pd.Timestamp
    entry_px: float
    exit_px: float
    qty: int
    pnl: float
    pnl_pct: float
    reason: str
    hold_days: int


def position_size(capital: float, price: float, sl: float, risk_fraction: float) -> int:
    risk = price - sl
    if risk <= 0 or price <= 0 or capital <= 0:
        return 0
    return max(0, min(int((capital * risk_fraction) // risk), int(capital // price)))


PANEL_COLS = ("open", "high", "low", "close", "score", "sl", "t1", "t2",
              "turnover", "sma200")


class Panel:
    """Feature dict flattened into aligned [date x ticker] numpy matrices.

    The event loop originally did `f.loc[date]` per ticker per day — ~124k
    pandas label lookups per run, which is what made a 36-point sweep take an
    hour. Same numbers, ~50x less time.
    """

    def __init__(self, feats: dict[str, pd.DataFrame], calendar: pd.DatetimeIndex):
        self.tickers = sorted(feats)
        self.calendar = calendar
        self.tidx = {t: i for i, t in enumerate(self.tickers)}
        self.m = {}
        for c in PANEL_COLS:
            self.m[c] = np.column_stack([
                feats[t][c].reindex(calendar).to_numpy(dtype=float)
                for t in self.tickers
            ])
        self.present = ~np.isnan(self.m["close"])


def run_backtest(feats: dict[str, pd.DataFrame], calendar: pd.DatetimeIndex,
                 p: Params, costs: Costs, initial_capital: float = 100_000.0,
                 min_turnover: float = 2e7,
                 panel: Panel | None = None,
                 tiebreak_seed: int | None = None) -> tuple[pd.Series, list[Trade]]:
    """`tiebreak_seed` randomises the order of equal-scoring candidates.

    The score is a rounded integer, so ties are extremely common and there are
    always more qualifying stocks than the 3 slots. Which of the tied names you
    buy is arbitrary. If the strategy's returns swing with the seed, the result
    is noise, not edge — see `--noise`.
    """
    rng = np.random.default_rng(tiebreak_seed) if tiebreak_seed is not None else None
    pan = panel if panel is not None else Panel(feats, calendar)
    if pan.calendar is not calendar:
        # Sub-window (train/test split) of a prefit panel: slice the matrices.
        pos_idx = pan.calendar.get_indexer(calendar)
        pan2 = object.__new__(Panel)
        pan2.tickers, pan2.calendar, pan2.tidx = pan.tickers, calendar, pan.tidx
        pan2.m = {c: v[pos_idx] for c, v in pan.m.items()}
        pan2.present = pan.present[pos_idx]
        pan = pan2

    M, T = pan.m, pan.tickers
    cash      = initial_capital
    open_pos: dict[str, Position] = {}
    trades: list[Trade] = []
    equity_dates, equity_vals = [], []
    dows = np.array([d.dayofweek for d in calendar])

    for di in range(1, len(calendar)):
        today = calendar[di]

        # ── 1. Manage open positions on TODAY's bar ──
        for tkr in list(open_pos.keys()):
            j = pan.tidx[tkr]
            if not pan.present[di, j]:
                continue
            pos = open_pos[tkr]

            op = M["open"][di, j]
            hi = M["high"][di, j]
            lo = M["low"][di, j]
            exit_px = exit_reason = None

            # Pessimistic: if both stop and target are inside the bar's range,
            # assume the stop filled first. Daily bars cannot tell us the order.
            # A gap THROUGH the stop fills at the open, not at the stop price —
            # you do not get your stop price in a gap down.
            if lo <= pos.sl:
                exit_px = min(op, pos.sl)
                exit_reason = "stop"
            elif pos.half_booked and hi >= pos.t2:
                exit_px, exit_reason = max(op, pos.t2), "t2"
            elif (not pos.half_booked) and hi >= pos.t1:
                if p.book_half_at_t1 and pos.qty >= 2:
                    fill_px = max(op, pos.t1)
                    half = pos.qty // 2
                    gross = costs.sell_fill(fill_px) * half
                    net = gross - costs.exit_cost(gross)
                    cash += net
                    pos.realised += net
                    pos.qty -= half
                    pos.half_booked = True
                else:
                    exit_px, exit_reason = max(op, pos.t1), "t1"
            if exit_reason is None and (today - pos.entry_date).days >= p.max_hold_days:
                exit_px, exit_reason = M["close"][di, j], "timeout"

            if exit_reason:
                gross = costs.sell_fill(exit_px) * pos.qty
                net   = gross - costs.exit_cost(gross)
                cash += net
                # Basis is the ORIGINAL position, all-in. Reconstructing it as
                # qty*2 after a half-book silently mis-prices every odd lot.
                proceeds = pos.realised + net
                pnl = proceeds - pos.entry_total
                trades.append(Trade(
                    tkr, pos.entry_date, today, pos.entry_px, exit_px, pos.orig_qty,
                    pnl, (pnl / pos.entry_total * 100) if pos.entry_total else 0.0,
                    exit_reason, (today - pos.entry_date).days))
                del open_pos[tkr]

        # ── 2. Entries: decide on PREV close, fill at TODAY's open ──
        if dows[di] == p.rebalance_dow and len(open_pos) < p.max_positions:
            pi = di - 1
            score_p = M["score"][pi]
            sl_p    = M["sl"][pi]
            close_p = M["close"][pi]
            turn_p  = M["turnover"][pi]
            open_t  = M["open"][di]

            elig = (
                np.isfinite(score_p) & (score_p >= p.min_score)
                & np.isfinite(sl_p) & (sl_p < close_p)
                & np.isfinite(turn_p) & (turn_p >= min_turnover)
                & np.isfinite(open_t) & (open_t > 0)
                & pan.present[di]
            )
            if p.require_200dma:
                sma200_p = M["sma200"][pi]
                elig &= np.isfinite(sma200_p) & (close_p > sma200_p)

            cand = np.flatnonzero(elig)
            if rng is not None:
                # Random tie-break: jitter strictly smaller than one score point.
                key = -(score_p[cand] + rng.random(cand.size) * 0.999)
            else:
                # Deterministic tie-break: among equal scores prefer the more
                # liquid name. Arbitrary, but at least economically defensible.
                tn = turn_p[cand]
                key = -(score_p[cand] + tn / (tn.max() + 1e9) * 0.999) if cand.size else -score_p[cand]
            cand = cand[np.argsort(key, kind="stable")]

            for j in cand:
                if len(open_pos) >= p.max_positions:
                    break
                tkr = T[j]
                if tkr in open_pos:
                    continue
                open_px = open_t[j]
                # Levels were computed on prev close; keep them fixed (that is
                # what the app shows the user). Size on the actual fill price.
                fill = costs.buy_fill(open_px)
                sl, t1, t2 = sl_p[j], M["t1"][pi, j], M["t2"][pi, j]
                if sl >= fill or not np.isfinite(t1) or not np.isfinite(t2):
                    continue
                qty = position_size(cash, fill, sl, p.risk_fraction)
                if qty < 1:
                    continue
                gross = fill * qty
                total = gross + costs.entry_cost(gross)
                if total > cash:
                    qty = int((cash * 0.98) // fill)
                    if qty < 1:
                        continue
                    gross = fill * qty
                    total = gross + costs.entry_cost(gross)
                    if total > cash:
                        continue
                cash -= total
                open_pos[tkr] = Position(tkr, today, float(open_px), qty,
                                         float(sl), float(t1), float(t2),
                                         orig_qty=qty, entry_total=total)

        # ── 3. Mark to market ──
        mtm = cash
        for tkr, pos in open_pos.items():
            j = pan.tidx[tkr]
            px = M["close"][di, j] if pan.present[di, j] else pos.entry_px
            mtm += px * pos.qty
        equity_dates.append(today)
        equity_vals.append(mtm)

    return pd.Series(equity_vals, index=equity_dates), trades


# ══════════════════════════════════════════════════════════════
#  METRICS
# ══════════════════════════════════════════════════════════════
def metrics(equity: pd.Series, trades: list[Trade]) -> dict:
    if len(equity) < 2:
        return {}
    rets = equity.pct_change().dropna()
    years = (equity.index[-1] - equity.index[0]).days / 365.25
    total = equity.iloc[-1] / equity.iloc[0] - 1
    cagr = (equity.iloc[-1] / equity.iloc[0]) ** (1 / years) - 1 if years > 0 else 0
    dd = (equity / equity.cummax() - 1)
    sharpe = (rets.mean() / rets.std() * np.sqrt(TRADING_DAYS)) if rets.std() > 0 else 0
    downside = rets[rets < 0].std()
    sortino = (rets.mean() / downside * np.sqrt(TRADING_DAYS)) if downside and downside > 0 else 0

    wins = [t for t in trades if t.pnl > 0]
    losses = [t for t in trades if t.pnl <= 0]
    avg_w = np.mean([t.pnl for t in wins]) if wins else 0
    avg_l = abs(np.mean([t.pnl for t in losses])) if losses else 0
    win_rate = len(wins) / len(trades) if trades else 0
    expectancy = win_rate * avg_w - (1 - win_rate) * avg_l
    pf = (sum(t.pnl for t in wins) / abs(sum(t.pnl for t in losses))) if losses and sum(t.pnl for t in losses) else float("inf")

    return {
        "total_return": total, "cagr": cagr, "max_dd": dd.min(),
        "sharpe": sharpe, "sortino": sortino, "years": years,
        "n_trades": len(trades), "win_rate": win_rate,
        "avg_win": avg_w, "avg_loss": avg_l,
        "payoff": (avg_w / avg_l) if avg_l else float("inf"),
        "expectancy": expectancy, "profit_factor": pf,
        "final": equity.iloc[-1],
    }


def benchmark_equity(start, end, initial: float) -> pd.Series | None:
    bm = load_history(BENCHMARK, start="2010-01-01")
    if bm is None:
        return None
    bm = bm.loc[(bm.index >= start) & (bm.index <= end), "Close"]
    if bm.empty:
        return None
    return bm / bm.iloc[0] * initial


def fmt(m: dict, label: str) -> str:
    if not m:
        return f"{label}: no data"
    return (f"{label:26} CAGR {m['cagr']*100:6.2f}%  |  Total {m['total_return']*100:8.2f}%  |  "
            f"MaxDD {m['max_dd']*100:7.2f}%  |  Sharpe {m['sharpe']:5.2f}  |  "
            f"Trades {m['n_trades']:4d}  Win {m['win_rate']*100:5.1f}%  "
            f"Payoff {m['payoff']:.2f}  PF {m['profit_factor']:.2f}")


# ══════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2010-01-01")
    ap.add_argument("--quick", action="store_true", help="2018+, 60 tickers")
    ap.add_argument("--sweep", action="store_true", help="train/test param sweep")
    ap.add_argument("--noise", type=int, default=0,
                    help="N runs with randomised tie-breaking among equal scores")
    ap.add_argument("--capital", type=float, default=100_000.0)
    ap.add_argument("--slippage", type=float, default=0.0010)
    args = ap.parse_args()

    start = "2018-01-01" if args.quick else args.start
    tickers = FALLBACK_UNIVERSE[:60] if args.quick else FALLBACK_UNIVERSE
    costs = Costs(slippage_pct=args.slippage)

    print("=" * 108)
    print("NSE Trader Pro — strategy validation")
    print("=" * 108)
    print(f"Universe: {len(tickers)} tickers (SURVIVORSHIP-BIASED: today's survivors)")
    print(f"Period:   {start} onward   |   Slippage: {args.slippage*1e4:.0f} bps/side   "
          f"|   Capital: Rs {args.capital:,.0f}")
    print("\nLoading data (cached to .cache/)...")
    data = load_universe(tickers, start)
    if not data:
        print("No data. Aborting.")
        return 1
    print(f"  {len(data)} tickers usable\n")

    base = Params()
    print("Building features (indicators + score, no look-ahead)...")
    feats = {t: build_features(df, base) for t, df in data.items()}
    feats = {t: f.loc[f.index >= start] for t, f in feats.items()}
    feats = {t: f for t, f in feats.items() if len(f) > 60}

    calendar = pd.DatetimeIndex(sorted(set().union(*[f.index for f in feats.values()])))
    print(f"  {len(calendar)} trading days, {calendar[0].date()} -> {calendar[-1].date()}\n")

    # ── Baseline: the app's own default parameters ──
    print("-" * 108)
    print("BASELINE — the app's current defaults (min_score=55, stop=2 ATR, T1=1.5 ATR -> R/R 0.75)")
    print("-" * 108)
    eq, trades = run_backtest(feats, calendar, base, costs, args.capital)
    m = metrics(eq, trades)
    print(fmt(m, "STRATEGY"))

    bm = benchmark_equity(eq.index[0], eq.index[-1], args.capital)
    if bm is not None:
        bm_m = metrics(bm, [])
        print(fmt(bm_m, "NIFTY 50 buy & hold"))
        print(f"\n  Strategy final: Rs {m['final']:,.0f}   |   Nifty final: Rs {bm_m['final']:,.0f}")
        verdict = "BEATS" if m["cagr"] > bm_m["cagr"] else "LOSES TO"
        print(f"  => Strategy {verdict} the benchmark on CAGR.")

    if trades:
        print(f"\n  Expectancy per trade: Rs {m['expectancy']:,.0f}  "
              f"(win {m['win_rate']*100:.1f}%, avg win Rs {m['avg_win']:,.0f}, "
              f"avg loss Rs {m['avg_loss']:,.0f})")
        be = 1 / (1 + m["payoff"]) * 100 if m["payoff"] not in (0, float("inf")) else float("nan")
        print(f"  Breakeven win rate at this payoff ratio: {be:.1f}%  "
              f"(actual {m['win_rate']*100:.1f}%)")
        rc = pd.Series([t.reason for t in trades]).value_counts()
        print(f"  Exit reasons: {dict(rc)}")
        pd.DataFrame([t.__dict__ for t in trades]).to_csv("backtest_trades.csv", index=False)
        print("  Trade log -> backtest_trades.csv")

    # ── Per-year (regime sensitivity) ──
    print("\n" + "-" * 108)
    print("PER-YEAR — does it work in every regime, or only in bull markets?")
    print("-" * 108)
    print(f"{'Year':6} {'Strategy':>12} {'Nifty':>12} {'Diff':>12}   {'Trades':>7}")
    for yr in sorted({d.year for d in eq.index}):
        e = eq[eq.index.year == yr]
        if len(e) < 20:
            continue
        s_ret = e.iloc[-1] / e.iloc[0] - 1
        b_ret = float("nan")
        if bm is not None:
            b = bm[bm.index.year == yr]
            if len(b) >= 2:
                b_ret = b.iloc[-1] / b.iloc[0] - 1
        n = len([t for t in trades if t.entry_date.year == yr])
        diff = s_ret - b_ret
        flag = "  <-- lost to index" if diff < 0 else ""
        print(f"{yr:6} {s_ret*100:11.2f}% {b_ret*100:11.2f}% {diff*100:11.2f}%   {n:7d}{flag}")

    # ── Noise test: is the edge distinguishable from arbitrary choice? ──
    if args.noise:
        print("\n" + "-" * 108)
        print(f"NOISE TEST — {args.noise} runs, identical rules, only the tie-break "
              f"among equal scores randomised")
        print("-" * 108)
        panel = Panel(feats, calendar)
        cagrs, dds = [], []
        for s in range(args.noise):
            e, tr = run_backtest(feats, calendar, base, costs, args.capital,
                                 panel=panel, tiebreak_seed=s)
            mm = metrics(e, tr)
            cagrs.append(mm["cagr"] * 100)
            dds.append(mm["max_dd"] * 100)
            print(f"  seed {s:2d}: CAGR {mm['cagr']*100:6.2f}%  MaxDD {mm['max_dd']*100:7.2f}%  "
                  f"trades {mm['n_trades']:4d}  PF {mm['profit_factor']:.2f}", flush=True)
        c = np.array(cagrs)
        print(f"\n  CAGR across seeds: mean {c.mean():.2f}%  std {c.std():.2f}%  "
              f"min {c.min():.2f}%  max {c.max():.2f}%")
        if bm is not None:
            beat = (c > bm_m["cagr"] * 100).sum()
            print(f"  Beat Nifty ({bm_m['cagr']*100:.2f}%) in {beat}/{len(c)} runs.")
        print("\n  A strategy with real edge barely moves when you shuffle which of two\n"
              "  equally-scored stocks it buys. A wide spread means the P&L is being\n"
              "  driven by which coin came up, not by the score.")

    # ── Walk-forward sweep ──
    if args.sweep:
        print("\n" + "-" * 108)
        print("WALK-FORWARD — optimise on TRAIN, report on TEST (never seen by the optimiser)")
        print("-" * 108)
        split = calendar[int(len(calendar) * 0.6)]
        print(f"Train: {calendar[0].date()} -> {split.date()}   "
              f"Test: {split.date()} -> {calendar[-1].date()}\n")

        train_cal = calendar[calendar <= split]
        test_cal  = calendar[calendar > split]

        grid = [Params(min_score=ms, t1_atr=t1, stop_atr=st, require_200dma=r200)
                for ms in (55, 65, 75)
                for t1 in (1.5, 3.0, 4.0)
                for st in (1.5, 2.0)
                for r200 in (False, True)]

        # Only t1_atr/t2_atr/stop_atr change the feature matrices; min_score and
        # require_200dma are filters applied at selection time. Cache accordingly.
        panel_cache: dict[tuple, Panel] = {}

        def panel_for(g: Params) -> Panel:
            key = (g.t1_atr, g.t2_atr, g.stop_atr)
            if key not in panel_cache:
                gf = {t: build_features(df, g) for t, df in data.items()}
                gf = {t: f.loc[f.index >= start] for t, f in gf.items() if len(f) > 60}
                panel_cache[key] = Panel(gf, calendar)
            return panel_cache[key]

        rows = []
        for i, g in enumerate(grid, 1):
            e, tr = run_backtest(feats, train_cal, g, costs, args.capital,
                                 panel=panel_for(g))
            mm = metrics(e, tr)
            rows.append((mm.get("sharpe", 0), mm.get("cagr", 0), g, mm))
            print(f"  [{i:2d}/{len(grid)}] score>={g.min_score} T1={g.t1_atr} SL={g.stop_atr} "
                  f"200dma={g.require_200dma}: train CAGR {mm.get('cagr',0)*100:6.2f}% "
                  f"Sharpe {mm.get('sharpe',0):5.2f} trades {mm.get('n_trades',0)}", flush=True)

        rows = [r for r in rows if r[3].get("n_trades", 0) >= 20]
        if not rows:
            print("\n  No parameter set produced >= 20 trades in-sample. Nothing to validate.")
            return 0
        rows.sort(key=lambda r: -r[0])
        best = rows[0][2]
        print(f"\n  Best on TRAIN by Sharpe: score>={best.min_score} T1={best.t1_atr} "
              f"SL={best.stop_atr} 200dma={best.require_200dma}")

        bp = panel_for(best)
        e_tr, t_tr = run_backtest(feats, train_cal, best, costs, args.capital, panel=bp)
        e_te, t_te = run_backtest(feats, test_cal,  best, costs, args.capital, panel=bp)
        print("\n  " + fmt(metrics(e_tr, t_tr), "IN-SAMPLE (train)"))
        print("  " + fmt(metrics(e_te, t_te), "OUT-OF-SAMPLE (test)"))
        if bm is not None and len(e_te) > 2:
            bt = benchmark_equity(e_te.index[0], e_te.index[-1], args.capital)
            if bt is not None:
                print("  " + fmt(metrics(bt, []), "NIFTY over test window"))
        print("\n  If out-of-sample collapses versus in-sample, the parameters were "
              "fitted to noise.")

    print("\n" + "=" * 108)
    print("REMEMBER: universe is survivorship-biased -> real-world results will be WORSE than shown.")
    print("=" * 108)
    return 0


if __name__ == "__main__":
    sys.exit(main())
