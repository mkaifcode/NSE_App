"""
NSE Trading Dashboard — Final Version
Pages: 🏠 Home | 🔍 Screener | 📊 Chart | 💼 Demat Analysis | ⭐ Watchlist | 📔 Trade Journal
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import json, io, os, html
from datetime import datetime, date
import yfinance as yf
import warnings
warnings.filterwarnings("ignore")

from engine import (
    parallel_scan, get_nse_universe, fetch_ohlcv, fetch_ohlcv_checked,
    compute_indicators, score_stock, compute_levels,
    get_signal, analyse_holding, position_size, sessions_to_target,
    DataError, WEIGHTS, MIN_SCORE, FALLBACK_UNIVERSE,
    RISK_FRACTION, MODEL_DISCLAIMER, SCAN_PERIOD,
)


def esc(v) -> str:
    """Escape anything interpolated into an unsafe_allow_html string.

    Symbols and names reach the UI from user-uploaded broker CSVs; without
    this, a crafted Symbol column injects markup into the page.
    """
    return html.escape(str(v), quote=True)

# ══════════════════════════════════════════════════════════════
#  CONFIG & GLOBAL STYLES
# ══════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="NSE Trader Pro",
    page_icon="📈",
    layout="wide",
    # "auto" = expanded on desktop, collapsed on phones. "expanded" made the
    # sidebar overlay the top nav on a 390px screen, covering the whole app.
    initial_sidebar_state="auto",
)

# Meta tags go in their own call. Markdown ends a bare-tag HTML block at the
# first blank line; when <meta> and <style> shared one string, every CSS rule
# after the first blank line was re-parsed as markdown and rendered as text.
st.markdown(
    '<meta name="viewport" content="width=device-width,initial-scale=1,maximum-scale=1">'
    '<meta name="mobile-web-app-capable" content="yes">'
    '<meta name="theme-color" content="#0a0e1a">',
    unsafe_allow_html=True,
)

# `<style>` must be the very first character: that makes this a raw-HTML block
# that markdown passes through verbatim until </style>, blank lines and all.
st.markdown("""<style>
/* ── Layout ── */
/* Hide the toolbar and footer, but NOT <header> itself — the sidebar
   expand arrow lives inside the header, and hiding it left a collapsed
   sidebar with no way to reopen. */
footer { visibility: hidden; }
[data-testid="stDecoration"] { display: none !important; }

/* Do NOT hide stToolbar: the sidebar's expand arrow is rendered inside it.
   Hiding the toolbar (or the whole <header>) is what left a collapsed sidebar
   with no way to reopen — worst on touch screens, where Streamlit's hover-to-
   reveal never fires. Hide only the toolbar's own actions. */
[data-testid="stToolbar"] { display: flex !important; }
[data-testid="stToolbarActions"],
[data-testid="stAppDeployButton"],
[data-testid="stStatusWidget"],
[data-testid="stMainMenu"], #MainMenu { display: none !important; }

/* The header is sticky and spans the full width. Left solid, its empty area
   swallows clicks on whatever sits under it — which was the top nav. Make the
   bar itself click-through, but keep its children (the expand arrow) clickable. */
header[data-testid="stHeader"] {
  background: transparent !important;
  pointer-events: none;
}
header[data-testid="stHeader"] * { pointer-events: auto; }

/* Always-on, never hover-gated. */
[data-testid="stExpandSidebarButton"],
[data-testid="stExpandSidebarButton"] button {
  display: flex !important;
  visibility: visible !important;
  opacity: 1 !important;
  width: auto !important; height: auto !important;
  z-index: 1000;
}

/* Top padding must clear the sticky header, or the first widget hides beneath it. */
.block-container { padding: 4.2rem 1.5rem 2rem !important; }
@media(max-width:768px){ .block-container{ padding:3.4rem 0.6rem 2rem !important; } }

/* ── Cards ── */
.card {
  background: linear-gradient(135deg,#131929,#0f1522);
  border: 1px solid #1e2d45;
  border-radius: 14px;
  padding: 18px 20px;
  margin-bottom: 12px;
}
.card-sm {
  background: #131929;
  border: 1px solid #1e2d45;
  border-radius: 10px;
  padding: 14px 16px;
  text-align: center;
  height: 100%;
}
.kpi-val   { font-size:24px; font-weight:700; margin:4px 0 2px; }
.kpi-label { font-size:11px; color:#6b7a99; text-transform:uppercase; letter-spacing:.06em; }

/* ── Signals ── */
.sig-strong{ color:#00d26a; font-weight:700; }
.sig-buy   { color:#3b9eff; font-weight:600; }
.sig-watch { color:#f59e0b; font-weight:500; }
.sig-skip  { color:#f87171; }

/* ── Badges ── */
.badge {
  display:inline-block; padding:3px 10px;
  border-radius:20px; font-size:11px; font-weight:600;
}
.badge-green { background:#0d2b1a; color:#00d26a; border:1px solid #00d26a44; }
.badge-blue  { background:#0d1f36; color:#3b9eff; border:1px solid #3b9eff44; }
.badge-amber { background:#2b1d08; color:#f59e0b; border:1px solid #f59e0b44; }
.badge-red   { background:#2b0d0d; color:#f87171; border:1px solid #f87171aa; }

/* ── Table ── */
.stDataFrame { font-size:13px !important; }

/* ── Nav pill ── */
div[data-testid="stRadio"] > div {
  display:flex; flex-direction:row; flex-wrap:wrap; gap:4px;
}
div[data-testid="stRadio"] label {
  border-radius:8px !important; padding:6px 14px !important;
  font-size:13px !important;
}

/* ── Holding card ── */
.hold-card {
  background:#131929;
  border:1px solid #1e2d45;
  border-radius:12px;
  padding:16px 18px;
  margin-bottom:10px;
  transition: border-color .2s;
}
.hold-card:hover { border-color:#3b9eff55; }
.hold-profit { color:#00d26a; font-weight:700; }
.hold-loss   { color:#f87171; font-weight:700; }

/* ── Divider ── */
hr { border-color:#1e2d45 !important; margin:1rem 0 !important; }

/* ── Scrollbar ── */
::-webkit-scrollbar { width:5px; }
::-webkit-scrollbar-track { background:#0a0e1a; }
::-webkit-scrollbar-thumb { background:#1e2d45; border-radius:3px; }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
#  SESSION STATE
# ══════════════════════════════════════════════════════════════
def ss(key, default):
    if key not in st.session_state:
        st.session_state[key] = default

PAGES = [
    "🏠 Home",
    "🔍 Screener",
    "📊 Chart",
    "💼 Demat Analysis",
    "⭐ Watchlist",
    "📔 Trade Journal",
]

ss("nav",           PAGES[0])
ss("scan_results",  [])
ss("scan_stats",    None)
ss("watchlist",     [])
ss("journal",       [])
ss("demat_df",      None)
ss("demat_analysis",[])
ss("chart_ticker",  "")
ss("capital",       100000)


# ══════════════════════════════════════════════════════════════
#  SIDEBAR NAVIGATION
# ══════════════════════════════════════════════════════════════
def goto(p: str):
    """Navigate from anywhere.

    Writes the radio's own session key. Streamlit forbids this after the widget
    is instantiated, but permits it inside a callback — which is why this is only
    ever passed as `on_click`, never called inline.
    """
    st.session_state.nav = p


# Navigation lives in the MAIN body, not the sidebar. Streamlit only reveals the
# sidebar's expand arrow on header hover — which does not exist on touch screens,
# so a collapsed sidebar used to strand the user with no way to navigate.
#
# The radio is bound by `key`, not by `index=`. With `index=` the widget keeps
# its own retained value and silently ignores the index on rerun, so navigating
# back to a previously-visited page did nothing.
st.radio(
    "Navigation", PAGES, key="nav",
    horizontal=True, label_visibility="collapsed",
)
page = st.session_state.nav

# Measured, not guessed: see backtest.py. 2010-2026, 137 NSE stocks, net of
# costs, on a survivorship-biased (i.e. flattering) universe.
st.error(
    "**Do not trade these signals.** Backtested 2010–2026: this scoring model "
    "returns **−2.25% CAGR** versus **+9.60%** for a Nifty 50 index fund, with a "
    "49% drawdown and negative expectancy (−₹36/trade over 920 trades). No "
    "parameter setting tested was profitable. Run `python backtest.py` to verify. "
    "Educational use only — not SEBI-registered, not investment advice.",
    icon="🚨",
)

with st.sidebar:
    st.markdown("""
    <div style='text-align:center;padding:16px 0 8px'>
      <div style='font-size:28px'>📈</div>
      <div style='font-size:18px;font-weight:700;color:#e8eaf0'>NSE Trader Pro</div>
      <div style='font-size:11px;color:#6b7a99;margin-top:2px'>Swing Trading Dashboard</div>
    </div>
    """, unsafe_allow_html=True)
    st.divider()

    st.session_state.capital = st.number_input(
        "💰 Trading Capital (₹)",
        min_value=1000, max_value=10_000_000,
        value=st.session_state.capital, step=5000
    )
    st.caption(f"2% risk = ₹{int(st.session_state.capital*0.02):,} per trade")
    st.divider()
    # Two slots, filled later. A container renders where it is DECLARED, not
    # where it is written to, so settings must be declared above the footer.
    sidebar_settings = st.container()
    sidebar_footer   = st.container()


capital = st.session_state.capital


# ══════════════════════════════════════════════════════════════
#  HELPER: full chart for any stock
# ══════════════════════════════════════════════════════════════
VIEW_BARS = {"1mo": 22, "3mo": 63, "6mo": 126, "1y": 252, "2y": 10_000}


def render_chart(ticker_ns: str, period: str = "6mo"):
    # Always pull the full history the indicators need, then slice for display.
    # Fetching only `period` meant a 6-month chart computed a 200-DMA over 126
    # bars — it came back NaN, so the 200-DMA criterion silently scored zero
    # and this page disagreed with the Demat page on the very same stock.
    try:
        df_full = fetch_ohlcv_checked(ticker_ns, period=SCAN_PERIOD)
    except DataError as e:
        st.error(f"Failed to fetch chart data for {esc(ticker_ns)}: {esc(e)}")
        return

    ind       = compute_indicators(df_full)
    score, bd = score_stock(ind)
    lvl       = compute_levels(ind)
    p         = ind["price"]
    name      = ticker_ns.replace(".NS", "")

    view = min(VIEW_BARS.get(period, 126), len(df_full))
    df   = df_full.tail(view)
    tv   = lambda s: s.tail(view)   # slice an indicator series to the view window

    # Header
    prev  = float(df_full["Close"].iloc[-2])
    chg1d = ((p - prev) / prev) * 100
    cc    = "#00d26a" if chg1d >= 0 else "#f87171"
    arr   = "▲" if chg1d >= 0 else "▼"

    rng_label = "52W" if ind["full_52w"] else f"{ind['range_window']}D"
    warn = ""
    if not ind["liquid"]:
        warn += f'<div class="badge badge-red">Illiquid — ₹{ind["turnover"]/1e7:.2f} cr/day</div>'
    if not ind["has_200dma"]:
        warn += '<div class="badge badge-amber">No 200-DMA — short history</div>'

    st.markdown(f"""
    <div class="card" style="padding:14px 20px;">
      <div style="display:flex;align-items:center;gap:20px;flex-wrap:wrap;">
        <div style="font-size:22px;font-weight:700">{esc(name)}</div>
        <div>
          <span style="font-size:26px;font-weight:600">₹{p:.2f}</span>
          <span style="color:{cc};font-size:14px;margin-left:8px">{arr} {abs(chg1d):.2f}%</span>
        </div>
        <div class="badge {'badge-green' if score>=65 else 'badge-amber' if score>=55 else 'badge-red'}">
          Score {score}/100 — {esc(get_signal(score, ind).split(' ',1)[1])}
        </div>
        <div style="color:#6b7a99;font-size:12px">
          {rng_label}: ₹{ind['low_52w']:.0f} – ₹{ind['high_52w']:.0f}
        </div>
        {warn}
      </div>
    </div>
    """, unsafe_allow_html=True)

    # KPI strip
    k1,k2,k3,k4,k5,k6 = st.columns(6)
    kpis = [
        (f"₹{lvl['sl']}",  "Stop Loss",   "#f87171"),
        (f"₹{lvl['t1']}",  "Target 1",    "#00d26a"),
        (f"₹{lvl['t2']}",  "Target 2",    "#00d26a"),
        (f"{ind['rsi']:.1f}", "RSI",       "#f59e0b"),
        (f"{ind['volume']/ind['avg_vol_20']:.2f}×", "Vol Ratio", "#3b9eff"),
        (f"{lvl['rr']}×",  "Risk/Reward", "#a78bfa"),
    ]
    for col,(val,lbl,clr) in zip([k1,k2,k3,k4,k5,k6], kpis):
        col.markdown(f"""<div class="card-sm">
          <div style="color:{clr}" class="kpi-val">{val}</div>
          <div class="kpi-label">{lbl}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("")

    # Main chart
    fig = make_subplots(
        rows=4, cols=1, shared_xaxes=True,
        row_heights=[0.52,0.16,0.16,0.16],
        vertical_spacing=0.015,
        subplot_titles=("","","MACD","RSI")
    )

    # Candlestick
    fig.add_trace(go.Candlestick(
        x=df.index, open=df["Open"], high=df["High"],
        low=df["Low"], close=df["Close"], name="Price",
        increasing_line_color="#00d26a", decreasing_line_color="#f87171",
        increasing_fillcolor="#00d26a22", decreasing_fillcolor="#f8717122",
    ), row=1, col=1)

    # MAs
    for series, name_s, color, width, dash in [
        (ind["_sma20"],  "20 SMA",  "#f59e0b", 1.2, "solid"),
        (ind["_sma50"],  "50 SMA",  "#3b9eff", 1.5, "solid"),
        (ind["_ema9"],   "9 EMA",   "#a78bfa", 1.0, "dot"),
    ]:
        fig.add_trace(go.Scatter(x=df.index, y=tv(series), name=name_s,
            line=dict(color=color, width=width, dash=dash)), row=1, col=1)

    if ind["sma200"] is not None:
        fig.add_trace(go.Scatter(x=df.index, y=tv(ind["_sma200"]), name="200 SMA",
            line=dict(color="#f87171", width=1.5)), row=1, col=1)

    # Bollinger Bands
    fig.add_trace(go.Scatter(x=df.index, y=tv(ind["_bb_upper"]), name="BB Upper",
        line=dict(color="rgba(59,158,255,0.3)", width=1)), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=tv(ind["_bb_lower"]), name="BB Lower",
        line=dict(color="rgba(59,158,255,0.3)", width=1),
        fill="tonexty", fillcolor="rgba(59,158,255,0.04)"), row=1, col=1)

    # Levels
    for lvl_val, lvl_name, clr in [
        (lvl["sl"],  f"SL ₹{lvl['sl']}",  "#f87171"),
        (lvl["t1"],  f"T1 ₹{lvl['t1']}",  "#00d26a"),
        (lvl["t2"],  f"T2 ₹{lvl['t2']}",  "#86efac"),
    ]:
        fig.add_hline(y=lvl_val, line_dash="dot", line_color=clr,
                      annotation_text=lvl_name,
                      annotation_font_color=clr,
                      annotation_position="right",
                      row=1, col=1)

    # Volume
    vcols = ["#00d26a" if c >= o else "#f87171"
             for c,o in zip(df["Close"], df["Open"])]
    fig.add_trace(go.Bar(x=df.index, y=df["Volume"], name="Volume",
        marker_color=vcols, opacity=0.6), row=2, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=tv(ind["_avg_vol_20"]),
        name="Vol MA20", line=dict(color="#f59e0b", width=1)), row=2, col=1)

    # MACD
    macd_h = tv(ind["_macd_h"])
    hcols  = ["#00d26a" if v >= 0 else "#f87171" for v in macd_h]
    fig.add_trace(go.Bar(x=df.index, y=macd_h, name="MACD Hist",
        marker_color=hcols, opacity=0.7), row=3, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=tv(ind["_macd_l"]), name="MACD",
        line=dict(color="#3b9eff", width=1.5)), row=3, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=tv(ind["_macd_s"]), name="Signal",
        line=dict(color="#f59e0b", width=1.2)), row=3, col=1)

    # RSI
    fig.add_trace(go.Scatter(x=df.index, y=tv(ind["_rsi"]), name="RSI",
        line=dict(color="#a78bfa", width=1.5)), row=4, col=1)
    fig.add_hrect(y0=50, y1=65, fillcolor="rgba(0,210,106,0.05)",
                  line_width=0, row=4, col=1)
    for lvl_r, clr_r in [(70,"#f87171"),(30,"#00d26a"),(50,"#6b7a99")]:
        fig.add_hline(y=lvl_r, line_dash="dot",
                      line_color=clr_r, line_width=0.8, row=4, col=1)

    fig.update_layout(
        height=680,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#0a0e1a",
        xaxis_rangeslider_visible=False,
        legend=dict(orientation="h", yanchor="bottom", y=1.01,
                    x=0, font=dict(size=10), bgcolor="rgba(0,0,0,0)"),
        margin=dict(t=10, b=10, l=10, r=90),
        font=dict(color="#e8eaf0", size=11),
    )
    for i in range(1,5):
        fig.update_xaxes(gridcolor="#1a2235", showgrid=True, row=i, col=1)
        fig.update_yaxes(gridcolor="#1a2235", showgrid=True, row=i, col=1)

    st.plotly_chart(fig, use_container_width=True)

    # Score + levels
    col_bd, col_lv = st.columns(2)
    with col_bd:
        st.markdown("#### 🎯 Score Breakdown")
        # A factor scored None was not applicable (e.g. no 200-DMA yet). It is
        # excluded from the denominator, so it must not be drawn as a zero.
        skipped = [k for k, v in bd.items() if v is None]
        bd_df = pd.DataFrame([{
            "Criterion": k.replace("_"," ").title(),
            "Earned": v, "Max": WEIGHTS[k],
        } for k, v in bd.items() if v is not None])
        if skipped:
            st.caption("Not scored (insufficient history): "
                       + ", ".join(k.replace("_", " ") for k in skipped))
        fig_bd = px.bar(bd_df, x="Earned", y="Criterion", orientation="h",
                        color="Earned", color_continuous_scale="RdYlGn",
                        range_color=[0,20],
                        text=bd_df.apply(lambda r:f"{r['Earned']}/{r['Max']}",axis=1))
        fig_bd.update_traces(textposition="outside")
        fig_bd.update_layout(height=260, margin=dict(t=5,b=5,l=5,r=60),
                             paper_bgcolor="rgba(0,0,0,0)",
                             plot_bgcolor="rgba(0,0,0,0)",
                             coloraxis_showscale=False,
                             font=dict(color="#e8eaf0",size=11))
        st.plotly_chart(fig_bd, use_container_width=True)

    with col_lv:
        st.markdown("#### 📐 Trade Plan")
        qty = position_size(capital, p, lvl["sl"])
        req = round(qty * p, 0)
        if qty == 0:
            st.warning("Position size is 0 — the 2% risk budget cannot buy even "
                       "one share at this stop distance. Reduce risk or skip.")
        st.markdown(f"""
<div class="card">
<table style="width:100%;font-size:13px;border-collapse:collapse;">
<tr><td style="color:#6b7a99;padding:5px 0">🔴 Stop Loss</td><td style="text-align:right;color:#f87171;font-weight:600">₹{lvl['sl']} &nbsp; (-{lvl['risk_pct']}%)</td></tr>
<tr><td style="color:#6b7a99;padding:5px 0">🟡 Buy Zone</td><td style="text-align:right;font-weight:600">₹{lvl['buy_low']} – ₹{lvl['buy_high']}</td></tr>
<tr><td style="color:#6b7a99;padding:5px 0">🟢 Target 1</td><td style="text-align:right;color:#00d26a;font-weight:600">₹{lvl['t1']} &nbsp; (+{lvl['reward_pct']}%)</td></tr>
<tr><td style="color:#6b7a99;padding:5px 0">🟢 Target 2</td><td style="text-align:right;color:#00d26a;font-weight:600">₹{lvl['t2']} &nbsp; (+{round(((lvl['t2']-p)/p)*100,1)}%)</td></tr>
<tr><td style="color:#6b7a99;padding:5px 0">🟢 Target 3</td><td style="text-align:right;color:#86efac;font-weight:600">₹{lvl['t3']} &nbsp; (+{round(((lvl['t3']-p)/p)*100,1)}%)</td></tr>
<tr style="border-top:1px solid #1e2d45"><td style="color:#6b7a99;padding:8px 0 5px">⚖️ Risk/Reward</td><td style="text-align:right;color:#a78bfa;font-weight:600">{lvl['rr']}×</td></tr>
<tr><td style="color:#6b7a99;padding:5px 0">📦 Qty (2% rule)</td><td style="text-align:right;font-weight:700;color:#3b9eff">{qty} shares</td></tr>
<tr><td style="color:#6b7a99;padding:5px 0">💰 Capital needed</td><td style="text-align:right;font-weight:600">₹{req:,.0f}</td></tr>
</table>
</div>
""", unsafe_allow_html=True)

    # Watchlist button
    ticker_clean = ticker_ns.replace(".NS","")
    if st.button(f"⭐ Add {ticker_clean} to Watchlist", use_container_width=False):
        if ticker_ns not in st.session_state.watchlist:
            st.session_state.watchlist.append(ticker_ns)
            st.success("Added to watchlist!")
        else:
            st.info("Already in watchlist.")


# ══════════════════════════════════════════════════════════════
#  PAGE: HOME
# ══════════════════════════════════════════════════════════════
if page == "🏠 Home":
    st.markdown("## 🏠 Welcome to NSE Trader Pro")

    # Quick market pulse
    st.markdown("### 📡 Market Pulse")
    # "NIFTY_MIDCAP_150.NS" returns an empty frame from Yahoo, so the fourth
    # card silently rendered "Loading..." forever. ^NSEMDCP50 resolves.
    indices = {
        "NIFTY 50":     "^NSEI",
        "NIFTY BANK":   "^NSEBANK",
        "SENSEX":       "^BSESN",
        "NIFTY MIDCAP": "^NSEMDCP50",
    }
    idx_cols = st.columns(4)
    for col, (name, sym) in zip(idx_cols, indices.items()):
        try:
            t   = yf.Ticker(sym)
            h   = t.history(period="2d")
            if len(h) >= 2:
                c  = float(h["Close"].iloc[-1])
                pc = float(h["Close"].iloc[-2])
                ch = ((c-pc)/pc)*100
                clr = "#00d26a" if ch >= 0 else "#f87171"
                arr = "▲" if ch >= 0 else "▼"
                col.markdown(f"""<div class="card-sm">
                  <div style="font-size:11px;color:#6b7a99">{name}</div>
                  <div style="font-size:18px;font-weight:700;margin:4px 0">{c:,.0f}</div>
                  <div style="color:{clr};font-size:13px">{arr} {abs(ch):.2f}%</div>
                </div>""", unsafe_allow_html=True)
        except Exception:
            col.markdown(f"""<div class="card-sm">
              <div style="font-size:11px;color:#6b7a99">{name}</div>
              <div style="color:#6b7a99;font-size:12px;margin-top:8px">Loading...</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Features grid
    st.markdown("### 🚀 What You Can Do")
    f1, f2, f3 = st.columns(3)
    # These were plain <div> cards — the "Run a scan →" text looked like a link
    # but nothing was clickable. Each card now carries a real button.
    features = [
        ("🔍 Screener", "Scan the NSE universe. The algorithm scores every stock 0–100.",
         "Run a scan", "🔍 Screener"),
        ("📊 Chart",    "Candlestick, moving averages, Bollinger Bands, MACD, RSI, volume, trade levels.",
         "View chart", "📊 Chart"),
        ("💼 Demat",    "Upload your broker holdings CSV. Get a Hold / Exit / Book call on each stock.",
         "Analyse portfolio", "💼 Demat Analysis"),
        ("⭐ Watchlist", "Save stocks you're tracking. One-click refresh of scores and signals.",
         "View watchlist", "⭐ Watchlist"),
        ("📔 Journal",  "Log every trade with entry, exit and reason. Track P&L and learn from history.",
         "Open journal", "📔 Trade Journal"),
        ("📐 Position", "Auto-calculates quantity from the 2% risk rule and your capital.",
         "Set capital", "🔍 Screener"),
    ]
    for i, (title, desc, cta, target) in enumerate(features):
        col = [f1, f2, f3][i % 3]
        with col:
            st.markdown(f"""<div class="card" style="min-height:118px;margin-bottom:4px">
              <div style="font-size:16px;font-weight:600;margin-bottom:6px">{title}</div>
              <div style="font-size:12px;color:#8a9ab5;line-height:1.6">{desc}</div>
            </div>""", unsafe_allow_html=True)
            st.button(f"{cta} →", key=f"feat_{i}", use_container_width=True,
                      on_click=goto, args=(target,))

    st.markdown("<br>", unsafe_allow_html=True)

    # Algorithm rules
    st.markdown("### 📋 Algorithm Rules — Always Follow")
    rules = [
        ("1", "Never buy if score < 55", "No exceptions, regardless of tips or news"),
        ("2", "Always set SL before buying", "Before order goes in, SL must be decided"),
        ("3", "Book 50% profit at T1", "Let other 50% run to T2 for bigger gains"),
        ("4", "Never average down", "If SL hits, exit. Don't add to a losing trade"),
        ("5", "Max 2–3 positions at once", "With ₹10K capital, 2 positions is ideal"),
        ("6", "Re-run screener every Monday", "Fresh week = fresh signals"),
        ("7", "If Nifty below 200 DMA", "Reduce all position sizes by 50%"),
        ("8", "Log every trade in journal", "Review monthly. Learn from losses"),
    ]
    r1, r2 = st.columns(2)
    for i, (num, rule, sub) in enumerate(rules):
        col = r1 if i%2==0 else r2
        col.markdown(f"""<div class="card" style="padding:12px 16px;margin-bottom:8px;">
          <div style="display:flex;gap:12px;align-items:flex-start">
            <div style="background:#1e2d45;border-radius:50%;width:24px;height:24px;display:flex;align-items:center;justify-content:center;font-size:11px;font-weight:700;flex-shrink:0">{num}</div>
            <div>
              <div style="font-weight:600;font-size:13px">{rule}</div>
              <div style="font-size:11px;color:#6b7a99;margin-top:2px">{sub}</div>
            </div>
          </div>
        </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
#  PAGE: SCREENER
# ══════════════════════════════════════════════════════════════
elif page == "🔍 Screener":
    st.markdown("## 🔍 NSE Swing Trade Screener")

    # Settings. Two controls are enough: how much you risk, and how strict you
    # are. Everything else has a sane default and is tucked away — including the
    # "Sector Filter", which was collected and then never used by any code path.
    with sidebar_settings:
        st.markdown("### ⚙️ Scan Settings")
        st.caption("Capital is set above ⬆️")
        min_score = st.slider(
            "Minimum Score", 40, 90, MIN_SCORE, 5,
            help="Only show stocks scoring at least this. Higher = fewer, "
                 "stricter picks. Backtested edge at every level: none.")
        run_btn = st.button("🚀 Run Scan", type="primary", use_container_width=True)

        with st.expander("Advanced"):
            universe_sz  = st.selectbox("Universe size", [200, 500, 1000, 2000], index=1)
            use_fallback = st.checkbox("Curated list only (faster)", False)
            req_liq = st.checkbox("Require ₹2 cr+ daily turnover", True,
                                  help="Excludes stocks you cannot enter or exit "
                                       "without moving the price.")
            min_rsi = st.slider("Min RSI", 30, 60, 45)
            max_rsi = st.slider("Max RSI", 55, 85, 75)
            min_vol = st.slider("Min volume ratio", 0.5, 3.0, 0.8, 0.1)

    if run_btn:
        with st.spinner("Fetching NSE universe list..."):
            universe = FALLBACK_UNIVERSE if use_fallback else get_nse_universe()
            if not universe:
                universe = FALLBACK_UNIVERSE
            universe = universe[:universe_sz]

        pb  = st.progress(0.0, text="Initialising...")
        def cb(done, total):
            pb.progress(done/total, text=f"Scanning... {done}/{total}")

        results, scan_stats = parallel_scan(universe, max_workers=8,
                                            progress_cb=cb, require_liquid=req_liq)
        pb.progress(1.0, text=f"✅ Done — usable data for {scan_stats['succeeded']} "
                              f"of {scan_stats['requested']} stocks.")
        st.session_state.scan_results = results
        st.session_state.scan_stats   = scan_stats

    results    = st.session_state.scan_results
    scan_stats = st.session_state.get("scan_stats")

    # A scan where most tickers were rate-limited used to render a clean, short
    # table that looked exactly like a complete one. Say what was actually fetched.
    if scan_stats and scan_stats["failed"]:
        pct = scan_stats["succeeded"] / max(scan_stats["requested"], 1) * 100
        detail = ", ".join(f"{v}× {esc(k)}" for k, v in
                           sorted(scan_stats["reasons"].items(), key=lambda x: -x[1])[:4])
        msg = (f"Scanned {scan_stats['succeeded']}/{scan_stats['requested']} "
               f"({pct:.0f}%). Skipped {scan_stats['failed']}: {detail}.")
        (st.warning if pct < 80 else st.info)(msg)
    if not results:
        st.markdown("""<div class="card" style="text-align:center;padding:40px">
          <div style="font-size:36px">🔍</div>
          <div style="font-size:16px;margin-top:12px;color:#8a9ab5">
            Configure settings in the sidebar and click <strong>Run Full Scan</strong>
          </div>
        </div>""", unsafe_allow_html=True)
    else:
        df_all = pd.DataFrame(results)
        df_fil = df_all[
            (df_all["score"]     >= min_score) &
            (df_all["rsi"]       >= min_rsi)   &
            (df_all["rsi"]       <= max_rsi)   &
            (df_all["vol_ratio"] >= min_vol)
        ].copy()

        # KPI
        k1,k2,k3,k4,k5 = st.columns(5)
        kd = [
            (len(df_all),                                "Scanned",       "#6b7a99"),
            (len(df_all[df_all["score"]>=min_score]),    f"Score≥{min_score}", "#3b9eff"),
            (len(df_all[df_all["score"]>=80]),           "🔥 Strong Buy", "#00d26a"),
            (len(df_all[df_all["score"]>=65]),           "✅ Buy",        "#00d26a"),
            (len(df_fil),                                "After Filters", "#f59e0b"),
        ]
        for col,(v,l,c) in zip([k1,k2,k3,k4,k5],kd):
            col.markdown(f"""<div class="card-sm">
              <div class="kpi-val" style="color:{c}">{v}</div>
              <div class="kpi-label">{l}</div>
            </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Charts
        ch1, ch2 = st.columns(2)
        with ch1:
            fig_h = px.histogram(df_all, x="score", nbins=25,
                                 color_discrete_sequence=["#3b9eff"],
                                 title="Score Distribution")
            fig_h.add_vline(x=min_score, line_dash="dash",
                            line_color="#f59e0b", annotation_text=f"Min {min_score}")
            fig_h.add_vline(x=80, line_dash="dash",
                            line_color="#00d26a", annotation_text="Strong Buy")
            fig_h.update_layout(height=240, margin=dict(t=35,b=10,l=10,r=10),
                                paper_bgcolor="rgba(0,0,0,0)",
                                plot_bgcolor="#0a0e1a",
                                font=dict(color="#e8eaf0"))
            st.plotly_chart(fig_h, use_container_width=True)

        with ch2:
            top20 = df_all.nlargest(20,"score")
            fig_b = px.bar(top20, x="name", y="score",
                           color="score", color_continuous_scale="RdYlGn",
                           range_color=[40,100], title="Top 20 by Score")
            fig_b.update_layout(height=240, margin=dict(t=35,b=40,l=10,r=10),
                                paper_bgcolor="rgba(0,0,0,0)",
                                plot_bgcolor="#0a0e1a",
                                coloraxis_showscale=False,
                                xaxis_tickangle=-45,
                                font=dict(color="#e8eaf0"))
            st.plotly_chart(fig_b, use_container_width=True)

        st.divider()
        st.markdown(f"### 📋 Candidates — {len(df_fil)} stocks")

        if df_fil.empty:
            st.warning("No stocks pass all filters. Try relaxing RSI range or Volume Ratio.")
        else:
            df_show = df_fil.copy()
            df_show["Qty"]   = df_show.apply(
                lambda r: position_size(capital, r["price"], r["sl"]), axis=1)
            df_show[">50D"]  = df_show["above_50"].map({True:"✅",False:"❌"})
            # "—" not "❌": no 200-DMA means unknown, not below it.
            df_show[">200D"] = df_show.apply(
                lambda r: ("✅" if r["above_200"] else "❌") if r["has_200"] else "—", axis=1)

            show_cols = {
                "name":"Stock","price":"CMP ₹","score":"Score","signal_s":"Signal",
                "rsi":"RSI","vol_ratio":"Vol×","chg_5d":"5D%","chg_20d":"20D%",
                ">50D":">50D",">200D":">200D",
                "turnover_cr":"₹cr/day",
                "sl":"SL ₹","t1":"T1 ₹","t2":"T2 ₹",
                "rr":"R/R","risk_pct":"Risk%","Qty":"Qty(2%)",
                "from_52h":"52H%",
            }
            st.dataframe(
                df_show[list(show_cols.keys())].rename(columns=show_cols),
                use_container_width=True, height=460,
            )

            ca, cb_, cc = st.columns(3)
            with ca:
                csv = df_show[list(show_cols.keys())].rename(columns=show_cols).to_csv(index=False)
                st.download_button("⬇️ Download CSV", csv,
                    f"scan_{datetime.now().strftime('%d%b%Y')}.csv",
                    "text/csv", use_container_width=True)
            with cb_:
                if st.button("⭐ Add Top 10 to Watchlist", use_container_width=True):
                    added = 0
                    for t in df_fil.nlargest(10,"score")["ticker"].tolist():
                        if t not in st.session_state.watchlist:
                            st.session_state.watchlist.append(t)
                            added += 1
                    st.success(f"Added {added} stocks to watchlist!")

        # Quick chart access
        st.divider()
        st.markdown("### 📊 Quick Chart")
        all_names = df_fil["name"].tolist() if not df_fil.empty else df_all["name"].tolist()
        if all_names:
            sc1, sc2 = st.columns([3,1])
            with sc1:
                sel_chart = st.selectbox("Pick stock", all_names)
            with sc2:
                per_sel = st.selectbox("Period", ["3mo","6mo","1y"], index=1)
            if st.button("📊 Load Chart", type="primary", use_container_width=False):
                st.session_state.chart_ticker = sel_chart
                render_chart(sel_chart + ".NS", per_sel)


# ══════════════════════════════════════════════════════════════
#  PAGE: CHART
# ══════════════════════════════════════════════════════════════
elif page == "📊 Chart":
    st.markdown("## 📊 Stock Chart & Analysis")

    ci, cp, cb_ = st.columns([3,1,1])
    with ci:
        ticker_in = st.text_input("NSE Symbol",
            value=st.session_state.chart_ticker,
            placeholder="e.g. HDFCBANK, SSWL, JMFINANCIL").upper().strip()
    with cp:
        period_in = st.selectbox("Period", ["1mo","3mo","6mo","1y","2y"], index=2)
    with cb_:
        st.markdown("<br>", unsafe_allow_html=True)
        go_btn = st.button("📥 Load Chart", type="primary", use_container_width=True)

    if ticker_in and go_btn:
        st.session_state.chart_ticker = ticker_in
        with st.spinner(f"Loading {ticker_in}..."):
            render_chart(ticker_in + ".NS", period_in)
    elif st.session_state.chart_ticker and not go_btn:
        render_chart(st.session_state.chart_ticker + ".NS", "6mo")
    else:
        st.markdown("""<div class="card" style="text-align:center;padding:50px">
          <div style="font-size:48px">📊</div>
          <div style="font-size:15px;color:#8a9ab5;margin-top:12px">
            Enter any NSE symbol above and click Load Chart
          </div>
          <div style="font-size:12px;color:#6b7a99;margin-top:6px">
            e.g. HDFCBANK · RELIANCE · INFY · SSWL · JMFINANCIL
          </div>
        </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
#  PAGE: DEMAT ANALYSIS
# ══════════════════════════════════════════════════════════════
elif page == "💼 Demat Analysis":
    st.markdown("## 💼 Demat Portfolio Analysis")
    st.caption("Upload your holdings CSV from CDSL / HDFC Securities / any broker → Get Hold/Exit/Buy-more signals")

    # Instructions
    with st.expander("📖 How to download your holdings CSV", expanded=False):
        st.markdown("""
**HDFC Securities:**
Settings → Portfolio → Holdings → Export / Download CSV

**Zerodha Kite:**
Console → Portfolio → Holdings → Download

**Upstox:**
Portfolio → Holdings → Export

**CDSL Myeasi:**
Login at myeasi.cdsl.com → Holdings → Download

**Manual entry** (if no CSV): Use the manual input below ↓

> The CSV should have columns: **Symbol, Quantity, Average Price**
> (Column names can vary — the app auto-detects common formats)
        """)

    tab_upload, tab_manual = st.tabs(["📤 Upload CSV", "✏️ Manual Entry"])

    with tab_upload:
        uploaded = st.file_uploader(
            "Drop your holdings CSV here",
            type=["csv","xlsx","xls"],
            help="CSV from HDFC Securities, Zerodha, Upstox, CDSL, Angel One etc."
        )

        if uploaded:
            try:
                if uploaded.name.endswith(".csv"):
                    raw = pd.read_csv(uploaded)
                else:
                    raw = pd.read_excel(uploaded)

                st.markdown("**Preview (first 5 rows):**")
                st.dataframe(raw.head(), use_container_width=True)

                # Auto-detect columns
                col_map = {}
                for c in raw.columns:
                    cl = c.lower().strip()
                    if any(x in cl for x in ["symbol","scrip","stock","name","isin"]):
                        col_map["symbol"] = c
                    if any(x in cl for x in ["qty","quantity","shares","holding"]):
                        col_map["qty"] = c
                    if any(x in cl for x in ["avg","average","buy","cost","price","rate"]):
                        col_map["avg"] = c

                st.markdown("**Column Mapping** (auto-detected — correct if wrong):")
                mc1, mc2, mc3 = st.columns(3)
                with mc1:
                    sym_col = st.selectbox("Symbol column", raw.columns.tolist(),
                        index=raw.columns.tolist().index(col_map.get("symbol", raw.columns[0])))
                with mc2:
                    qty_col = st.selectbox("Quantity column", raw.columns.tolist(),
                        index=raw.columns.tolist().index(col_map.get("qty", raw.columns[1] if len(raw.columns)>1 else raw.columns[0])))
                with mc3:
                    avg_col = st.selectbox("Avg Price column", raw.columns.tolist(),
                        index=raw.columns.tolist().index(col_map.get("avg", raw.columns[2] if len(raw.columns)>2 else raw.columns[0])))

                if st.button("🔬 Analyse My Portfolio", type="primary"):
                    holdings = []
                    for _, row in raw.iterrows():
                        try:
                            sym = str(row[sym_col]).strip().upper()
                            qty = int(float(str(row[qty_col]).replace(",","")))
                            avg = float(str(row[avg_col]).replace(",","").replace("₹",""))
                            if sym and qty > 0 and avg > 0:
                                holdings.append((sym, qty, avg))
                        except Exception:
                            continue

                    st.session_state.demat_df = holdings
                    st.success(f"Found {len(holdings)} holdings. Analysing...")

            except Exception as e:
                st.error(f"Could not read file: {e}")
                st.info("Try saving as CSV (UTF-8) and re-uploading.")

    with tab_manual:
        st.markdown("Enter your holdings manually:")
        manual_text = st.text_area(
            "Format: SYMBOL, QTY, AVG_PRICE  (one per line)",
            placeholder="HDFCBANK, 10, 1650\nJMFINANCIL, 50, 124\nSSWL, 20, 215",
            height=150
        )
        if st.button("🔬 Analyse Manual Entry", type="primary"):
            holdings = []
            for line in manual_text.strip().split("\n"):
                parts = [p.strip() for p in line.split(",")]
                if len(parts) == 3:
                    try:
                        holdings.append((parts[0].upper(), int(parts[1]), float(parts[2])))
                    except Exception:
                        pass
            if holdings:
                st.session_state.demat_df = holdings
            else:
                st.warning("No valid entries found. Use format: SYMBOL, QTY, AVG_PRICE")

    # ── Run analysis ──
    if st.session_state.demat_df:
        holdings = st.session_state.demat_df
        st.divider()
        st.markdown(f"### 📊 Portfolio Analysis — {len(holdings)} Holdings")

        if not st.session_state.demat_analysis or st.button("🔄 Refresh Analysis"):
            pb = st.progress(0.0, text="Analysing holdings...")
            analyses = []
            for i, (sym, qty, avg) in enumerate(holdings):
                pb.progress((i+1)/len(holdings), text=f"Analysing {sym}...")
                r = analyse_holding(sym, avg, qty)
                analyses.append(r)
            pb.empty()
            st.session_state.demat_analysis = analyses

        analyses = st.session_state.demat_analysis
        valid    = [a for a in analyses if not a.get("error")]
        broken   = [a for a in analyses if a.get("error")]

        if broken:
            # These rows are missing from every total below. Say so — a silently
            # dropped holding makes the portfolio P&L quietly wrong.
            st.warning(
                f"No data for {len(broken)} holding(s); they are excluded from all "
                "totals below: "
                + ", ".join(f"{esc(a['symbol'])} ({esc(a.get('msg','unknown'))})"
                            for a in broken)
            )

        if not valid:
            st.error("Could not fetch data for any holdings. Check symbols and internet connection.")
        else:
            # Summary KPIs
            total_inv  = sum(a["invested"] for a in valid)
            total_curr = sum(a["curr_val"] for a in valid)
            total_pnl  = total_curr - total_inv
            total_pct  = (total_pnl / total_inv * 100) if total_inv > 0 else 0
            gainers    = len([a for a in valid if a["pnl"] > 0])
            losers     = len([a for a in valid if a["pnl"] < 0])

            pk1,pk2,pk3,pk4,pk5 = st.columns(5)
            pkd = [
                (f"₹{total_inv:,.0f}",                      "Invested",   "#6b7a99"),
                (f"₹{total_curr:,.0f}",                     "Curr Value", "#3b9eff"),
                (f"₹{total_pnl:+,.0f}",                     "Total P&L",
                 "#00d26a" if total_pnl>=0 else "#f87171"),
                (f"{total_pct:+.1f}%",                      "Return",
                 "#00d26a" if total_pct>=0 else "#f87171"),
                (f"🟢{gainers}  🔴{losers}",                "G/L Ratio",  "#f59e0b"),
            ]
            for col,(v,l,c) in zip([pk1,pk2,pk3,pk4,pk5],pkd):
                col.markdown(f"""<div class="card-sm">
                  <div class="kpi-val" style="color:{c};font-size:18px">{v}</div>
                  <div class="kpi-label">{l}</div>
                </div>""", unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)

            # P&L bar chart
            pnl_df = pd.DataFrame([{
                "Stock": a["symbol"], "P&L ₹": a["pnl"], "P&L %": a["pnl_pct"]
            } for a in valid])
            fig_pnl = px.bar(pnl_df, x="Stock", y="P&L ₹",
                             color="P&L ₹", color_continuous_scale="RdYlGn",
                             title="P&L by Position", text="P&L %")
            fig_pnl.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
            fig_pnl.update_layout(height=280, paper_bgcolor="rgba(0,0,0,0)",
                                  plot_bgcolor="#0a0e1a", coloraxis_showscale=False,
                                  font=dict(color="#e8eaf0"),
                                  margin=dict(t=35,b=10))
            st.plotly_chart(fig_pnl, use_container_width=True)

            st.divider()

            # Individual holding cards
            st.markdown("### 📋 Stock-by-Stock Recommendation")

            # Sort by action urgency
            action_order = {"🔴 EXIT": 0, "🟡 PARTIAL": 1, "🟡 HOLD": 2,
                           "🟢 TRAIL": 3, "🟢 HOLD": 4}
            valid_sorted = sorted(valid,
                key=lambda a: action_order.get(a["action"][:12], 5))

            for a in valid_sorted:
                pnl_col  = "#00d26a" if a["pnl"] >= 0 else "#f87171"
                pnl_sign = "+" if a["pnl"] >= 0 else ""
                sig_cls  = ("sig-strong" if a["score"]>=80 else
                            "sig-buy"    if a["score"]>=65 else
                            "sig-watch"  if a["score"]>=55 else "sig-skip")

                action_col = ("#f87171" if "🔴" in a["action"] else
                              "#f59e0b" if "🟡" in a["action"] else "#00d26a")

                st.markdown(f"""
<div class="hold-card">
  <div style="display:flex;justify-content:space-between;align-items:flex-start;flex-wrap:wrap;gap:10px">
    <div>
      <span style="font-size:17px;font-weight:700">{esc(a['symbol'])}</span>
      <span style="margin-left:10px;font-size:12px;color:#6b7a99">{a['qty']} shares @ ₹{a['avg_price']}</span>
    </div>
    <div style="text-align:right">
      <span style="font-size:16px;font-weight:700">₹{a['cmp']}</span>
      <span style="margin-left:8px;color:{pnl_col};font-weight:600">{pnl_sign}₹{a['pnl']:,.0f} ({pnl_sign}{a['pnl_pct']:.1f}%)</span>
    </div>
  </div>
  <div style="display:flex;gap:20px;margin:10px 0;flex-wrap:wrap">
    <div><span style="color:#6b7a99;font-size:11px">SCORE</span><br>
         <span class="{sig_cls}" style="font-size:14px">{a['score']}/100</span></div>
    <div><span style="color:#6b7a99;font-size:11px">RSI</span><br>
         <span style="font-size:14px">{a['rsi']}</span></div>
    <div><span style="color:#6b7a99;font-size:11px">SIGNAL</span><br>
         <span class="{sig_cls}" style="font-size:13px">{a['signal']}</span></div>
    <div><span style="color:#6b7a99;font-size:11px">STOP LOSS</span><br>
         <span style="font-size:13px;color:#f87171">₹{a['sl']}</span></div>
    <div><span style="color:#6b7a99;font-size:11px">TARGET 1</span><br>
         <span style="font-size:13px;color:#00d26a">₹{a['t1']}</span></div>
    <div><span style="color:#6b7a99;font-size:11px">>50 DMA</span><br>
         <span style="font-size:13px">{'✅' if a['above_50'] else '❌'}</span></div>
    <div><span style="color:#6b7a99;font-size:11px">>200 DMA</span><br>
         <span style="font-size:13px">{('✅' if a['above_200'] else '❌') if a['has_200'] else '—'}</span></div>
    <div><span style="color:#6b7a99;font-size:11px">R/R</span><br>
         <span style="font-size:13px">{a['rr']}×</span></div>
  </div>
  <div style="background:#0a0e1a;border-radius:8px;padding:10px 14px;border-left:3px solid {action_col}">
    <div style="color:{action_col};font-weight:600;font-size:13px">{esc(a['action'])}</div>
    <div style="color:#8a9ab5;font-size:12px;margin-top:3px">{esc(a['reason'])}</div>
  </div>
</div>
""", unsafe_allow_html=True)

            # Export
            st.divider()
            export_df = pd.DataFrame([{
                "Symbol": a["symbol"], "Qty": a["qty"], "Avg ₹": a["avg_price"],
                "CMP ₹": a["cmp"], "P&L ₹": a["pnl"], "P&L %": a["pnl_pct"],
                "Score": a["score"], "Signal": a["signal"],
                "SL ₹": a["sl"], "T1 ₹": a["t1"], "Action": a["action"],
            } for a in valid])
            st.download_button("⬇️ Export Analysis CSV",
                export_df.to_csv(index=False),
                f"demat_analysis_{datetime.now().strftime('%d%b%Y')}.csv",
                "text/csv", use_container_width=False)


# ══════════════════════════════════════════════════════════════
#  PAGE: WATCHLIST
# ══════════════════════════════════════════════════════════════
elif page == "⭐ Watchlist":
    st.markdown("## ⭐ My Watchlist")

    # Export / Import
    with st.expander("💾 Export / Import Watchlist"):
        wl_json = json.dumps(st.session_state.watchlist, indent=2)
        st.download_button("⬇️ Export Watchlist JSON", wl_json,
                           "watchlist.json", "application/json")
        up_wl = st.file_uploader("📤 Import Watchlist JSON", type="json")
        if up_wl:
            imported = json.load(up_wl)
            st.session_state.watchlist = imported
            st.success(f"Imported {len(imported)} stocks!")

    # Add stock
    wa, wb = st.columns([4,1])
    with wa:
        new_sym = st.text_input("Add stock", placeholder="e.g. RELIANCE",
                                label_visibility="collapsed").upper().strip()
    with wb:
        if st.button("➕ Add", use_container_width=True):
            t = new_sym + ".NS" if new_sym and not new_sym.endswith(".NS") else new_sym
            if t and t not in st.session_state.watchlist:
                st.session_state.watchlist.append(t)
                st.success(f"Added {new_sym}!")
            elif t in st.session_state.watchlist:
                st.info("Already in watchlist.")

    if not st.session_state.watchlist:
        st.markdown("""<div class="card" style="text-align:center;padding:40px">
          <div style="font-size:36px">⭐</div>
          <div style="font-size:14px;color:#8a9ab5;margin-top:10px">
            Watchlist is empty. Add stocks above or from the Screener.
          </div>
        </div>""", unsafe_allow_html=True)
    else:
        if st.button("🔄 Refresh All Signals", type="primary"):
            with st.spinner("Fetching live data..."):
                # require_liquid=False: the user explicitly chose to track these.
                wl_results, wl_stats = parallel_scan(
                    st.session_state.watchlist, max_workers=6, require_liquid=False)
                st.session_state.wl_data = wl_results
                if wl_stats["failed"]:
                    st.warning(f"No data for {wl_stats['failed']} of "
                               f"{wl_stats['requested']} watchlist symbols.")

        wl_data = st.session_state.get("wl_data", [])
        if not wl_data:
            st.info("Click **Refresh All Signals** to load latest data.")
        else:
            df_wl = pd.DataFrame(wl_data)
            df_wl = df_wl.sort_values("score", ascending=False)

            for _, r in df_wl.iterrows():
                sc = r["score"]
                sig_cls = ("sig-strong" if sc>=80 else "sig-buy" if sc>=65
                           else "sig-watch" if sc>=55 else "sig-skip")
                chg_col = "#00d26a" if r["chg_5d"]>=0 else "#f87171"

                c1,c2,c3,c4,c5 = st.columns([2,1.5,2,1.5,0.5])
                c1.markdown(f"**{r['name']}**  \n₹{r['price']:.1f}")
                c2.markdown(f"Score: `{sc}/100`  \nRSI: {r['rsi']}")
                c3.markdown(f"<span class='{sig_cls}'>{r['signal']}</span>  \nSL ₹{r['sl']} → T1 ₹{r['t1']}",
                            unsafe_allow_html=True)
                c4.markdown(f"5D: <span style='color:{chg_col}'>{r['chg_5d']:+.1f}%</span>  \nVol: {r['vol_ratio']}×",
                            unsafe_allow_html=True)
                with c5:
                    if st.button("🗑️", key=f"wdel_{r['ticker']}"):
                        st.session_state.watchlist.remove(r["ticker"])
                        if "wl_data" in st.session_state:
                            del st.session_state["wl_data"]
                        st.rerun()
                st.divider()


# ══════════════════════════════════════════════════════════════
#  PAGE: TRADE JOURNAL
# ══════════════════════════════════════════════════════════════
elif page == "📔 Trade Journal":
    st.markdown("## 📔 Trade Journal")

    # Export / Import
    with st.expander("💾 Export / Import Journal"):
        jj = json.dumps(st.session_state.journal, indent=2, default=str)
        st.download_button("⬇️ Export Journal JSON", jj,
                           "trade_journal.json", "application/json")
        up_j = st.file_uploader("📤 Import Journal JSON", type="json")
        if up_j:
            st.session_state.journal = json.load(up_j)
            st.success("Journal imported!")

    # Add trade
    with st.expander("➕ Log New Trade", expanded=len(st.session_state.journal)==0):
        jc1,jc2,jc3 = st.columns(3)
        with jc1:
            j_sym    = st.text_input("Symbol").upper()
            j_type   = st.selectbox("Trade Type", ["Buy","Sell"])
            j_qty    = st.number_input("Quantity", 1, 100000, 10)
        with jc2:
            j_entry  = st.number_input("Entry Price ₹", 0.1, 100000.0, 100.0, 0.5)
            j_sl     = st.number_input("Stop Loss ₹",   0.1, 100000.0,  95.0, 0.5)
            j_target = st.number_input("Target ₹",      0.1, 100000.0, 110.0, 0.5)
        with jc3:
            j_date   = st.date_input("Date", value=date.today())
            j_status = st.selectbox("Status", ["Open","Target Hit","SL Hit","Manual Exit"])
            j_exit   = st.number_input("Exit Price ₹ (if closed)", 0.0, 100000.0, 0.0, 0.5)
            j_notes  = st.text_area("Notes / Reason", placeholder="Why did you buy? What signal?", height=80)

        if st.button("💾 Save Trade", type="primary"):
            if j_sym:
                invested = j_entry * j_qty
                pnl      = (j_exit - j_entry) * j_qty if j_exit > 0 else 0
                pnl_pct  = ((j_exit - j_entry) / j_entry * 100) if j_exit > 0 else 0
                st.session_state.journal.append({
                    "date": str(j_date), "symbol": j_sym, "type": j_type,
                    "qty": j_qty, "entry": j_entry, "sl": j_sl,
                    "target": j_target, "exit": j_exit if j_exit > 0 else None,
                    "status": j_status, "notes": j_notes,
                    "invested": round(invested,0), "pnl": round(pnl,0),
                    "pnl_pct": round(pnl_pct,1),
                })
                st.success(f"Trade logged: {j_qty} × {j_sym} @ ₹{j_entry}")
            else:
                st.warning("Please enter a symbol.")

    if not st.session_state.journal:
        st.markdown("""<div class="card" style="text-align:center;padding:40px">
          <div style="font-size:36px">📔</div>
          <div style="font-size:14px;color:#8a9ab5;margin-top:10px">
            No trades logged yet. Add your first trade above.
          </div>
        </div>""", unsafe_allow_html=True)
    else:
        df_j = pd.DataFrame(st.session_state.journal)
        closed = df_j[df_j["status"] != "Open"]

        # Stats
        total_trades  = len(df_j)
        open_trades   = len(df_j[df_j["status"]=="Open"])
        winners       = len(closed[closed["pnl"] > 0])
        losers        = len(closed[closed["pnl"] < 0])
        win_rate      = (winners/len(closed)*100) if len(closed) > 0 else 0
        total_pnl     = closed["pnl"].sum() if len(closed) > 0 else 0
        avg_win       = closed[closed["pnl"]>0]["pnl"].mean() if winners > 0 else 0
        avg_loss      = closed[closed["pnl"]<0]["pnl"].mean() if losers  > 0 else 0

        jk1,jk2,jk3,jk4,jk5,jk6 = st.columns(6)
        jkd = [
            (total_trades,           "Total Trades", "#6b7a99"),
            (open_trades,            "Open",         "#3b9eff"),
            (f"{win_rate:.0f}%",     "Win Rate",     "#00d26a"),
            (f"₹{total_pnl:+,.0f}", "Closed P&L",
             "#00d26a" if total_pnl>=0 else "#f87171"),
            (f"₹{avg_win:,.0f}",    "Avg Win",      "#00d26a"),
            (f"₹{avg_loss:,.0f}",   "Avg Loss",     "#f87171"),
        ]
        for col,(v,l,c) in zip([jk1,jk2,jk3,jk4,jk5,jk6],jkd):
            col.markdown(f"""<div class="card-sm">
              <div class="kpi-val" style="color:{c};font-size:18px">{v}</div>
              <div class="kpi-label">{l}</div>
            </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # P&L over time
        if len(closed) > 1:
            closed_sorted = closed.sort_values("date")
            closed_sorted["cumulative_pnl"] = closed_sorted["pnl"].cumsum()
            fig_j = px.area(closed_sorted, x="date", y="cumulative_pnl",
                            title="Cumulative P&L",
                            color_discrete_sequence=["#00d26a"])
            fig_j.add_bar(x=closed_sorted["date"], y=closed_sorted["pnl"],
                          name="Trade P&L",
                          marker_color=["#00d26a" if v>=0 else "#f87171"
                                        for v in closed_sorted["pnl"]])
            fig_j.update_layout(height=280, paper_bgcolor="rgba(0,0,0,0)",
                                plot_bgcolor="#0a0e1a", font=dict(color="#e8eaf0"),
                                margin=dict(t=35,b=10))
            st.plotly_chart(fig_j, use_container_width=True)

        st.divider()
        st.markdown("### 📋 All Trades")
        st.dataframe(df_j[[
            "date","symbol","type","qty","entry","sl","target",
            "exit","status","pnl","pnl_pct","notes"
        ]], use_container_width=True, height=350)

        col_del, col_exp = st.columns(2)
        with col_del:
            if st.button("🗑️ Clear All Trades", type="secondary"):
                st.session_state.journal = []
                st.rerun()
        with col_exp:
            st.download_button("⬇️ Export as CSV",
                df_j.to_csv(index=False),
                f"journal_{datetime.now().strftime('%d%b%Y')}.csv",
                "text/csv", use_container_width=True)


# ══════════════════════════════════════════════════════════════
#  SIDEBAR FOOTER (rendered last, pinned below per-page settings)
# ══════════════════════════════════════════════════════════════
with sidebar_footer:
    st.divider()
    st.caption(f"🕐 {datetime.now().strftime('%d %b %Y  %H:%M')}")
    st.caption("⚠️ Not SEBI-registered. Educational use only.")
    st.caption(f"⚠️ {MODEL_DISCLAIMER}")
