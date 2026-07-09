# NSE Trader Pro — Deployment Guide

## Run Locally
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy to Streamlit Cloud (FREE — 10 minutes)

### Step 1: Push to GitHub
1. Go to github.com → New Repository → Name: `nse-trader-pro` → Public → Create
2. Upload all files:
   - app.py
   - engine.py
   - requirements.txt
   - .streamlit/config.toml  ← upload inside .streamlit folder

### Step 2: Deploy
1. Go to share.streamlit.io
2. Sign in with GitHub
3. New App → select repo `nse-trader-pro` → Main file: `app.py`
4. Click Deploy → Wait 3–5 minutes

### Step 3: Install on Android
1. Open Chrome on Android
2. Go to your Streamlit URL
3. Tap 3-dot menu → Add to Home Screen → Add
4. Done! App icon on home screen.

## Important Notes
- Streamlit Cloud free tier resets when idle (30 min inactivity)
- Use Export/Import JSON buttons in Watchlist and Journal to save your data
- Download your watchlist/journal JSON periodically as backup
- For persistent storage upgrade to Streamlit Cloud paid plan (~$25/mo) or use Railway.app

## File Structure
```
nse-trader-pro/
├── app.py              ← Main app (run this)
├── engine.py           ← Screener logic
├── requirements.txt    ← Dependencies
├── DEPLOY.md           ← This file
└── .streamlit/
    └── config.toml     ← Dark theme config
```

## Demat CSV Format
Your broker CSV should have these columns (any order):
- Symbol / Scrip / Stock name
- Quantity / Qty / Shares
- Average Price / Avg Cost / Buy Price

Supported brokers: HDFC Securities, Zerodha, Upstox, Angel One, 5paisa, CDSL Myeasi
