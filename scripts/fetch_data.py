"""
fetch_data.py  —  Fetch end-of-day VIX price and option bid/ask from yfinance.
Returns a dict written to data/fetched.json for use by run_agent.py and gen_dashboard.py.
"""

import json, math, sys, csv as _csv
from pathlib import Path
from datetime import date, datetime

import yfinance as yf
import numpy as np

BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / 'data'

R      = 0.045
TENOR  = 180   # calendar days from entry to option expiry


def bs_call(S, K, T, sigma, r=R):
    """Black-Scholes call price."""
    if T <= 0:
        return max(0.0, S - K)
    from scipy import stats
    d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    return max(0.01, S * stats.norm.cdf(d1) - K * math.exp(-r * T) * stats.norm.cdf(d2))


def fetch_vix():
    """Download latest VIX close."""
    ticker = yf.Ticker('^VIX')
    hist = ticker.history(period='5d')
    if hist.empty:
        raise RuntimeError('VIX history empty from yfinance')
    vix = float(hist['Close'].iloc[-1])
    fetch_date = hist.index[-1].strftime('%Y-%m-%d')
    return vix, fetch_date


def fetch_option(strike: int, entry_date_str: str):
    """
    Fetch bid/ask for the VIX call at the given strike.
    Finds the option chain expiry closest to entry_date + TENOR days.
    Returns (bid, ask, implied_vol, expiry_used).
    """
    entry_date = datetime.strptime(entry_date_str, '%Y-%m-%d').date()
    target_expiry = entry_date + __import__('datetime').timedelta(days=TENOR)

    ticker = yf.Ticker('^VIX')
    expiries = ticker.options
    if not expiries:
        raise RuntimeError('No VIX option expiries found')

    # Pick closest expiry to target
    expiry = min(expiries, key=lambda e: abs(
        (datetime.strptime(e, '%Y-%m-%d').date() - target_expiry).days
    ))

    chain = ticker.option_chain(expiry)
    calls = chain.calls

    # Filter to our strike
    row = calls[calls['strike'] == float(strike)]
    if row.empty:
        # Fall back to nearest available strike
        row = calls.iloc[(calls['strike'] - float(strike)).abs().argsort()[:1]]

    bid  = float(row['bid'].iloc[0])
    ask  = float(row['ask'].iloc[0])
    iv   = float(row['impliedVolatility'].iloc[0])
    last = float(row['lastPrice'].iloc[0])

    # If market is closed bid/ask may be 0; fall back to last price ±3%
    if bid <= 0 and ask <= 0:
        spread_est = max(0.10, last * 0.06)
        bid = round(last - spread_est / 2, 2)
        ask = round(last + spread_est / 2, 2)

    return round(bid, 2), round(ask, 2), round(iv, 4), expiry


def estimate_sigma_from_iv(iv_raw):
    """yfinance IV is annualised log-vol; use directly as sigma for BS."""
    return round(float(iv_raw), 4)


def main():
    pos_path = DATA_DIR / 'position.json'
    if not pos_path.exists():
        print('No position.json found — skipping option fetch')
        out = {'fetch_date': str(date.today()), 'vix': None, 'error': 'no position.json'}
        (DATA_DIR / 'fetched.json').write_text(json.dumps(out, indent=2))
        return

    position = json.loads(pos_path.read_text())

    print('Fetching VIX close...')
    vix, fetch_date = fetch_vix()
    print(f'  VIX: {vix:.2f}  ({fetch_date})')

    result = {
        'fetch_date': fetch_date,
        'vix': vix,
    }

    # Always fetch option data — use actual position strike if in position,
    # otherwise compute hypothetical entry strike from today's VIX.
    if position.get('in_position'):
        strike     = int(position['strike'])
        entry_date = position['entry_date']
    else:
        strike     = math.ceil(vix * 1.2)   # hypothetical entry today
        entry_date = fetch_date

    print(f'Fetching option chain  strike={strike}  entry={entry_date}...')
    try:
        bid, ask, iv, expiry = fetch_option(strike, entry_date)
        sigma = estimate_sigma_from_iv(iv)
        print(f'  Expiry: {expiry}  Bid: {bid}  Ask: {ask}  IV: {sigma:.4f}')
    except Exception as e:
        print(f'  Option fetch failed: {e}  — using BS estimate')
        daily_log = list(__import__('csv').DictReader(open(DATA_DIR / 'daily_log.csv')))
        last_sigma = float(daily_log[-1]['sigma']) if daily_log else float(position.get('entry_sigma', 1.2964))
        sigma = last_sigma
        if position.get('in_position'):
            entry_date_dt = datetime.strptime(entry_date, '%Y-%m-%d').date()
            days_held_so_far = (datetime.strptime(fetch_date, '%Y-%m-%d').date() - entry_date_dt).days
            T = max(0, (TENOR - days_held_so_far) / 365)
        else:
            T = TENOR / 365
        mid = bs_call(vix, strike, T, sigma)
        sp  = max(0.10, mid * 0.06)
        bid = round(mid - sp / 2, 2)
        ask = round(mid + sp / 2, 2)
        expiry = 'estimated'

    result.update({
        'strike':     strike,
        'option_bid': bid,
        'option_ask': ask,
        'sigma':      sigma,
        'expiry_used': expiry,
    })

    out_path = DATA_DIR / 'fetched.json'
    out_path.write_text(json.dumps(result, indent=2))
    print(f'Saved → {out_path}')
    return result


if __name__ == '__main__':
    main()
