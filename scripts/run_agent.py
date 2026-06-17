"""
run_agent.py  —  Load v1210 (entry) + v1243 (exit) models and produce today's signal.
Signal written to data/signal.json: {signal: 'BUY'|'HOLD'|'SELL', action: int, exit_prob: float}

Requires: data/fetched.json (from fetch_data.py) and data/position.json.
"""

import json, math, sys, warnings
from pathlib import Path
from datetime import datetime
from collections import deque

import numpy as np
import pandas as pd
import scipy.stats as si
import yfinance as yf

warnings.filterwarnings('ignore')

BASE_DIR  = Path(__file__).parent.parent
DATA_DIR  = BASE_DIR / 'data'
MODEL_DIR = BASE_DIR / 'models'

# ── paths to the production models (copies of v1210 entry + v1243 exit) ────────
ENTRY_MODEL_PATH = MODEL_DIR / 'v1210_entry.zip'
EXIT_MODEL_PATH  = MODEL_DIR / 'v1243_exit.zip'

# ── constants (must match training) ────────────────────────────────────────────
R_OPT           = 0.045
SIGMA_OPT       = 1.2964
TENOR           = 180
SPIKE_THRESHOLD = 20.0
MAX_HOLD        = 91
MAX_ENTRY_VIX   = 30.0
COOLDOWN        = 5


def bs_call(S, K, T, sigma=SIGMA_OPT, r=R_OPT):
    if T <= 0:
        return max(0.0, S - K)
    d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    return max(0.01, S * si.norm.cdf(d1) - K * math.exp(-r * T) * si.norm.cdf(d2))


def load_vix_history():
    """Download full VIX history for feature computation."""
    raw = yf.download('^VIX', period='max', progress=False)
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)
    raw = raw.reset_index()
    hist = pd.DataFrame({'Date': pd.to_datetime(raw['Date']), 'VIX': raw['Close'].astype(float)})
    hist = hist.dropna().reset_index(drop=True)

    v   = hist['VIX']
    rm  = v.rolling(252).mean()
    rs  = v.rolling(252).std().clip(lower=1.)
    a14 = v.diff(1).abs().rolling(14).mean().clip(lower=0.1)

    hist['vix_from_normal']    = (v - rm) / rs
    hist['vix_percentile_1y']  = [float((v.iloc[max(0,i-252):i] < v.iloc[i]).mean()) if i > 0 else 0.5 for i in range(len(hist))]
    hist['vix_percentile_3m']  = [float((v.iloc[max(0,i-63):i]  < v.iloc[i]).mean()) if i > 0 else 0.5 for i in range(len(hist))]
    hist['vix_change_1d_atr']  = v.diff(1) / a14
    hist['vix_change_3d_atr']  = v.diff(3) / a14
    hist['vix_change_30d_atr'] = v.diff(30) / a14
    hist['vix_accel']          = hist['vix_change_1d_atr'].diff(1)

    # Spike rolling features
    def rolling_spike_avg(series, window=None):
        vals = series.values; n = len(vals); result = np.full(n, SPIKE_THRESHOLD*1.5); dq = deque()
        win_sum = 0.; in_spike = False; cur_peak = 0.
        for i in range(n):
            if vals[i] >= SPIKE_THRESHOLD: in_spike = True; cur_peak = max(cur_peak, vals[i])
            else:
                if in_spike: dq.append((i, cur_peak)); win_sum += cur_peak
                in_spike = False; cur_peak = 0.
            if window is not None:
                while dq and i - dq[0][0] > window: win_sum -= dq.popleft()[1]
            result[i] = win_sum / len(dq) if dq else SPIKE_THRESHOLD * 1.5
        return pd.Series(result, index=series.index)

    def rolling_spike_pct(series, window=252):
        vals = series.values; n = len(vals); result = np.full(n, 0.5); dq = deque()
        in_spike = False; cur_peak = 0.
        for i in range(n):
            if vals[i] >= SPIKE_THRESHOLD: in_spike = True; cur_peak = max(cur_peak, vals[i])
            else:
                if in_spike: dq.append((i, cur_peak))
                in_spike = False; cur_peak = 0.
            while dq and i - dq[0][0] > window: dq.popleft()
            result[i] = float(np.mean([vals[i] > p[1] for p in dq])) if len(dq) >= 2 else 0.5
        return pd.Series(result, index=series.index)

    hist['spike_avg_1y']        = rolling_spike_avg(hist['VIX'], 252)
    hist['vix_vs_spike_avg_1y'] = (v - hist['spike_avg_1y']) / hist['spike_avg_1y']
    hist['spike_pct_1y']        = rolling_spike_pct(hist['VIX'], 252)

    di = []; c = 0
    for vi in v:
        c = c + 1 if vi >= SPIKE_THRESHOLD else 0; di.append(c)
    hist['days_in_spike'] = pd.Series(di, index=hist.index) / 30.

    return hist.dropna().reset_index(drop=True)


ENTRY_FEATURES = [
    'vix_from_normal','vix_percentile_1y','vix_percentile_3m',
    'vix_change_1d_atr','vix_change_3d_atr','vix_accel',
    'vix_vs_spike_avg_1y','days_in_spike','spike_pct_1y',
]
EXIT_FEATURES = [
    'vix_from_entry','vix_change_30d_atr','loss_slow_drift',
    'days_held_norm','peak_spike','peak_vs_norm','spike_reversal',
    'fib382','loss_urgency','fib618','fib_retrace',
    'days_in_spike','spike_pct_1y',
    'vix_change_3d_atr','vix_accel',
    'current_roi_pct','spike_progress',
]


def build_exit_obs(row, position, peak_vix, swing_low, spike_avg_at_entry, peak_roi):
    vix         = float(row['VIX'])
    entry_vix   = float(position['entry_vix'])
    entry_ask   = float(position['entry_ask'])
    strike      = int(position['strike'])
    entry_date  = datetime.strptime(position['entry_date'], '%Y-%m-%d')
    cal_days    = (datetime.strptime(str(row['Date'])[:10], '%Y-%m-%d') - entry_date).days
    rem_T       = max(0., (TENOR - cal_days) / 365.)
    curr_opt    = bs_call(vix, strike, rem_T)
    curr_roi    = (curr_opt - entry_ask) / entry_ask
    _peak_roi   = max(peak_roi, curr_roi)

    vix_from_entry  = (vix - entry_vix) / entry_vix
    days_held_norm  = min(cal_days / MAX_HOLD, 1.)
    peak_spike      = float(np.clip(peak_vix / entry_vix, 0.5, 3.))
    peak_vs_norm    = float(np.clip(peak_vix / max(spike_avg_at_entry, 20.), 0., 3.))
    spike_amp_rev   = max(peak_vix - entry_vix, 1.0)
    spike_reversal  = float(np.clip((peak_vix - vix) / spike_amp_rev, 0., 2.))
    _swing          = peak_vix - swing_low; scale = max(entry_vix * 0.1, 1.0)
    if _swing > 0:
        f382 = (peak_vix - 0.382 * _swing - entry_vix) / scale
        f618 = (peak_vix - 0.618 * _swing - entry_vix) / scale
        fr   = float(np.clip((peak_vix - vix) / _swing, 0., 3.))
    else:
        f382 = f618 = fr = 0.
    curr_roi_pct   = float(np.clip(curr_roi, -1.5, 8.0))
    spike_amp      = max(spike_avg_at_entry - entry_vix, 1.0)
    spike_progress = float(np.clip((vix - entry_vix) / spike_amp, -2., 4.))
    below_entry    = float(np.clip(max(0., entry_vix - vix) / entry_vix, 0., 1.))
    loss_slow_drift = float(np.clip(below_entry * (1. + days_held_norm), 0., 1.))
    adverse_30d    = max(0., -float(row['vix_change_30d_atr']))
    adverse_3d     = max(0., -float(row['vix_change_3d_atr']))
    loss_urgency   = float(np.clip(below_entry * (3. * adverse_30d + adverse_3d), 0., 5.))

    obs = np.array([
        vix_from_entry, float(row['vix_change_30d_atr']), loss_slow_drift,
        days_held_norm, peak_spike, peak_vs_norm, spike_reversal,
        float(np.clip(f382, -10., 10.)), loss_urgency,
        float(np.clip(f618, -10., 10.)), fr,
        float(row['days_in_spike']), float(row['spike_pct_1y']),
        float(row['vix_change_3d_atr']), float(row['vix_accel']),
        curr_roi_pct, spike_progress,
    ], dtype=np.float32)
    return np.clip(obs, -10., 10.), _peak_roi


def main():
    from stable_baselines3 import PPO

    fetched  = json.loads((DATA_DIR / 'fetched.json').read_text())
    position = json.loads((DATA_DIR / 'position.json').read_text())

    today_str = fetched['fetch_date']
    vix       = float(fetched['vix'])
    signal_out = {'fetch_date': today_str, 'vix': vix}

    print('Loading VIX history for feature computation...')
    hist = load_vix_history()

    # Get today's row (or last available row if today is not a trading day)
    today_rows = hist[hist['Date'].dt.strftime('%Y-%m-%d') == today_str]
    if today_rows.empty:
        row = hist.iloc[-1]
        print(f'  Note: {today_str} not in history — using last row {row["Date"].strftime("%Y-%m-%d")}')
    else:
        row = today_rows.iloc[0]

    if not position.get('in_position'):
        # ── Check entry signal ───────────────────────────────────────────────
        if not ENTRY_MODEL_PATH.exists():
            print(f'Entry model not found at {ENTRY_MODEL_PATH} — defaulting HOLD')
            signal_out.update({'signal': 'HOLD', 'action': 0, 'note': 'no entry model'})
        else:
            entry_model = PPO.load(str(ENTRY_MODEL_PATH), device='cpu')
            obs = np.clip(np.array([float(row[f]) for f in ENTRY_FEATURES], dtype=np.float32), -10., 10.)
            action, _ = entry_model.predict(obs, deterministic=True)
            action = int(action)
            if action == 1 and vix <= MAX_ENTRY_VIX:
                signal_out.update({'signal': 'BUY', 'action': action})
            else:
                signal_out.update({'signal': 'HOLD', 'action': action})
        print(f'Entry signal: {signal_out["signal"]}')
    else:
        # ── Check exit signal ────────────────────────────────────────────────
        peak_vix          = float(position.get('peak_vix', vix))
        entry_vix         = float(position['entry_vix'])
        entry_date        = datetime.strptime(position['entry_date'], '%Y-%m-%d')
        # swing_low: minimum VIX since entry
        entry_idx = hist[hist['Date'].dt.strftime('%Y-%m-%d') >= position['entry_date']].index
        if len(entry_idx):
            slice_ = hist.loc[entry_idx[0]:]
            swing_low = float(slice_['VIX'].min())
        else:
            swing_low = entry_vix * 0.85
        # spike_avg_at_entry
        entry_row_cands = hist[hist['Date'].dt.strftime('%Y-%m-%d') == position['entry_date']]
        spike_avg_at_entry = float(entry_row_cands['spike_avg_1y'].iloc[0]) if not entry_row_cands.empty else SPIKE_THRESHOLD * 1.5

        if not EXIT_MODEL_PATH.exists():
            print(f'Exit model not found at {EXIT_MODEL_PATH} — defaulting HOLD')
            signal_out.update({'signal': 'HOLD', 'action': 0, 'note': 'no exit model'})
        else:
            exit_model = PPO.load(str(EXIT_MODEL_PATH), device='cpu')
            # peak_roi: approximated from daily_log
            import csv as csv_mod
            daily_log = list(csv_mod.DictReader(open(DATA_DIR / 'daily_log.csv')))
            peak_roi = max((float(r['roi_bid']) / 100 for r in daily_log), default=0.0)

            obs, _ = build_exit_obs(row, position, peak_vix, swing_low, spike_avg_at_entry, peak_roi)
            action, _ = exit_model.predict(obs, deterministic=True)
            action = int(action)
            # Hard deadline check
            cal_days = (datetime.strptime(today_str, '%Y-%m-%d') - entry_date).days
            if cal_days >= MAX_HOLD:
                action = 1  # force exit

            # Exit prob: run 20 stochastic samples
            exit_probs = []
            for _ in range(20):
                a, _ = exit_model.predict(obs, deterministic=False)
                exit_probs.append(1 if int(a) == 1 else 0)
            exit_prob = sum(exit_probs) / len(exit_probs)

            if action == 1:
                signal_out.update({'signal': 'SELL', 'action': action, 'exit_prob': round(exit_prob, 2)})
            else:
                signal_out.update({'signal': 'HOLD', 'action': action, 'exit_prob': round(exit_prob, 2)})
        print(f'Exit signal: {signal_out["signal"]}  exit_prob={signal_out.get("exit_prob","n/a")}')

    out_path = DATA_DIR / 'signal.json'
    out_path.write_text(json.dumps(signal_out, indent=2))
    print(f'Saved → {out_path}')
    return signal_out


if __name__ == '__main__':
    main()
