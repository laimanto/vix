"""
mock_5day.py  —  Simulate 5 trading days with flat VIX to sanity-check the model.

Entry: VIX = 16.41 (yesterday's close), ask = $4.20, strike = 20
Days 1-5: VIX stays flat at 16.41
Reports the exit model's signal each day.
"""
import math, sys, warnings
from pathlib import Path
from datetime import datetime, timedelta
from collections import deque

import numpy as np
import pandas as pd
import scipy.stats as si

warnings.filterwarnings('ignore')

BASE_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(Path(__file__).parent))

# ── Parameters (must match training) ──────────────────────────────────────────
R_OPT           = 0.045
SIGMA_OPT       = 1.2964
TENOR           = 180
SPIKE_THRESHOLD = 20.0
MAX_HOLD        = 91
ENTRY_VIX       = 16.41
ENTRY_ASK       = 4.20
STRIKE          = 20       # ceil(16.41 * 1.2)
MOCK_VIX        = 16.41   # flat VIX for all 5 days
ENTRY_DATE_STR  = '2026-06-16'

EXIT_FEATURES = [
    'vix_from_entry','vix_change_30d_atr','loss_slow_drift',
    'days_held_norm','peak_spike','peak_vs_norm','spike_reversal',
    'fib382','loss_urgency','fib618','fib_retrace',
    'days_in_spike','spike_pct_1y',
    'vix_change_3d_atr','vix_accel',
    'current_roi_pct','spike_progress',
]

def bs_call(S, K, T, sigma=SIGMA_OPT, r=R_OPT):
    if T <= 0:
        return max(0.0, S - K)
    d1 = (math.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    return max(0.01, S * si.norm.cdf(d1) - K * math.exp(-r * T) * si.norm.cdf(d2))

def option_bid(mid):
    return round(mid - max(0.10, mid * 0.06) / 2, 2)

def build_exit_obs(cal_days, vix, peak_vix, swing_low, spike_avg_at_entry, peak_roi,
                   vix_change_30d_atr=0.0, vix_change_3d_atr=0.0,
                   vix_accel=0.0, days_in_spike=0.0, spike_pct_1y=0.5):
    rem_T      = max(0., (TENOR - cal_days) / 365.)
    curr_opt   = bs_call(vix, STRIKE, rem_T)
    curr_roi   = (curr_opt - ENTRY_ASK) / ENTRY_ASK
    _peak_roi  = max(peak_roi, curr_roi)

    vix_from_entry  = (vix - ENTRY_VIX) / ENTRY_VIX
    days_held_norm  = min(cal_days / MAX_HOLD, 1.)
    peak_spike      = float(np.clip(peak_vix / ENTRY_VIX, 0.5, 3.))
    peak_vs_norm    = float(np.clip(peak_vix / max(spike_avg_at_entry, 20.), 0., 3.))
    spike_amp_rev   = max(peak_vix - ENTRY_VIX, 1.0)
    spike_reversal  = float(np.clip((peak_vix - vix) / spike_amp_rev, 0., 2.))
    _swing          = peak_vix - swing_low; scale = max(ENTRY_VIX * 0.1, 1.0)
    if _swing > 0:
        f382 = (peak_vix - 0.382 * _swing - ENTRY_VIX) / scale
        f618 = (peak_vix - 0.618 * _swing - ENTRY_VIX) / scale
        fr   = float(np.clip((peak_vix - vix) / _swing, 0., 3.))
    else:
        f382 = f618 = fr = 0.
    curr_roi_pct   = float(np.clip(curr_roi, -1.5, 8.0))
    spike_amp      = max(spike_avg_at_entry - ENTRY_VIX, 1.0)
    spike_progress = float(np.clip((vix - ENTRY_VIX) / spike_amp, -2., 4.))
    below_entry    = float(np.clip(max(0., ENTRY_VIX - vix) / ENTRY_VIX, 0., 1.))
    loss_slow_drift = float(np.clip(below_entry * (1. + days_held_norm), 0., 1.))
    adverse_30d    = max(0., -vix_change_30d_atr)
    adverse_3d     = max(0., -vix_change_3d_atr)
    loss_urgency   = float(np.clip(below_entry * (3. * adverse_30d + adverse_3d), 0., 5.))

    obs = np.array([
        vix_from_entry, vix_change_30d_atr, loss_slow_drift,
        days_held_norm, peak_spike, peak_vs_norm, spike_reversal,
        float(np.clip(f382, -10., 10.)), loss_urgency,
        float(np.clip(f618, -10., 10.)), fr,
        days_in_spike, spike_pct_1y,
        vix_change_3d_atr, vix_accel,
        curr_roi_pct, spike_progress,
    ], dtype=np.float32)
    return np.clip(obs, -10., 10.), _peak_roi


def main():
    try:
        from stable_baselines3 import PPO
    except ImportError:
        print('stable_baselines3 not available — cannot run mock test')
        return

    exit_model_path = BASE_DIR / 'models' / 'v1243_exit.zip'
    if not exit_model_path.exists():
        print(f'Exit model not found: {exit_model_path}')
        return

    exit_model = PPO.load(str(exit_model_path), device='cpu')
    print(f'\nExit model loaded: {exit_model_path.name}')
    print(f'Entry: VIX={ENTRY_VIX}, ask=${ENTRY_ASK}, strike={STRIKE}')
    print(f'Mock VIX for all 5 days: {MOCK_VIX} (flat)\n')

    entry_dt        = datetime.strptime(ENTRY_DATE_STR, '%Y-%m-%d')
    spike_avg_at_entry = SPIKE_THRESHOLD * 1.5  # typical value when VIX < 20
    peak_vix        = ENTRY_VIX
    swing_low       = ENTRY_VIX
    peak_roi        = 0.0

    header = f"{'Day':>4}  {'Date':>12}  {'VIX':>6}  {'Bid':>6}  {'ROI%':>7}  {'Signal':>8}  {'ExitProb':>9}  Note"
    print(header)
    print('-' * 80)

    # Day 1 = entry day: signal is always BUY, exit model not consulted
    rem_T0    = max(0., TENOR / 365.)
    mid0      = bs_call(ENTRY_VIX, STRIKE, rem_T0)
    bid0      = option_bid(mid0)
    roi0      = (bid0 - ENTRY_ASK) / ENTRY_ASK * 100
    print(f"   1  {ENTRY_DATE_STR:>12}  {ENTRY_VIX:>6.2f}  ${bid0:>5.2f}  {roi0:>+6.1f}%  {'BUY':>8}  {'—':>9}  Entry day — exit model not run")

    # Days 2-6 = cal_days 1-5: first exit evaluation starts Day 2
    for day in range(2, 7):
        cal_days  = day - 1   # calendar days since entry
        sim_date  = (entry_dt + timedelta(days=cal_days)).strftime('%Y-%m-%d')
        vix       = MOCK_VIX
        peak_vix  = max(peak_vix, vix)
        swing_low = min(swing_low, vix)

        rem_T     = max(0., (TENOR - cal_days) / 365.)
        curr_mid  = bs_call(vix, STRIKE, rem_T)
        curr_bid  = option_bid(curr_mid)
        roi_pct   = (curr_bid - ENTRY_ASK) / ENTRY_ASK * 100

        obs, peak_roi = build_exit_obs(
            cal_days, vix, peak_vix, swing_low, spike_avg_at_entry, peak_roi,
        )

        action, _ = exit_model.predict(obs, deterministic=True)
        action = int(action)

        # Stochastic exit probability (20 samples)
        probs = [int(exit_model.predict(obs, deterministic=False)[0]) for _ in range(20)]
        exit_prob = sum(probs) / len(probs)

        signal = 'SELL' if action == 1 else 'HOLD'
        note   = f'cal_days={cal_days}  roi_obs={obs[15]:.3f}'
        print(f"  {day:>2}  {sim_date:>12}  {vix:>6.2f}  ${curr_bid:>5.2f}  {roi_pct:>+6.1f}%  {signal:>8}  {exit_prob*100:>7.0f}%   {note}")

    print()
    print('Day 1  = entry day, signal locked to BUY, exit model not consulted.')
    print('Day 2+ = exit model evaluates each 4:35pm close.')
    print('First valid exit window: Day 2 (after 24h since entry close).')


if __name__ == '__main__':
    main()
