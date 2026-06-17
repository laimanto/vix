"""
run_daily.py  —  Master daily runner.  Called by GitHub Actions at 20:35 UTC (4:35pm EDT).

Sequence:
  1. fetch_data.py   — get VIX + option bid/ask from yfinance (always, in or out of position)
  2. run_agent.py    — load v1243 model, compute today's signal
  3. auto_manage_trade — open trade on BUY signal / close trade on SELL signal automatically
  4. gen_dashboard   — append daily_log (if in position), then regenerate index.html
  5. (optional) copy index.html to Google Drive folder
"""

import argparse, csv, json, math, shutil, sys
from datetime import date, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / 'data'
DASH_DIR = BASE_DIR / 'dashboard'
TENOR    = 180
MAX_HOLD = 91


def run_step(name, func):
    print(f'\n{"="*60}')
    print(f'  {name}')
    print('='*60)
    try:
        result = func()
        print(f'  OK {name} complete')
        return result
    except Exception as e:
        print(f'  FAIL {name} FAILED: {e}')
        import traceback; traceback.print_exc()
        return None


def auto_manage_trade(fetched, signal_info, position):
    """
    Open a trade on BUY signal (when out of position) or close on SELL (when in position).
    Updates trades.csv and position.json in place.
    Returns the updated position dict.
    """
    if not fetched:
        print('  [auto-trade] No fetched data — skipping')
        return position

    signal   = signal_info.get('signal', 'HOLD')
    in_pos   = position.get('in_position', False)
    today    = fetched.get('fetch_date', str(date.today()))
    vix      = float(fetched.get('vix', 0))
    ask      = float(fetched.get('option_ask', 0))
    bid_px   = float(fetched.get('option_bid', 0))
    sigma    = float(fetched.get('sigma', 1.2964))
    trades_path = DATA_DIR / 'trades.csv'

    if not in_pos and signal == 'BUY':
        # ── Open new trade ───────────────────────────────────────────────────────
        strike = math.ceil(vix * 1.2)

        rows = list(csv.DictReader(open(trades_path, encoding='utf-8')))
        next_id = max(int(r['trade_id']) for r in rows) + 1 if rows else 1

        with open(trades_path, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([next_id, today, round(vix, 2), round(ask, 2),
                             strike, '', '', '', 0, '', 'OPEN', 'auto'])

        new_pos = {
            'in_position':  True,
            'trade_id':     next_id,
            'entry_date':   today,
            'entry_vix':    round(vix, 2),
            'entry_ask':    round(ask, 2),
            'entry_sigma':  sigma,
            'strike':       strike,
            'tenor':        TENOR,
            'max_hold':     MAX_HOLD,
            'peak_vix':     round(vix, 2),
            'expiry':       fetched.get('expiry_used', ''),  # actual option expiry from yfinance
        }
        (DATA_DIR / 'position.json').write_text(json.dumps(new_pos, indent=2))

        # Reset daily_log for new trade
        (DATA_DIR / 'daily_log.csv').write_text(
            'date,vix,sigma,option_bid,option_ask,signal,in_position,days_held,roi_bid\n'
        )

        print(f'  AUTO-BUY: trade #{next_id} opened  VIX={vix:.2f}  ask=${ask:.2f}  strike={strike}')
        return new_pos

    elif in_pos and signal == 'SELL':
        # ── Close current trade ──────────────────────────────────────────────────
        entry_ask  = float(position.get('entry_ask', 1))
        entry_date = position.get('entry_date', today)
        trade_id   = position.get('trade_id')
        days_held  = (datetime.strptime(today, '%Y-%m-%d') -
                      datetime.strptime(entry_date, '%Y-%m-%d')).days

        # Grace period: each trading day runs 4:35pm → next 4:35pm.
        # Don't exit until at least one full trading day has elapsed since entry close.
        entry_4pm = datetime.strptime(entry_date, '%Y-%m-%d').replace(hour=16, minute=35)
        trading_days_held = max(0, int((datetime.now() - entry_4pm).total_seconds() / 86400))
        if trading_days_held < 1:
            print(f'  [auto-trade] Grace period — SELL ignored (Day 1 still in progress, {trading_days_held:.2f} trading days elapsed)')
            return position
        roi_bid    = round((bid_px - entry_ask) / entry_ask * 100, 1) if entry_ask > 0 else 0.0

        rows = list(csv.DictReader(open(trades_path, encoding='utf-8')))
        fieldnames = list(rows[0].keys()) if rows else []
        with open(trades_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows:
                if int(row['trade_id']) == trade_id:
                    row['exit_date']   = today
                    row['exit_vix']    = round(vix, 2)
                    row['exit_bid']    = round(bid_px, 2)
                    row['days_held']   = days_held
                    row['roi_bid']     = roi_bid
                    row['exit_reason'] = 'AE'
                writer.writerow(row)

        new_pos = {**position, 'in_position': False}
        (DATA_DIR / 'position.json').write_text(json.dumps(new_pos, indent=2))

        print(f'  AUTO-SELL: trade #{trade_id} closed  bid=${bid_px:.2f}  ROI={roi_bid:+.1f}%  days={days_held}')
        return new_pos

    elif in_pos and (datetime.strptime(today, '%Y-%m-%d') -
                     datetime.strptime(position.get('entry_date', today), '%Y-%m-%d')).days >= MAX_HOLD \
                 and (datetime.strptime(today, '%Y-%m-%d') -
                      datetime.strptime(position.get('entry_date', today), '%Y-%m-%d')).days > 0:
        # ── Hard deadline hit ────────────────────────────────────────────────────
        entry_ask  = float(position.get('entry_ask', 1))
        entry_date = position.get('entry_date', today)
        trade_id   = position.get('trade_id')
        days_held  = (datetime.strptime(today, '%Y-%m-%d') -
                      datetime.strptime(entry_date, '%Y-%m-%d')).days
        roi_bid    = round((bid_px - entry_ask) / entry_ask * 100, 1) if entry_ask > 0 else 0.0

        rows = list(csv.DictReader(open(trades_path, encoding='utf-8')))
        fieldnames = list(rows[0].keys()) if rows else []
        with open(trades_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows:
                if int(row['trade_id']) == trade_id:
                    row['exit_date']   = today
                    row['exit_vix']    = round(vix, 2)
                    row['exit_bid']    = round(bid_px, 2)
                    row['days_held']   = days_held
                    row['roi_bid']     = roi_bid
                    row['exit_reason'] = 'HD'
                writer.writerow(row)

        new_pos = {**position, 'in_position': False}
        (DATA_DIR / 'position.json').write_text(json.dumps(new_pos, indent=2))

        print(f'  HARD-DEADLINE: trade #{trade_id} force-closed  bid=${bid_px:.2f}  ROI={roi_bid:+.1f}%  days={days_held}')
        return new_pos

    else:
        print(f'  [auto-trade] signal={signal}  in_pos={in_pos}  no action')
        return position


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--skip-fetch',  action='store_true')
    parser.add_argument('--skip-agent',  action='store_true')
    parser.add_argument('--gdrive-path', default='')
    parser.add_argument('--mock',        action='store_true')
    args = parser.parse_args()

    print(f'\nVIX Call Strategy — Daily Run  [{date.today()}]')
    print(f'BASE_DIR: {BASE_DIR}')

    # Detect Day 1 lock: if an active trade's entry close (4:35pm) hasn't rolled to Day 2 yet,
    # freeze fetch + agent so the dashboard always shows the entry-day VIX and BUY signal.
    position_early = json.loads((DATA_DIR / 'position.json').read_text()) if (DATA_DIR / 'position.json').exists() else {}
    in_day1 = False
    if position_early.get('in_position'):
        entry_date_early = position_early.get('entry_date', '')
        if entry_date_early:
            entry_4pm_early = datetime.strptime(entry_date_early, '%Y-%m-%d').replace(hour=16, minute=35)
            secs_since_close = (datetime.now() - entry_4pm_early).total_seconds()
            if 0 <= secs_since_close < 86400:  # less than 24h since entry close
                in_day1 = True
                print(f'\n[Day 1 lock] {secs_since_close/3600:.1f}h since entry close — skipping fetch/agent, using entry-day data')
                # Still refresh spot VIX so dashboard shows live price and ROI
                try:
                    import yfinance as _yf
                    _hist = _yf.Ticker('^VIX').history(period='2d')
                    if not _hist.empty:
                        _spot = round(float(_hist['Close'].iloc[-1]), 2)
                        _fp = DATA_DIR / 'fetched.json'
                        if _fp.exists():
                            _fd = json.loads(_fp.read_text())
                            _fd['spot_vix'] = _spot
                            _fp.write_text(json.dumps(_fd, indent=2))
                            print(f'  [Day 1 spot] spot_vix updated to {_spot}')
                except Exception as _e:
                    print(f'  [Day 1 spot] update skipped: {_e}')

    # Step 1: Fetch VIX + option data
    if not args.skip_fetch and not in_day1:
        import fetch_data
        run_step('fetch_data', fetch_data.main)
    else:
        print('\n[skip] fetch_data' + (' (Day 1 lock)' if in_day1 else ''))

    # Step 2: Run agent
    if not args.skip_agent and not in_day1:
        try:
            import run_agent
            run_step('run_agent', run_agent.main)
        except ImportError as e:
            print(f'\n[warn] run_agent import failed ({e}) — using existing signal.json or HOLD default')
            sig_path = DATA_DIR / 'signal.json'
            if not sig_path.exists():
                sig_path.write_text(json.dumps({'signal': 'HOLD', 'exit_prob': None, 'note': 'model unavailable'}))
    else:
        print('\n[skip] run_agent' + (' (Day 1 lock)' if in_day1 else ''))

    # Step 3: Auto-manage trade (open on BUY, close on SELL / hard deadline)
    fetched     = json.loads((DATA_DIR / 'fetched.json').read_text()) if (DATA_DIR / 'fetched.json').exists() else {}
    signal_info = json.loads((DATA_DIR / 'signal.json').read_text())  if (DATA_DIR / 'signal.json').exists()  else {'signal': 'HOLD'}
    position    = json.loads((DATA_DIR / 'position.json').read_text())

    position = run_step('auto_manage_trade', lambda: auto_manage_trade(fetched, signal_info, position))
    if position is None:
        position = json.loads((DATA_DIR / 'position.json').read_text())

    # Step 4: Reload position (may have been updated by auto_manage_trade) then generate dashboard
    position = json.loads((DATA_DIR / 'position.json').read_text())
    import gen_dashboard as gd
    run_step('append_daily_log',      lambda: gd.append_daily_log(fetched, signal_info, position))
    run_step('append_option_price_log', lambda: gd.append_option_price_log(fetched, position))
    run_step('gen_dashboard',         lambda: gd.main(is_mock=args.mock))

    # Step 5: Copy to Google Drive (optional)
    if args.gdrive_path:
        dest = Path(args.gdrive_path) / 'index.html'
        run_step('copy_to_gdrive', lambda: shutil.copy(DASH_DIR / 'index.html', dest))
        print(f'  Copied to {dest}')

    print('\nDaily run complete')


if __name__ == '__main__':
    main()
