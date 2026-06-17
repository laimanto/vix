"""
run_daily.py  —  Master daily runner.  Called by Windows Task Scheduler at ~16:30 ET.

Sequence:
  1. fetch_data.py   — get VIX + option bid/ask from yfinance
  2. run_agent.py    — load v1243 model, compute today's signal
  3. gen_dashboard   — append daily_log, then regenerate index.html
  4. (optional) copy index.html to Google Drive folder

Usage:
  python run_daily.py [--skip-fetch] [--skip-agent] [--gdrive-path "C:/..."]
"""

import argparse, json, shutil, sys
from datetime import date, datetime
from pathlib import Path

# Allow sibling scripts to be imported regardless of CWD
sys.path.insert(0, str(Path(__file__).parent))

BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / 'data'
DASH_DIR = BASE_DIR / 'dashboard'


def run_step(name, func):
    print(f'\n{"="*60}')
    print(f'  {name}')
    print('='*60)
    try:
        result = func()
        print(f'  ✓ {name} complete')
        return result
    except Exception as e:
        print(f'  ✗ {name} FAILED: {e}')
        import traceback; traceback.print_exc()
        return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--skip-fetch',  action='store_true', help='Skip yfinance fetch (use existing fetched.json)')
    parser.add_argument('--skip-agent',  action='store_true', help='Skip RL model inference (use existing signal.json)')
    parser.add_argument('--gdrive-path', default='', help='Google Drive folder to copy index.html into')
    parser.add_argument('--mock',        action='store_true', help='Show mock-data banner in dashboard')
    args = parser.parse_args()

    print(f'\nVIX Call Strategy — Daily Run  [{date.today()}]')
    print(f'BASE_DIR: {BASE_DIR}')

    # Step 1: Fetch data
    if not args.skip_fetch:
        import fetch_data
        run_step('fetch_data', fetch_data.main)
    else:
        print('\n[skip] fetch_data')

    # Step 2: Run agent
    if not args.skip_agent:
        try:
            import run_agent
            run_step('run_agent', run_agent.main)
        except ImportError as e:
            print(f'\n[warn] run_agent import failed ({e}) — using existing signal.json or HOLD default')
            sig_path = DATA_DIR / 'signal.json'
            if not sig_path.exists():
                sig_path.write_text(json.dumps({'signal': 'HOLD', 'exit_prob': None, 'note': 'model unavailable'}))
    else:
        print('\n[skip] run_agent')

    # Step 3: Load latest data, append daily_log, generate dashboard
    import gen_dashboard as gd
    fetched     = json.loads((DATA_DIR / 'fetched.json').read_text()) if (DATA_DIR / 'fetched.json').exists() else {}
    signal_info = json.loads((DATA_DIR / 'signal.json').read_text())  if (DATA_DIR / 'signal.json').exists()  else {'signal': 'HOLD'}
    position    = json.loads((DATA_DIR / 'position.json').read_text())

    run_step('append_daily_log', lambda: gd.append_daily_log(fetched, signal_info, position))
    run_step('gen_dashboard',    lambda: gd.main(is_mock=args.mock))

    # Step 4: Copy to Google Drive (optional)
    if args.gdrive_path:
        dest = Path(args.gdrive_path) / 'index.html'
        run_step('copy_to_gdrive', lambda: shutil.copy(DASH_DIR / 'index.html', dest))
        print(f'  Copied to {dest}')

    print('\n✓ Daily run complete')


if __name__ == '__main__':
    main()
