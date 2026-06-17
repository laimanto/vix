"""
gen_dashboard.py  —  Read CSVs/JSONs from data/ and regenerate dashboard/index.html.
Reads index_base.html (permanent template with %%SENTINEL%% markers) and replaces
each sentinel region with freshly computed content.  Writes dashboard/index.html.

Can be run standalone for testing:  python gen_dashboard.py [--mock]
"""

import csv, json, math, re, sys, argparse
from pathlib import Path
from datetime import datetime, timedelta, date

BASE_DIR  = Path(__file__).parent.parent
DATA_DIR  = BASE_DIR / 'data'
DASH_DIR  = BASE_DIR / 'dashboard'
TMPL_PATH = DASH_DIR / 'index_base.html'
OUT_PATH  = DASH_DIR / 'index.html'

R      = 0.045
SIGMA0 = 1.2964   # training-time IV used in BS model
TENOR  = 180
MAX_HOLD = 91


# ── Black-Scholes helpers ──────────────────────────────────────────────────────

def _ncdf(x):
    t = 1 / (1 + 0.2316419 * abs(x))
    d = 0.3989423 * math.exp(-x * x / 2)
    p = d * t * (0.3193815 + t * (-0.3565638 + t * (1.7814779 + t * (-1.8212560 + t * 1.3302744))))
    return 1 - p if x > 0 else p


def bs_call(S, K, T, sigma=SIGMA0, r=R):
    if T <= 0:
        return max(0.0, S - K)
    d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    return max(0.01, S * _ncdf(d1) - K * math.exp(-r * T) * _ncdf(d2))


def option_bid(mid):
    sp = max(0.10, mid * 0.06)
    return round(mid - sp / 2, 2)


def option_ask(mid):
    sp = max(0.10, mid * 0.06)
    return round(mid + sp / 2, 2)


def theta_pct(S, K, T, sigma):
    if T <= 1 / 365:
        return 0.0
    mid  = bs_call(S, K, T, sigma)
    nxt  = bs_call(S, K, T - 1 / 365, sigma)
    return (nxt - mid) / mid * 100


# ── Data loading ───────────────────────────────────────────────────────────────

def load_trades():
    rows = list(csv.DictReader(open(DATA_DIR / 'trades.csv', encoding='utf-8')))
    return rows  # chronological order


def load_daily_log():
    rows = list(csv.DictReader(open(DATA_DIR / 'daily_log.csv', encoding='utf-8')))
    return rows  # chronological order


def load_position():
    return json.loads((DATA_DIR / 'position.json').read_text())


def load_signal():
    p = DATA_DIR / 'signal.json'
    if p.exists():
        return json.loads(p.read_text())
    return {'signal': 'HOLD', 'exit_prob': None}


# ── Computed summary stats ────────────────────────────────────────────────────

def compute_perf(trades):
    closed = [t for t in trades if t['exit_reason'] not in ('OPEN', '')]
    if not closed:
        return dict(n=0, wr=0, avg=0, total=0, hd=0, avg_hold=0)
    rois = [float(t['roi_bid']) for t in closed]
    n    = len(closed)
    wins = sum(1 for r in rois if r > 0)
    hd   = sum(1 for t in closed if t['exit_reason'] == 'HD')
    avg_hold = sum(int(t['days_held']) for t in closed) / n
    return dict(
        n=n,
        wr=round(wins / n * 100, 1),
        avg=round(sum(rois) / n, 1),
        total=round(sum(rois), 0),
        hd=hd,
        avg_hold=round(avg_hold, 1),
    )


# ── Sentinel replacement ──────────────────────────────────────────────────────

def replace_sentinel(html, tag, new_content):
    """Replace content between <!-- %%TAG_START%% --> and <!-- %%TAG_END%% --> (or JS-style)."""
    # Try HTML comment style first, then JS style
    for start_marker, end_marker in [
        (f'<!-- %%{tag}_START%% -->', f'<!-- %%{tag}_END%% -->'),
        (f'// %%{tag}_START%%',       f'// %%{tag}_END%%'),
    ]:
        if start_marker in html and end_marker in html:
            before = html[:html.index(start_marker) + len(start_marker)]
            after  = html[html.index(end_marker):]
            html   = before + '\n' + new_content + '\n' + after
            return html
    raise ValueError(f'Sentinel {tag} not found in template')


# ── Content generators ────────────────────────────────────────────────────────

def gen_banner(is_mock=False):
    if is_mock:
        return '<div class="mock-banner">⚠ MOCK DATA — Design Reference Only — Not Live Production Data</div>'
    return '<div style="background:#0c2d12;color:#3fb950;text-align:center;padding:7px;font-weight:700;font-size:12px;letter-spacing:1px;">● LIVE — Data sourced from yfinance EOD</div>'


def gen_header_meta(today_str):
    return f'''  <div class="header-right">
    <div>Last updated: <strong>{today_str} EOD</strong></div>
    <div>OOS period: <strong>2019-01-01 → present</strong></div>
    <div>Training cutoff: <strong>2018-12-31</strong></div>
  </div>'''


def gen_status(position, daily_log, signal_info, today_str):
    """Generate the full status section: cards + alert + position detail groups."""
    if not position.get('in_position'):
        return _gen_status_out(signal_info, today_str)

    entry_vix   = float(position['entry_vix'])
    entry_ask   = float(position['entry_ask'])
    entry_sigma = float(position.get('entry_sigma', SIGMA0))
    strike      = int(position['strike'])
    entry_date  = position['entry_date']
    peak_vix    = float(position.get('peak_vix', entry_vix))

    last = daily_log[-1]
    curr_vix   = float(last['vix'])
    curr_sigma = float(last['sigma'])
    curr_bid   = float(last['option_bid'])
    curr_ask_p = float(last['option_ask'])
    days_held  = int(last['days_held'])
    roi_bid    = float(last['roi_bid'])

    # Derived
    spread_dollar = round(curr_ask_p - curr_bid, 2)
    mid_price     = (curr_bid + curr_ask_p) / 2
    spread_pct    = round(spread_dollar / mid_price * 100, 1) if mid_price > 0 else 0
    days_remaining = MAX_HOLD - days_held
    entry_dt      = datetime.strptime(entry_date, '%Y-%m-%d')
    hard_deadline = (entry_dt + timedelta(days=MAX_HOLD)).strftime('%Y-%m-%d')
    rem_T         = max(0, (TENOR - days_held) / 365)
    theta         = theta_pct(curr_vix, strike, rem_T, curr_sigma)
    vix_change    = round(curr_vix - entry_vix, 2)
    vix_change_pct = round(vix_change / entry_vix * 100, 1)
    sigma_change_pp = round((curr_sigma - entry_sigma) * 100, 1)

    signal   = signal_info.get('signal', 'HOLD')
    exit_prob = signal_info.get('exit_prob', None)
    ep_str   = f'{int(exit_prob*100)}%' if exit_prob is not None else '—'

    # Card styles
    signal_class = {'HOLD': 'c-hold', 'SELL': 'c-sell', 'BUY': 'c-buy'}.get(signal, 'c-hold')
    signal_color = {'HOLD': 'orange', 'SELL': 'red', 'BUY': 'green'}.get(signal, 'orange')

    roi_color = 'green' if roi_bid >= 0 else 'red'
    roi_str   = f'+{roi_bid:.1f}%' if roi_bid >= 0 else f'{roi_bid:.1f}%'
    roi_hint  = f'Ask ${entry_ask:.2f} paid → Bid ${curr_bid:.2f} now'

    if roi_bid >= 0:
        risk_level, risk_color = 'LOW', 'green'
        risk_hint = f'ROI positive, day {days_held}'
    elif roi_bid >= -20 and days_held < 60:
        risk_level, risk_color = 'MEDIUM', 'orange'
        risk_hint = f'Negative ROI, {days_remaining}d remaining'
    else:
        risk_level, risk_color = 'HIGH', 'red'
        risk_hint = f'Deep loss or late hold, day {days_held}'

    # Alert
    if roi_bid >= 5 and signal == 'HOLD':
        alert_class = 'alert-ok'
        alert_txt   = f'✓ Position healthy — ROI {roi_str}, VIX {"up" if vix_change >= 0 else "down"} from entry, day {days_held}. No stop-loss concern.'
    elif signal == 'SELL':
        alert_class = 'alert-warn'
        alert_txt   = f'⚠ EXIT SIGNAL — Agent recommends selling today. Exit prob: {ep_str}. Bid: ${curr_bid:.2f}  ROI: {roi_str}'
    elif roi_bid < -20:
        alert_class = 'alert-warn'
        alert_txt   = f'⚠ Deep loss ({roi_str}), day {days_held}. Consider manual review.'
    elif days_remaining <= 10:
        alert_class = 'alert-warn'
        alert_txt   = f'⚠ Hard deadline in {days_remaining} days ({hard_deadline}). Prepare to exit.'
    else:
        alert_class = 'alert-ok'
        alert_txt   = f'✓ Position monitored — ROI {roi_str}, day {days_held}, {days_remaining}d remaining.'

    vix_badge_dir   = 'up' if vix_change >= 0 else 'down'
    vix_badge_color = 'green' if vix_change >= 0 else 'red'
    vix_badge_sign  = '+' if vix_change_pct >= 0 else ''
    sigma_sign      = '+' if sigma_change_pp >= 0 else ''
    sigma_badge_dir = 'up' if sigma_change_pp >= 0 else 'down'
    theta_sign      = '+' if theta >= 0 else ''

    return f'''<!-- STATUS CARDS -->
<div class="sec">Today's Status</div>
<div class="status-row">
  <div class="card c-in">
    <div class="lbl">Position</div>
    <div class="val green">IN</div>
    <div class="hint">Day {days_held} of {MAX_HOLD} max</div>
  </div>
  <div class="card {signal_class}">
    <div class="lbl">Exit Signal</div>
    <div class="val {signal_color}">{signal}</div>
    <div class="hint">Exit prob: {ep_str}</div>
  </div>
  <div class="card">
    <div class="lbl">Current ROI</div>
    <div class="val {roi_color}">{roi_str}</div>
    <div class="hint">{roi_hint}</div>
  </div>
  <div class="card">
    <div class="lbl">Stop-Loss Risk</div>
    <div class="val {risk_color}">{risk_level}</div>
    <div class="hint">{risk_hint}</div>
  </div>
</div>

<div class="alert {alert_class}">{alert_txt}</div>

<!-- POSITION DETAIL -->
<div class="sec">Current Position Detail</div>
<div class="pos-groups">

  <!-- GROUP 1: VIX & VOLATILITY -->
  <div class="pos-group">
    <h3>VIX &amp; Volatility</h3>
    <div class="pos-row"><span class="k">Entry VIX</span><span class="v">{entry_vix:.2f}</span></div>
    <div class="pos-row"><span class="k">Current VIX</span><span class="v {vix_badge_color}">{curr_vix:.2f} <span class="badge badge-{vix_badge_dir}">{vix_badge_sign}{vix_change_pct:.1f}%</span></span></div>
    <div class="pos-row"><span class="k">Entry Implied Vol</span><span class="v">{entry_sigma*100:.1f}%</span></div>
    <div class="pos-row"><span class="k">Current Implied Vol</span><span class="v orange">{curr_sigma*100:.1f}% <span class="badge badge-{sigma_badge_dir}">{sigma_sign}{sigma_change_pp:.1f}pp</span></span></div>
    <div class="pos-row"><span class="k">VIX vs Entry</span><span class="v {'green' if vix_change >= 0 else 'red'}">{'+' if vix_change >= 0 else ''}{vix_change:.2f} pts</span></div>
    <div class="pos-row"><span class="k">Strike (&#8968;VIX&times;1.2&#8969;)</span><span class="v">{strike}</span></div>
  </div>

  <!-- GROUP 2: OPTION PRICES -->
  <div class="pos-group">
    <h3>Option Prices</h3>
    <div class="pos-row"><span class="k">Entry Ask (paid)</span><span class="v">${entry_ask:.2f}</span></div>
    <div class="pos-row"><span class="k">Current Bid (sell at)</span><span class="v {'green' if roi_bid >= 0 else 'red'}">${curr_bid:.2f}</span></div>
    <div class="pos-row"><span class="k">Current Ask (buy at)</span><span class="v">${curr_ask_p:.2f}</span></div>
    <div class="pos-row"><span class="k">Bid/Ask Spread</span><span class="v">${spread_dollar:.2f}</span></div>
    <div class="pos-row"><span class="k">Spread %</span><span class="v orange">{spread_pct:.1f}%</span></div>
    <div class="pos-row"><span class="k">Current ROI (bid exit)</span><span class="v {'green' if roi_bid >= 0 else 'red'}">{roi_str}</span></div>
    <div class="pos-row"><span class="k">Theta</span><span class="v red" id="thetaDisp">{theta_sign}{theta:.2f}%/day</span></div>
  </div>

  <!-- GROUP 3: TIME & DATES -->
  <div class="pos-group">
    <h3>Time &amp; Dates</h3>
    <div class="pos-row"><span class="k">Entry Date</span><span class="v">{entry_date}</span></div>
    <div class="pos-row"><span class="k">Today</span><span class="v">{today_str}</span></div>
    <div class="pos-row"><span class="k">Days Held</span><span class="v">{days_held}</span></div>
    <div class="pos-row"><span class="k">Days Remaining</span><span class="v orange">{days_remaining}</span></div>
    <div class="pos-row"><span class="k">Tenor</span><span class="v gray">180 calendar days</span></div>
    <div class="pos-row"><span class="k">Hard Deadline</span><span class="v red">{hard_deadline}</span></div>
  </div>

</div>'''


def _gen_status_out(signal_info, today_str):
    """Status section when not in position."""
    signal = signal_info.get('signal', 'HOLD')
    signal_class = 'c-buy' if signal == 'BUY' else 'c-hold'
    signal_color = 'green' if signal == 'BUY' else 'orange'
    alert_class  = 'alert-warn' if signal == 'BUY' else 'alert-ok'
    alert_txt    = ('⚠ BUY SIGNAL — Agent recommends entering today. Check VIX option ask price.'
                    if signal == 'BUY' else
                    '✓ Waiting for entry signal — no position open.')
    return f'''<!-- STATUS CARDS -->
<div class="sec">Today's Status ({today_str})</div>
<div class="status-row">
  <div class="card">
    <div class="lbl">Position</div>
    <div class="val gray">OUT</div>
    <div class="hint">No open trade</div>
  </div>
  <div class="card {signal_class}">
    <div class="lbl">Entry Signal</div>
    <div class="val {signal_color}">{signal}</div>
    <div class="hint">—</div>
  </div>
  <div class="card">
    <div class="lbl">Current ROI</div>
    <div class="val gray">—</div>
    <div class="hint">No open position</div>
  </div>
  <div class="card">
    <div class="lbl">Stop-Loss Risk</div>
    <div class="val gray">—</div>
    <div class="hint">—</div>
  </div>
</div>
<div class="alert {alert_class}">{alert_txt}</div>
<!-- POSITION DETAIL -->
<div class="sec">Current Position Detail</div>
<div style="color:#8b949e;padding:20px;text-align:center;">No open position.</div>'''


def gen_perf(perf):
    val_color = 'green' if perf['total'] >= 0 else 'red'
    return f'''<div class="perf-row">
  <div class="perf-cell"><div class="lbl">Closed Trades</div><div class="val white">{perf["n"]}</div></div>
  <div class="perf-cell"><div class="lbl">Win Rate</div><div class="val {val_color}">{perf["wr"]:.1f}%</div></div>
  <div class="perf-cell"><div class="lbl">Avg ROI / Trade</div><div class="val {val_color}">{"+"+str(perf["avg"]) if perf["avg"]>=0 else str(perf["avg"])}%</div></div>
  <div class="perf-cell"><div class="lbl">Total ROI</div><div class="val {val_color}">{"+"+str(int(perf["total"])) if perf["total"]>=0 else str(int(perf["total"]))}%</div></div>
  <div class="perf-cell"><div class="lbl">Hard Deadlines</div><div class="val {'red' if perf["hd"]>0 else 'green'}">{perf["hd"]}</div></div>
  <div class="perf-cell"><div class="lbl">Avg Hold</div><div class="val white">{perf["avg_hold"]:.1f}d</div></div>
</div>'''


def gen_jsdata(position, daily_log):
    """Generate the JS constants + histVIX data block."""
    if not position.get('in_position') or not daily_log:
        return f'const SIGMA0={SIGMA0}, R={R}, TENOR={TENOR}, STRIKE=0;\nconst ENTRY_ASK=0;\nconst ENTRY_DATE=new Date();\nconst DAYS_HELD=0;\nconst CURR_VIX=18, CURR_SIGMA={SIGMA0};\nconst histVIX=[];'

    entry_ask   = float(position['entry_ask'])
    entry_sigma = float(position.get('entry_sigma', SIGMA0))
    strike      = int(position['strike'])
    entry_date  = position['entry_date']
    curr_vix    = float(daily_log[-1]['vix'])
    curr_sigma  = float(daily_log[-1]['sigma'])
    days_held   = int(daily_log[-1]['days_held'])
    hist_vix    = [float(r['vix']) for r in daily_log]
    vix_js      = ', '.join(str(v) for v in hist_vix)
    n_per_line  = 10
    vix_lines   = []
    for i in range(0, len(hist_vix), n_per_line):
        vix_lines.append('  ' + ', '.join(str(v) for v in hist_vix[i:i+n_per_line]))
    vix_block = '\n'.join(vix_lines)

    return f'''// ── Constants ──────────────────────────────────────────────────────────────────
const SIGMA0={entry_sigma}, R={R}, TENOR={TENOR}, STRIKE={strike};
const ENTRY_ASK={entry_ask};   // ask price paid at entry
const ENTRY_DATE=new Date('{entry_date}');
const DAYS_HELD={days_held};
const CURR_VIX={curr_vix}, CURR_SIGMA={curr_sigma};

// ── Historical VIX path (days 0–{days_held-1}) ────────────────────────────────
const histVIX=[
{vix_block}
];'''


def gen_trades_js(trades):
    """Generate the JS trades array (reverse-chronological, newest first)."""
    # Sort newest first
    def sort_key(t):
        return t['entry_date']
    sorted_trades = sorted(trades, key=sort_key, reverse=True)

    items = []
    for i, t in enumerate(sorted_trades):
        n        = len(sorted_trades) - i
        entry    = t['entry_date']
        evix     = float(t['entry_vix'])
        strike   = int(t['strike'])
        is_open  = t['exit_reason'] == 'OPEN'
        exit_d   = t['exit_date'] if t['exit_date'] else '—'
        exvix    = float(t['exit_vix']) if t['exit_vix'] else 'null'
        days     = int(t['days_held']) if t['days_held'] else 0
        roi_bid  = float(t['roi_bid']) if t['roi_bid'] else 0.0
        rsn      = t['exit_reason'] if t['exit_reason'] else 'AE'
        note     = t.get('note', '').replace("'", "\\'")
        items.append(
            f"  {{n:{n},entry:'{entry}',evix:{evix},strike:{strike},"
            f"exit:'{exit_d}',exvix:{exvix},days:{days},"
            f"roi:{roi_bid},roiBid:{roi_bid},rsn:'{rsn}',note:'{note}'}}"
        )
    return 'const trades=[\n' + ',\n'.join(items) + '\n];'


# ── Update daily_log and position after a trade ───────────────────────────────

def append_daily_log(fetched, signal_info, position):
    """Append today's row to daily_log.csv and update position.json."""
    if not fetched or not position.get('in_position'):
        return

    today_str  = fetched['fetch_date']
    vix        = float(fetched['vix'])
    sigma      = float(fetched.get('sigma', SIGMA0))
    curr_bid   = float(fetched.get('option_bid', 0))
    curr_ask_p = float(fetched.get('option_ask', 0))
    signal     = signal_info.get('signal', 'HOLD')
    entry_ask  = float(position['entry_ask'])
    entry_date = position['entry_date']
    entry_dt   = datetime.strptime(entry_date, '%Y-%m-%d')
    today_dt   = datetime.strptime(today_str, '%Y-%m-%d')
    days_held  = (today_dt - entry_dt).days

    roi_bid    = round((curr_bid - entry_ask) / entry_ask * 100, 1) if entry_ask > 0 else 0.0
    in_pos     = position['in_position']

    # Check if today already logged
    log = list(csv.DictReader(open(DATA_DIR / 'daily_log.csv', encoding='utf-8')))
    if any(r['date'] == today_str for r in log):
        print(f'  {today_str} already in daily_log — skipping append')
        return

    with open(DATA_DIR / 'daily_log.csv', 'a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow([today_str, vix, sigma, curr_bid, curr_ask_p,
                         signal, in_pos, days_held, roi_bid])

    # Update peak_vix in position.json
    peak_vix = max(float(position.get('peak_vix', vix)), vix)
    position['peak_vix'] = peak_vix
    position['days_held'] = days_held
    (DATA_DIR / 'position.json').write_text(json.dumps(position, indent=2))
    print(f'  daily_log updated: {today_str}  VIX={vix}  ROI={roi_bid}%  Signal={signal}')


# ── Main ──────────────────────────────────────────────────────────────────────

def main(is_mock=False):
    today_str = date.today().strftime('%Y-%m-%d')
    print(f'gen_dashboard.py  today={today_str}  mock={is_mock}')

    trades    = load_trades()
    daily_log = load_daily_log()
    position  = load_position()
    signal_info = load_signal()

    perf = compute_perf(trades)

    # Read template
    if not TMPL_PATH.exists():
        raise FileNotFoundError(f'Template not found: {TMPL_PATH}')
    html = TMPL_PATH.read_text(encoding='utf-8')

    # Replace sentinels
    html = replace_sentinel(html, 'BANNER',      gen_banner(is_mock))
    html = replace_sentinel(html, 'HEADER_META', gen_header_meta(today_str))
    html = replace_sentinel(html, 'STATUS',      gen_status(position, daily_log, signal_info, today_str))
    html = replace_sentinel(html, 'PERF',        gen_perf(perf))
    html = replace_sentinel(html, 'JSDATA',      gen_jsdata(position, daily_log))
    html = replace_sentinel(html, 'TRADES',      gen_trades_js(trades))

    OUT_PATH.write_text(html, encoding='utf-8')
    print(f'Dashboard written -> {OUT_PATH}  ({len(html):,} bytes)')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mock', action='store_true', help='Show mock-data banner')
    args = parser.parse_args()
    main(is_mock=args.mock)
