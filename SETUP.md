# VIX Call Strategy — Production Setup Guide

## Overview

The production system runs a daily pipeline on GitHub's servers (no PC required).
Each weekday at ~17:00 ET it fetches live VIX and option data, runs the RL model
to produce a BUY / HOLD / SELL signal, and publishes an updated HTML dashboard to
a GitHub Pages URL that you can share with anyone.

```
21:00 UTC (Mon–Fri)
  └─ fetch_data.py    VIX close + option bid/ask from yfinance
  └─ run_agent.py     v1210 entry + v1243 exit model inference
  └─ gen_dashboard.py append daily_log.csv  →  regenerate index.html
  └─ git commit+push  CSVs and index.html updated in repo
  └─ GitHub Pages     serves the new dashboard automatically
```

---

## Folder Structure

```
production_v1/
├── .github/
│   └── workflows/
│       └── daily_run.yml       GitHub Actions workflow
├── dashboard/
│   ├── index.html              Generated daily (do not edit manually)
│   └── index_base.html         Permanent HTML template (edit only for layout changes)
├── data/
│   ├── trades.csv              Trade history — one row per trade
│   ├── daily_log.csv           Daily price + signal log for current open trade
│   └── position.json           Current position state
├── models/
│   ├── v1210_entry.zip         Entry model (frozen v1210 Stage 2 best)
│   └── v1243_exit.zip          Exit model (v1243 Stage 2 best)
├── scripts/
│   ├── fetch_data.py           yfinance EOD fetch
│   ├── run_agent.py            RL model inference
│   ├── gen_dashboard.py        HTML dashboard generator
│   └── run_daily.py            Master orchestrator
├── index.html                  Root redirect to dashboard/index.html
├── requirements.txt            Python dependencies
└── SETUP.md                    This file
```

---

## One-Time Setup

### Step 1 — Create the GitHub Repository

1. Go to [github.com](https://github.com) and sign in.
2. Click **New repository**.
3. Name it `vix-strategy` (or any name you prefer).
4. Set visibility to **Public** (required for free GitHub Pages).
   - If you prefer Private, GitHub Pages requires a Pro account (~$4/mo).
5. Do **not** tick "Add a README" — you will push existing files.
6. Click **Create repository**.

### Step 2 — Push the Production Folder

Open a terminal in this folder and run:

```bash
cd "D:\Backup D\Weekly\USB drive\Invest\AI invest\VIX2\production_v1"

git init
git add .
git commit -m "Initial production setup"
git remote add origin https://github.com/YOUR_USERNAME/vix-strategy.git
git push -u origin main
```

Replace `YOUR_USERNAME` with your GitHub username.

> **Tip:** In Claude Code you can type `! <command>` in the prompt to run a
> shell command directly in the session.

### Step 3 — Enable GitHub Pages

1. In your repo on GitHub, go to **Settings → Pages**.
2. Under **Source**, choose **Deploy from branch**.
3. Set branch to `main`, folder to `/ (root)`.
4. Click **Save**.
5. After about 60 seconds your dashboard is live at:
   ```
   https://YOUR_USERNAME.github.io/vix-strategy/
   ```
   Visitors are redirected automatically from the root to `dashboard/index.html`.

### Step 4 — Verify the Workflow

1. Go to your repo → **Actions** tab.
2. Click **VIX Strategy Daily Run** in the left panel.
3. Click **Run workflow** → **Run workflow** (green button) to trigger manually.
4. Watch the run complete (~5–8 min, mostly PyTorch install on first run).
5. Check that a new commit appears with message `Daily update YYYY-MM-DD`.
6. Open your GitHub Pages URL and confirm the dashboard loaded.

> The pip cache kicks in from the second run onward — subsequent runs take
> ~2–3 minutes total.

---

## What Runs Automatically

The workflow file `.github/workflows/daily_run.yml` triggers at **21:00 UTC
every weekday (Mon–Fri)**, which is approximately:

| Season | Local time |
|--------|-----------|
| EDT (summer, Apr–Oct) | 17:00 ET |
| EST (winter, Nov–Mar) | 16:00 ET |

yfinance EOD data is fully settled by this time. No action is required on your
part — GitHub's servers run the job whether your PC is on or off.

To trigger a run manually at any time: Actions → VIX Strategy Daily Run →
Run workflow.

---

## Daily Data Flow

| File | Updated by | How |
|------|-----------|-----|
| `data/fetched.json` | `fetch_data.py` | Overwritten each run (not committed) |
| `data/signal.json` | `run_agent.py` | Overwritten each run (not committed) |
| `data/daily_log.csv` | `gen_dashboard.py` | One row appended per trading day |
| `data/position.json` | `gen_dashboard.py` | peak_vix and days_held updated |
| `data/trades.csv` | **Manual only** | Append a row when a trade opens or closes |
| `dashboard/index.html` | `gen_dashboard.py` | Fully regenerated from template + CSVs |

---

## Managing Trades Manually

The agent produces signals but **you execute the actual trade**.
When you act on a signal, update the CSV files:

### Opening a new trade (BUY signal)

1. Note the ask price you paid and the date.
2. Append a row to `data/trades.csv`:
   ```
   14,2026-08-15,17.20,5.10,21,,,,0,,OPEN,
   ```
   Columns: `trade_id, entry_date, entry_vix, entry_ask, strike, exit_date,
   exit_vix, exit_bid, days_held, roi_bid, exit_reason, note`

3. Update `data/position.json`:
   ```json
   {
     "in_position": true,
     "trade_id": 14,
     "entry_date": "2026-08-15",
     "entry_vix": 17.20,
     "entry_ask": 5.10,
     "entry_sigma": 1.2964,
     "strike": 21,
     "tenor": 180,
     "max_hold": 91,
     "peak_vix": 17.20
   }
   ```

4. Clear `data/daily_log.csv` (keep the header row only) and add the entry day:
   ```
   date,vix,sigma,option_bid,option_ask,signal,in_position,days_held,roi_bid
   2026-08-15,17.20,1.2964,4.82,5.10,BUY,True,0,0.0
   ```

5. Commit and push — the next workflow run will use the new position state.

### Closing a trade (SELL signal)

1. Note the bid price you received and the date.
2. Update the open row in `data/trades.csv` — fill in exit columns and
   change `exit_reason` from `OPEN` to `AE` (agent exit) or `HD` (hard deadline):
   ```
   14,2026-08-15,17.20,5.10,21,2026-09-20,22.50,7.85,36,53.9,AE,
   ```
3. Update `data/position.json` — set `"in_position": false`.
4. Commit and push.

---

## Strike Price Formula

```
strike = ceil(entry_vix * 1.2)
```

Examples: VIX 16.57 → ceil(19.884) = **20** | VIX 17.8 → ceil(21.36) = **22**

Always verify the strike is available in the IB option chain before entering.

---

## Model Files

| File | Source |
|------|--------|
| `models/v1210_entry.zip` | `output_milestones/output_v1210_full/entry_checkpoints/vix_ppo_v1210_EntryS2_roi_best.zip` |
| `models/v1243_exit.zip`  | `output_milestones/output_v1243_full/exit_checkpoints/vix_ppo_v1243_ExitS2_roi_best.zip` |

These are committed to the repo (~490 KB each) so GitHub Actions can load them
during inference without any external download.

---

## Updating the Dashboard Layout

The template file `dashboard/index_base.html` controls all styling, chart code,
and calculator logic. The generated `dashboard/index.html` is overwritten daily
and should never be edited directly.

To change the layout:
1. Edit `dashboard/index_base.html`.
2. Run `python production_v1/scripts/gen_dashboard.py --mock` locally to preview.
3. Commit and push — the next daily run picks up the new template.

Do not remove or rename the `%% SENTINEL %%` comment markers inside
`index_base.html` — `gen_dashboard.py` uses them to inject fresh data.

---

## Sharing the Dashboard

Send viewers the GitHub Pages URL:
```
https://YOUR_USERNAME.github.io/vix-strategy/
```

Anyone with the link can view it. No login required (public repo).
The page updates automatically each weekday evening — viewers just refresh.

---

## Troubleshooting

| Symptom | Likely cause | Fix |
|---------|-------------|-----|
| Workflow fails at "Install dependencies" | pip cache miss on first run | Re-run manually; it will succeed |
| `fetched.json` has `option_bid: 0` | Market closed / yfinance outage | Script falls back to BS estimate automatically |
| Signal stuck on HOLD | Model file not found at expected path | Verify `models/v1243_exit.zip` exists in repo |
| Dashboard shows old data | Commit step skipped ("Nothing changed") | Check that daily_log.csv was actually appended |
| GitHub Pages shows 404 | Pages not enabled, or branch mismatch | Settings → Pages → confirm branch = main, folder = / (root) |
