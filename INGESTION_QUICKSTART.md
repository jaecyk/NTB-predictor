# Ingestion Quickstart

This file gives the fastest path to load snapshot rows into the API without typing them manually in the frontend.

## Files added
- `ingest_snapshots.py`
- `sample_snapshot_template.csv`

## What it does
The script reads the CSV file, validates the required fields, and posts each row to:
- `POST /snapshots`

The backend then stores those rows in the database, and the frontend can read them through:
- `/snapshots/latest`
- `/predict/latest`
- `/predictions/history`

## Required columns
- `auction_date`
- `tenor_days`
- `lag1_stop`
- `lag2_stop`
- `lag3_stop`
- `offer_amt`
- `prev_offer`
- `prev_bid_cover`
- `sec_rate`
- `sec_rate_5d_ago`
- `system_liquidity`
- `mpr`
- `inflation`
- `source`

## Step-by-step run

### 1. Make sure the backend is running
```bash
python -m uvicorn backend_main:app --reload
```

### 2. Preview the CSV payload without posting
```bash
python ingest_snapshots.py --print-only
```

### 3. Post the CSV rows into the backend
```bash
python ingest_snapshots.py
```

### 4. Check what landed
Open in browser:
- `http://127.0.0.1:8000/snapshots/latest`
- `http://127.0.0.1:8000/predict/latest`

Or use the Streamlit frontend buttons:
- `Refresh latest snapshots`
- `Predict from latest snapshots`
- `Load prediction history`

## Alternative CSV path
If your CSV is elsewhere:
```bash
python ingest_snapshots.py --csv your_file.csv
```

## Alternative API base URL
If the backend is hosted elsewhere:
```bash
python ingest_snapshots.py --api-base-url https://your-backend-url
```

## Practical next step
Replace the sample CSV with a daily export or a generated file from your live market source.
That lets you move from manual entry to repeatable semi-automated ingestion first.
