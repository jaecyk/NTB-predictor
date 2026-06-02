# NTB Live API Reference

FastAPI service (`backend_main.py`) that stores live market data, serves model
predictions, records actual auction outcomes, and **retrains the models from the
accumulated history stored in the database**.

Run locally:

```bash
pip install -r requirements_backend.txt
uvicorn backend_main:app --reload
# Interactive docs at http://127.0.0.1:8000/docs
```

The database is configured via `DATABASE_URL` (defaults to `sqlite:///./ntb_live.db`;
set a PostgreSQL URL in production).

---

## The live-training loop

```
1. POST /snapshots          → store pre-auction market features
2. POST /predict/latest     → model predicts; prediction logged to DB
3. POST /auctions/results   → after the auction, record the ACTUAL stop rate
4. GET  /accuracy           → compare logged predictions vs actuals
5. POST /train              → retrain models on all matched snapshot+result pairs
                              (new models are saved and loaded immediately)
```

Each retrain joins every `MarketSnapshot` with the `AuctionResult` for the same
date and tenor, fits a GradientBoostingRegressor per tenor, evaluates it with
time-series cross-validation, writes a versioned `.pkl` plus the canonical
`gti_ntb_v5_<tenor>D.pkl`, and logs metrics to `training_runs`.

---

## Endpoints

### `GET /health`
Returns service status and which tenor models are loaded.

### `POST /snapshots`
Store a pre-auction market snapshot. Body: `MarketSnapshotIn`
(`auction_date`, `tenor_days` ∈ {91,182,364}, the engineered feature fields,
optional `source`).

### `GET /snapshots/latest`
Latest snapshot per tenor, keyed `"91D" | "182D" | "364D"`.

### `POST /predict/latest`
Predicts the stop rate for the latest snapshot of each tenor and logs each
successful prediction to `prediction_runs`.

### `GET /predictions/history?limit=20`
Recent prediction runs, newest first.

### `POST /auctions/results`
Record the realised stop rate (training target). Body: `AuctionResultIn`
(`auction_date`, `tenor_days`, `actual_stop_rate`, optional `source`).

### `GET /auctions/results?limit=50`
Recorded auction results, newest first.

### `GET /accuracy`
Per-tenor error of logged predictions vs realised results:
`n_compared`, `mae`, `rmse`, and `bias` (mean signed error; positive means the
model over-predicts).

### `POST /train`
Retrain every tenor that has at least `MIN_TRAINING_SAMPLES` (12) matched
snapshot+result pairs. Tenors below the threshold return `status: "skip"`.
Returns per-tenor `TrainingResultOut` (status, n_samples, model_version, mae,
rmse, r2). Freshly trained models are reloaded in place — the next
`/predict/latest` uses them with no restart.

### `GET /training/history?limit=20`
Recent training runs with their metrics, newest first.
