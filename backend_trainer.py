"""
backend_trainer.py — Retrain NTB stop-rate models from stored history.

Builds a supervised dataset by joining each MarketSnapshot (features) with the
AuctionResult recorded for the same auction date and tenor (target). Trains a
GradientBoostingRegressor per tenor, evaluates with time-series cross-validation,
saves a versioned .pkl plus the canonical file the predictor loads, and logs
metrics to the training_runs table.
"""

import math
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sqlalchemy import and_
from sqlalchemy.orm import Session
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from backend_models import MarketSnapshot, AuctionResult, TrainingRun
from backend_predictor import FEATURE_ORDER, MODEL_FILES, TENORS, _feature_row

MODEL_DIR = Path(".")

# Minimum matched (snapshot, result) pairs before we attempt to train a tenor.
MIN_TRAINING_SAMPLES = 12

GBR_PARAMS = {
    "n_estimators": 300,
    "learning_rate": 0.05,
    "max_depth": 3,
    "subsample": 0.8,
    "min_samples_leaf": 3,
    "random_state": 42,
}


def build_training_frame(db: Session, tenor: int) -> pd.DataFrame:
    """
    Join snapshots and actual results for one tenor into a feature+target frame,
    ordered chronologically so time-series CV is valid.
    """
    pairs = (
        db.query(MarketSnapshot, AuctionResult)
        .join(
            AuctionResult,
            and_(
                MarketSnapshot.auction_date == AuctionResult.auction_date,
                MarketSnapshot.tenor_days == AuctionResult.tenor_days,
            ),
        )
        .filter(MarketSnapshot.tenor_days == tenor)
        .order_by(MarketSnapshot.auction_date.asc())
        .all()
    )

    rows = []
    for snapshot, result in pairs:
        feat = _feature_row(snapshot)
        feat["stop_rate"] = float(result.actual_stop_rate)
        feat["auction_date"] = snapshot.auction_date
        rows.append(feat)

    return pd.DataFrame(rows)


def train_tenor(db: Session, tenor: int) -> dict:
    """Train one tenor model. Returns a result dict (also logged to DB)."""
    frame = build_training_frame(db, tenor)
    n_samples = len(frame)

    if n_samples < MIN_TRAINING_SAMPLES:
        result = {
            "tenor_days": tenor,
            "status": "skip",
            "message": (
                f"Only {n_samples} matched snapshot/result pairs "
                f"(need {MIN_TRAINING_SAMPLES}+). Record more auction results."
            ),
            "n_samples": n_samples,
        }
        _log_run(db, tenor, "n/a", n_samples, None, None, None, "skip")
        return result

    X = frame[FEATURE_ORDER]
    y = frame["stop_rate"]

    n_splits = min(5, max(2, n_samples // 5))
    tscv = TimeSeriesSplit(n_splits=n_splits)

    maes, rmses, r2s = [], [], []
    for train_idx, val_idx in tscv.split(X):
        model = GradientBoostingRegressor(**GBR_PARAMS)
        model.fit(X.iloc[train_idx], y.iloc[train_idx])
        preds = model.predict(X.iloc[val_idx])
        maes.append(mean_absolute_error(y.iloc[val_idx], preds))
        rmses.append(math.sqrt(mean_squared_error(y.iloc[val_idx], preds)))
        r2s.append(r2_score(y.iloc[val_idx], preds))

    mae = round(float(np.mean(maes)), 4)
    rmse = round(float(np.mean(rmses)), 4)
    r2 = round(float(np.mean(r2s)), 4)

    # Fit final model on all available data and persist it.
    final_model = GradientBoostingRegressor(**GBR_PARAMS)
    final_model.fit(X, y)

    version = f"v{datetime.utcnow().strftime('%Y%m%d%H%M%S')}"
    versioned_path = MODEL_DIR / f"gti_ntb_{version}_{tenor}D.pkl"
    canonical_path = MODEL_DIR / MODEL_FILES[tenor]

    joblib.dump(final_model, versioned_path)
    joblib.dump(final_model, canonical_path)  # what the predictor loads

    _log_run(db, tenor, version, n_samples, mae, rmse, r2, "ok")

    return {
        "tenor_days": tenor,
        "status": "ok",
        "message": f"Trained on {n_samples} samples; saved {canonical_path.name}.",
        "n_samples": n_samples,
        "model_version": version,
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
    }


def train_all(db: Session) -> list[dict]:
    return [train_tenor(db, tenor) for tenor in TENORS]


def _log_run(db, tenor, version, n_samples, mae, rmse, r2, status) -> None:
    db.add(
        TrainingRun(
            tenor_days=tenor,
            model_version=version,
            n_samples=n_samples,
            mae=mae,
            rmse=rmse,
            r2=r2,
            status=status,
        )
    )
    db.commit()
