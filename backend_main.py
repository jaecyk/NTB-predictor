import math

from fastapi import Depends, FastAPI, HTTPException
from sqlalchemy import and_
from sqlalchemy.orm import Session

from backend_db import Base, engine, get_db
from backend_models import AuctionResult, MarketSnapshot, PredictionRun, TrainingRun
from backend_predictor import TENORS, load_models, predict_from_snapshot
from backend_schemas import (
    AccuracyItem,
    AuctionResultIn,
    AuctionResultOut,
    MarketSnapshotIn,
    MarketSnapshotOut,
    PredictionOut,
    PredictionRunOut,
    TrainingResultOut,
    TrainingRunOut,
)
import backend_trainer

app = FastAPI(title="NTB Live API", version="0.2.0")
models = load_models()
Base.metadata.create_all(bind=engine)


@app.get("/health")
def health():
    return {
        "status": "ok",
        "available_models": sorted(list(models.keys())),
    }


@app.post("/snapshots", response_model=MarketSnapshotOut)
def create_snapshot(payload: MarketSnapshotIn, db: Session = Depends(get_db)):
    item = MarketSnapshot(**payload.model_dump())
    db.add(item)
    db.commit()
    db.refresh(item)
    return item


@app.get("/snapshots/latest")
def latest_snapshots(db: Session = Depends(get_db)):
    result = {}
    for tenor in [91, 182, 364]:
        row = (
            db.query(MarketSnapshot)
            .filter(MarketSnapshot.tenor_days == tenor)
            .order_by(MarketSnapshot.created_at.desc())
            .first()
        )
        result[f"{tenor}D"] = MarketSnapshotOut.model_validate(row).model_dump() if row else None
    return result


@app.post("/predict/latest", response_model=list[PredictionOut])
def predict_latest(db: Session = Depends(get_db)):
    outputs = []

    for tenor in [91, 182, 364]:
        row = (
            db.query(MarketSnapshot)
            .filter(MarketSnapshot.tenor_days == tenor)
            .order_by(MarketSnapshot.created_at.desc())
            .first()
        )

        if not row:
            outputs.append(PredictionOut(tenor_days=tenor, predicted_stop_rate=None, status="no snapshot found"))
            continue

        value, status = predict_from_snapshot(row, models)

        if value is not None:
            run = PredictionRun(
                auction_date=row.auction_date,
                tenor_days=tenor,
                predicted_stop_rate=value,
                model_name=f"gti_ntb_v5_{tenor}D.pkl",
                status=status,
            )
            db.add(run)
            db.commit()

        outputs.append(PredictionOut(tenor_days=tenor, predicted_stop_rate=value, status=status))

    return outputs


@app.get("/predictions/history", response_model=list[PredictionRunOut])
def prediction_history(limit: int = 20, db: Session = Depends(get_db)):
    rows = (
        db.query(PredictionRun)
        .order_by(PredictionRun.created_at.desc())
        .limit(limit)
        .all()
    )
    return rows


# ---------------------------------------------------------------------------
# Auction results (training targets)
# ---------------------------------------------------------------------------

@app.post("/auctions/results", response_model=AuctionResultOut)
def record_auction_result(payload: AuctionResultIn, db: Session = Depends(get_db)):
    if payload.tenor_days not in TENORS:
        raise HTTPException(status_code=400, detail=f"tenor_days must be one of {TENORS}")
    item = AuctionResult(**payload.model_dump())
    db.add(item)
    db.commit()
    db.refresh(item)
    return item


@app.get("/auctions/results", response_model=list[AuctionResultOut])
def list_auction_results(limit: int = 50, db: Session = Depends(get_db)):
    return (
        db.query(AuctionResult)
        .order_by(AuctionResult.auction_date.desc())
        .limit(limit)
        .all()
    )


# ---------------------------------------------------------------------------
# Accuracy: compare logged predictions against realised auction results
# ---------------------------------------------------------------------------

@app.get("/accuracy", response_model=list[AccuracyItem])
def accuracy(db: Session = Depends(get_db)):
    items = []
    for tenor in TENORS:
        pairs = (
            db.query(PredictionRun, AuctionResult)
            .join(
                AuctionResult,
                and_(
                    PredictionRun.auction_date == AuctionResult.auction_date,
                    PredictionRun.tenor_days == AuctionResult.tenor_days,
                ),
            )
            .filter(PredictionRun.tenor_days == tenor)
            .all()
        )

        errors = [
            pred.predicted_stop_rate - result.actual_stop_rate
            for pred, result in pairs
        ]

        if not errors:
            items.append(AccuracyItem(tenor_days=tenor, n_compared=0, mae=None, rmse=None, bias=None))
            continue

        n = len(errors)
        mae = round(sum(abs(e) for e in errors) / n, 4)
        rmse = round(math.sqrt(sum(e * e for e in errors) / n), 4)
        bias = round(sum(errors) / n, 4)
        items.append(AccuracyItem(tenor_days=tenor, n_compared=n, mae=mae, rmse=rmse, bias=bias))

    return items


# ---------------------------------------------------------------------------
# Training: retrain models from stored snapshots + actual results
# ---------------------------------------------------------------------------

@app.post("/train", response_model=list[TrainingResultOut])
def train(db: Session = Depends(get_db)):
    results = backend_trainer.train_all(db)

    # Reload freshly-trained models in place so /predict uses them immediately.
    if any(r["status"] == "ok" for r in results):
        models.clear()
        models.update(load_models())

    return results


@app.get("/training/history", response_model=list[TrainingRunOut])
def training_history(limit: int = 20, db: Session = Depends(get_db)):
    return (
        db.query(TrainingRun)
        .order_by(TrainingRun.created_at.desc())
        .limit(limit)
        .all()
    )
