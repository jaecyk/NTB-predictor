from fastapi import Depends, FastAPI, HTTPException
from sqlalchemy.orm import Session

from backend_db import Base, engine, get_db
from backend_models import MarketSnapshot, PredictionRun
from backend_predictor import load_models, predict_from_snapshot
from backend_schemas import MarketSnapshotIn, MarketSnapshotOut, PredictionOut, PredictionRunOut

app = FastAPI(title="NTB Live API", version="0.1.0")
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
