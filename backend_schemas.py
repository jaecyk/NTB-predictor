from datetime import date, datetime
from pydantic import BaseModel


class MarketSnapshotIn(BaseModel):
    auction_date: date
    tenor_days: int
    lag1_stop: float
    lag2_stop: float
    lag3_stop: float
    offer_amt: float
    prev_offer: float
    prev_bid_cover: float
    sec_rate: float
    sec_rate_5d_ago: float
    system_liquidity: float
    mpr: float
    inflation: float
    source: str | None = None


class MarketSnapshotOut(MarketSnapshotIn):
    id: int
    created_at: datetime

    class Config:
        from_attributes = True


class PredictionOut(BaseModel):
    tenor_days: int
    predicted_stop_rate: float | None
    status: str


class PredictionRunOut(BaseModel):
    id: int
    auction_date: date
    tenor_days: int
    predicted_stop_rate: float
    model_name: str
    status: str
    created_at: datetime

    class Config:
        from_attributes = True


class AuctionResultIn(BaseModel):
    auction_date: date
    tenor_days: int
    actual_stop_rate: float
    source: str | None = None


class AuctionResultOut(AuctionResultIn):
    id: int
    created_at: datetime

    class Config:
        from_attributes = True


class AccuracyItem(BaseModel):
    tenor_days: int
    n_compared: int
    mae: float | None
    rmse: float | None
    bias: float | None  # mean(predicted - actual); +ve = model over-predicts


class TrainingResultOut(BaseModel):
    tenor_days: int
    status: str
    message: str
    n_samples: int
    model_version: str | None = None
    mae: float | None = None
    rmse: float | None = None
    r2: float | None = None


class TrainingRunOut(BaseModel):
    id: int
    tenor_days: int
    model_version: str
    n_samples: int
    mae: float | None
    rmse: float | None
    r2: float | None
    status: str
    created_at: datetime

    class Config:
        from_attributes = True
