from sqlalchemy import Column, Date, DateTime, Float, Integer, String
from sqlalchemy.sql import func
from backend_db import Base


class MarketSnapshot(Base):
    __tablename__ = "market_snapshots"

    id = Column(Integer, primary_key=True, index=True)
    auction_date = Column(Date, nullable=False)
    tenor_days = Column(Integer, nullable=False, index=True)
    lag1_stop = Column(Float, nullable=False)
    lag2_stop = Column(Float, nullable=False)
    lag3_stop = Column(Float, nullable=False)
    offer_amt = Column(Float, nullable=False)
    prev_offer = Column(Float, nullable=False)
    prev_bid_cover = Column(Float, nullable=False)
    sec_rate = Column(Float, nullable=False)
    sec_rate_5d_ago = Column(Float, nullable=False)
    system_liquidity = Column(Float, nullable=False)
    mpr = Column(Float, nullable=False)
    inflation = Column(Float, nullable=False)
    source = Column(String(100), nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())


class PredictionRun(Base):
    __tablename__ = "prediction_runs"

    id = Column(Integer, primary_key=True, index=True)
    auction_date = Column(Date, nullable=False)
    tenor_days = Column(Integer, nullable=False, index=True)
    predicted_stop_rate = Column(Float, nullable=False)
    model_name = Column(String(100), nullable=False)
    status = Column(String(50), nullable=False, default="ok")
    created_at = Column(DateTime(timezone=True), server_default=func.now())
