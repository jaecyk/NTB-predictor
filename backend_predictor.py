from pathlib import Path
import joblib
import pandas as pd

TENORS = [91, 182, 364]
MODEL_FILES = {
    91: "gti_ntb_v5_91D.pkl",
    182: "gti_ntb_v5_182D.pkl",
    364: "gti_ntb_v5_364D.pkl",
}
FEATURE_ORDER = [
    "lag1_stop",
    "lag2_stop",
    "lag3_stop",
    "ma3_stop",
    "delta_stop_1",
    "offer_amt",
    "offer_change",
    "prev_bid_cover",
    "sec_rate",
    "sec_rate_change_5d",
    "sec_minus_lag1",
    "system_liquidity",
    "mpr",
    "inflation",
]


def _feature_row(snapshot) -> dict:
    return {
        "lag1_stop": float(snapshot.lag1_stop),
        "lag2_stop": float(snapshot.lag2_stop),
        "lag3_stop": float(snapshot.lag3_stop),
        "ma3_stop": round((float(snapshot.lag1_stop) + float(snapshot.lag2_stop) + float(snapshot.lag3_stop)) / 3.0, 6),
        "delta_stop_1": round(float(snapshot.lag1_stop) - float(snapshot.lag2_stop), 6),
        "offer_amt": float(snapshot.offer_amt),
        "offer_change": round(float(snapshot.offer_amt) - float(snapshot.prev_offer), 6),
        "prev_bid_cover": float(snapshot.prev_bid_cover),
        "sec_rate": float(snapshot.sec_rate),
        "sec_rate_change_5d": round(float(snapshot.sec_rate) - float(snapshot.sec_rate_5d_ago), 6),
        "sec_minus_lag1": round(float(snapshot.sec_rate) - float(snapshot.lag1_stop), 6),
        "system_liquidity": float(snapshot.system_liquidity),
        "mpr": float(snapshot.mpr),
        "inflation": float(snapshot.inflation),
    }


def load_models():
    models = {}
    for tenor, fname in MODEL_FILES.items():
        if Path(fname).exists():
            models[tenor] = joblib.load(fname)
    return models


def predict_from_snapshot(snapshot, models: dict):
    tenor = int(snapshot.tenor_days)
    if tenor not in models:
        return None, "model file not available"

    features = _feature_row(snapshot)
    X = pd.DataFrame([features])[FEATURE_ORDER]

    try:
        value = float(models[tenor].predict(X)[0])
        return round(value, 4), "ok"
    except Exception as exc:
        return None, f"{type(exc).__name__}: {exc}"
