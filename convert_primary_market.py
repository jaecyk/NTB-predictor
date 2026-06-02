"""
convert_primary_market.py
=========================
Convert the CBN Primary Market Excel download into a bulk-import CSV that
the NTB-predictor backend can ingest via POST /import/historical.

Usage:
    python convert_primary_market.py \
        --excel Primary_Market_in_Excel.xlsx \
        --out   ntb_historical_import.csv \
        --start 2020-01-01

The auction-derived features (lags, offer amounts, bid cover) are computed
directly from the Excel.  Macro variables (sec_rate, system_liquidity, MPR,
inflation) are approximated from published CBN/NBS historical data; verify
and correct any values before running the model retrain.
"""

import argparse
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Tenor normalisation — handles every label variant seen in CBN exports
# ---------------------------------------------------------------------------
TENOR_MAP = {91: set(), 182: set(), 364: set()}


def normalize_tenor(raw: str) -> int | None:
    t = str(raw).strip().upper().replace(" ", "").replace("DAY", "").replace("DAYS", "")
    try:
        n = int(t)
        if 88 <= n <= 95:
            return 91
        if 178 <= n <= 185:
            return 182
        if 340 <= n <= 370:
            return 364
    except ValueError:
        pass
    return None


# ---------------------------------------------------------------------------
# Macro variable lookup tables (step-function by effective date)
# Sources: CBN MPC communiqués, NBS CPI releases
# These are APPROXIMATE — update with precise data if available.
# ---------------------------------------------------------------------------

# CBN Monetary Policy Rate (MPR) effective dates
_MPR_STEPS = [
    ("2020-01-01", 13.50),
    ("2020-05-28", 12.50),
    ("2021-01-01", 11.50),
    ("2022-01-01", 11.50),
    ("2022-05-24", 13.00),
    ("2022-07-19", 14.00),
    ("2022-09-27", 15.50),
    ("2023-02-23", 17.50),
    ("2023-03-22", 18.00),
    ("2023-05-23", 18.50),
    ("2023-07-25", 18.75),
    ("2024-02-27", 22.75),
    ("2024-03-26", 24.75),
    ("2024-05-21", 26.25),
    ("2024-07-23", 26.75),
    ("2024-09-24", 27.25),
    ("2024-11-26", 27.50),
    ("2025-02-19", 27.50),
    ("2025-05-20", 27.00),
    ("2025-07-22", 26.75),
    ("2025-09-23", 26.50),
    ("2026-01-01", 26.50),
]

# Nigeria headline inflation (NBS CPI, year-on-year %) by approximate month
_INFLATION_STEPS = [
    ("2020-01-01", 12.13),
    ("2020-04-01", 12.26),
    ("2020-07-01", 12.82),
    ("2020-10-01", 14.23),
    ("2021-01-01", 15.75),
    ("2021-04-01", 18.12),
    ("2021-07-01", 17.38),
    ("2021-10-01", 15.99),
    ("2022-01-01", 15.60),
    ("2022-04-01", 16.82),
    ("2022-07-01", 19.64),
    ("2022-10-01", 21.09),
    ("2023-01-01", 21.82),
    ("2023-04-01", 22.22),
    ("2023-07-01", 24.08),
    ("2023-10-01", 27.33),
    ("2024-01-01", 29.90),
    ("2024-04-01", 33.20),
    ("2024-07-01", 36.00),
    ("2024-10-01", 32.70),
    ("2025-01-01", 24.48),
    ("2025-04-01", 23.71),
    ("2025-07-01", 26.00),
    ("2025-10-01", 32.70),
    ("2026-01-01", 24.48),
    ("2026-04-01", 15.06),
]

# System liquidity (NGN bn) — very rough monthly averages; update with
# actual CBN banking system liquidity data for better model accuracy.
_LIQUIDITY_STEPS = [
    ("2020-01-01", 500.0),
    ("2021-01-01", 700.0),
    ("2022-01-01", 800.0),
    ("2023-01-01", 1200.0),
    ("2024-01-01", 1800.0),
    ("2024-07-01", 2200.0),
    ("2025-01-01", 2600.0),
    ("2025-07-01", 2780.0),
    ("2026-01-01", 3000.0),
]


def _step_lookup(steps: list[tuple], dt: date) -> float:
    """Return the most recent value from a step-function table."""
    val = steps[0][1]
    for date_str, v in steps:
        if pd.Timestamp(date_str).date() <= dt:
            val = v
        else:
            break
    return val


def lookup_mpr(dt: date) -> float:
    return _step_lookup(_MPR_STEPS, dt)


def lookup_inflation(dt: date) -> float:
    return _step_lookup(_INFLATION_STEPS, dt)


def lookup_liquidity(dt: date) -> float:
    return _step_lookup(_LIQUIDITY_STEPS, dt)


# ---------------------------------------------------------------------------
# Main conversion
# ---------------------------------------------------------------------------

def load_and_clean(excel_path: Path) -> pd.DataFrame:
    df = pd.read_excel(excel_path, header=0)
    df["tenor_norm"] = df["tenor"].apply(normalize_tenor)
    df = df[df["tenor_norm"].notna()].copy()
    df["auction_date"] = pd.to_datetime(df["auctionDate"], format="mixed")
    df["tenor_norm"] = df["tenor_norm"].astype(int)
    df = df[df["rate"] > 0].copy()
    return df


def build_import_frame(df: pd.DataFrame, start_date: str) -> pd.DataFrame:
    rows = []

    for tenor in [91, 182, 364]:
        sub = (
            df[df["tenor_norm"] == tenor]
            .sort_values("auction_date")
            .reset_index(drop=True)
        )

        for i, row in sub.iterrows():
            dt = row["auction_date"].date()

            if str(dt) < start_date:
                continue

            # Lags: previous auction stop rates for this tenor
            prev_rates = sub.loc[:i - 1, "rate"].values
            prev_offers = sub.loc[:i - 1, "amtOffered"].values
            prev_subs = sub.loc[:i - 1, "totalSubscription"].values

            if len(prev_rates) < 3:
                # Skip rows where we can't compute all three lags
                continue

            lag1 = round(float(prev_rates[-1]), 4)
            lag2 = round(float(prev_rates[-2]), 4)
            lag3 = round(float(prev_rates[-3]), 4)

            prev_offer = round(float(prev_offers[-1]), 2) if len(prev_offers) >= 1 else float(row["amtOffered"])
            offer_amt = round(float(row["amtOffered"]), 2)

            # Bid cover = total subscription / amount offered (previous auction)
            prev_bid_cover = round(float(prev_subs[-1]) / float(prev_offers[-1]), 4) if prev_offers[-1] > 0 else 1.0

            # Macro approximations
            mpr = lookup_mpr(dt)
            inflation = lookup_inflation(dt)
            system_liquidity = lookup_liquidity(dt)

            # Secondary rate proxy: use lag1_stop + a small tenor spread
            tenor_spread = {91: 0.00, 182: 0.05, 364: 0.15}[tenor]
            sec_rate = round(lag1 + tenor_spread, 4)
            sec_rate_5d_ago = round(lag2 + tenor_spread, 4)

            rows.append({
                "auction_date": str(dt),
                "tenor_days": tenor,
                "lag1_stop": lag1,
                "lag2_stop": lag2,
                "lag3_stop": lag3,
                "offer_amt": offer_amt,
                "prev_offer": prev_offer,
                "prev_bid_cover": prev_bid_cover,
                "sec_rate": sec_rate,
                "sec_rate_5d_ago": sec_rate_5d_ago,
                "system_liquidity": system_liquidity,
                "mpr": mpr,
                "inflation": inflation,
                "actual_stop_rate": round(float(row["rate"]), 4),
                "source": "cbn_primary_market",
            })

    return pd.DataFrame(rows).sort_values("auction_date").reset_index(drop=True)


def main():
    parser = argparse.ArgumentParser(description="Convert CBN Primary Market Excel to NTB import CSV")
    parser.add_argument("--excel", default="Primary_Market_in_Excel.xlsx")
    parser.add_argument("--out", default="ntb_historical_import.csv")
    parser.add_argument("--start", default="2020-01-01", help="Earliest auction date to include")
    args = parser.parse_args()

    print(f"Loading {args.excel}...")
    df = load_and_clean(Path(args.excel))
    print(f"  Raw rows after cleaning: {len(df)}")

    print(f"Building import frame (start={args.start})...")
    out = build_import_frame(df, args.start)
    print(f"  Export rows: {len(out)}")
    print(f"  Per tenor:\n{out['tenor_days'].value_counts().sort_index().to_string()}")
    print(f"  Date range: {out['auction_date'].min()} -> {out['auction_date'].max()}")

    out.to_csv(args.out, index=False)
    print(f"\nSaved → {args.out}")
    print("\nNOTE: sec_rate, sec_rate_5d_ago, system_liquidity, mpr, inflation are")
    print("approximate. Verify these columns against actual CBN/NBS data for best")
    print("model accuracy, then import via the Streamlit Bulk Import tab.")


if __name__ == "__main__":
    main()
