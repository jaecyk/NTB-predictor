import argparse
import csv
import json
from pathlib import Path
from typing import Iterable

import requests

DEFAULT_API_BASE_URL = "http://127.0.0.1:8000"
REQUIRED_FIELDS = [
    "auction_date",
    "tenor_days",
    "lag1_stop",
    "lag2_stop",
    "lag3_stop",
    "offer_amt",
    "prev_offer",
    "prev_bid_cover",
    "sec_rate",
    "sec_rate_5d_ago",
    "system_liquidity",
    "mpr",
    "inflation",
    "source",
]
INT_FIELDS = {"tenor_days"}
FLOAT_FIELDS = {
    "lag1_stop",
    "lag2_stop",
    "lag3_stop",
    "offer_amt",
    "prev_offer",
    "prev_bid_cover",
    "sec_rate",
    "sec_rate_5d_ago",
    "system_liquidity",
    "mpr",
    "inflation",
}


def load_csv_rows(csv_path: Path) -> list[dict]:
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    with csv_path.open("r", newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        missing = [field for field in REQUIRED_FIELDS if field not in (reader.fieldnames or [])]
        if missing:
            raise ValueError(f"CSV is missing required column(s): {', '.join(missing)}")

        rows = []
        for idx, row in enumerate(reader, start=2):
            normalized = normalize_row(row, row_number=idx)
            rows.append(normalized)

    if not rows:
        raise ValueError("CSV contains no data rows.")

    return rows


def normalize_row(row: dict, row_number: int) -> dict:
    output = {}

    for field in REQUIRED_FIELDS:
        value = row.get(field, "")
        if value is None or str(value).strip() == "":
            raise ValueError(f"Row {row_number}: '{field}' is blank.")
        value = str(value).strip()

        if field in INT_FIELDS:
            output[field] = int(value)
        elif field in FLOAT_FIELDS:
            output[field] = float(value)
        else:
            output[field] = value

    if output["tenor_days"] not in {91, 182, 364}:
        raise ValueError(f"Row {row_number}: tenor_days must be one of 91, 182, 364.")

    return output


def post_snapshots(rows: Iterable[dict], api_base_url: str, timeout: int = 20) -> list[dict]:
    results = []

    for row in rows:
        response = requests.post(f"{api_base_url}/snapshots", json=row, timeout=timeout)
        status_ok = 200 <= response.status_code < 300
        payload = None

        try:
            payload = response.json()
        except Exception:
            payload = response.text

        results.append(
            {
                "tenor_days": row["tenor_days"],
                "status_code": response.status_code,
                "ok": status_ok,
                "response": payload,
            }
        )

        if not status_ok:
            raise RuntimeError(
                f"Snapshot post failed for tenor {row['tenor_days']}: "
                f"HTTP {response.status_code} | {payload}"
            )

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Load NTB snapshot rows from a CSV file and post them into the FastAPI backend."
    )
    parser.add_argument(
        "--csv",
        default="sample_snapshot_template.csv",
        help="Path to the CSV file containing snapshot rows.",
    )
    parser.add_argument(
        "--api-base-url",
        default=DEFAULT_API_BASE_URL,
        help="Base URL for the FastAPI backend.",
    )
    parser.add_argument(
        "--print-only",
        action="store_true",
        help="Validate and print the payloads without posting them to the API.",
    )
    args = parser.parse_args()

    rows = load_csv_rows(Path(args.csv))

    if args.print_only:
        print(json.dumps(rows, indent=2))
        return

    results = post_snapshots(rows, api_base_url=args.api_base_url)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
