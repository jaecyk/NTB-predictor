import argparse
import csv
import json
from pathlib import Path
from typing import Iterable

import requests

DEFAULT_API_BASE_URL = "http://127.0.0.1:8000"
DEFAULT_OUTPUT_CSV = "latest_snapshots.csv"
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
VALID_TENORS = {91, 182, 364}


def fetch_remote_text(source_url: str, timeout: int = 30) -> tuple[str, str]:
    response = requests.get(source_url, timeout=timeout)
    response.raise_for_status()
    return response.text, response.headers.get("content-type", "")


def detect_format(source_format: str, content_type: str, source_url: str, text: str) -> str:
    if source_format != "auto":
        return source_format

    lowered_ct = content_type.lower()
    lowered_url = source_url.lower()
    stripped = text.lstrip()

    if "json" in lowered_ct or lowered_url.endswith(".json") or stripped.startswith("{") or stripped.startswith("["):
        return "json"
    return "csv"


def normalize_scalar(field: str, value):
    if value is None or str(value).strip() == "":
        raise ValueError(f"'{field}' is blank.")

    if field in INT_FIELDS:
        return int(str(value).strip())
    if field in FLOAT_FIELDS:
        return float(str(value).strip())
    return str(value).strip()


def normalize_row(row: dict, row_label: str) -> dict:
    output = {}
    for field in REQUIRED_FIELDS:
        if field not in row:
            raise ValueError(f"{row_label}: missing field '{field}'.")
        output[field] = normalize_scalar(field, row[field])

    if output["tenor_days"] not in VALID_TENORS:
        raise ValueError(f"{row_label}: tenor_days must be one of 91, 182, 364.")

    return output


def parse_csv_text(text: str) -> list[dict]:
    rows = []
    reader = csv.DictReader(text.splitlines())
    missing = [field for field in REQUIRED_FIELDS if field not in (reader.fieldnames or [])]
    if missing:
        raise ValueError(f"CSV is missing required column(s): {', '.join(missing)}")

    for idx, row in enumerate(reader, start=2):
        rows.append(normalize_row(row, f"CSV row {idx}"))

    if not rows:
        raise ValueError("CSV source contained no data rows.")
    return rows


def parse_json_payload(payload) -> list[dict]:
    if isinstance(payload, list):
        return [normalize_row(row, f"JSON row {idx}") for idx, row in enumerate(payload, start=1)]

    if not isinstance(payload, dict):
        raise ValueError("JSON source must be either a list of rows or an object.")

    if "snapshots" in payload and isinstance(payload["snapshots"], list):
        return [normalize_row(row, f"JSON snapshots row {idx}") for idx, row in enumerate(payload["snapshots"], start=1)]

    if "shared" in payload and "tenors" in payload:
        shared = payload["shared"] or {}
        tenors = payload["tenors"] or []
        rows = []
        for idx, row in enumerate(tenors, start=1):
            merged = {**shared, **row}
            rows.append(normalize_row(merged, f"JSON shared/tenors row {idx}"))
        return rows

    if all(field in payload for field in REQUIRED_FIELDS):
        return [normalize_row(payload, "JSON object")]

    raise ValueError(
        "JSON source was not recognized. Supported shapes: list[rows], "
        "{'snapshots': [...]}, or {'shared': {...}, 'tenors': [...]}"
    )


def write_csv(rows: Iterable[dict], output_csv: Path) -> None:
    with output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=REQUIRED_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def post_snapshots(rows: Iterable[dict], api_base_url: str, timeout: int = 20) -> list[dict]:
    results = []
    for row in rows:
        response = requests.post(f"{api_base_url}/snapshots", json=row, timeout=timeout)
        try:
            payload = response.json()
        except Exception:
            payload = response.text

        results.append(
            {
                "tenor_days": row["tenor_days"],
                "status_code": response.status_code,
                "ok": 200 <= response.status_code < 300,
                "response": payload,
            }
        )

        if not (200 <= response.status_code < 300):
            raise RuntimeError(
                f"Snapshot post failed for tenor {row['tenor_days']}: "
                f"HTTP {response.status_code} | {payload}"
            )
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Fetch a remote JSON/CSV market feed, normalize snapshot rows, optionally write CSV, and optionally post to the backend API."
    )
    parser.add_argument("--source-url", required=True, help="Remote URL for the source feed.")
    parser.add_argument(
        "--source-format",
        choices=["auto", "json", "csv"],
        default="auto",
        help="Input format for the source feed.",
    )
    parser.add_argument(
        "--output-csv",
        default=DEFAULT_OUTPUT_CSV,
        help="Path to write the normalized CSV file.",
    )
    parser.add_argument(
        "--skip-csv",
        action="store_true",
        help="Do not write the normalized CSV output.",
    )
    parser.add_argument(
        "--post-api",
        action="store_true",
        help="Post the normalized rows into the backend API.",
    )
    parser.add_argument(
        "--api-base-url",
        default=DEFAULT_API_BASE_URL,
        help="Base URL for the FastAPI backend.",
    )
    parser.add_argument(
        "--print-only",
        action="store_true",
        help="Print the normalized rows and exit.",
    )
    args = parser.parse_args()

    text, content_type = fetch_remote_text(args.source_url)
    source_format = detect_format(args.source_format, content_type, args.source_url, text)

    if source_format == "json":
        rows = parse_json_payload(json.loads(text))
    else:
        rows = parse_csv_text(text)

    if args.print_only:
        print(json.dumps(rows, indent=2))
        return

    if not args.skip_csv:
        write_csv(rows, Path(args.output_csv))

    results = {
        "row_count": len(rows),
        "csv_written": None if args.skip_csv else str(Path(args.output_csv).resolve()),
        "api_post_results": None,
    }

    if args.post_api:
        results["api_post_results"] = post_snapshots(rows, api_base_url=args.api_base_url)

    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
