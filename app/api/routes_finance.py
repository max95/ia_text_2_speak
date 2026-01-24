from __future__ import annotations

import csv
from io import StringIO
from typing import Any

import requests
from fastapi import APIRouter, HTTPException

router = APIRouter()


@router.get("/v1/finance/price")
def get_finance_price(symbol: str):
    if not symbol.strip():
        raise HTTPException(status_code=400, detail="symbol is required")

    try:
        response = requests.get(
            "https://stooq.com/q/l/",
            params={"s": symbol, "f": "sd2t2ohlcv", "h": "", "e": "csv"},
            timeout=10,
        )
        response.raise_for_status()
    except requests.RequestException as exc:
        raise HTTPException(status_code=502, detail=f"price lookup failed: {exc}") from exc

    reader = csv.DictReader(StringIO(response.text))
    row: dict[str, Any] | None = next(reader, None)
    if not row or row.get("Close") in (None, "N/A"):
        raise HTTPException(status_code=404, detail="symbol not found")

    return {
        "symbol": row.get("Symbol") or symbol,
        "date": row.get("Date"),
        "time": row.get("Time"),
        "open": row.get("Open"),
        "high": row.get("High"),
        "low": row.get("Low"),
        "close": row.get("Close"),
        "volume": row.get("Volume"),
    }
