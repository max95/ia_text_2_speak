from __future__ import annotations

import os
from typing import Any

import requests
from fastapi import APIRouter, HTTPException

router = APIRouter()


def fetch_line_l_departures(stop_area_id: str, count: int = 5) -> dict[str, Any]:
    api_key = (os.environ.get("SNCF_API_KEY") or "").strip()
    if not api_key:
        raise HTTPException(status_code=503, detail="SNCF_API_KEY is not configured")

    if not stop_area_id.strip():
        raise HTTPException(status_code=400, detail="stop_area_id is required")

    url = f"https://api.sncf.com/v1/coverage/sncf/stop_areas/{stop_area_id}/departures"
    params = {
        "count": max(1, min(count, 20)),
        "line": "line:L",
    }

    try:
        response = requests.get(url, params=params, auth=(api_key, ""), timeout=10)
        response.raise_for_status()
    except requests.RequestException as exc:
        raise HTTPException(status_code=502, detail=f"sncf lookup failed: {exc}") from exc

    data = response.json()
    departures = []
    for item in data.get("departures", []):
        departure = item.get("departure", {})
        departures.append(
            {
                "direction": departure.get("direction", {}).get("name"),
                "departure_time": departure.get("stop_date_time", {}).get("departure_date_time"),
                "base_departure_time": departure.get("stop_date_time", {}).get("base_departure_date_time"),
                "line": departure.get("route", {}).get("line", {}).get("name"),
                "stop_area": departure.get("stop_point", {}).get("stop_area", {}).get("name"),
            }
        )

    return {
        "stop_area_id": stop_area_id,
        "count": params["count"],
        "departures": departures,
    }


@router.get("/v1/trains/line-l/departures")
def get_line_l_departures(stop_area_id: str, count: int = 5):
    return fetch_line_l_departures(stop_area_id, count)
