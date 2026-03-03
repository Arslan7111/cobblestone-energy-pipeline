import os
import time
import json
from datetime import datetime, timedelta
from typing import Iterator, Tuple, Dict, Any

import requests




def ensure_dirs() -> None:
    for p in ["data/raw", "data/processed", "logs", "reports"]:
        os.makedirs(p, exist_ok=True)


def date_chunks(start: datetime, end: datetime, chunk_days: int = 7) -> Iterator[Tuple[datetime, datetime]]:
    """
    Yield [chunk_start, chunk_end) intervals.
    Many Elexon endpoints limit how much data you can pull per request (often ~7 days).
    """
    cur = start
    while cur < end:
        nxt = min(cur + timedelta(days=chunk_days), end)
        yield cur, nxt
        cur = nxt


def iso(dt: datetime) -> str:
    
    return dt.strftime("%Y-%m-%dT%H:%M:%SZ")


def get_json(url: str, params: Dict[str, Any], retries: int = 3, backoff_sec: float = 1.5) -> Dict[str, Any]:
    """
    Basic resilient GET request with retries.
    """
    last_err = None
    for attempt in range(1, retries + 1):
        try:
            r = requests.get(url, params=params, timeout=60)
            r.raise_for_status()
            return r.json()
        except Exception as e:
            last_err = e
            time.sleep(backoff_sec * attempt)
    raise RuntimeError(f"Failed after {retries} retries: {url} params={params}. Last error: {last_err}")


def write_jsonl(path: str, record: Dict[str, Any]) -> None:
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")