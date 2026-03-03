from datetime import datetime, timedelta
from typing import Any

import pandas as pd

from src.utils import ensure_dirs, date_chunks, get_json

BASE_URL = "https://data.elexon.co.uk/bmrs/api/v1"


def _extract_data(payload: Any) -> list:
    if isinstance(payload, dict) and "data" in payload and isinstance(payload["data"], list):
        return payload["data"]
    if isinstance(payload, list):
        return payload
    return []


def _date_str(dt: datetime) -> str:
    return dt.strftime("%Y-%m-%d")


def _parse_date(s: str) -> datetime:
    return datetime.fromisoformat(s)


def _chunk_end_inclusive(exclusive_end: datetime) -> str:
    """
    Our date_chunks yields [a, b) where b is exclusive.
    Many Elexon dataset filters behave like inclusive dates.
    So for a chunk [a,b) we query end_day = b-1 day.
    """
    return (exclusive_end - timedelta(days=1)).strftime("%Y-%m-%d")



# FETCHERS 

def fetch_mid(start: datetime, end_exclusive: datetime, chunk_days: int = 7) -> pd.DataFrame:
    """
    MID: requires date range between from/to <= 7 days.
    We'll chunk and call /datasets/MID with from/to (date-only).
    """
    url = f"{BASE_URL}/datasets/MID"
    all_rows = []

    for a, b in date_chunks(start, end_exclusive, chunk_days=chunk_days):
        params = {"from": _date_str(a), "to": _chunk_end_inclusive(b), "format": "json"}
        payload = get_json(url, params)
        all_rows.extend(_extract_data(payload))

    return pd.DataFrame(all_rows)


def fetch_fuelhh(start: datetime, end_exclusive: datetime, chunk_days: int = 7) -> pd.DataFrame:
    """
    FUELHH: also range-limited; we chunk by 7 days.
    Uses settlementDateFrom/To + settlementPeriodFrom/To.
    """
    url = f"{BASE_URL}/datasets/FUELHH"
    all_rows = []

    for a, b in date_chunks(start, end_exclusive, chunk_days=chunk_days):
        params = {
            "settlementDateFrom": _date_str(a),
            "settlementDateTo": _chunk_end_inclusive(b),
            "settlementPeriodFrom": 1,
            "settlementPeriodTo": 48,
            "format": "json",
        }
        payload = get_json(url, params)
        all_rows.extend(_extract_data(payload))

    return pd.DataFrame(all_rows)


def fetch_temp(start: datetime, end_exclusive: datetime) -> pd.DataFrame:
    """
    TEMP: pull once (works over the month). We'll filter locally to our window.
    """
    url = f"{BASE_URL}/datasets/TEMP"
    params = {"from": _date_str(start), "to": _date_str(end_exclusive), "format": "json"}
    payload = get_json(url, params)
    return pd.DataFrame(_extract_data(payload))


# TRANSFORMS

def _std_settlement_keys(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    # Normalize key names
    col_map = {}
    for c in out.columns:
        lc = c.lower()
        if lc == "settlementdate":
            col_map[c] = "settlementDate"
        if lc == "settlementperiod":
            col_map[c] = "settlementPeriod"
        if lc == "fueltype":
            col_map[c] = "fuelType"
    out = out.rename(columns=col_map)

    if "settlementDate" in out.columns:
        out["settlementDate"] = pd.to_datetime(out["settlementDate"], errors="coerce").dt.date.astype(str)

    if "settlementPeriod" in out.columns:
        out["settlementPeriod"] = pd.to_numeric(out["settlementPeriod"], errors="coerce").astype("Int64")

    return out


def _pick_mid_price_col(mid: pd.DataFrame) -> str:
    for c in ["price", "marketIndexPrice", "marketIndexPriceGbp", "marketIndexPriceGBP"]:
        if c in mid.columns:
            return c
    raise ValueError(f"MID: no price column found. Columns={list(mid.columns)[:80]}")


def _to_hourly_from_settlement(df_sp: pd.DataFrame, value_col: str, out_name: str, agg: str = "mean") -> pd.DataFrame:
    """
    Convert settlement periods (30-min) to hourly.
    - hour = (settlementPeriod+1)//2
    - aggregate within hour (mean)
    """
    x = _std_settlement_keys(df_sp)
    x[out_name] = pd.to_numeric(x[value_col], errors="coerce")
    x["hour"] = ((x["settlementPeriod"].astype(int) + 1) // 2).astype(int)

    if agg == "sum":
        f = "sum"
    else:
        f = "mean"

    hourly = (
        x.dropna(subset=["settlementDate", "hour"])
        .groupby(["settlementDate", "hour"], as_index=False)[out_name]
        .agg(f)
        .sort_values(["settlementDate", "hour"])
    )
    return hourly


def build_hourly_dataset(mid: pd.DataFrame, fuelhh: pd.DataFrame, temp: pd.DataFrame,
                         start_date: str, end_date: str) -> pd.DataFrame:
    # MID price hourly
    mid_std = _std_settlement_keys(mid)
    price_col = _pick_mid_price_col(mid_std)

    # If multiple providers exist, average per settlement period first via grouping by date/period implicitly (hourly mean) 
    price_hourly = _to_hourly_from_settlement(mid_std, price_col, "da_price", agg="mean")

    # Wind+Solar hourly from FUELHH
    fuel_std = _std_settlement_keys(fuelhh)
    if "fuelType" not in fuel_std.columns:
        raise ValueError(f"FUELHH: missing fuelType. Columns={list(fuel_std.columns)}")
    if "generation" not in fuel_std.columns:
        raise ValueError(f"FUELHH: missing generation. Columns={list(fuel_std.columns)}")

    ft = fuel_std["fuelType"].astype(str).str.upper().str.strip()
    ws = fuel_std[ft.isin(["WIND", "SOLAR"])].copy()

    # Sum WIND+SOLAR at settlement period level, then to hourly mean
    ws["ws_mw"] = pd.to_numeric(ws["generation"], errors="coerce")
    ws_sp = (
        ws.dropna(subset=["settlementDate", "settlementPeriod"])
        .groupby(["settlementDate", "settlementPeriod"], as_index=False)["ws_mw"]
        .sum()
    )
    ws_hourly = _to_hourly_from_settlement(ws_sp, "ws_mw", "wind_solar_mw", agg="mean")

    
    temp_df = temp.copy()
    # TEMP uses measurementDate
    if "measurementDate" not in temp_df.columns or "temperature" not in temp_df.columns:
        raise ValueError(f"TEMP: expected measurementDate + temperature. Columns={list(temp_df.columns)}")

    temp_df["settlementDate"] = pd.to_datetime(temp_df["measurementDate"], errors="coerce").dt.date.astype(str)
    temp_df["temp_c"] = pd.to_numeric(temp_df["temperature"], errors="coerce")
    temp_daily = temp_df.groupby("settlementDate", as_index=False)["temp_c"].mean()

    # Merge
    out = price_hourly.merge(ws_hourly, on=["settlementDate", "hour"], how="left")
    out = out.merge(temp_daily, on="settlementDate", how="left")

    # Filter to our inclusive window
    out = out[(out["settlementDate"] >= start_date) & (out["settlementDate"] <= end_date)].copy()

    # keeps only full 24 hour days
    counts = out.groupby("settlementDate")["hour"].nunique()
    keep_days = counts[counts == 24].index
    out = out[out["settlementDate"].isin(keep_days)].copy()

    return out


# PIPELINE ENTRY

def run_ingest(start_date: str, end_date: str) -> None:
    """
    start_date/end_date are inclusive strings YYYY-MM-DD
    We'll download with end_exclusive = end_date + 1 day for chunking loops.
    """
    ensure_dirs()

    start_dt = _parse_date(start_date)
    end_dt_inclusive = _parse_date(end_date)
    end_exclusive = end_dt_inclusive + timedelta(days=1)

    print(f"[INFO] Window locked: {start_date} to {end_date} (inclusive)")

    print("Downloading MID (chunked)...")
    mid = fetch_mid(start_dt, end_exclusive, chunk_days=7)
    mid.to_parquet("data/raw/mid.parquet", index=False)

    print("Downloading FUELHH (chunked)...")
    fuelhh = fetch_fuelhh(start_dt, end_exclusive, chunk_days=7)
    fuelhh.to_parquet("data/raw/fuelhh.parquet", index=False)

    print("Downloading TEMP (single call)...")
    temp = fetch_temp(start_dt, end_exclusive)
    temp.to_parquet("data/raw/temp.parquet", index=False)

    print("Building hourly dataset...")
    hourly = build_hourly_dataset(mid, fuelhh, temp, start_date=start_date, end_date=end_date)
    hourly.to_parquet("data/processed/gb_hourly_dataset.parquet", index=False)

    print("DONE. Saved: data/processed/gb_hourly_dataset.parquet")