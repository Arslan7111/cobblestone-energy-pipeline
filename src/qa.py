import json
import pandas as pd


def run_qa(processed_path: str = "data/processed/gb_hourly_dataset.parquet") -> dict:
    df = pd.read_parquet(processed_path).copy()

    report = {}
    report["n_rows"] = int(len(df))
    report["date_min"] = str(df["settlementDate"].min())
    report["date_max"] = str(df["settlementDate"].max())

    report["missing_by_col"] = {c: int(df[c].isna().sum()) for c in df.columns}
    report["duplicate_date_hour_rows"] = int(df.duplicated(subset=["settlementDate", "hour"]).sum())

    hours_per_day = df.groupby("settlementDate")["hour"].nunique()
    report["hours_per_day_counts"] = hours_per_day.value_counts().sort_index().to_dict()

    report["price_stats"] = {
        "min": float(df["da_price"].min()),
        "p01": float(df["da_price"].quantile(0.01)),
        "median": float(df["da_price"].median()),
        "p99": float(df["da_price"].quantile(0.99)),
        "max": float(df["da_price"].max()),
    }

    report["pct_missing_wind_solar_mw"] = float(df["wind_solar_mw"].isna().mean())
    report["pct_missing_temp_c"] = float(df["temp_c"].isna().mean())

    with open("logs/qa_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    return report