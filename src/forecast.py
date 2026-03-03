from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.ensemble import HistGradientBoostingRegressor


@dataclass
class BacktestConfig:
    data_path: str = "data/processed/gb_hourly_dataset.parquet"
    test_days: int = 7  # last N days for walk forward evaluation
    reports_dir: str = "reports"


def _make_timestamp(df: pd.DataFrame) -> pd.Series:
    # hour is 1..24, convert to 0..23 offset
    return pd.to_datetime(df["settlementDate"]) + pd.to_timedelta(df["hour"] - 1, unit="h")


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build leakage-safe features:
    - calendar features (hour, day-of-week)
    - lagged price and rolling stats (use only past information)
    - lagged fundamentals (wind_solar_mw, temp_c)
    """
    x = df.copy()
    x["timestamp"] = _make_timestamp(x)
    x = x.sort_values("timestamp").reset_index(drop=True)

    # Calendar features
    x["dow"] = pd.to_datetime(x["settlementDate"]).dt.dayofweek  # Mon=0
    x["is_weekend"] = (x["dow"] >= 5).astype(int)

    # Price lags (available up to yesterday)
    x["lag_24_price"] = x["da_price"].shift(24)
    x["lag_48_price"] = x["da_price"].shift(48)
    x["lag_168_price"] = x["da_price"].shift(168)

    # Rolling stats (shift first to avoid using current hour)
    x["roll_mean_24"] = x["da_price"].shift(1).rolling(24).mean()
    x["roll_std_24"] = x["da_price"].shift(1).rolling(24).std()
    x["roll_mean_168"] = x["da_price"].shift(1).rolling(168).mean()

    # Fundamentals lags 
    x["lag_24_wind"] = x["wind_solar_mw"].shift(24)
    x["lag_24_temp"] = x["temp_c"].shift(24)

    # Target
    x["y"] = x["da_price"]

    return x


def baseline_predictions(df_feat: pd.DataFrame) -> pd.DataFrame:
    """
    Two baselines:
    1) Naive-24: same hour yesterday
    2) Blend: average of yesterday and last-week same hour (if available)
    """
    out = df_feat.copy()
    out["pred_naive_24"] = out["lag_24_price"]
    out["pred_blend_24_168"] = np.where(
        out["lag_168_price"].notna(),
        0.5 * out["lag_24_price"] + 0.5 * out["lag_168_price"],
        out["lag_24_price"],
    )
    return out


def walk_forward_backtest(df_feat: pd.DataFrame, feature_cols: List[str], test_days: int) -> pd.DataFrame:
    """
    Walk-forward by day:
    For each forecast day D in the last `test_days` days:
      train on all data with settlementDate < D
      predict all 24 hours of D
    """
    df_feat = df_feat.copy()
    all_days = sorted(df_feat["settlementDate"].unique())

    # Only keep days that have all 24 hours present in df_feat
    counts = df_feat.groupby("settlementDate")["hour"].nunique()
    full_days = [d for d in all_days if counts.get(d, 0) == 24]

    if len(full_days) <= test_days + 1:
        raise ValueError(f"Not enough full days for backtest. Full days={len(full_days)}")

    test_days_list = full_days[-test_days:]
    preds = []

    model = HistGradientBoostingRegressor(
        max_depth=6,
        learning_rate=0.05,
        max_iter=400,
        random_state=42,
    )

    for d in test_days_list:
        train = df_feat[df_feat["settlementDate"] < d].dropna(subset=feature_cols + ["y"])
        test = df_feat[df_feat["settlementDate"] == d].dropna(subset=feature_cols)
        # skip incomplete day 
        if len(test) != 24:
            
            continue

        model.fit(train[feature_cols], train["y"])
        y_pred = model.predict(test[feature_cols])

        tmp = test[["settlementDate", "hour", "y"]].copy()
        tmp["y_pred_model"] = y_pred
        preds.append(tmp)

    return pd.concat(preds, ignore_index=True)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    mae = mean_absolute_error(y_true, y_pred)
    # sklearn version safe RMSE:
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    return {"MAE": float(mae), "RMSE": rmse}


def run_forecast(cfg: BacktestConfig) -> None:
    reports = Path(cfg.reports_dir)
    reports.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(cfg.data_path)
    df_feat = build_features(df)

    # Choose feature set 
    feature_cols = [
        "hour",
        "dow",
        "is_weekend",
        "lag_24_price",
        "lag_168_price",
        "roll_mean_24",
        "roll_std_24",
        "roll_mean_168",
        "lag_24_wind",
        "lag_24_temp",
    ]

    # Baselines
    base = baseline_predictions(df_feat)
    base_eval = base.dropna(subset=["y", "pred_naive_24", "pred_blend_24_168"])

    # Model walk forward
    model_preds = walk_forward_backtest(df_feat, feature_cols, test_days=cfg.test_days)

    # baseline evaluation to same days as model predictions
    eval_days = sorted(model_preds["settlementDate"].unique())
    base_eval_same = base_eval[base_eval["settlementDate"].isin(eval_days)].copy()

    metrics = {
        "baseline_naive_24": compute_metrics(base_eval_same["y"].values, base_eval_same["pred_naive_24"].values),
        "baseline_blend_24_168": compute_metrics(base_eval_same["y"].values, base_eval_same["pred_blend_24_168"].values),
        "improved_model": compute_metrics(model_preds["y"].values, model_preds["y_pred_model"].values),
        "eval_days": eval_days,
        "n_eval_rows": int(len(model_preds)),
    }

    # Save metrics
    (reports / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    # Save predictions 
    out = model_preds.copy()
    out["id"] = out["settlementDate"].astype(str) + "_" + out["hour"].astype(int).astype(str).str.zfill(2)
    out = out[["id", "settlementDate", "hour", "y", "y_pred_model"]]
    out.to_csv(reports / "submission.csv", index=False)

    # Plot: actual vs predicted for last ~3 days in eval window
    plot_days = eval_days[-3:] if len(eval_days) >= 3 else eval_days
    plot_df = model_preds[model_preds["settlementDate"].isin(plot_days)].copy()
    plot_df["ts"] = pd.to_datetime(plot_df["settlementDate"]) + pd.to_timedelta(plot_df["hour"] - 1, unit="h")
    plot_df = plot_df.sort_values("ts")

    plt.figure()
    plt.plot(plot_df["ts"], plot_df["y"], label="Actual")
    plt.plot(plot_df["ts"], plot_df["y_pred_model"], label="Predicted")
    plt.legend()
    plt.title("Actual vs Predicted (last days of evaluation window)")
    plt.xticks(rotation=30)
    plt.tight_layout()
    plt.savefig(reports / "actual_vs_pred.png")
    plt.close()

    # Print key results
    print("\nSaved:")
    print(" - reports/metrics.json")
    print(" - reports/submission.csv")
    print(" - reports/actual_vs_pred.png")
    print("\nMetrics summary:")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    run_forecast(BacktestConfig())