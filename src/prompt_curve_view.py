import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def peak_hours_mask(hours: pd.Series) -> pd.Series:
    """
    Define PEAK as 08:00–20:00 (UK time).
    Since hour=1 means 00:00–01:00, 08:00–20:00 corresponds to hours 9..20.
    """
    return (hours >= 9) & (hours <= 20)


def load_rmse(metrics_path: str = "reports/metrics.json") -> float:
    m = json.loads(Path(metrics_path).read_text(encoding="utf-8"))
    return float(m["improved_model"]["RMSE"])


def compute_base_peak(df: pd.DataFrame, price_col: str) -> dict:
    base = float(df[price_col].mean())
    peak = float(df.loc[peak_hours_mask(df["hour"]), price_col].mean())
    return {"base": base, "peak": peak}


def last_n_full_days(df_hist: pd.DataFrame, before_date: str, n: int = 7) -> list:
    """
    Returns the last n dates (strings) that have 24 hours, strictly before before_date.
    """
    hist = df_hist[df_hist["settlementDate"] < before_date].copy()
    counts = hist.groupby("settlementDate")["hour"].nunique()
    full_days = counts[counts == 24].index.tolist()
    full_days = sorted(full_days)
    return full_days[-n:]


def run_prompt_curve_view(
    processed_path: str = "data/processed/gb_hourly_dataset.parquet",
    preds_path: str = "reports/submission.csv",
    metrics_path: str = "reports/metrics.json",
    out_dir: str = "reports",
) -> None:
    outp = Path(out_dir)
    outp.mkdir(parents=True, exist_ok=True)

    df_hist = pd.read_parquet(processed_path)
    preds = pd.read_csv(preds_path)

    #  latest forecast day available in submission.csv
    forecast_day = sorted(preds["settlementDate"].unique())[-1]
    df_fc = preds[preds["settlementDate"] == forecast_day].copy()
    df_fc["hour"] = df_fc["hour"].astype(int)
    df_fc = df_fc.sort_values("hour")

    #  24 hours
    if df_fc["hour"].nunique() != 24:
        raise ValueError(f"Forecast day {forecast_day} is not a full 24-hour day in predictions.")

    #  prompt proxy using last 7 available full days before forecast_day
    proxy_days = last_n_full_days(df_hist, before_date=forecast_day, n=7)
    if len(proxy_days) < 3:
        raise ValueError(f"Not enough history before {forecast_day} to compute prompt proxy. Found {len(proxy_days)} days.")

    df_proxy_hist = df_hist[df_hist["settlementDate"].isin(proxy_days)].copy()

    # baseload/peak = average over last 7 available days
    proxy_base_peak = compute_base_peak(df_proxy_hist, "da_price")

    
    proxy_hourly = (
        df_proxy_hist.groupby("hour", as_index=False)["da_price"]
        .mean()
        .rename(columns={"da_price": "prompt_proxy_hourly"})
    )

    # Forecast baseload/peak
    fc_base_peak = compute_base_peak(df_fc, "y_pred_model")

    # Spread
    spread_base = fc_base_peak["base"] - proxy_base_peak["base"]
    spread_peak = fc_base_peak["peak"] - proxy_base_peak["peak"]

    # Threshold: using half RMSE as a simple “action band”
    rmse = load_rmse(metrics_path)
    threshold = 0.5 * rmse

    def signal(x: float) -> str:
        if x > threshold:
            return "DA RICH vs prompt → SELL DA / BUY prompt (or reduce longs)"
        if x < -threshold:
            return "DA CHEAP vs prompt → BUY DA / SELL prompt (or add longs)"
        return "NEUTRAL → no strong edge"

    signal_base = signal(spread_base)
    signal_peak = signal(spread_peak)

    # Assemble hourly view table
    out_hourly = df_fc[["settlementDate", "hour", "y_pred_model"]].merge(proxy_hourly, on="hour", how="left")
    out_hourly["spread_vs_prompt_hourly"] = out_hourly["y_pred_model"] - out_hourly["prompt_proxy_hourly"]

    #  CSV
    out_hourly.to_csv(outp / "prompt_curve_hourly_view.csv", index=False)

    #  JSON summary
    summary = {
        "forecast_day": forecast_day,
        "proxy_days_used": proxy_days,
        "model_rmse": rmse,
        "action_threshold": threshold,
        "forecast_base": fc_base_peak["base"],
        "forecast_peak": fc_base_peak["peak"],
        "prompt_proxy_base": proxy_base_peak["base"],
        "prompt_proxy_peak": proxy_base_peak["peak"],
        "spread_base": spread_base,
        "spread_peak": spread_peak,
        "signal_base": signal_base,
        "signal_peak": signal_peak,
        "peak_definition": "08:00–20:00 => hours 9..20",
    }
    (outp / "prompt_curve_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    # Plot hourly forecast vs prompt proxy hourly profile
    plt.figure()
    plt.plot(out_hourly["hour"], out_hourly["y_pred_model"], label="Forecast FV (DA hourly)")
    plt.plot(out_hourly["hour"], out_hourly["prompt_proxy_hourly"], label="Prompt proxy (last 7 days hourly avg)")
    plt.title(f"DA Fair Value vs Prompt Proxy — {forecast_day}")
    plt.xlabel("Hour (1=00:00-01:00)")
    plt.ylabel("Price")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outp / "prompt_curve_plot.png")
    plt.close()

    
    note = f"""# Prompt curve translation (prototype)

**Forecast day:** {forecast_day}  
**Prompt proxy:** average of last 7 *available full days* before {forecast_day}: {', '.join(proxy_days)}

## Fair value summary
- Forecast **Baseload FV**: {fc_base_peak["base"]:.2f}
- Forecast **Peak FV** (08:00–20:00): {fc_base_peak["peak"]:.2f}
- Prompt proxy **Baseload**: {proxy_base_peak["base"]:.2f}
- Prompt proxy **Peak**: {proxy_base_peak["peak"]:.2f}

## DA-to-prompt view
- Baseload spread (FV - prompt): {spread_base:.2f}  → **{signal_base}**
- Peak spread (FV - prompt): {spread_peak:.2f}  → **{signal_peak}**

**Action band:** ±{threshold:.2f} (0.5 × model RMSE={rmse:.2f})

## How this would be used (and invalidated)
Use:
- Express “DA vs prompt” view using baseload/peak spreads and hourly shape (see `prompt_curve_hourly_view.csv`).
- Size trades only when spread exceeds the action band and aligns with fundamentals (wind/temperature regime).

Invalidate / de-risk when:
- Data QA flags missing days / incomplete hours.
- Exceptional system events (large outages, interconnector constraints) not represented in drivers.
- Large forecast error regime: recent RMSE spikes vs normal.
"""
    (outp / "trading_guidance.md").write_text(note, encoding="utf-8")

    print("Saved:")
    print(" - reports/prompt_curve_summary.json")
    print(" - reports/prompt_curve_hourly_view.csv")
    print(" - reports/prompt_curve_plot.png")
    print(" - reports/trading_guidance.md")
    print("\nKey signals:")
    print(json.dumps({k: summary[k] for k in ["forecast_day", "spread_base", "signal_base", "spread_peak", "signal_peak"]}, indent=2))


if __name__ == "__main__":
    run_prompt_curve_view()