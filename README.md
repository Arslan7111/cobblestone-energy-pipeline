# Cobblestone Energy Pipeline

An end-to-end machine learning pipeline for **GB day-ahead power price forecasting** and **energy trading signal generation**. The system ingests real-time market data from the Elexon BMRS API, builds predictive models for hourly electricity prices, compares forecasts against prompt proxy curves, and produces actionable trading guidance — including LLM-generated desk notes via Google Gemini.

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
- [Pipeline Steps](#pipeline-steps)
- [Dashboard](#dashboard)
- [Model Details](#model-details)
- [Data Sources](#data-sources)

## Overview

The pipeline automates the daily workflow of an energy trading desk:

1. **Ingest** GB power market data (prices, generation by fuel type, temperature)
2. **QA** the dataset for completeness and anomalies
3. **Forecast** day-ahead hourly prices using a gradient boosting model
4. **Generate trading signals** by comparing forecasts to a prompt proxy curve
5. **Produce a desk note** using an LLM for trader consumption
6. **Visualize** everything in an interactive Streamlit dashboard

## Architecture

```
Elexon BMRS API
       |
  [Data Ingestion] ──> data/raw/*.parquet
       |
  [QA Checks] ──> logs/qa_report.json
       |
  [Feature Engineering + Forecasting] ──> reports/metrics.json, submission.csv
       |
  [Prompt Curve Analysis] ──> reports/prompt_curve_summary.json, trading_guidance.md
       |
  [LLM Desk Note (Gemini)] ──> reports/daily_note_*.md
       |
  [Streamlit Dashboard] ──> http://localhost:8501
```

## Project Structure

```
cobblestone-energy-pipeline/
├── app.py                        # Streamlit dashboard (Bloomberg-style UI)
├── run_pipeline.py               # Data ingestion & QA orchestrator
├── run_forecast.py               # Forecasting runner
├── run_prompt_curve_view.py      # Prompt curve analysis runner
├── run_llm_note.py               # LLM desk note generator
├── requirements.txt              # Python dependencies
├── src/
│   ├── ingest_elexon.py          # Elexon BMRS API client & data processing
│   ├── qa.py                     # Data quality assurance checks
│   ├── forecast.py               # ML model, feature engineering, backtesting
│   ├── prompt_curve_view.py      # Trading signal generation
│   ├── llm_note_gemini_rest.py   # Gemini API integration for desk notes
│   └── utils.py                  # Shared helpers (retries, date chunking, I/O)
├── data/
│   ├── raw/                      # Raw API data (parquet)
│   └── processed/                # Cleaned hourly dataset (parquet)
├── reports/                      # Model outputs, charts, trading guidance
└── logs/                         # QA reports, LLM call logs
```

## Installation

### Prerequisites

- Python 3.9+
- pip

### Setup

```bash
git clone <repository-url>
cd cobblestone-energy-pipeline

pip install -r requirements.txt
pip install streamlit scikit-learn
```

## Configuration

### Environment Variables

| Variable | Required | Description |
|---|---|---|
| `GEMINI_API_KEY` | For LLM step only | Google Gemini API key for desk note generation |
| `GOOGLE_API_KEY` | Alternative | Alternative env var name for the Gemini key |

The Elexon BMRS API is public and requires no authentication.

### Pipeline Parameters

| Parameter | Default | Description |
|---|---|---|
| `--start` | `2026-01-29` | Start date for data ingestion |
| `--end` | `2026-02-28` | End date for data ingestion |
| `--test_days` | `7` | Number of days for walk-forward backtesting |
| `--show_prompt` | `false` | Print the full LLM prompt to stdout |

## Usage

### Run the Full Pipeline

Execute each step in sequence:

```bash
# 1. Ingest data from Elexon API & run QA
python run_pipeline.py --start 2026-01-29 --end 2026-02-28

# 2. Train model & generate forecasts
python run_forecast.py --test_days 7

# 3. Compare forecasts against prompt proxy & generate trading signals
python run_prompt_curve_view.py

# 4. Generate LLM desk note (requires GEMINI_API_KEY)
export GEMINI_API_KEY='your-key-here'
python run_llm_note.py

# 5. Launch the dashboard
streamlit run app.py
```

### Run Individual Steps

Each step is independent and can be run on its own, provided its input artifacts exist from a prior run.

## Pipeline Steps

### 1. Data Ingestion (`run_pipeline.py`)

Fetches three data streams from the Elexon BMRS REST API:

- **MID** — Market Index Price (day-ahead power prices)
- **FUELHH** — Half-hourly generation by fuel type (wind, solar, etc.)
- **TEMP** — Temperature observations

Data is fetched in 7-day chunks with exponential backoff retries, converted from 30-minute settlement periods to hourly resolution, merged into a single dataset, and saved as `data/processed/gb_hourly_dataset.parquet`.

**Outputs:** `data/raw/*.parquet`, `data/processed/gb_hourly_dataset.parquet`

### 2. Data QA (`run_pipeline.py`)

Validates the processed dataset:

- Row count and date range verification
- Missing value detection per column
- Duplicate date-hour checks
- Hours-per-day distribution
- Price statistics (min, p01, median, p99, max)
- Wind/solar and temperature coverage

**Output:** `logs/qa_report.json`

### 3. Forecasting (`run_forecast.py`)

Builds and evaluates a `HistGradientBoostingRegressor` model with walk-forward backtesting.

**Features engineered:**
- Calendar: hour-of-day, day-of-week, weekend flag
- Price lags: 24h, 48h, 168h (1 week)
- Rolling statistics: 24h and 168h mean/std (shifted to avoid leakage)
- Fundamental lags: wind+solar generation, temperature (24h lag)

**Baselines compared:**
- Naive-24 (same hour yesterday)
- Blend (50% yesterday + 50% same hour last week)

**Outputs:** `reports/metrics.json`, `reports/submission.csv`, `reports/actual_vs_pred.png`

### 4. Prompt Curve Analysis (`run_prompt_curve_view.py`)

Constructs a "prompt proxy" from the average hourly price profile over the last 7 available full days, then computes:

- **Baseload & peak fair values** (peak = hours 08:00–20:00)
- **Spreads** (forecast FV minus prompt proxy)
- **Trading signals**: `RICH` / `CHEAP` / `NEUTRAL` based on an action band of 0.5 x model RMSE

**Outputs:** `reports/prompt_curve_summary.json`, `reports/prompt_curve_hourly_view.csv`, `reports/prompt_curve_plot.png`, `reports/trading_guidance.md`

### 5. LLM Desk Note (`run_llm_note.py`)

Sends forecast results and trading signals to **Google Gemini (gemini-2.5-flash)** via REST API. The structured prompt instructs the LLM to produce a professional trading desk note with:

- A headline
- Model summary
- Trade implications
- Invalidation checks

All API calls are logged to `logs/llm_calls.jsonl` for auditability.

**Output:** `reports/daily_note_*.md`

## Dashboard

The Streamlit dashboard (`app.py`) provides a Bloomberg-inspired dark terminal UI with five views:

| Tab | Content |
|---|---|
| **Overview** | KPI cards (baseload FV, prompt proxy, spreads), hourly chart, method notes |
| **Data QA** | Row counts, date ranges, missing values, dataset preview |
| **Forecasting** | Model vs baseline metrics, actual vs predicted chart, predictions table |
| **Prompt View** | Baseload/peak KPIs, hourly spread chart, largest deviations, trading guidance |
| **LLM Note** | Full desk note, LLM metadata and call logs |

Launch with:

```bash
streamlit run app.py
# Opens at http://localhost:8501
```

## Model Details

| Parameter | Value |
|---|---|
| Algorithm | HistGradientBoostingRegressor |
| Max depth | 6 |
| Learning rate | 0.05 |
| Max iterations | 400 |
| Backtesting | Walk-forward by day |
| Leakage prevention | All features use lagged values; rolling stats shifted by 1 |

### Sample Performance (7-day backtest)

| Model | MAE (GBP/MWh) | RMSE (GBP/MWh) |
|---|---|---|
| Naive-24 | 13.20 | 17.04 |
| Blend (24h+168h) | 10.59 | 13.95 |
| **ML Model** | **10.41** | **13.34** |

## Data Sources

- **[Elexon BMRS API](https://data.elexon.co.uk/bmrs/api/v1)** — GB electricity market data (public, no authentication required)
- **[Google Gemini API](https://ai.google.dev/)** — LLM for desk note generation (API key required)