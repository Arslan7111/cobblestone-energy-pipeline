# app.py — Bloomberg-style Streamlit dashboard (dark, dense, terminal-like)
import json
from pathlib import Path
from datetime import datetime

import pandas as pd
import streamlit as st


# -----------------------------
# Page config
# -----------------------------
st.set_page_config(page_title="GB POWER FV | Terminal", layout="wide")

REPORTS = Path("reports")
LOGS = Path("logs")
DATA = Path("data/processed")


# -----------------------------
# Helpers
# -----------------------------
def safe_read_json(path: Path):
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def safe_read_csv(path: Path):
    if not path.exists():
        return None
    return pd.read_csv(path)


def safe_read_parquet(path: Path):
    if not path.exists():
        return None
    return pd.read_parquet(path)


def latest_daily_note_path():
    notes = sorted(REPORTS.glob("daily_note_*.md"))
    return notes[-1] if notes else None


def latest_llm_log_line():
    log_path = LOGS / "llm_calls.jsonl"
    if not log_path.exists():
        return None
    lines = log_path.read_text(encoding="utf-8").strip().splitlines()
    if not lines:
        return None
    return json.loads(lines[-1])


def fmt(x, nd=2):
    if x is None:
        return "—"
    try:
        return f"{float(x):.{nd}f}"
    except Exception:
        return str(x)


def status_badge(ok: bool) -> str:
    return "AVAILABLE" if ok else "MISSING"


def safe_minmax_dates(df: pd.DataFrame):
    if df is None or df.empty or "settlementDate" not in df.columns:
        return ("—", "—")
    return (str(df["settlementDate"].min()), str(df["settlementDate"].max()))


def safe_rows(df: pd.DataFrame):
    if df is None:
        return "—"
    return f"{len(df):,}"


# -----------------------------
# Bloomberg-style CSS
# -----------------------------
st.markdown(
    """
<style>
/* --- Terminal palette --- */
:root{
  --bg: #0b0f14;
  --panel: #111822;
  --panel2: #0f1620;
  --text: #e7eef8;
  --muted: rgba(231,238,248,0.70);
  --muted2: rgba(231,238,248,0.55);
  --border: rgba(231,238,248,0.10);
  --accent: #ff9900;       /* Bloomberg-ish orange */
  --accent2: #f2b84b;
  --danger: #f06464;
  --ok: #62c98b;
}

/* Streamlit base */
html, body, [class*="stApp"] {
  background: var(--bg) !important;
  color: var(--text) !important;
}

.block-container {
  padding-top: 0.75rem !important;
  padding-bottom: 2rem !important;
  max-width: 1400px !important;
}

h1, h2, h3, h4, h5, h6, p, div, span, label {
  color: var(--text) !important;
}

small, .stCaption, .stMarkdown p {
  color: var(--muted) !important;
}

/* Reduce whitespace / make dense */
div[data-testid="stVerticalBlock"] > div {
  gap: 0.6rem !important;
}
div[data-testid="stHorizontalBlock"] > div {
  gap: 0.6rem !important;
}

/* Panels */
.bb-panel {
  background: linear-gradient(180deg, var(--panel), var(--panel2));
  border: 1px solid var(--border);
  border-radius: 14px;
  padding: 12px 14px;
}
.bb-title {
  display:flex;
  justify-content:space-between;
  align-items:center;
  gap: 10px;
  margin-bottom: 8px;
}
.bb-title-left {
  font-size: 12px;
  letter-spacing: 1px;
  text-transform: uppercase;
  color: var(--muted) !important;
}
.bb-badge {
  font-size: 11px;
  padding: 2px 8px;
  border-radius: 999px;
  border: 1px solid var(--border);
  color: var(--muted) !important;
}
.bb-badge.ok { border-color: rgba(98,201,139,0.55); color: rgba(98,201,139,0.95) !important; }
.bb-badge.miss { border-color: rgba(240,100,100,0.55); color: rgba(240,100,100,0.95) !important; }

.bb-kpi {
  display:flex;
  flex-direction:column;
  gap: 2px;
}
.bb-kpi .k { font-size: 11px; color: var(--muted2) !important; text-transform: uppercase; letter-spacing: 0.8px;}
.bb-kpi .v { font-size: 22px; font-weight: 700; line-height: 1.1; }
.bb-kpi .s { font-size: 12px; color: var(--muted) !important; }

.bb-accent { color: var(--accent) !important; }
.bb-muted { color: var(--muted) !important; }

/* Header bar */
.bb-header {
  background: rgba(17,24,34,0.70);
  border: 1px solid var(--border);
  border-radius: 14px;
  padding: 10px 14px;
  margin-bottom: 10px;
}
.bb-header-top {
  display:flex;
  justify-content:space-between;
  align-items:center;
  gap: 14px;
}
.bb-brand {
  font-weight: 800;
  letter-spacing: 1px;
  font-size: 16px;
}
.bb-sub {
  font-size: 12px;
  color: var(--muted) !important;
}
.bb-ticker {
  margin-top: 10px;
  padding: 8px 10px;
  border-radius: 12px;
  border: 1px solid var(--border);
  background: rgba(15,22,32,0.70);
  overflow: hidden;
  white-space: nowrap;
}
.bb-ticker span {
  display: inline-block;
  padding-left: 100%;
  animation: bbscroll 16s linear infinite;
  color: var(--muted) !important;
}
@keyframes bbscroll {
  0% { transform: translateX(0); }
  100% { transform: translateX(-100%); }
}

/* Dataframe containers (subtle border) */
div[data-testid="stDataFrame"] {
  border: 1px solid var(--border);
  border-radius: 12px;
  overflow: hidden;
}
</style>
""",
    unsafe_allow_html=True,
)


# -----------------------------
# Load artifacts
# -----------------------------
qa = safe_read_json(LOGS / "qa_report.json")
metrics = safe_read_json(REPORTS / "metrics.json")
prompt_summary = safe_read_json(REPORTS / "prompt_curve_summary.json")
hourly_view = safe_read_csv(REPORTS / "prompt_curve_hourly_view.csv")
preds = safe_read_csv(REPORTS / "submission.csv")
processed = safe_read_parquet(DATA / "gb_hourly_dataset.parquet")

note_path = latest_daily_note_path()
llm_last = latest_llm_log_line()

plot_actual_vs_pred = REPORTS / "actual_vs_pred.png"
plot_prompt_curve = REPORTS / "prompt_curve_plot.png"
trading_guidance = REPORTS / "trading_guidance.md"


# -----------------------------
# Small UI helpers (HTML panels)
# -----------------------------
def panel(title: str, body_html: str, badge: str | None = None):
    badge_html = f'<span class="bb-badge">{badge}</span>' if badge else ""
    st.markdown(
        f"""
<div class="bb-panel">
  <div class="bb-title">
    <div class="bb-title-left">{title}</div>
    {badge_html}
  </div>
  {body_html}
</div>
""",
        unsafe_allow_html=True,
    )


def kpi_row(items):
    # items = [(label, value, subtext, accent_bool), ...]
    cols = st.columns(len(items))
    for c, (k, v, s, accent) in zip(cols, items):
        cls = "bb-accent" if accent else ""
        c.markdown(
            f"""
<div class="bb-panel">
  <div class="bb-kpi">
    <div class="k">{k}</div>
    <div class="v {cls}">{v}</div>
    <div class="s">{s}</div>
  </div>
</div>
""",
            unsafe_allow_html=True,
        )


# -----------------------------
# Header bar (Bloomberg-ish)
# -----------------------------
now_utc = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")

data_min, data_max = safe_minmax_dates(processed)
forecast_day = prompt_summary["forecast_day"] if prompt_summary else "—"

header_ticker = " | ".join(
    [
        f"WINDOW {data_min}→{data_max}",
        f"FORECAST DAY {forecast_day}",
        f"RMSE {fmt(prompt_summary['model_rmse']) if prompt_summary else '—'}",
        f"ACTION BAND ±{fmt(prompt_summary['action_threshold']) if prompt_summary else '—'}",
        f"BASE SPREAD {fmt(prompt_summary['spread_base']) if prompt_summary else '—'}",
        f"PEAK SPREAD {fmt(prompt_summary['spread_peak']) if prompt_summary else '—'}",
    ]
)

st.markdown(
    f"""
<div class="bb-header">
  <div class="bb-header-top">
    <div>
      <div class="bb-brand">GB POWER FV <span class="bb-accent">TERMINAL</span></div>
      <div class="bb-sub">Fair value forecasting, prompt proxy translation, and LLM desk note generation</div>
    </div>
    <div class="bb-sub">{now_utc}</div>
  </div>
  <div class="bb-ticker"><span>{header_ticker}</span></div>
</div>
""",
    unsafe_allow_html=True,
)


# -----------------------------
# Controls (top row)
# -----------------------------
c1, c2, c3, c4 = st.columns([1.2, 1.2, 1.2, 1.4])

with c1:
    st.write("**View**")
    view = st.selectbox("Section", ["Overview", "Data QA", "Forecasting", "Prompt View", "LLM Note"], label_visibility="collapsed")

with c2:
    st.write("**Forecast day**")
    if preds is not None and not preds.empty:
        days = sorted(preds["settlementDate"].unique())
        sel_day = st.selectbox("Forecast day", days, index=len(days)-1, label_visibility="collapsed")
    else:
        sel_day = None
        st.selectbox("Forecast day", ["—"], label_visibility="collapsed")

with c3:
    st.write("**Artifacts**")
    st.caption(f"QA: {status_badge((LOGS / 'qa_report.json').exists())}  •  Metrics: {status_badge((REPORTS / 'metrics.json').exists())}")

with c4:
    st.write("**Files**")
    st.caption(f"Note: {note_path.name if note_path else '—'}  •  LLM log: {status_badge((LOGS / 'llm_calls.jsonl').exists())}")


# -----------------------------
# Overview
# -----------------------------
if view == "Overview":
    if prompt_summary:
        kpi_row([
            ("Forecast Baseload FV", fmt(prompt_summary["forecast_base"]), "average of 24 hours", True),
            ("Prompt Proxy Baseload", fmt(prompt_summary["prompt_proxy_base"]), "last 7 available full days", False),
            ("Baseload Spread", fmt(prompt_summary["spread_base"]), prompt_summary.get("signal_base", ""), True),
            ("Peak Spread", fmt(prompt_summary["spread_peak"]), prompt_summary.get("signal_peak", ""), True),
        ])

        st.write("")
        left, right = st.columns([1.25, 1.0])

        with left:
            if plot_prompt_curve.exists():
                panel(
                    "Hourly shape: Forecast FV vs Prompt Proxy",
                    f'<div class="bb-muted">Forecast day: {prompt_summary["forecast_day"]} | Peak: {prompt_summary.get("peak_definition","—")}</div>',
                    badge="CHART",
                )
                st.image(str(plot_prompt_curve), use_container_width=True)
            else:
                panel("Hourly shape", '<div class="bb-muted">Missing: reports/prompt_curve_plot.png</div>', badge="MISSING")

        with right:
            proxy_days = prompt_summary.get("proxy_days_used", [])
            panel(
                "Prompt proxy construction",
                "<div class='bb-muted'>Proxy is the average hourly profile over the last 7 available full days (pre-forecast day). "
                "This is a prototype stand-in for a true traded prompt curve snapshot.</div>"
                + f"<div style='margin-top:10px; font-size:12px; color: var(--muted)'>Days: {', '.join(proxy_days)}</div>",
                badge="METHOD",
            )

            panel(
                "Signal framing",
                "<div class='bb-muted'>Signals use an action band = 0.5 × RMSE to avoid over-trading noise. "
                "For production, replace prompt proxy with broker/exchange curve and add outage/interconnector features.</div>",
                badge="NOTE",
            )
    else:
        panel("Overview", "<div class='bb-muted'>Prompt summary not found. Run prompt-curve step first.</div>", badge="MISSING")


# -----------------------------
# Data QA
# -----------------------------
elif view == "Data QA":
    left, right = st.columns([1.0, 1.0])

    with left:
        if qa:
            kpi_row([
                ("Rows", f"{qa.get('n_rows', 0):,}", "processed hourly rows", False),
                ("Date min", qa.get("date_min", "—"), "from qa_report.json", False),
                ("Date max", qa.get("date_max", "—"), "from qa_report.json", False),
                ("Duplicates", str(qa.get("duplicate_date_hour_rows", "—")), "date-hour duplicates", False),
            ])
            panel(
                "Hours per day",
                f"<div class='bb-muted'>{qa.get('hours_per_day_counts', {})}</div>",
                badge="CHECK",
            )
            panel(
                "Missing values (by column)",
                "<div class='bb-muted'>All key columns should be 0 for modeling readiness.</div>",
                badge="CHECK",
            )
            st.dataframe(pd.DataFrame([qa.get("missing_by_col", {})]), use_container_width=True, height=180)
        else:
            panel("QA report", "<div class='bb-muted'>Missing logs/qa_report.json</div>", badge="MISSING")

    with right:
        if processed is not None and not processed.empty:
            panel("Dataset preview", "<div class='bb-muted'>Top rows from data/processed/gb_hourly_dataset.parquet</div>", badge="DATA")
            st.dataframe(processed.head(60), use_container_width=True, height=520)
        else:
            panel("Dataset preview", "<div class='bb-muted'>Missing processed parquet</div>", badge="MISSING")


# -----------------------------
# Forecasting
# -----------------------------
elif view == "Forecasting":
    left, right = st.columns([1.05, 0.95])

    with left:
        if metrics:
            rows = []
            for k in ["baseline_naive_24", "baseline_blend_24_168", "improved_model"]:
                if k in metrics:
                    rows.append({"Model": k, "MAE": metrics[k]["MAE"], "RMSE": metrics[k]["RMSE"]})
            panel("Model performance", "<div class='bb-muted'>Walk-forward daily evaluation</div>", badge="METRICS")
            st.dataframe(pd.DataFrame(rows), use_container_width=True, height=160)

            panel("Evaluation window", f"<div class='bb-muted'>{metrics.get('eval_days', [])}</div>", badge="INFO")
        else:
            panel("Model performance", "<div class='bb-muted'>Missing reports/metrics.json</div>", badge="MISSING")

        if plot_actual_vs_pred.exists():
            panel("Actual vs predicted", "<div class='bb-muted'>Last days of evaluation window</div>", badge="CHART")
            st.image(str(plot_actual_vs_pred), use_container_width=True)
        else:
            panel("Actual vs predicted", "<div class='bb-muted'>Missing reports/actual_vs_pred.png</div>", badge="MISSING")

    with right:
        if preds is not None and not preds.empty and sel_day:
            df_day = preds[preds["settlementDate"] == sel_day].copy().sort_values("hour")
            panel("Hourly predictions", f"<div class='bb-muted'>Forecast day: {sel_day}</div>", badge="TABLE")
            st.dataframe(df_day, use_container_width=True, height=640)
        else:
            panel("Hourly predictions", "<div class='bb-muted'>Missing reports/submission.csv</div>", badge="MISSING")


# -----------------------------
# Prompt View
# -----------------------------
elif view == "Prompt View":
    left, right = st.columns([1.05, 0.95])

    with left:
        if prompt_summary:
            kpi_row([
                ("Forecast Base FV", fmt(prompt_summary["forecast_base"]), "avg 24h", True),
                ("Prompt Proxy Base", fmt(prompt_summary["prompt_proxy_base"]), "7d hourly avg", False),
                ("Forecast Peak FV", fmt(prompt_summary["forecast_peak"]), "hours 9..20", True),
                ("Prompt Proxy Peak", fmt(prompt_summary["prompt_proxy_peak"]), "7d hourly avg", False),
            ])
        else:
            panel("Prompt summary", "<div class='bb-muted'>Missing reports/prompt_curve_summary.json</div>", badge="MISSING")

        if plot_prompt_curve.exists():
            panel("Curve view", "<div class='bb-muted'>Forecast FV vs prompt proxy hourly profile</div>", badge="CHART")
            st.image(str(plot_prompt_curve), use_container_width=True)
        else:
            panel("Curve view", "<div class='bb-muted'>Missing reports/prompt_curve_plot.png</div>", badge="MISSING")

        if trading_guidance.exists():
            panel("Trading guidance", "<div class='bb-muted'>Short desk guidance generated by pipeline</div>", badge="TEXT")
            st.markdown(trading_guidance.read_text(encoding="utf-8"))
        else:
            panel("Trading guidance", "<div class='bb-muted'>Missing reports/trading_guidance.md</div>", badge="MISSING")

    with right:
        if hourly_view is not None and not hourly_view.empty:
            panel("Hourly FV vs prompt proxy", "<div class='bb-muted'>Includes hourly spread (FV − proxy)</div>", badge="TABLE")
            st.dataframe(hourly_view.sort_values("hour"), use_container_width=True, height=520)

            tmp = hourly_view.dropna(subset=["spread_vs_prompt_hourly"]).copy()
            cheap = tmp.sort_values("spread_vs_prompt_hourly").head(5)
            rich = tmp.sort_values("spread_vs_prompt_hourly", ascending=False).head(5)

            panel("Largest deviations", "<div class='bb-muted'>Cheapest and richest hours vs proxy</div>", badge="RANK")
            cA, cB = st.columns(2)
            with cA:
                st.caption("Cheapest hours")
                st.dataframe(cheap, use_container_width=True, height=220)
            with cB:
                st.caption("Richest hours")
                st.dataframe(rich, use_container_width=True, height=220)
        else:
            panel("Hourly FV vs prompt proxy", "<div class='bb-muted'>Missing reports/prompt_curve_hourly_view.csv</div>", badge="MISSING")


# -----------------------------
# LLM Note
# -----------------------------
else:  # "LLM Note"
    left, right = st.columns([1.15, 0.85])

    with left:
        if note_path and note_path.exists():
            panel("Daily desk note (Gemini)", f"<div class='bb-muted'>{note_path.name}</div>", badge="NOTE")
            st.markdown(note_path.read_text(encoding="utf-8"))
        else:
            panel("Daily desk note", "<div class='bb-muted'>Missing reports/daily_note_*.md</div>", badge="MISSING")

    with right:
        panel("LLM metadata", "<div class='bb-muted'>Latest call metadata (safe view)</div>", badge="LOG")
        if llm_last:
            safe_view = {
                "ts_utc": llm_last.get("ts_utc"),
                "provider": llm_last.get("provider"),
                "model": llm_last.get("model"),
                "finishReason": (
                    llm_last.get("raw_response", {})
                    .get("candidates", [{}])[0]
                    .get("finishReason", "—")
                    if isinstance(llm_last.get("raw_response", {}), dict)
                    else "—"
                ),
                "output_preview": (llm_last.get("output", "")[:500] + "...") if llm_last.get("output") else None,
            }
            st.json(safe_view, expanded=False)
        else:
            st.caption("No logs/llm_calls.jsonl found.")