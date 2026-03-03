from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests

# Gemini generateContent REST endpoint (v1beta)
GEMINI_ENDPOINT_TMPL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"


def _print_block(title: str, text: str, max_chars: int | None = None) -> None:
    print("\n" + "=" * 90)
    print(title)
    print("=" * 90)
    if max_chars is not None and len(text) > max_chars:
        print(text[:max_chars] + "\n...[TRUNCATED]...")
    else:
        print(text)
    print("=" * 90 + "\n")


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def get_api_key() -> str:
    key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not key:
        raise RuntimeError(
            "Missing GEMINI_API_KEY (or GOOGLE_API_KEY). "
            "In VS Code terminal run: export GEMINI_API_KEY='YOUR_KEY'"
        )
    return key


def load_summary(path: str = "reports/prompt_curve_summary.json") -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_hourly_view(path: str = "reports/prompt_curve_hourly_view.csv") -> pd.DataFrame:
    return pd.read_csv(path)


def top_spreads(df: pd.DataFrame, n: int = 3) -> tuple[pd.DataFrame, pd.DataFrame]:
    x = df.dropna(subset=["spread_vs_prompt_hourly"]).copy()
    richest = x.sort_values("spread_vs_prompt_hourly", ascending=False).head(n)
    cheapest = x.sort_values("spread_vs_prompt_hourly", ascending=True).head(n)
    return richest, cheapest


def build_prompt(summary: dict, hourly: pd.DataFrame) -> str:
    forecast_day = summary["forecast_day"]
    richest, cheapest = top_spreads(hourly, n=3)

    def fmt_rows(d: pd.DataFrame) -> str:
        rows = []
        for _, r in d.iterrows():
            rows.append(
                f"- hour={int(r['hour'])}, fv={float(r['y_pred_model']):.2f}, "
                f"prompt={float(r['prompt_proxy_hourly']):.2f}, spread={float(r['spread_vs_prompt_hourly']):.2f}"
            )
        return "\n".join(rows)

    peak_definition = summary.get("peak_definition", "08:00–20:00 (hours 9..20)")
    model_rmse = float(summary["model_rmse"])
    action_threshold = float(summary["action_threshold"])
    forecast_base = float(summary["forecast_base"])
    forecast_peak = float(summary["forecast_peak"])
    prompt_proxy_base = float(summary["prompt_proxy_base"])
    prompt_proxy_peak = float(summary["prompt_proxy_peak"])
    spread_base = float(summary["spread_base"])
    spread_peak = float(summary["spread_peak"])
    proxy_days_used = ", ".join(summary["proxy_days_used"])

    top_rich_hours = fmt_rows(richest)
    top_cheap_hours = fmt_rows(cheapest)

    return f"""
You are an energy trading analyst writing a short daily desk note (max ~180 words).
Write in clear, professional tone. No fluff. Return Markdown only.

IMPORTANT TRADING INTERPRETATION RULE (must follow):
- Define spread = (Forecast FV − Prompt Proxy).
- If spread is NEGATIVE → DA is CHEAP vs prompt → bias is BUY DA / SELL prompt.
- If spread is POSITIVE → DA is RICH vs prompt → bias is SELL DA / BUY prompt.
- For hourly comments, apply the same rule to each hour’s spread.

Context:
- Market: GB power
- Forecast day: {forecast_day}
- Peak definition: {peak_definition}
- Model RMSE: {model_rmse:.2f}; action threshold: ±{action_threshold:.2f}
- Forecast baseload FV: {forecast_base:.2f}
- Forecast peak FV: {forecast_peak:.2f}
- Prompt proxy baseload: {prompt_proxy_base:.2f}
- Prompt proxy peak: {prompt_proxy_peak:.2f}
- Baseload spread (FV - prompt): {spread_base:.2f}
- Peak spread (FV - prompt): {spread_peak:.2f}
- Prompt proxy days used: {proxy_days_used}

Hourly shape highlights:
Top 3 RICHEST hours (highest positive spread):
{top_rich_hours}

Top 3 CHEAPEST hours (most negative spread):
{top_cheap_hours}

Task:
Produce a desk note with these sections (in this order):
1) Headline (one line)
2) What the model says (2–3 bullets: base/peak + shape)
3) Trade implication (1–2 bullets; must use the interpretation rule; mention “neutral/borderline” if spread magnitude < threshold)
4) Invalidation checks (2 bullets; concrete reasons to ignore or de-risk the view)

Constraints:
- Do NOT invent new numbers.
- Use the values given in Context/Hourly highlights.
- Keep within ~180 words.
""".strip()


def call_gemini_rest(
    prompt: str,
    model: str = "gemini-2.5-flash",
    max_output_tokens: int = 3000,
    temperature: float = 0.2,
) -> dict:
    key = get_api_key()
    url = GEMINI_ENDPOINT_TMPL.format(model=model) + f"?key={key}"

    body = {
        "contents": [{"role": "user", "parts": [{"text": prompt}]}],
        "generationConfig": {
            "maxOutputTokens": max_output_tokens,
            "temperature": temperature,
            
            "thinkingConfig": {"thinkingBudget": 0},
        },
    }

    r = requests.post(url, headers={"Content-Type": "application/json"}, json=body, timeout=90)

    try:
        data = r.json()
    except Exception:
        data = {"raw_text": r.text}

    if r.status_code != 200:
        raise RuntimeError(f"Gemini REST error {r.status_code}: {data}")

    return data


def extract_text(resp_json: dict) -> str:
    cands = resp_json.get("candidates", [])
    if not cands:
        return json.dumps(resp_json, indent=2)

    parts = cands[0].get("content", {}).get("parts", [])
    texts = [p.get("text", "") for p in parts if isinstance(p, dict)]
    out = "\n".join([t for t in texts if t.strip()])
    return out or json.dumps(resp_json, indent=2)


def log_call(
    prompt: str,
    output_text: str,
    model: str,
    raw_response: dict,
    log_path: str = "logs/llm_calls.jsonl",
) -> None:
    Path("logs").mkdir(parents=True, exist_ok=True)
    record = {
        "ts_utc": utc_now_iso(),
        "provider": "gemini_rest",
        "model": model,
        "prompt": prompt,
        "output": output_text,
        "raw_response": raw_response,
    }
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def run_llm_note(
    summary_path: str = "reports/prompt_curve_summary.json",
    hourly_path: str = "reports/prompt_curve_hourly_view.csv",
    out_dir: str = "reports",
    model: str = "gemini-2.5-flash",
    show_prompt: bool = False,
) -> None:
    summary = load_summary(summary_path)
    hourly = load_hourly_view(hourly_path)

    prompt = build_prompt(summary, hourly)
    resp_json = call_gemini_rest(prompt, model=model, max_output_tokens=3000, temperature=0.2)
    output_text = extract_text(resp_json)

    # Save files first 
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    forecast_day = summary["forecast_day"]
    note_path = Path(out_dir) / f"daily_note_{forecast_day}.md"
    note_path.write_text(output_text, encoding="utf-8")

    log_call(prompt, output_text, model=model, raw_response=resp_json)

    # Terminal display 
    if show_prompt:
        _print_block("PROMPT SENT TO GEMINI (FULL)", prompt, max_chars=None)
    else:
        _print_block("PROMPT PREVIEW (use --show_prompt for full)", prompt, max_chars=1200)

    _print_block("GEMINI OUTPUT (DAILY NOTE)", output_text, max_chars=None)

    finish = "UNKNOWN"
    try:
        finish = resp_json.get("candidates", [{}])[0].get("finishReason", "UNKNOWN")
    except Exception:
        pass

    _print_block(
        "RESPONSE META",
        f"model={model}\nfinishReason={finish}\nsaved_note={note_path}\nlog=logs/llm_calls.jsonl",
        max_chars=None,
    )

    print("Saved:")
    print(f" - {note_path}")
    print(" - logs/llm_calls.jsonl")