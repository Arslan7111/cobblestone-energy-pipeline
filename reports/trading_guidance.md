# Prompt curve translation (prototype)

**Forecast day:** 2026-02-27  
**Prompt proxy:** average of last 7 *available full days* before 2026-02-27: 2026-02-19, 2026-02-20, 2026-02-21, 2026-02-22, 2026-02-23, 2026-02-24, 2026-02-26

## Fair value summary
- Forecast **Baseload FV**: 28.45
- Forecast **Peak FV** (08:00–20:00): 37.27
- Prompt proxy **Baseload**: 34.61
- Prompt proxy **Peak**: 38.58

## DA-to-prompt view
- Baseload spread (FV - prompt): -6.16  → **NEUTRAL → no strong edge**
- Peak spread (FV - prompt): -1.31  → **NEUTRAL → no strong edge**

**Action band:** ±6.67 (0.5 × model RMSE=13.34)

## How this would be used (and invalidated)
Use:
- Express “DA vs prompt” view using baseload/peak spreads and hourly shape (see `prompt_curve_hourly_view.csv`).
- Size trades only when spread exceeds the action band and aligns with fundamentals (wind/temperature regime).

Invalidate / de-risk when:
- Data QA flags missing days / incomplete hours.
- Exceptional system events (large outages, interconnector constraints) not represented in drivers.
- Large forecast error regime: recent RMSE spikes vs normal.
