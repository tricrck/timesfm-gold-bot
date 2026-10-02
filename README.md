# TimesFM Gold Bot: XAUUSD Rise/Fall on Deriv

## Install

```bash
pip install websockets numpy
# TimesFM 3.0 + 2.5 from the zip you have (unzip it first):
pip install -e "./timesfm-3.0.0[torch]"
python gold_bot_gui.py
```

The first start downloads the TimesFM weights from Hugging Face (about 1.3 GB for 3.0).
If neither model loads, the bot falls back to a statistical forecaster automatically.
The top bar shows which model is running.

## Use it in this order

1. **Connect tab:** enter your token, App ID and the **demo** Account ID (VRTC…).
2. **Events tab:** paste this week's NFP, CPI, FOMC and PCE times in **UTC**.
3. **Backtest:** pulls up to 5,000 M5 bars and runs the *same* decision code bar by bar.
   Read the summary. It shows the win rate against the break-even rate, the expectancy, and the stake you'd need for $500/day.
   If it says "No positive edge", do not go live with those settings.
4. **Start** on demo for 2–4 weeks. Compare the live win rate with the backtest.
5. Only then consider a real account, and keep risk per trade at 1–2%.

## How a trade is decided (every M5 close)

| Setup | What it does |
|---|---|
| Prob | Takes P(price at expiry > now) from TimesFM's 9 quantiles |
| S1 Band | Close outside the previous bar's p10/p90 band means breakout. Tight-band regime fades it, wide-band regime follows it |
| S2 Macro | Joint Gold + DXY-proxy (built from EUR/JPY/GBP/CAD/SEK/CHF) + Silver forecast. A falling dollar or rising silver supports gold longs |
| S3 Events | Event proximity is fed in as known-future covariates (TimesFM 3.0), with a hard no-trade window around each release |
| S4 MTF | M15 and H1 forecasts must agree with the direction, and M5 must sit in the correct half of its band |
| S5 Sizing | Stake = balance × risk% × (normal band width ÷ current band width), clamped. In Multiplier mode, SL/TP go at p10/p90 |

A trade fires only when the ensemble confidence is at least the minimum confidence, the directional probability is at least the minimum probability, all gates pass, the time is inside the London/NY session, and the payout ratio is acceptable.

## Notes
- DXY, US10Y and VIX are not on Deriv. The bot rebuilds a DXY replica from Deriv FX pairs and uses silver in place of yields/VIX.
- Gold Rise/Fall durations depend on the account. The bot reads `contracts_for` at start and raises the expiry if your setting is below the broker minimum.
- TimesFM 3.0 weights are licensed non-commercial / non-production. Choose model "2.5" (Apache-2.0) if that matters to you.
- Martingale is off by default. On a fixed-payout binary it raises the risk of ruin without improving expectancy.
