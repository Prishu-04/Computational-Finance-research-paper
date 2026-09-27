# Dashboard Update — Historical + Future Forecasting

## What changed

### 1. Better price visualisation
The simulator chart is now a market-style intraday view with:
- OHLC candlesticks
- Volume bars
- Price grid and session-time guides
- Selected decision-time cursor
- 60-minute forecast arrows
- Forecast target markers
- Moirai q10/q90 uncertainty whiskers where available

### 2. Future 2025–2026 mode
The dashboard now supports a second mode:
- **Historical / observed** — uses the real dates in the supplied historical data.
- **Future scenario (2025–2026)** — accepts any weekday from 2025-01-01 through 2026-12-31.

Because the supplied market data stops in 2024-03-22, 2025/2026 cannot be treated as observed market data. Future mode therefore uses a selected historical seed day and recursively generates a scenario from the selected trained model's rolling 45-minute context → 60-minute forecast.

The UI explicitly labels these outputs as **ML scenario projections**.

### 3. Upward and downward forecast visibility
The graph now shows both directions with explicit arrows:
- `↗ UP` / green forecast path
- `↘ DOWN` / red forecast path
- `→ NEUTRAL`

The forecast card also displays the direction and a plain-language message such as **"Downward forecast — model projects lower target price"**.

Where a model's regression return and classification head disagree, the visualisation shows both, e.g. `DOWN · cls UP`, rather than silently hiding the disagreement.

### 4. Saved model artifacts
Future mode uses trained artifacts saved under:
- `results/models/classical/`
- `results/models/deep/`

The model-training scripts now save these artifacts automatically for later future inference.

## Run

```bash
pip install -r requirements.txt
pip install -r webapp/requirements.txt
python webapp/app.py
```

Open `http://localhost:5000/`.

## Important research interpretation
A future date in 2025/2026 is a **scenario**, not a claim about what the market actually did or will do. No future market observations are fabricated into the historical dataset.
