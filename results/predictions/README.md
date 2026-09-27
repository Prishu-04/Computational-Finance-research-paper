# Historical prediction trace

`prediction_trace.csv` contains one row per model and decision window.

Columns include:
- decision/target timestamps and walk-forward split
- true and predicted direction
- predicted return and confidence
- class probabilities for direction-capable models
- q10/q50/q90 for Moirai-family probabilistic forecasts

The web dashboard uses this file for the historical simulator. It does not
generate predictions from aggregate accuracy. Re-running the model stages
regenerates this trace for the current dataset.
