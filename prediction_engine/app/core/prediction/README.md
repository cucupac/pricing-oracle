# Vendor Error Model: Key Design Decisions

## What We're Predicting

**`predicted_adjustment_factor`** — How much worse actual execution will be vs. the vendor quote (0-4%). Directly captures vendor reliability for risk-adjusted pricing.

## Features Included

-   **Vendor** _(odos vs 1inch)_ — Different aggregators have different execution quality
-   **Trade size** _(log-scaled)_ — Larger trades face more slippage; log handles $1 to $1M+ range
-   **Hop count** — More routing hops = higher execution risk
-   **Pool liquidity** _(log-scaled)_ — Small pools are more vulnerable to slippage
-   **Gas cost** — Signals network conditions and route complexity
-   **Interactions** _(size×hops, vendor×size, vendor×hops)_ — Captures compounding effects

## Features Excluded

-   **Constants** _(chain, tokens, gas price)_ — No variation = no predictive power
-   **Post-quote data** _(actual results, derived prices)_ — Not available at quote time

## Model Design

-   **Small neural network** _(32→16 neurons)_ — Fast, practical, less overfitting
-   **Bounded output** _[0, 4%]_ — Ensures business-realistic predictions
-   **Robust loss function** — Handles outliers while staying precise for small errors

## Training

-   **5-fold cross-validation** — Picks best model, ensures generalization
-   **Early stopping** — Prevents overfitting
-   **Linear baseline** — Ensures neural network actually improves over simple methods

## Evaluation

-   **Basis points accuracy** — How traders think about execution error
-   **Coverage bands** — How often predictions are "close enough"
-   **Vendor/size stratification** — Works across all trading scenarios

**Conclusion:** Prioritizes practical reliability for real-time quote adjustment over theoretical sophistication.
