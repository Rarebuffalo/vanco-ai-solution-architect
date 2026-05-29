# Developer Guide: Grocery Sales Forecasting Pipeline

This guide provides a detailed walkthrough of the architectural design, features pipeline, validation framework, and training scripts implemented for the Grocery Sales Forecasting project.

---

## 1. Pipeline Architecture & Data Flow

Instead of fitting 1,782 separate statistical models (which is slow, computationally heavy, and prevents learning patterns across series), we build a **single global LightGBM Regressor** that trains on all store-product combinations simultaneously.

The high-level data alignment flow is as follows:
```
                      [raw tables]
   train.csv / test.csv   stores.csv   oil.csv   holidays_events.csv
           │                  │           │               │
           ├──────────────────┘           │               │
           ▼ (joined by store_nbr)        │               │
   Store Metadata Merged                  │               │
           │                              │               │
           ├──────────────────────────────┘               │
           ▼ (linear interpolated & rolling mean)        │
   Oil Prices Aligned                                     │
           │                                              │
           ├──────────────────────────────────────────────┘
           ▼ (transferred check + local/state/national hierarchy)
   Holiday Mapping
           │
           ├──► [Feature Extraction]
           │      - Calendar variables (day of week, month, etc.)
           │      - Lags (minimum shift t-16 to prevent leakage)
           │      - Rolling statistics computed on lag-16
           │
           ├──► [Chronological Out-of-Sample Split]
           │      - Train: < 2017-08-01
           │      - Validation: 2017-08-01 to 2017-08-15
           │
           └──► [Model Execution]
                  - LightGBM Regressor (native category handling)
                  - NumPy Ridge Regressor fallback (if offline sandbox lacks LightGBM)
```

---

## 2. File-by-File Blueprint

All code is structured within `grocery_forecasting/src/` to isolate logic and promote reuse:

### A. [features.py](file:///home/Krishna-Singh/vanco-ai-solution-architect/grocery_forecasting/src/features.py)
This script manages table merging, gap-filling, holiday mapping, and feature extraction.
* `load_and_align_oil(oil_path)`: Loads crude oil prices. It generates a continuous daily timeline spanning the min/max dates, fills weekend/holiday price gaps using **linear interpolation**, and calculates 3-day and 7-day rolling means to capture short-term oil trends.
* `map_holidays(df, holidays_path, stores)`: Implements a localization mapping hierarchy. First, it drops transferred holidays (`transferred == True`). Next, it applies holidays sequentially: National holidays apply to all stores, Regional holidays match store state names, and Local holidays match store city names.
* `build_features(...)`: The primary orchestrator. Concatenates train and test rows (with sales filled as `NaN` in the test set) to calculate lags across the boundary. Extracts calendar components (`day_of_week`, `day_of_month`, `month`, `year`, `is_weekend`), computes the log target (`log1p`), calculates lags, and creates rolling window averages.

### B. [validation.py](file:///home/Krishna-Singh/vanco-ai-solution-architect/grocery_forecasting/src/validation.py)
This script establishes chronological splitting to validate predictions accurately without leaking future details.
* `get_time_series_split(df, val_start, val_end)`: Splits data at the `2017-08-01` boundary. The validation window (Aug 1 to Aug 15) exactly matches the 15-day length and weekly cycle of the Kaggle evaluation window (Aug 16 to Aug 31). It asserts that the max train date is strictly less than the min validation date.

### C. [model.py](file:///home/Krishna-Singh/vanco-ai-solution-architect/grocery_forecasting/src/model.py)
This is the training orchestrator, supporting native GBDT training and baseline fallbacks.
* `FallbackLinearRegression`: A pure-NumPy class implementing a Ridge Regressor using the closed-form normal equation:
  $$\beta = (X^T X + \lambda I)^{-1} X^T y$$
  It standardizes continuous features, encodes categoricals as integer codes, and acts as a zero-dependency fallback.
* `compute_rmsle(y_true, y_pred)`: Computes Root Mean Squared Logarithmic Error. Since training runs on `log1p(sales)`, this is computed as a simple root-mean-squared-error of the log-transformed predictions.
* `run_pipeline(...)`: Parses command-line inputs, coordinates feature extraction, runs the validation split, trains the selected model, computes metrics, prints subpopulation residual errors, and outputs a formatted competition-ready `submission.csv`.

### D. [test_run.py](file:///home/Krishna-Singh/vanco-ai-solution-architect/grocery_forecasting/src/test_run.py)
* Integration test script. Synthesizes a mini-version of all Kaggle tables (sales, oil, stores, holidays) with realistic numeric columns, and runs the entire modeling script from start to finish to confirm the environment is configured correctly.

---

## 3. Core Feature Engineering Math

### A. Oil price Interpolation
Raw oil tables lack prices on weekends and holidays. Dropping these dates would create misaligned dates in our time-series index. We apply linear interpolation:
$$y = y_0 + (x - x_0) \frac{y_1 - y_0}{x_1 - x_0}$$
This creates a continuous indicator of macroeconomic health that aligns daily sales figures with daily oil indices.

### B. Leakage-Free Lags & Rolling Statistics
A common time-series leakage occurs when models use short lags (e.g. `sales_lag_1`) on a future test window where actual sales are unknown.
Since the test window spans 16 days, the minimum lag shift must be **16 days** (`sales_lag_16`). 
Any rolling statistics (e.g., 7, 14, or 28-day moving averages) are computed strictly on top of `sales_lag_16`:
$$\text{sales\_roll\_mean\_7}(t) = \frac{1}{7} \sum_{i=0}^{6} \text{sales\_lag\_16}(t - i)$$
This guarantees that features are fully computable for the entire future test window without accessing any future sales values.

---

## 4. Failure Modes & Limitations

1. **Promotional Volatility (Onpromotion)**: Grocery items are highly sensitive to sudden discount campaigns. While LightGBM learns general promotion coefficients, massive single-day promotions can result in large model residuals if historical lags were recorded during non-promotional periods.
2. **Holiday Hoarding vs. Stockouts**: Before major national holidays, consumer demand spikes (hoarding), followed by days of zero sales because stores close. If a store runs out of inventory (stockout), actual sales drop to 0, which is a supply-chain constraint that the model cannot predict using historical buyer demand alone.
3. **Macro Economic Shifts**: While oil is a strong long-term indicator for Ecuador, daily fluctuations in crude oil index prices have zero immediate impact on whether a shopper buys milk or bread on a Tuesday.
