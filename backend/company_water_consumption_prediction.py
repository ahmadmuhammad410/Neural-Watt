import os
import joblib
import numpy as np
import pandas as pd
from pathlib import Path

# ============================================================
# HELPER FUNCTIONS (From AQUAGUARD AI)
# ============================================================

def safe_mape(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    mask = np.abs(y_true) > 1e-9
    if mask.sum() == 0:
        return np.nan
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100


def create_time_features(df):
    df = df.copy()
    df["year"] = df["period"].dt.year
    df["month"] = df["period"].dt.month
    df["quarter"] = df["period"].dt.quarter
    df["weekofyear"] = df["period"].dt.isocalendar().week.astype(int)
    df["dayofyear"] = df["period"].dt.dayofyear
    df["dayofmonth"] = df["period"].dt.day
    df["days_in_month"] = df["period"].dt.days_in_month
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
    df["quarter_sin"] = np.sin(2 * np.pi * df["quarter"] / 4)
    df["quarter_cos"] = np.cos(2 * np.pi * df["quarter"] / 4)
    df["week_sin"] = np.sin(2 * np.pi * df["weekofyear"] / 52)
    df["week_cos"] = np.cos(2 * np.pi * df["weekofyear"] / 52)
    return df


def create_lag_features(df, lags, rolling_windows):
    df = df.copy()
    grouped = df.groupby("building_id")["consumption"]

    for lag in lags:
        df[f"lag_{lag}"] = grouped.shift(lag)

    for window in rolling_windows:
        df[f"rolling_mean_{window}"] = grouped.transform(
            lambda x: x.shift(1).rolling(window=window, min_periods=max(1, window // 2)).mean()
        )
        df[f"rolling_std_{window}"] = grouped.transform(
            lambda x: x.shift(1).rolling(window=window, min_periods=max(2, window // 2)).std()
        )
        df[f"rolling_max_{window}"] = grouped.transform(
            lambda x: x.shift(1).rolling(window=window, min_periods=max(1, window // 2)).max()
        )
        df[f"rolling_min_{window}"] = grouped.transform(
            lambda x: x.shift(1).rolling(window=window, min_periods=max(1, window // 2)).min()
        )
    return df


def load_and_preprocess(data_path, config):
    df = pd.read_csv(data_path)

    timestamp_candidates = ["timestamp", "datetime", "date_time", "date", "time", "DateTime", "Timestamp"]
    timestamp_col = next((c for c in timestamp_candidates if c in df.columns), None)
    if timestamp_col is None:
        timestamp_col = next((c for c in df.columns if "time" in c.lower() or "date" in c.lower()), None)
    if timestamp_col is None:
        raise ValueError("Could not detect timestamp column in dataset.")

    df.rename(columns={timestamp_col: "timestamp"}, inplace=True)
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp"]).drop_duplicates(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)

    if "building_id" in df.columns and "consumption" in df.columns:
        long_df = df[["timestamp", "building_id", "consumption"]].copy()
    else:
        building_cols = [c for c in df.columns if c != "timestamp"]
        for c in building_cols:
            df[c] = pd.to_numeric(df[c], errors="coerce")
            df.loc[df[c] < 0, c] = np.nan
        long_df = df.melt(id_vars=["timestamp"], value_vars=building_cols, var_name="building_id", value_name="consumption")

    long_df["consumption"] = pd.to_numeric(long_df["consumption"], errors="coerce")
    
    aggregated = (
        long_df.set_index("timestamp")
        .groupby("building_id")["consumption"]
        .resample(config["rule"])
        .sum()
        .reset_index()
        .rename(columns={"timestamp": "period"})
    )
    aggregated = aggregated.dropna(subset=["consumption"]).sort_values(["building_id", "period"]).reset_index(drop=True)
    return aggregated

# ============================================================
# MAIN FORECASTING FUNCTION
# ============================================================

"""
test_water_forecast.py
=======================
Standalone evaluator for the water-consumption forecasting models trained
by train_water_forecast.py.

Unlike test_train_water_forecast.py, this script does NOT import or need
train_water_forecast.py at all. All it needs is:

    1. A CSV of raw meter-level hourly readings to evaluate on (same
       column layout as the original training CSV: a "timestamp" column
       plus one column per meter).
    2. The best_model_{day,week,month,quarter}.joblib files that a prior
       run of train_water_forecast.py already saved to --outdir.

It reimplements, self-contained, exactly the pieces of feature engineering
and recursive-forecast logic needed to turn a raw CSV into the feature
columns each frozen model expects, runs inference with the already-fitted
model/strategy, and reports R2 / MAE / RMSE per horizon.

Each best_model_{horizon}.joblib was saved by train_water_forecast.py as
one of four "method" shapes, all handled below:

    "direct"           -> {"model": <fitted estimator>, "feature_cols": [...]}
    "ensemble"          -> {"components": [name1, name2],
                             "component_models": {name: <fitted estimator>},
                             "feature_cols": [...]}
    "seasonal_naive"    -> {"seasonal_lag": int, "series": <training series>}
    "aggregated_daily"  -> {"day_model": <fitted daily estimator>,
                             "day_feature_cols": [...]}

CAVEATS (inherent to reusing a frozen model on a *different* CSV, not
something this script introduces):

  * For "direct"/"ensemble" models, the exact set of lag_/roll_mean_/
    roll_std_ columns that get engineered depends on how much history the
    input series has (train_water_forecast.py trims lags that don't leave
    enough rows). If your new CSV spans a very different amount of time
    than the original training CSV, the engineered columns may not match
    feature_cols exactly -- if that happens this script prints which
    columns are missing and skips that horizon rather than guessing.
  * Unpickling a saved XGBoost/LightGBM/CatBoost model still requires that
    library to be installed in this environment -- that's a joblib/pickle
    requirement of the saved object itself, unrelated to whether
    train_water_forecast.py is present. If a horizon's winning model needs
    a library you don't have installed, loading that one .joblib will
    raise ImportError; catch it, skip that horizon, and move on.

Usage
-----
    python test_water_forecast.py --input new_water_data.csv \
        --models-dir ./outputs --outdir ./test_outputs
"""
import argparse
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from pandas.tseries.holiday import USFederalHolidayCalendar
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

warnings.filterwarnings("ignore")

FREQ_MAP = {"day": "D", "week": "W-SUN", "month": "MS", "quarter": "QS"}


# --------------------------------------------------------------------------- #
# 1. Data loading & aggregation (mirrors train_water_forecast.py exactly)
# --------------------------------------------------------------------------- #
def load_total_series(csv_path: str) -> pd.Series:
    """Read the raw meter-level CSV and collapse it into a single hourly
    total-consumption series (NaN readings are treated as 0 consumption)."""
    df = pd.read_csv(csv_path)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    meter_cols = [c for c in df.columns if c != "timestamp"]
    total = df[meter_cols].sum(axis=1, skipna=True)
    s = pd.Series(total.values, index=df["timestamp"], name="total").sort_index()
    s = s.asfreq("h")
    s = s.interpolate(limit_direction="both")
    return s


def aggregate(hourly: pd.Series, horizon: str) -> pd.Series:
    """Sum the hourly total series into the target horizon's period sums."""
    freq = FREQ_MAP[horizon]
    return hourly.resample(freq).sum()


# --------------------------------------------------------------------------- #
# 2. Feature engineering (mirrors train_water_forecast.py's make_features
#    exactly, so a new CSV produces the same columns the frozen models were
#    fitted on)
# --------------------------------------------------------------------------- #
def make_features(series: pd.Series, horizon: str) -> pd.DataFrame:
    """Build a supervised-learning table: features derived from history up to
    time t, target = value at t+1 (one-step-ahead forecast)."""
    df = pd.DataFrame({"y": series})

    n = len(series)
    if horizon == "day":
        lags = [1, 2, 3, 7, 14, 21, 28]
        roll_windows = [3, 7, 14, 28]
    elif horizon == "week":
        lags = [1, 2, 3, 4, 8, 13, 52]
        roll_windows = [2, 4, 8]
    elif horizon == "month":
        lags = [1, 2, 3, 6, 12]
        roll_windows = [2, 3, 6]
    else:  # quarter
        lags = [1]
        roll_windows = []

    if horizon == "day":
        min_rows_keep = 200
        lags = [l for l in lags if l < n - min_rows_keep] or [1]
        roll_windows = [w for w in roll_windows if w < n - min_rows_keep] or [3]
    elif horizon != "quarter":
        lags = [l for l in lags if l < max(3, n // 3)] or [1]
        roll_windows = [w for w in roll_windows if w < max(3, n // 3)] or [2]

    for lag in lags:
        df[f"lag_{lag}"] = series.shift(lag)

    for w in roll_windows:
        df[f"roll_mean_{w}"] = series.shift(1).rolling(w).mean()
        df[f"roll_std_{w}"] = series.shift(1).rolling(w).std()

    df["t_index"] = np.arange(len(series))

    idx = series.index
    if horizon in ("day", "week"):
        df["month"] = idx.month
        df["quarter"] = idx.quarter
        df["weekofyear"] = idx.isocalendar().week.astype(int)
        if horizon == "day":
            df["dayofweek"] = idx.dayofweek
            df["is_weekend"] = (idx.dayofweek >= 5).astype(int)
            df["day"] = idx.day
            df["dow_sin"] = np.sin(2 * np.pi * df["dayofweek"] / 7)
            df["dow_cos"] = np.cos(2 * np.pi * df["dayofweek"] / 7)

            cal = USFederalHolidayCalendar()
            holidays = cal.holidays(start=idx.min() - pd.Timedelta(days=3), end=idx.max() + pd.Timedelta(days=3))
            holiday_set = set(holidays.normalize())
            df["is_holiday"] = idx.normalize().isin(holiday_set).astype(int)
            df["is_day_before_holiday"] = (idx.normalize() + pd.Timedelta(days=1)).isin(holiday_set).astype(int)
            df["is_day_after_holiday"] = (idx.normalize() - pd.Timedelta(days=1)).isin(holiday_set).astype(int)

            doy = idx.dayofyear.values.astype(float)
            df["doy_sin"] = np.sin(2 * np.pi * doy / 365.25)
            df["doy_cos"] = np.cos(2 * np.pi * doy / 365.25)
        df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
        df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
    elif horizon == "month":
        df["month"] = idx.month
        df["quarter"] = idx.quarter
        df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
        df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
    else:  # quarter
        df["quarter_num"] = idx.quarter

    df["target"] = df["y"].shift(-1)
    df = df.drop(columns=["y"])
    df = df.dropna()
    return df


# --------------------------------------------------------------------------- #
# 3. Recursive daily-model aggregation (mirrors train_water_forecast.py
#    exactly -- needed to replay the "aggregated_daily" strategy)
# --------------------------------------------------------------------------- #
def _single_day_feature_row(series_ext: pd.Series, target_date: pd.Timestamp, feature_cols):
    row = {}
    cal = USFederalHolidayCalendar()
    holidays = cal.holidays(
        start=target_date - pd.Timedelta(days=3), end=target_date + pd.Timedelta(days=3)
    )
    holiday_set = set(holidays.normalize())

    for col in feature_cols:
        if col.startswith("lag_"):
            lag = int(col.split("_")[1])
            d = target_date - pd.Timedelta(days=lag)
            row[col] = series_ext.get(d, np.nan)
        elif col.startswith("roll_mean_"):
            w = int(col.split("_")[-1])
            window = series_ext.loc[: target_date - pd.Timedelta(days=1)].iloc[-w:]
            row[col] = window.mean() if len(window) == w else np.nan
        elif col.startswith("roll_std_"):
            w = int(col.split("_")[-1])
            window = series_ext.loc[: target_date - pd.Timedelta(days=1)].iloc[-w:]
            row[col] = window.std() if len(window) == w else np.nan
        elif col == "t_index":
            row[col] = series_ext.index.get_loc(series_ext.index[series_ext.index <= target_date][-1]) \
                if target_date in series_ext.index else len(series_ext)
        elif col == "month":
            row[col] = target_date.month
        elif col == "quarter":
            row[col] = target_date.quarter
        elif col == "weekofyear":
            row[col] = int(target_date.isocalendar()[1])
        elif col == "dayofweek":
            row[col] = target_date.dayofweek
        elif col == "is_weekend":
            row[col] = int(target_date.dayofweek >= 5)
        elif col == "day":
            row[col] = target_date.day
        elif col == "dow_sin":
            row[col] = np.sin(2 * np.pi * target_date.dayofweek / 7)
        elif col == "dow_cos":
            row[col] = np.cos(2 * np.pi * target_date.dayofweek / 7)
        elif col == "month_sin":
            row[col] = np.sin(2 * np.pi * target_date.month / 12)
        elif col == "month_cos":
            row[col] = np.cos(2 * np.pi * target_date.month / 12)
        elif col == "doy_sin":
            row[col] = np.sin(2 * np.pi * target_date.dayofyear / 365.25)
        elif col == "doy_cos":
            row[col] = np.cos(2 * np.pi * target_date.dayofyear / 365.25)
        elif col == "is_holiday":
            row[col] = int(target_date.normalize() in holiday_set)
        elif col == "is_day_before_holiday":
            row[col] = int((target_date.normalize() + pd.Timedelta(days=1)) in holiday_set)
        elif col == "is_day_after_holiday":
            row[col] = int((target_date.normalize() - pd.Timedelta(days=1)) in holiday_set)
        else:
            row[col] = np.nan
    return row


def recursive_daily_forecast(daily_model, feature_cols, history_daily: pd.Series, start_date, n_days):
    series_ext = history_daily.copy()
    preds = []
    cur = pd.Timestamp(start_date)
    for _ in range(n_days):
        row = _single_day_feature_row(series_ext, cur, feature_cols)
        X_row = pd.DataFrame([row])[feature_cols]
        if X_row.isna().any(axis=None):
            pred = series_ext.iloc[-1]
        else:
            pred = float(daily_model.predict(X_row)[0])
        preds.append(pred)
        series_ext.loc[cur] = pred
        cur = cur + pd.Timedelta(days=1)
    return preds


def evaluate(y_true, y_pred):
    r2 = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    return r2, mae, rmse


def evaluate_aggregated_from_daily(daily_model, day_feature_cols, hourly: pd.Series, horizon: str, test_periods_index):
    daily_history = aggregate(hourly, "day")
    preds, actuals, kept_starts = [], [], []
    for period_start in test_periods_index:
        period_start = pd.Timestamp(period_start)
        if horizon == "week":
            period_end = period_start + pd.Timedelta(days=6)
        elif horizon == "month":
            period_end = period_start + pd.offsets.MonthEnd(0)
        else:  # quarter
            period_end = period_start + pd.offsets.QuarterEnd(0)

        history_before = daily_history.loc[: period_start - pd.Timedelta(days=1)]
        if len(history_before) < 30:
            continue  # not enough daily history to seed a recursive forecast
        n_days = (period_end - period_start).days + 1
        day_preds = recursive_daily_forecast(daily_model, day_feature_cols, history_before, period_start, n_days)

        actual_slice = hourly.loc[period_start: period_end + pd.Timedelta(hours=23)]
        if actual_slice.empty:
            continue
        preds.append(sum(day_preds))
        actuals.append(actual_slice.sum())
        kept_starts.append(period_start)

    if len(preds) < 2:
        return None
    return evaluate(np.array(actuals), np.array(preds)), preds, actuals, kept_starts


def seasonal_naive_predict(series: pd.Series, seasonal_lag: int) -> pd.Series:
    """Same-period-last-year lookup, falling back to last-known-value
    persistence -- identical formula to train_water_forecast.py."""
    naive = series.shift(seasonal_lag - 1)
    naive = naive.fillna(series.shift(1))
    return naive


# --------------------------------------------------------------------------- #
# 4. Predict one horizon using frozen best_model_{horizon}.joblib
# --------------------------------------------------------------------------- #
def predict_horizon(horizon: str, hourly: pd.Series, models_dir: str) -> float:
    model_path = Path(models_dir) / f"best_model_{horizon}.joblib"
    if not model_path.exists():
        print(f"[{horizon:8s}] no saved model found at {model_path}")
        return 0.0

    try:
        bundle = joblib.load(model_path)
    except ImportError as e:
        print(f"[{horizon:8s}] Import error: {e}")
        return 0.0

    method = bundle["method"]
    series = aggregate(hourly, horizon)
    
    if method in ("direct", "ensemble"):
        # Append dummy row to keep the last actual period during make_features' dropna()
        dummy_idx = series.index[-1] + pd.Timedelta(days=1)
        series_ext = pd.concat([series, pd.Series([0.0], index=[dummy_idx])])
        data = make_features(series_ext, horizon)
        
        feature_cols = bundle["feature_cols"]
        missing = [c for c in feature_cols if c not in data.columns]
        if missing:
            for c in missing:
                data[c] = 0.0
                
        X = data[feature_cols].iloc[[-1]]
        
        if method == "direct":
            pred = bundle["model"].predict(X)[0]
        else:
            valid_models = [m for m in bundle["component_models"].values() if m is not None]
            preds_list = [m.predict(X)[0] for m in valid_models]
            pred = np.mean(preds_list) if preds_list else 0.0
            
        return float(pred)

    elif method == "seasonal_naive":
        seasonal_lag = bundle["seasonal_lag"]
        train_series = bundle.get("series")
        if train_series is not None:
            combined = pd.concat([train_series, series])
            combined = combined[~combined.index.duplicated(keep="last")].sort_index()
        else:
            combined = series
            
        idx_to_use = len(combined) - seasonal_lag
        if idx_to_use < 0:
            pred = combined.iloc[-1]
        else:
            pred = combined.iloc[idx_to_use]
        return float(pred)

    elif method == "aggregated_daily":
        day_model, day_feature_cols = bundle["day_model"], bundle["day_feature_cols"]
        daily_history = aggregate(hourly, "day")
        
        last_period = series.index[-1]
        if horizon == "week":
            period_start = last_period + pd.Timedelta(days=1)
            n_days = 7
        elif horizon == "month":
            period_start = last_period + pd.offsets.MonthBegin(1)
            period_end = period_start + pd.offsets.MonthEnd(0)
            n_days = (period_end - period_start).days + 1
        else:  # quarter
            period_start = last_period + pd.offsets.QuarterBegin(1)
            period_end = period_start + pd.offsets.QuarterEnd(0)
            n_days = (period_end - period_start).days + 1
            
        history_before = daily_history.loc[: period_start - pd.Timedelta(days=1)]
        day_preds = recursive_daily_forecast(day_model, day_feature_cols, history_before, period_start, n_days)
        return float(sum(day_preds))

    return 0.0


def main(csv_path, duration_months=0):
    import os
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    models_dir = os.path.join(BASE_DIR, "..", "Models", "Water", "Company")

    print("Loading and aggregating raw data...")
    hourly = load_total_series(csv_path)
    
    forecast_results = {}
    h_map = {
        "day": "Target_Next_Day",
        "week": "Target_Next_Week",
        "month": "Target_Next_Month",
        "quarter": "Target_Next_Quarter"
    }

    print("=" * 75)
    print("      AQUAGUARD AI - WATER INFERENCE ENGINE (MULTI-HORIZON)      ")
    print("=" * 75)

    try:
        duration_val = float(duration_months)
    except (ValueError, TypeError):
        duration_val = 0.0

    horizons = ["day", "week", "month"]
    if duration_val > 5:
        horizons.append("quarter")

    for horizon in horizons:
        target_key = h_map[horizon]
        try:
            pred_liters = predict_horizon(horizon, hourly, models_dir)
            pred_m3 = pred_liters / 1000.0
            forecast_results[target_key] = max(0.0, float(pred_m3))
            print(f"[*] {target_key:<20} | {pred_m3:>10,.2f} m3")
        except Exception as e:
            print(f"[!] Error running {target_key}: {e}")
            forecast_results[target_key] = 0.0

    print("=" * 75)
    return forecast_results


if __name__ == "__main__":
    pass


def process_company_water(df, facility_subtype, holiday_usage, holiday_days, duration_months, data_handling_method, facility_size="small"):
    """
    Processes company water consumption data.
    """
    csv_path = "Actual_daily_consumption.csv"
    if not os.path.exists(csv_path):
        df.to_csv(csv_path, index=False)
            
    try:
        forecast = main(csv_path, duration_months)
    except Exception as e:
        print(f"Error in utility pipeline: {e}")
        forecast = {}

    def calc_waste(pred, factor=1.0):
        if not pred: return 0.0, 0.0
        # Since water thresholds weren't provided differently, we use the same electricity ones as agreed
        water_thresholds = {
            'Bakery':      {'small': 8,   'medium': 30,  'large': 90},
            'Office':      {'small': 10,  'medium': 40,  'large': 120},
            'Hotel':       {'small': 150, 'medium': 600, 'large': 2000},
            'Restaurant':  {'small': 20,  'medium': 80,  'large': 250},
            'School':      {'small': 150, 'medium': 500, 'large': 1200},
            'SuperMarket': {'small': 40,  'medium': 200, 'large': 600},
        }
        base_threshold = 2.5
        if facility_subtype in water_thresholds:
            size = facility_size.lower() if facility_size else 'small'
            if size not in water_thresholds[facility_subtype]:
                size = 'small'
            base_threshold = water_thresholds[facility_subtype][size]
        
        limit = base_threshold * factor
        if pred > limit:
            waste_amt = pred - limit
            waste_pct = (waste_amt / limit) * 100
            return round(waste_amt, 2), round(waste_pct, 2)
        return 0.0, 0.0

    day_amt, day_pct = calc_waste(forecast.get("Target_Next_Day"), factor=1/30)
    week_amt, week_pct = calc_waste(forecast.get("Target_Next_Week"), factor=1/4)
    month_amt, month_pct = calc_waste(forecast.get("Target_Next_Month"), factor=1.0)
    quarter_amt, quarter_pct = calc_waste(forecast.get("Target_Next_Quarter"), factor=3.0)

    waste_dict = {
        "waste_day": day_amt,
        "waste_day_pct": day_pct,
        "waste_week": week_amt,
        "waste_week_pct": week_pct,
        "waste_month": month_amt,
        "waste_month_pct": month_pct,
        "waste_quarter": quarter_amt,
        "waste_quarter_pct": quarter_pct,
    }

    prediction_result = {
        "next_day": forecast.get("Target_Next_Day"),
        "next_week": forecast.get("Target_Next_Week"),
        "next_month": forecast.get("Target_Next_Month"),
        "next_quarter": forecast.get("Target_Next_Quarter"),
        "next_semi_annual": forecast.get("Target_Next_SemiAnnual", 0.0),
        "next_annual": forecast.get("Target_Next_Annual", 0.0),
        "waste": waste_dict
    }
    
    return prediction_result
