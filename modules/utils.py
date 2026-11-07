from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None or str(value).strip() == "":
            return float(default)
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def safe_int(value: Any, default: int = 0) -> int:
    try:
        if value is None or str(value).strip() == "":
            return int(default)
        return int(value)
    except (TypeError, ValueError):
        return int(default)


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = [str(col).strip().lower() for col in out.columns]
    return out


def ensure_numeric_column(df: pd.DataFrame, column: str, default: float = 0.0) -> pd.DataFrame:
    out = df.copy()
    if column not in out.columns:
        out[column] = default
    out[column] = pd.to_numeric(out[column], errors="coerce").fillna(default)
    return out


def clamp(value: float, min_value: float, max_value: float) -> float:
    return max(min_value, min(max_value, value))


def rolling_mean(values: list[float], window: int = 7) -> float:
    if not values:
        return 0.0
    arr = np.asarray(values, dtype=float)
    if len(arr) < window:
        return float(np.mean(arr))
    return float(np.mean(arr[-window:]))


def series_is_valid(series: pd.Series) -> bool:
    if series is None or series.empty:
        return False
    cleaned = pd.to_numeric(series, errors="coerce").dropna()
    return not cleaned.empty


def percentile_interval(
    values: np.ndarray, lower_pct: float = 0.10, upper_pct: float = 0.90
) -> tuple[Any, Any]:
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return arr, arr
    lower = np.percentile(arr, lower_pct * 100)
    upper = np.percentile(arr, upper_pct * 100)
    return lower, upper


def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    if abs(denominator) < 1e-12:
        return default
    return numerator / denominator


def round_numeric_df(
    df: pd.DataFrame, cols: list[str] | None = None, decimals: int = 2
) -> pd.DataFrame:
    out = df.copy()
    if cols is None:
        cols = out.select_dtypes(include=["number"]).columns.tolist()
    for col in cols:
        if col in out.columns:
            out[col] = out[col].round(decimals)
    return out


def lag_features(values: pd.Series, max_lag: int = 30) -> pd.DataFrame:
    series = pd.Series(values).astype(float).reset_index(drop=True)
    rows = []
    for idx in range(len(series)):
        if idx == 0:
            continue
        row = {}
        for lag in range(1, min(max_lag, len(series) - 1) + 1):
            if idx - lag >= 0:
                row[f"lag_{lag}"] = float(series.iloc[idx - lag])
            else:
                row[f"lag_{lag}"] = float(series.iloc[0])
        rows.append(row)
    return pd.DataFrame(rows)