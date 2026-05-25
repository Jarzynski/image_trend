# -*- coding: utf-8 -*-
"""
02_make_labels_and_baselines.py

Purpose
-------
Generate future-return labels and traditional baseline features.

Input
-----
data/processed/panel_daily.parquet

Output
------
data/features/baseline_features.parquet

Key outputs
-----------
future_ret_{h}d and label_{h}d for horizons configured in EXPERIMENTS
ret_1d, ret_5d, ret_20d, etc.
open_to_close_ret_1d for executable next-open entry portfolio returns
ma gap features
price-position features
volatility features
liquidity features
tradable flag
"""

import numpy as np
import pandas as pd

from config import (
    PANEL_PATH,
    BASELINE_FEATURE_PATH,
    MIN_AMOUNT,
    MIN_LIST_DAYS,
    EXPERIMENTS,
    LOW_VOLUME_LIMIT_RATIO,
)


def safe_divide(a, b):
    """
    Numerically safe division.
    """
    out = a / b
    out = out.replace([np.inf, -np.inf], np.nan)
    return out


def configured_horizons():
    return sorted({cfg["horizon"] for cfg in EXPERIMENTS.values()})


def configured_windows():
    return sorted({cfg["window"] for cfg in EXPERIMENTS.values()})


def configured_ma_windows():
    windows = {5, 10, 20, 30, 120, 250}

    for cfg in EXPERIMENTS.values():
        ma_col = cfg.get("ma_col")
        if not ma_col:
            continue
        if ma_col.startswith("ma") and ma_col.endswith("_adj"):
            try:
                windows.add(int(ma_col[2:-4]))
            except ValueError:
                pass

    return sorted(windows)


def calc_limit_pct(df):
    """
    Board-aware daily limit threshold in percentage points.

    pct_chg is expected to use percentage units, e.g. 9.8 means 9.8%.
    """
    code = df["code"].astype(str).str.zfill(6)
    date = pd.to_datetime(df["date"])

    limit_pct = pd.Series(9.8, index=df.index, dtype="float64")

    limit_pct.loc[code.str.startswith("688")] = 19.8

    is_chinext = code.str.startswith(("300", "301"))
    chinext_reform = date >= pd.Timestamp("2020-08-24")
    limit_pct.loc[is_chinext & chinext_reform] = 19.8

    if "is_st" in df.columns:
        limit_pct.loc[pd.to_numeric(df["is_st"], errors="coerce").fillna(0) == 1] = 4.8

    return limit_pct


def add_returns_and_labels(df):
    """
    Add past returns and future returns by stock.

    Important:
    All shift operations must be grouped by code to avoid leakage
    across different stocks.

    Future-return convention:
    A signal formed at date t buys at the next trading day's adjusted open
    and exits at the horizon date's adjusted close:

        future_ret_h = close_adj[t + h] / open_adj[t + 1] - 1

    Therefore h=5 holds trading days t+1 through t+5.
    """
    g = df.groupby("code", group_keys=False)

    # Past returns used by baseline features and experiment windows.
    return_windows = sorted({1, 3, 5, 10, 20, 25, 60} | set(configured_windows()))
    for n in return_windows:
        df[f"ret_{n}d"] = g["close_adj"].pct_change(n)

    df["open_to_close_ret_1d"] = safe_divide(
        df["close_adj"],
        df["open_adj"].where(df["open_adj"] > 0),
    ) - 1.0

    # Future returns used by configured prediction horizons.
    for h in configured_horizons():
        buy_open = g["open_adj"].shift(-1).where(lambda x: x > 0)
        sell_close = g["close_adj"].shift(-h).where(lambda x: x > 0)
        future_ret = safe_divide(sell_close, buy_open) - 1.0

        df[f"future_ret_{h}d"] = future_ret
        df[f"label_{h}d"] = np.where(future_ret.notna(), (future_ret > 0).astype(int), np.nan)

    return df


def add_trading_constraint_features(df):
    """
    Add board-aware low-volume limit-up/down execution flags.

    The rolling volume baseline excludes the current day so the low-volume
    check uses only information available before that trading session.
    """
    g = df.groupby("code", group_keys=False)

    df["limit_pct"] = calc_limit_pct(df)
    df["volume_mean_20d_prev"] = g["volume"].transform(
        lambda x: x.shift(1).rolling(20, min_periods=20).mean()
    )
    df["volume_ratio_to_20d_prev"] = safe_divide(
        df["volume"],
        df["volume_mean_20d_prev"].where(df["volume_mean_20d_prev"] > 0),
    )

    pct_chg = pd.to_numeric(df["pct_chg"], errors="coerce")
    volume = pd.to_numeric(df["volume"], errors="coerce")
    low_volume = (
        df["volume_mean_20d_prev"].notna()
        & (df["volume_mean_20d_prev"] > 0)
        & (volume <= df["volume_mean_20d_prev"] * LOW_VOLUME_LIMIT_RATIO)
    )

    df["is_low_volume_limit_up"] = (
        low_volume
        & pct_chg.notna()
        & (pct_chg >= df["limit_pct"])
    ).astype("int8")
    df["is_low_volume_limit_down"] = (
        low_volume
        & pct_chg.notna()
        & (pct_chg <= -df["limit_pct"])
    ).astype("int8")

    return df


def add_baseline_features(df):
    """
    Add standard price/volume baseline features.
    """
    g = df.groupby("code", group_keys=False)

    # Reversal and momentum definitions.
    df["reversal_5d"] = -df["ret_5d"]
    df["reversal_10d"] = -df["ret_10d"]
    df["momentum_20d"] = df["ret_20d"]
    df["momentum_60d"] = df["ret_60d"]

    # Moving-average gaps.
    for n in configured_ma_windows():
        ma_col = f"ma{n}_adj"
        gap_col = f"ma{n}_gap"

        if ma_col in df.columns:
            df[gap_col] = safe_divide(df["close_adj"], df[ma_col]) - 1.0
        else:
            # If MA is absent, recompute from adjusted close.
            ma = g["close_adj"].transform(lambda x: x.rolling(n, min_periods=n).mean())
            df[gap_col] = safe_divide(df["close_adj"], ma) - 1.0
            df[ma_col] = ma

    # Price position within recent high-low range.
    for n in [5, 20, 60]:
        rolling_high = g["high_adj"].transform(lambda x: x.rolling(n, min_periods=n).max())
        rolling_low = g["low_adj"].transform(lambda x: x.rolling(n, min_periods=n).min())
        denom = rolling_high - rolling_low
        df[f"position_{n}d"] = safe_divide(df["close_adj"] - rolling_low, denom)

    # Volatility.
    for n in [20, 60]:
        df[f"vol_{n}d"] = g["ret_1d"].transform(lambda x: x.rolling(n, min_periods=n).std())

    # Liquidity and turnover changes.
    df["amount_mean_20d"] = g["amount"].transform(lambda x: x.rolling(20, min_periods=20).mean())
    df["amount_change_20d"] = safe_divide(df["amount"], df["amount_mean_20d"]) - 1.0

    if "turnover_rate" in df.columns:
        df["turnover_mean_20d"] = g["turnover_rate"].transform(
            lambda x: x.rolling(20, min_periods=20).mean()
        )
        df["turnover_change_20d"] = safe_divide(df["turnover_rate"], df["turnover_mean_20d"]) - 1.0

    # Size controls.
    if "total_mktcap" in df.columns:
        df["log_total_mktcap"] = np.log(df["total_mktcap"].where(df["total_mktcap"] > 0))
    if "float_mktcap" in df.columns:
        df["log_float_mktcap"] = np.log(df["float_mktcap"].where(df["float_mktcap"] > 0))

    return df


def add_final_tradable_flag(df):
    """
    Define the main tradable flag for model training and backtest.

    This flag is deliberately simple.
    More realistic trading constraints are added in the backtest script.
    """
    df["is_tradable"] = 1

    if "is_tradable_basic" in df.columns:
        df.loc[df["is_tradable_basic"] == 0, "is_tradable"] = 0

    if "is_st" in df.columns:
        df.loc[df["is_st"] == 1, "is_tradable"] = 0

    if "is_suspended" in df.columns:
        df.loc[df["is_suspended"] == 1, "is_tradable"] = 0

    if "amount" in df.columns:
        df.loc[df["amount"].fillna(0) < MIN_AMOUNT, "is_tradable"] = 0

    if "days_since_list" in df.columns:
        # If days_since_list is missing, we keep the row for now.
        # You may choose to drop missing listing dates more aggressively.
        mask_new = df["days_since_list"].notna() & (df["days_since_list"] < MIN_LIST_DAYS)
        df.loc[mask_new, "is_tradable"] = 0

    return df


def main():
    print(f"Reading panel: {PANEL_PATH}")
    df = pd.read_parquet(PANEL_PATH)
    df = df.sort_values(["code", "date"]).reset_index(drop=True)

    print("Adding returns and labels...")
    df = add_returns_and_labels(df)

    print("Adding trading constraint features...")
    df = add_trading_constraint_features(df)

    print("Adding baseline features...")
    df = add_baseline_features(df)

    print("Adding final tradable flag...")
    df = add_final_tradable_flag(df)

    print(f"Saving baseline features to: {BASELINE_FEATURE_PATH}")
    df.to_parquet(BASELINE_FEATURE_PATH, index=False)

    print("Done.")
    print("Feature table shape:", df.shape)


if __name__ == "__main__":
    main()
