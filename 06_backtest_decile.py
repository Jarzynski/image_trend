# -*- coding: utf-8 -*-
"""
06_backtest_decile.py

Evaluate model predictions with IC, decile returns, long-only overlapping
decile portfolios, and D10-D1 spread portfolios.

Execution convention
--------------------
A signal formed at date t buys at the next trading day's adjusted open and
holds through the horizon date's adjusted close. For the first holding day,
portfolio returns use open-to-close returns. Later holding days use standard
close-to-close returns.

Inputs
------
outputs/predictions/pred_*.parquet
data/features/features_by_year/year=*/part-*.parquet

Outputs
-------
outputs/tables/ic_by_period.csv
outputs/tables/ic_summary.csv
outputs/tables/cumulative_ic.csv
outputs/tables/decile_returns.csv
outputs/tables/decile_summary.csv
outputs/tables/decile_monotonicity.csv
outputs/tables/portfolio_returns.csv
outputs/tables/portfolio_turnover.csv
outputs/tables/portfolio_nav.csv
outputs/tables/performance_summary.csv
outputs/tables/cost_sensitivity.csv
outputs/tables/return_attribution.csv
"""

import argparse
from collections import defaultdict
import glob
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from config import (
    FEATURE_BY_YEAR_DIR,
    PRED_DIR,
    TABLE_DIR,
    N_DECILES,
    COST_BPS_GRID,
    UNIVERSE_SPLIT_METHOD,
    TRADING_DAYS_PER_YEAR,
)


PRED_REQUIRED_COLS = {
    "date",
    "code",
    "pred_prob",
    "future_ret",
    "float_mktcap",
    "horizon",
    "experiment_name",
    "model_name",
}

DAILY_RETURN_REQUIRED_COLS = {
    "date",
    "code",
    "ret_1d",
    "open_to_close_ret_1d",
    "volume",
    "is_suspended",
    "is_low_volume_limit_up",
    "is_low_volume_limit_down",
}

RETURN_CHECK_MAX_ROWS = 10_000
RETURN_CHECK_TOL = 1e-6
LOW_COVERAGE_VALID_WEIGHT = 0.95
LONG_SHORT_DECILE = 0
LONG_SHORT_PORTFOLIO_NAME = "D10_minus_D1"

GROUP_COLS = ["experiment_name", "model_name", "universe_group"]
PERIOD_GROUP_COLS = ["date"] + GROUP_COLS

ROW_RET_1D = 0
ROW_OPEN_TO_CLOSE_RET_1D = 1
ROW_VOLUME = 2
ROW_IS_SUSPENDED = 3
ROW_IS_LOW_VOLUME_LIMIT_UP = 4
ROW_IS_LOW_VOLUME_LIMIT_DOWN = 5


def log(message):
    """
    Print progress immediately in Slurm logs.
    """
    print(f"[06] {message}", flush=True)


def require_columns(df, required_cols, context):
    """
    Fail early with a clear missing-column error.
    """
    missing = sorted(set(required_cols) - set(df.columns))
    if missing:
        raise RuntimeError(f"{context} missing required columns: {missing}")


def safe_corr(x, y):
    """
    Correlation that returns NaN for too-small or constant samples.
    """
    valid = pd.DataFrame({"x": x, "y": y}).dropna()
    if len(valid) < 2:
        return np.nan
    if valid["x"].nunique() < 2 or valid["y"].nunique() < 2:
        return np.nan
    return valid["x"].corr(valid["y"])


def max_drawdown(nav):
    """
    Compute maximum drawdown from a NAV series.
    """
    nav = nav.dropna()
    if len(nav) == 0:
        return np.nan
    running_max = nav.cummax()
    drawdown = nav / running_max - 1.0
    return drawdown.min()


def load_prediction_file(path):
    """
    Read and validate one prediction parquet file.
    """
    log(f"Reading predictions: {path}")
    pred = pd.read_parquet(path, columns=sorted(PRED_REQUIRED_COLS))
    require_columns(pred, PRED_REQUIRED_COLS, path)

    pred = pred.copy()
    pred["date"] = pd.to_datetime(pred["date"])
    pred["pred_prob"] = pd.to_numeric(pred["pred_prob"], errors="coerce")
    pred["future_ret"] = pd.to_numeric(pred["future_ret"], errors="coerce")
    pred["float_mktcap"] = pd.to_numeric(pred["float_mktcap"], errors="coerce")
    pred["horizon"] = pd.to_numeric(pred["horizon"], errors="coerce")

    pred = pred[
        pred["date"].notna()
        & pred["code"].notna()
        & pred["pred_prob"].notna()
        & pred["future_ret"].notna()
        & pred["horizon"].notna()
    ].copy()

    if pred.empty:
        raise RuntimeError(f"{path} has no valid prediction rows.")

    return pred


def load_earliest_prediction_date(pred_files):
    """
    Read only prediction dates to find the earliest required trading-calendar date.
    """
    earliest = None

    for path in pred_files:
        dates = pd.read_parquet(path, columns=["date"])["date"]
        one_min = pd.to_datetime(dates, errors="coerce").min()
        if pd.isna(one_min):
            continue
        one_min = pd.Timestamp(one_min)
        earliest = one_min if earliest is None else min(earliest, one_min)

    return earliest


def load_daily_returns(min_date=None):
    """
    Read daily realized returns used by overlapping portfolio evaluation.
    """
    log(f"Reading daily returns: {FEATURE_BY_YEAR_DIR}")
    files = sorted(FEATURE_BY_YEAR_DIR.glob("year=*/part-*.parquet"))
    if not files:
        raise RuntimeError(
            f"No year-partitioned feature files found in {FEATURE_BY_YEAR_DIR}. "
            "Run 02_make_labels_and_baselines.py first."
        )

    available = set(pq.read_schema(files[0]).names)
    missing = sorted(DAILY_RETURN_REQUIRED_COLS - available)
    if missing:
        raise RuntimeError(
            f"{FEATURE_BY_YEAR_DIR} must include {missing}. "
            "Rerun 02_make_labels_and_baselines.py after this execution-return update."
        )

    try:
        read_kwargs = {"columns": sorted(DAILY_RETURN_REQUIRED_COLS)}
        if min_date is not None:
            read_kwargs["filters"] = [("date", ">=", pd.Timestamp(min_date))]
        daily_ret = pd.read_parquet(FEATURE_BY_YEAR_DIR, **read_kwargs)
    except Exception as exc:
        raise RuntimeError(
            f"{FEATURE_BY_YEAR_DIR} must include {sorted(DAILY_RETURN_REQUIRED_COLS)}. "
            "Rerun 02_make_labels_and_baselines.py after this execution-return update."
        ) from exc
    require_columns(daily_ret, DAILY_RETURN_REQUIRED_COLS, str(FEATURE_BY_YEAR_DIR))

    daily_ret = daily_ret.copy()
    daily_ret["date"] = pd.to_datetime(daily_ret["date"])
    daily_ret["ret_1d"] = pd.to_numeric(daily_ret["ret_1d"], errors="coerce")
    daily_ret["open_to_close_ret_1d"] = pd.to_numeric(
        daily_ret["open_to_close_ret_1d"],
        errors="coerce",
    )
    daily_ret["volume"] = pd.to_numeric(daily_ret["volume"], errors="coerce")
    for col in ["is_suspended", "is_low_volume_limit_up", "is_low_volume_limit_down"]:
        daily_ret[col] = pd.to_numeric(daily_ret[col], errors="coerce").fillna(0).astype("int8")
    daily_ret = daily_ret.dropna(subset=["date", "code"])

    if daily_ret.empty:
        raise RuntimeError(f"{FEATURE_BY_YEAR_DIR} has no valid daily return rows.")

    return daily_ret


def add_universe_groups(pred):
    """
    Add all / large_cap / small_mid_cap rows.

    The daily tercile split assigns the top one-third by float_mktcap to
    large_cap and the bottom two-thirds to small_mid_cap.
    """
    if UNIVERSE_SPLIT_METHOD != "daily_tercile":
        raise RuntimeError(f"Unsupported universe split method: {UNIVERSE_SPLIT_METHOD}")

    all_pred = pred.copy()
    all_pred["universe_group"] = "all"

    cap_pred = pred[pred["float_mktcap"].notna()].copy()
    if cap_pred.empty:
        return all_pred

    rank_pct = cap_pred.groupby("date")["float_mktcap"].rank(method="first", pct=True)
    cap_pred["universe_group"] = np.where(
        rank_pct > (2.0 / 3.0),
        "large_cap",
        "small_mid_cap",
    )

    return pd.concat([all_pred, cap_pred], ignore_index=True)


def calc_ic_by_period(pred):
    """
    Calculate IC and RankIC for each date and universe group.
    """
    rows = []
    groups = pred.groupby(PERIOD_GROUP_COLS, sort=True)
    total_groups = groups.ngroups
    last_progress = perf_counter()

    for idx, (keys, group) in enumerate(groups, start=1):
        date, exp_name, model_name, universe_group = keys
        valid = group[["pred_prob", "future_ret"]].dropna()

        ic = safe_corr(valid["pred_prob"], valid["future_ret"])
        rankic = safe_corr(valid["pred_prob"].rank(method="average"), valid["future_ret"].rank(method="average"))

        rows.append({
            "date": date,
            "experiment_name": exp_name,
            "model_name": model_name,
            "universe_group": universe_group,
            "num_stocks": len(valid),
            "ic": ic,
            "rankic": rankic,
        })

        now = perf_counter()
        if idx == total_groups or idx % 500 == 0 or now - last_progress >= 60:
            log(
                "IC progress: "
                f"{idx:,}/{total_groups:,} groups, "
                f"current={exp_name}/{model_name}/{universe_group}/{pd.Timestamp(date).date()}"
            )
            last_progress = now

    return pd.DataFrame(rows)


def summarize_metric(series):
    """
    Summary statistics for an IC-like time series.
    """
    valid = series.dropna()
    n = len(valid)
    if n == 0:
        return {
            "n": 0,
            "mean": np.nan,
            "std": np.nan,
            "ir": np.nan,
            "t_stat": np.nan,
            "positive_rate": np.nan,
        }

    mean = valid.mean()
    std = valid.std(ddof=1)
    ir = mean / std if std > 0 else np.nan
    t_stat = mean / (std / np.sqrt(n)) if std > 0 else np.nan

    return {
        "n": n,
        "mean": mean,
        "std": std,
        "ir": ir,
        "t_stat": t_stat,
        "positive_rate": (valid > 0).mean(),
    }


def calc_ic_summary(ic_by_period):
    """
    Summarize IC and RankIC across time.
    """
    rows = []

    for keys, group in ic_by_period.groupby(GROUP_COLS, sort=True):
        exp_name, model_name, universe_group = keys
        ic_stats = summarize_metric(group["ic"])
        rankic_stats = summarize_metric(group["rankic"])

        rows.append({
            "experiment_name": exp_name,
            "model_name": model_name,
            "universe_group": universe_group,
            "n_periods": len(group),
            "n_ic": ic_stats["n"],
            "ic_mean": ic_stats["mean"],
            "ic_std": ic_stats["std"],
            "icir": ic_stats["ir"],
            "ic_t_stat": ic_stats["t_stat"],
            "ic_positive_rate": ic_stats["positive_rate"],
            "n_rankic": rankic_stats["n"],
            "rankic_mean": rankic_stats["mean"],
            "rankic_std": rankic_stats["std"],
            "rankicir": rankic_stats["ir"],
            "rankic_t_stat": rankic_stats["t_stat"],
            "rankic_positive_rate": rankic_stats["positive_rate"],
        })

    return pd.DataFrame(rows)


def calc_cumulative_ic(ic_by_period):
    """
    Cumulative sum of IC and RankIC over time.
    """
    out = ic_by_period.sort_values(PERIOD_GROUP_COLS).copy()
    group = out.groupby(GROUP_COLS, sort=False)
    out["cumulative_ic"] = group["ic"].transform(lambda x: x.fillna(0.0).cumsum())
    out["cumulative_rankic"] = group["rankic"].transform(lambda x: x.fillna(0.0).cumsum())
    return out


def assign_deciles(pred):
    """
    Assign cross-sectional deciles within date/model/universe group.

    D10 has the highest predicted probability.
    """
    pred = pred.copy()
    pred["decile"] = np.nan
    groups = pred.groupby(PERIOD_GROUP_COLS, sort=False).groups
    total_groups = len(groups)
    last_progress = perf_counter()

    for idx, (keys, index) in enumerate(groups.items(), start=1):
        if len(index) < N_DECILES:
            continue

        rank = pred.loc[index, "pred_prob"].rank(method="first")
        decile = pd.qcut(rank, N_DECILES, labels=False) + 1
        pred.loc[index, "decile"] = decile.astype(float).values

        now = perf_counter()
        if idx == total_groups or idx % 500 == 0 or now - last_progress >= 60:
            date, exp_name, model_name, universe_group = keys
            log(
                "Decile assignment progress: "
                f"{idx:,}/{total_groups:,} groups, "
                f"current={exp_name}/{model_name}/{universe_group}/{pd.Timestamp(date).date()}"
            )
            last_progress = now

    return pred


def calc_decile_returns(pred):
    """
    Average future return for D1-D10 on each date.
    """
    valid = pred.dropna(subset=["decile"]).copy()
    if valid.empty:
        return pd.DataFrame()

    valid["decile"] = valid["decile"].astype(int)

    decile_ret = (
        valid
        .groupby(PERIOD_GROUP_COLS + ["decile"], sort=True)
        .agg(
            mean_future_ret=("future_ret", "mean"),
            median_future_ret=("future_ret", "median"),
            avg_pred_prob=("pred_prob", "mean"),
            num_stocks=("code", "count"),
        )
        .reset_index()
    )

    decile_ret["portfolio_name"] = "D" + decile_ret["decile"].astype(str)
    return decile_ret


def calc_decile_summary(decile_returns):
    """
    Full-period D1-D10 average future return summary.
    """
    if decile_returns.empty:
        return pd.DataFrame()

    return (
        decile_returns
        .groupby(GROUP_COLS + ["decile", "portfolio_name"], sort=True)
        .agg(
            n_periods=("date", "count"),
            mean_future_ret=("mean_future_ret", "mean"),
            std_period_future_ret=("mean_future_ret", "std"),
            median_period_future_ret=("median_future_ret", "median"),
            avg_pred_prob=("avg_pred_prob", "mean"),
            avg_num_stocks=("num_stocks", "mean"),
        )
        .reset_index()
    )


def calc_decile_monotonicity(decile_summary):
    """
    Check whether average future return rises from D1 to D10.
    """
    rows = []

    if decile_summary.empty:
        return pd.DataFrame()

    for keys, group in decile_summary.groupby(GROUP_COLS, sort=True):
        exp_name, model_name, universe_group = keys
        group = group.sort_values("decile").dropna(subset=["mean_future_ret"])

        deciles = group["decile"].astype(float)
        returns = group["mean_future_ret"].astype(float)
        diffs = returns.diff().dropna()

        if len(group) >= 2:
            decile_return_spearman = safe_corr(
                deciles.rank(method="average"),
                returns.rank(method="average"),
            )
            strict = bool((diffs > 0).all())
            nonstrict = bool((diffs >= 0).all())
            violations = int((diffs < 0).sum())
        else:
            decile_return_spearman = np.nan
            strict = False
            nonstrict = False
            violations = np.nan

        rows.append({
            "experiment_name": exp_name,
            "model_name": model_name,
            "universe_group": universe_group,
            "valid_decile_count": len(group),
            "decile_return_spearman": decile_return_spearman,
            "strict_monotonic_increasing": strict,
            "nonstrict_monotonic_increasing": nonstrict,
            "adjacent_violation_count": violations,
        })

    return pd.DataFrame(rows)


def floating_lookup_dtype(series):
    """
    Preserve the loaded floating dtype when it supports NaN, otherwise use float64.
    """
    dtype = series.to_numpy().dtype
    return dtype if np.issubdtype(dtype, np.floating) else np.dtype("float64")


def prepare_daily_return_lookup(daily_returns):
    """
    Build a dense date/code array lookup for execution and return fields.

    Duplicate date/code rows retain the existing aggregation semantics:
    returns and volume use mean, while execution flags use max.
    """
    trading_dates = pd.DatetimeIndex(sorted(daily_returns["date"].dropna().unique()))
    date_to_pos = {date: i for i, date in enumerate(trading_dates)}
    codes = pd.Index(pd.unique(daily_returns["code"].dropna()))
    code_to_id = {code: i for i, code in enumerate(codes)}
    shape = (len(trading_dates), len(codes))

    lookup = {
        "code_to_id": code_to_id,
        "present": np.zeros(shape, dtype=bool),
        "ret_1d": np.full(
            shape,
            np.nan,
            dtype=floating_lookup_dtype(daily_returns["ret_1d"]),
        ),
        "open_to_close_ret_1d": np.full(
            shape,
            np.nan,
            dtype=floating_lookup_dtype(daily_returns["open_to_close_ret_1d"]),
        ),
        "volume": np.full(
            shape,
            np.nan,
            dtype=floating_lookup_dtype(daily_returns["volume"]),
        ),
        "is_suspended": np.zeros(shape, dtype=np.int8),
        "is_low_volume_limit_up": np.zeros(shape, dtype=np.int8),
        "is_low_volume_limit_down": np.zeros(shape, dtype=np.int8),
    }

    if daily_returns.duplicated(["date", "code"]).any():
        aggregated = (
            daily_returns
            .groupby(["date", "code"], sort=True, as_index=False)
            .agg(
                ret_1d=("ret_1d", "mean"),
                open_to_close_ret_1d=("open_to_close_ret_1d", "mean"),
                volume=("volume", "mean"),
                is_suspended=("is_suspended", "max"),
                is_low_volume_limit_up=("is_low_volume_limit_up", "max"),
                is_low_volume_limit_down=("is_low_volume_limit_down", "max"),
            )
        )
    else:
        aggregated = daily_returns

    date_positions = trading_dates.get_indexer(aggregated["date"])
    code_ids = aggregated["code"].map(code_to_id).to_numpy(dtype=np.int64)
    lookup["present"][date_positions, code_ids] = True
    for col in [
        "ret_1d",
        "open_to_close_ret_1d",
        "volume",
        "is_suspended",
        "is_low_volume_limit_up",
        "is_low_volume_limit_down",
    ]:
        lookup[col][date_positions, code_ids] = aggregated[col].to_numpy()

    return trading_dates, date_to_pos, lookup


def get_daily_info(lookup, trading_dates, return_pos):
    if return_pos < 0 or return_pos >= len(trading_dates):
        return None
    return lookup, return_pos


def code_row(daily_info, code):
    if daily_info is None:
        return None
    lookup, date_pos = daily_info
    code_id = lookup["code_to_id"].get(code)
    if code_id is None or not lookup["present"][date_pos, code_id]:
        return None
    return (
        lookup["ret_1d"][date_pos, code_id],
        lookup["open_to_close_ret_1d"][date_pos, code_id],
        lookup["volume"][date_pos, code_id],
        lookup["is_suspended"][date_pos, code_id],
        lookup["is_low_volume_limit_up"][date_pos, code_id],
        lookup["is_low_volume_limit_down"][date_pos, code_id],
    )


def row_return_value(row, return_col):
    if return_col == "open_to_close_ret_1d":
        return row[ROW_OPEN_TO_CLOSE_RET_1D]
    return row[ROW_RET_1D]


def is_valid_buy_row(row):
    if row is None:
        return False
    if int(row[ROW_IS_SUSPENDED]) == 1:
        return False
    if int(row[ROW_IS_LOW_VOLUME_LIMIT_UP]) == 1:
        return False
    if pd.isna(row[ROW_OPEN_TO_CLOSE_RET_1D]):
        return False
    if pd.isna(row[ROW_VOLUME]) or float(row[ROW_VOLUME]) <= 0:
        return False
    return True


def is_valid_sell_row(row, return_col):
    if row is None:
        return False
    if int(row[ROW_IS_SUSPENDED]) == 1:
        return False
    if int(row[ROW_IS_LOW_VOLUME_LIMIT_DOWN]) == 1:
        return False
    if pd.isna(row_return_value(row, return_col)):
        return False
    return True


def get_code_return(row, return_col):
    if row is None:
        return np.nan, "missing"

    ret = row_return_value(row, return_col)
    if not pd.isna(ret):
        return float(ret), "valid"

    if int(row[ROW_IS_SUSPENDED]) == 1:
        return 0.0, "suspended"

    return np.nan, "missing"


def filter_buyable_code_ids(code_ids, daily_info):
    """
    Keep only stocks that can be bought at the next open.

    Cohorts store integer code ids so the hot path can use vectorized array
    lookups instead of repeated code -> row dictionary access.
    """
    code_ids = np.asarray(code_ids, dtype=np.int32)
    if len(code_ids) == 0:
        return code_ids, 0, 0
    if daily_info is None:
        return np.empty(0, dtype=np.int32), int(len(code_ids)), int(len(code_ids))

    lookup, date_pos = daily_info
    valid_id = code_ids >= 0
    safe_ids = code_ids[valid_id]

    present = np.zeros(len(code_ids), dtype=bool)
    open_ret = np.full(len(code_ids), np.nan, dtype=lookup["open_to_close_ret_1d"].dtype)
    volume = np.full(len(code_ids), np.nan, dtype=lookup["volume"].dtype)
    suspended = np.ones(len(code_ids), dtype=bool)
    low_volume_limit_up = np.zeros(len(code_ids), dtype=bool)

    if len(safe_ids):
        present_valid = lookup["present"][date_pos, safe_ids]
        valid_positions = np.flatnonzero(valid_id)
        present[valid_positions] = present_valid
        open_ret[valid_positions] = lookup["open_to_close_ret_1d"][date_pos, safe_ids]
        volume[valid_positions] = lookup["volume"][date_pos, safe_ids]
        suspended[valid_positions] = lookup["is_suspended"][date_pos, safe_ids].astype(bool)
        low_volume_limit_up[valid_positions] = lookup["is_low_volume_limit_up"][date_pos, safe_ids].astype(bool)

    buyable_mask = (
        valid_id
        & present
        & (~suspended)
        & (~low_volume_limit_up)
        & np.isfinite(open_ret)
        & np.isfinite(volume)
        & (volume > 0)
    )
    data_missing_mask = (~valid_id) | (~present) | ~np.isfinite(open_ret)
    return (
        code_ids[buyable_mask].astype(np.int32, copy=False),
        int((~buyable_mask).sum()),
        int(data_missing_mask.sum()),
    )


def unique_nonnull(values):
    """
    Keep first occurrence of each non-null code.
    """
    result = []
    seen = set()

    for value in values:
        if pd.isna(value) or value in seen:
            continue
        seen.add(value)
        result.append(value)

    return result


def make_weight_state_from_cohorts(cohorts):
    """
    Convert independent subportfolio cohorts into vectorized target weights.

    Each signal cohort uses one horizon slot. A 20-day signal therefore receives
    1/20 total portfolio weight, split equally across stocks in that cohort.
    Before the first horizon days have elapsed, the strategy is intentionally
    under-invested. Forced-hold cohorts can extend realized exposure beyond the
    target horizon and are tracked through diagnostics.
    """
    id_parts = []
    weight_parts = []
    static_missing_weight = 0.0
    static_missing_count = 0

    for cohort in cohorts:
        n_codes = int(cohort.get("n_codes", 0))
        if n_codes <= 0:
            continue
        horizon = int(cohort.get("horizon", 0))
        if horizon <= 0:
            continue

        stock_weight = (1.0 / horizon) / n_codes
        code_ids = np.asarray(cohort.get("code_ids", np.empty(0, dtype=np.int32)), dtype=np.int32)
        known_ids = code_ids[code_ids >= 0]
        if len(known_ids):
            id_parts.append(known_ids)
            weight_parts.append(np.full(len(known_ids), stock_weight, dtype=np.float64))

        missing_count = n_codes - len(known_ids)
        if missing_count > 0:
            static_missing_count += int(missing_count)
            static_missing_weight += float(missing_count * stock_weight)

    if not id_parts:
        return {
            "ids": np.empty(0, dtype=np.int32),
            "weights": np.empty(0, dtype=np.float64),
            "static_missing_weight": static_missing_weight,
            "static_missing_count": static_missing_count,
        }

    ids = np.concatenate(id_parts).astype(np.int32, copy=False)
    weights = np.concatenate(weight_parts).astype(np.float64, copy=False)
    unique_ids, inverse = np.unique(ids, return_inverse=True)
    summed_weights = np.bincount(inverse, weights=weights).astype(np.float64, copy=False)
    return {
        "ids": unique_ids.astype(np.int32, copy=False),
        "weights": summed_weights,
        "static_missing_weight": static_missing_weight,
        "static_missing_count": static_missing_count,
    }


def make_weights_from_cohorts(cohorts):
    """
    Compatibility helper returning a dict-like target vector.
    """
    state = make_weight_state_from_cohorts(cohorts)
    return dict(zip(state["ids"].tolist(), state["weights"].tolist()))


def calc_weight_turnover(prev_weights, next_weights):
    """
    Sum absolute target-weight changes.
    """
    codes = set(prev_weights) | set(next_weights)
    return float(sum(abs(next_weights.get(code, 0.0) - prev_weights.get(code, 0.0)) for code in codes))


def calc_weight_turnover_states(prev_state, next_state):
    """
    Sum absolute target-weight changes for sorted vector weight states.
    """
    prev_ids = prev_state["ids"]
    next_ids = next_state["ids"]
    if len(prev_ids) == 0 and len(next_ids) == 0:
        return 0.0
    all_ids = np.union1d(prev_ids, next_ids)
    prev_aligned = np.zeros(len(all_ids), dtype=np.float64)
    next_aligned = np.zeros(len(all_ids), dtype=np.float64)
    if len(prev_ids):
        pos = np.searchsorted(all_ids, prev_ids)
        prev_aligned[pos] = prev_state["weights"]
    if len(next_ids):
        pos = np.searchsorted(all_ids, next_ids)
        next_aligned[pos] = next_state["weights"]
    return float(np.abs(next_aligned - prev_aligned).sum())


def start_ids_from_cohorts(cohorts, return_pos):
    """
    Code ids that require open-to-close returns on this holding day.
    """
    parts = [
        np.asarray(cohort.get("code_ids", np.empty(0, dtype=np.int32)), dtype=np.int32)
        for cohort in cohorts
        if int(cohort.get("start_pos", -1)) == int(return_pos)
    ]
    if not parts:
        return np.empty(0, dtype=np.int32)
    ids = np.concatenate(parts)
    ids = ids[ids >= 0]
    return np.unique(ids).astype(np.int32, copy=False)


def calc_weighted_return_state(weight_state, daily_info, start_ids):
    """
    Weighted average return using only valid marked prices.

    The returned gross return is normalized by valid_weight, so one missing stock
    no longer turns the entire portfolio day into NaN. valid_weight and
    missing_weight make the coverage loss explicit for performance diagnostics.
    """
    ids = weight_state["ids"]
    weights = weight_state["weights"]
    static_missing_weight = float(weight_state.get("static_missing_weight", 0.0))
    static_missing_count = int(weight_state.get("static_missing_count", 0))
    if len(ids) == 0:
        if static_missing_weight > 0:
            return {
                "return": np.nan,
                "weighted_return": np.nan,
                "valid_weight": 0.0,
                "missing_weight": static_missing_weight,
                "data_missing": static_missing_count,
            }
        return empty_return_info(0.0)

    if daily_info is None:
        return {
            "return": np.nan,
            "weighted_return": np.nan,
            "valid_weight": 0.0,
            "missing_weight": float(weights.sum() + static_missing_weight),
            "data_missing": int(len(ids) + static_missing_count),
        }

    lookup, date_pos = daily_info
    present = lookup["present"][date_pos, ids]
    use_open = np.isin(ids, start_ids, assume_unique=True)
    ret = lookup["ret_1d"][date_pos, ids].astype(np.float64, copy=True)
    if use_open.any():
        ret[use_open] = lookup["open_to_close_ret_1d"][date_pos, ids[use_open]]
    suspended = lookup["is_suspended"][date_pos, ids].astype(bool)

    finite_ret = np.isfinite(ret)
    suspended_zero = present & (~finite_ret) & suspended
    valid_mask = present & (finite_ret | suspended_zero)
    ret[suspended_zero] = 0.0

    if valid_mask.any():
        valid_weight = float(weights[valid_mask].sum())
        weighted_return = float(np.dot(weights[valid_mask], ret[valid_mask]))
    else:
        valid_weight = 0.0
        weighted_return = np.nan

    missing_mask = ~valid_mask
    missing_weight = float(weights[missing_mask].sum() + static_missing_weight)
    data_missing = int(missing_mask.sum() + static_missing_count)

    if valid_weight <= 0:
        return {
            "return": np.nan,
            "weighted_return": np.nan,
            "valid_weight": 0.0,
            "missing_weight": missing_weight,
            "data_missing": data_missing,
        }

    return {
        "return": float(weighted_return / valid_weight),
        "weighted_return": float(weighted_return),
        "valid_weight": float(valid_weight),
        "missing_weight": float(missing_weight),
        "data_missing": data_missing,
    }


def calc_weighted_return(weights, daily_info, return_pos, start_pos_by_code):
    """
    Compatibility wrapper for dict weights.
    """
    if not weights:
        return empty_return_info(0.0)
    state = {
        "ids": np.asarray(list(weights), dtype=np.int32),
        "weights": np.asarray(list(weights.values()), dtype=np.float64),
        "static_missing_weight": 0.0,
        "static_missing_count": 0,
    }
    start_ids = np.asarray(
        [code for code in weights if start_pos_by_code.get(code) == return_pos],
        dtype=np.int32,
    )
    return calc_weighted_return_state(state, daily_info, start_ids)


def empty_return_info(return_value=0.0):
    """
    Return diagnostics for an empty cash portfolio.
    """
    return {
        "return": return_value,
        "weighted_return": return_value,
        "valid_weight": 0.0,
        "missing_weight": 0.0,
        "data_missing": 0,
    }


def start_pos_by_code_from_cohorts(cohorts):
    """
    Map each code to the first holding date of its active cohort.

    If a code appears in multiple active cohorts, use the earliest start date
    that still requires open-to-close handling on its first holding day.
    """
    out = {}
    for cohort in cohorts:
        start_pos = cohort["start_pos"]
        for code in np.asarray(cohort.get("code_ids", np.empty(0, dtype=np.int32)), dtype=np.int32):
            if code < 0:
                continue
            out[int(code)] = min(out.get(int(code), start_pos), start_pos)
    return out


def calc_active_cohort_return(active_cohorts, daily_info, return_pos):
    """
    Daily portfolio return from active cohorts.
    """
    weight_state = make_weight_state_from_cohorts(active_cohorts)
    return calc_weighted_return_state(weight_state, daily_info, start_ids_from_cohorts(active_cohorts, return_pos))


def retire_expired_cohorts(active_cohorts, return_pos):
    """
    Remove cohorts after their planned holding period without execution delay.
    """
    return [
        cohort
        for cohort in active_cohorts
        if int(cohort.get("n_codes", 0)) > 0 and return_pos < cohort["target_end_pos"]
    ]


def finite_or_zero(value):
    return 0.0 if pd.isna(value) else float(value)


def empty_weight_state():
    return {
        "ids": np.empty(0, dtype=np.int32),
        "weights": np.empty(0, dtype=np.float64),
        "static_missing_weight": 0.0,
        "static_missing_count": 0,
    }


def process_sell_attempts(active_cohorts, daily_info, return_pos):
    """
    Sell codes whose holding period ended unless execution is blocked.
    """
    next_active = []
    blocked_sells = 0
    forced_holds = 0
    data_missing = 0

    for cohort in active_cohorts:
        if return_pos < cohort["target_end_pos"]:
            next_active.append(cohort)
            continue

        return_col = "open_to_close_ret_1d" if return_pos == cohort["start_pos"] else "ret_1d"
        code_ids = np.asarray(cohort.get("code_ids", np.empty(0, dtype=np.int32)), dtype=np.int32)
        if len(code_ids) == 0:
            continue

        if daily_info is None:
            keep_ids = code_ids
            data_missing += int(len(code_ids))
        else:
            lookup, date_pos = daily_info
            present = lookup["present"][date_pos, code_ids]
            if return_col == "open_to_close_ret_1d":
                ret = lookup["open_to_close_ret_1d"][date_pos, code_ids]
            else:
                ret = lookup["ret_1d"][date_pos, code_ids]
            suspended = lookup["is_suspended"][date_pos, code_ids].astype(bool)
            low_down = lookup["is_low_volume_limit_down"][date_pos, code_ids].astype(bool)
            valid_sell = present & (~suspended) & (~low_down) & np.isfinite(ret)
            keep_ids = code_ids[~valid_sell]
            data_missing += int(((~present) | ~np.isfinite(ret)).sum())

        blocked_sells += int(len(keep_ids))
        forced_holds += int(len(keep_ids))

        if len(keep_ids):
            next_cohort = cohort.copy()
            next_cohort["code_ids"] = keep_ids.astype(np.int32, copy=False)
            next_cohort["n_codes"] = int(len(keep_ids))
            next_active.append(next_cohort)

    return next_active, blocked_sells, forced_holds, data_missing


def compound_return_for_row(row, trading_dates, date_to_pos, lookup):
    """
    Rebuild one future return from daily open-to-close and close-to-close returns.
    """
    signal_pos = date_to_pos.get(pd.Timestamp(row["date"]))
    if signal_pos is None:
        return np.nan

    horizon = int(row["horizon"])
    start_pos = signal_pos + 1
    end_pos = signal_pos + horizon
    if start_pos >= len(trading_dates) or end_pos >= len(trading_dates):
        return np.nan

    gross = 1.0
    for pos in range(start_pos, end_pos + 1):
        info = get_daily_info(lookup, trading_dates, pos)
        return_col = "open_to_close_ret_1d" if pos == start_pos else "ret_1d"
        ret, status = get_code_return(code_row(info, row["code"]), return_col)
        if status == "missing" or pd.isna(ret):
            return np.nan
        gross *= 1.0 + ret

    return gross - 1.0


def warn_if_compound_returns_mismatch(pred, trading_dates, date_to_pos, lookup):
    """
    Check that configured future returns match compounded daily returns.
    """
    sample = pred[["date", "code", "horizon", "future_ret"]].dropna().head(RETURN_CHECK_MAX_ROWS)
    if sample.empty:
        return

    diffs = []
    checked = 0
    for row in sample.to_dict("records"):
        rebuilt = compound_return_for_row(row, trading_dates, date_to_pos, lookup)
        if pd.isna(rebuilt):
            continue
        checked += 1
        diffs.append(abs(float(row["future_ret"]) - rebuilt))

    if not diffs:
        log("[Warning] Return compounding check skipped: no rows with complete daily returns.")
        return

    max_diff = max(diffs)
    if max_diff > RETURN_CHECK_TOL:
        log(
            "[Warning] Future-return compounding check exceeded tolerance: "
            f"checked={checked}, max_abs_diff={max_diff:.8g}, tol={RETURN_CHECK_TOL}"
        )


def build_signal_cohorts(group, date_to_pos, max_date_pos, lookup):
    """
    Convert signal rows into cohorts keyed by first holding date.
    """
    group = group.copy()
    group["signal_pos"] = group["date"].map(date_to_pos)
    group = group[group["signal_pos"].notna()].copy()

    signals_by_pos = defaultdict(list)
    max_end_pos = -1

    if group.empty:
        return signals_by_pos, max_end_pos

    group["signal_pos"] = group["signal_pos"].astype(int)
    group["decile"] = group["decile"].astype(int)
    group["horizon"] = group["horizon"].astype(int)

    for (signal_pos, decile, horizon), cohort in group.groupby(["signal_pos", "decile", "horizon"]):
        signal_pos = int(signal_pos)
        decile = int(decile)
        horizon = int(horizon)
        if horizon <= 0:
            continue

        codes = unique_nonnull(cohort["code"])
        if not codes:
            continue
        mapped_ids = [lookup["code_to_id"].get(code, -1) for code in codes]
        code_ids = np.asarray(mapped_ids, dtype=np.int32)

        start_pos = signal_pos + 1
        end_pos = min(signal_pos + horizon, max_date_pos)
        if start_pos > end_pos:
            continue

        signals_by_pos[start_pos].append({
            "decile": decile,
            "start_pos": start_pos,
            "target_end_pos": end_pos,
            "horizon": horizon,
            "code_ids": code_ids,
            "n_codes": int(len(code_ids)),
        })
        max_end_pos = max(max_end_pos, end_pos)

    return signals_by_pos, max_end_pos


def expand_portfolio_cost_rows(base):
    """
    Expand a base daily portfolio DataFrame across the configured cost grid.

    Row-major repetition preserves the original ordering: all configured costs
    for one base row appear before the next base row.
    """
    if base.empty:
        return pd.DataFrame()

    costs = np.asarray(COST_BPS_GRID)
    expanded = base.loc[base.index.repeat(len(costs))].reset_index(drop=True)
    expanded.insert(6, "cost_bps", np.tile(costs, len(base)))
    net_return = (
        expanded["gross_return"]
        - expanded["turnover"] * (expanded["cost_bps"].astype(float) / 10_000.0)
    )
    expanded.insert(11, "net_return", net_return)
    expanded["turnover_cost"] = expanded["turnover"] * (expanded["cost_bps"].astype(float) / 10_000.0)
    expanded["missing_return_data_issue"] = (
        expanded["signal_gross_alpha"].map(finite_or_zero)
        - expanded["buy_blocked_loss"].map(finite_or_zero)
        - expanded["sell_blocked_forced_hold_loss"].map(finite_or_zero)
        - expanded["turnover_cost"].map(finite_or_zero)
        - expanded["net_return"].map(finite_or_zero)
    )
    expanded["attributed_net_return"] = (
        expanded["signal_gross_alpha"].map(finite_or_zero)
        - expanded["buy_blocked_loss"].map(finite_or_zero)
        - expanded["sell_blocked_forced_hold_loss"].map(finite_or_zero)
        - expanded["missing_return_data_issue"].map(finite_or_zero)
        - expanded["turnover_cost"].map(finite_or_zero)
    )
    return expanded


def add_long_short_return_rows(portfolio_returns):
    """
    Add D10-D1 spread rows derived from the long-only D10 and D1 legs.
    """
    if portfolio_returns.empty:
        return portfolio_returns

    key_cols = ["date"] + GROUP_COLS + ["cost_bps"]
    d10 = portfolio_returns[portfolio_returns["decile"] == N_DECILES].copy()
    d1 = portfolio_returns[portfolio_returns["decile"] == 1].copy()
    if d10.empty or d1.empty:
        return portfolio_returns

    merged = d10.merge(d1, on=key_cols, suffixes=("_d10", "_d1"), how="inner")
    if merged.empty:
        return portfolio_returns

    cost_rate = merged["cost_bps"].astype(float) / 10_000.0
    turnover = merged["turnover_d10"].fillna(0.0) + merged["turnover_d1"].fillna(0.0)
    gross_return = merged["gross_return_d10"] - merged["gross_return_d1"]
    turnover_cost = turnover * cost_rate

    rows = pd.DataFrame({
        "date": merged["date"],
        "experiment_name": merged["experiment_name"],
        "model_name": merged["model_name"],
        "universe_group": merged["universe_group"],
        "decile": LONG_SHORT_DECILE,
        "portfolio_name": LONG_SHORT_PORTFOLIO_NAME,
        "cost_bps": merged["cost_bps"],
        "gross_return": gross_return,
        "turnover": turnover,
        "start_turnover": merged["start_turnover_d10"].fillna(0.0) + merged["start_turnover_d1"].fillna(0.0),
        "end_turnover": merged["end_turnover_d10"].fillna(0.0) + merged["end_turnover_d1"].fillna(0.0),
        "net_return": gross_return - turnover_cost,
        "active_cohorts": merged["active_cohorts_d10"].fillna(0) + merged["active_cohorts_d1"].fillna(0),
        "num_holdings": merged["num_holdings_d10"].fillna(0) + merged["num_holdings_d1"].fillna(0),
        "num_blocked_buys": merged["num_blocked_buys_d10"].fillna(0) + merged["num_blocked_buys_d1"].fillna(0),
        "num_blocked_sells": merged["num_blocked_sells_d10"].fillna(0) + merged["num_blocked_sells_d1"].fillna(0),
        "num_forced_holds": merged["num_forced_holds_d10"].fillna(0) + merged["num_forced_holds_d1"].fillna(0),
        "num_data_missing_returns": merged["num_data_missing_returns_d10"].fillna(0) + merged["num_data_missing_returns_d1"].fillna(0),
        "valid_weight": np.minimum(
            merged["valid_weight_d10"].fillna(0.0),
            merged["valid_weight_d1"].fillna(0.0),
        ),
        "missing_weight": merged["missing_weight_d10"].fillna(0.0) + merged["missing_weight_d1"].fillna(0.0),
        "signal_valid_weight": np.minimum(
            merged["signal_valid_weight_d10"].fillna(0.0),
            merged["signal_valid_weight_d1"].fillna(0.0),
        ),
        "signal_missing_weight": (
            merged["signal_missing_weight_d10"].fillna(0.0)
            + merged["signal_missing_weight_d1"].fillna(0.0)
        ),
        "signal_gross_alpha": merged["signal_gross_alpha_d10"] - merged["signal_gross_alpha_d1"],
        "buy_constrained_gross_return": (
            merged["buy_constrained_gross_return_d10"] - merged["buy_constrained_gross_return_d1"]
        ),
        "buy_blocked_loss": merged["buy_blocked_loss_d10"] - merged["buy_blocked_loss_d1"],
        "sell_blocked_forced_hold_loss": (
            merged["sell_blocked_forced_hold_loss_d10"]
            - merged["sell_blocked_forced_hold_loss_d1"]
        ),
        "is_warmup": merged["is_warmup_d10"].fillna(False) | merged["is_warmup_d1"].fillna(False),
        "turnover_cost": turnover_cost,
    })
    rows["missing_return_data_issue"] = (
        rows["signal_gross_alpha"].map(finite_or_zero)
        - rows["buy_blocked_loss"].map(finite_or_zero)
        - rows["sell_blocked_forced_hold_loss"].map(finite_or_zero)
        - rows["turnover_cost"].map(finite_or_zero)
        - rows["net_return"].map(finite_or_zero)
    )
    rows["attributed_net_return"] = (
        rows["signal_gross_alpha"].map(finite_or_zero)
        - rows["buy_blocked_loss"].map(finite_or_zero)
        - rows["sell_blocked_forced_hold_loss"].map(finite_or_zero)
        - rows["missing_return_data_issue"].map(finite_or_zero)
        - rows["turnover_cost"].map(finite_or_zero)
    )

    return pd.concat([portfolio_returns, rows], ignore_index=True)


def add_long_short_turnover_rows(portfolio_turnover):
    """
    Add D10-D1 turnover diagnostics derived from both long-only legs.
    """
    if portfolio_turnover.empty:
        return portfolio_turnover

    key_cols = ["date"] + GROUP_COLS
    d10 = portfolio_turnover[portfolio_turnover["decile"] == N_DECILES].copy()
    d1 = portfolio_turnover[portfolio_turnover["decile"] == 1].copy()
    if d10.empty or d1.empty:
        return portfolio_turnover

    merged = d10.merge(d1, on=key_cols, suffixes=("_d10", "_d1"), how="inner")
    if merged.empty:
        return portfolio_turnover

    rows = pd.DataFrame({
        "date": merged["date"],
        "experiment_name": merged["experiment_name"],
        "model_name": merged["model_name"],
        "universe_group": merged["universe_group"],
        "decile": LONG_SHORT_DECILE,
        "portfolio_name": LONG_SHORT_PORTFOLIO_NAME,
        "turnover": merged["turnover_d10"].fillna(0.0) + merged["turnover_d1"].fillna(0.0),
        "start_turnover": merged["start_turnover_d10"].fillna(0.0) + merged["start_turnover_d1"].fillna(0.0),
        "end_turnover": merged["end_turnover_d10"].fillna(0.0) + merged["end_turnover_d1"].fillna(0.0),
        "active_cohorts": merged["active_cohorts_d10"].fillna(0) + merged["active_cohorts_d1"].fillna(0),
        "num_holdings": merged["num_holdings_d10"].fillna(0) + merged["num_holdings_d1"].fillna(0),
        "num_blocked_buys": merged["num_blocked_buys_d10"].fillna(0) + merged["num_blocked_buys_d1"].fillna(0),
        "num_blocked_sells": merged["num_blocked_sells_d10"].fillna(0) + merged["num_blocked_sells_d1"].fillna(0),
        "num_forced_holds": merged["num_forced_holds_d10"].fillna(0) + merged["num_forced_holds_d1"].fillna(0),
        "num_data_missing_returns": (
            merged["num_data_missing_returns_d10"].fillna(0)
            + merged["num_data_missing_returns_d1"].fillna(0)
        ),
        "valid_weight": np.minimum(
            merged["valid_weight_d10"].fillna(0.0),
            merged["valid_weight_d1"].fillna(0.0),
        ),
        "missing_weight": merged["missing_weight_d10"].fillna(0.0) + merged["missing_weight_d1"].fillna(0.0),
        "is_warmup": merged["is_warmup_d10"].fillna(False) | merged["is_warmup_d1"].fillna(False),
    })

    return pd.concat([portfolio_turnover, rows], ignore_index=True)


def calc_overlapping_portfolios(pred, daily_context):
    """
    Build D1-D10 long-only daily portfolio returns with overlapping cohorts.
    """
    signals = pred.dropna(subset=["decile", "horizon"]).copy()
    if signals.empty:
        return pd.DataFrame(), pd.DataFrame()

    trading_dates, date_to_pos, lookup = daily_context
    if len(trading_dates) == 0:
        return pd.DataFrame(), pd.DataFrame()

    warn_if_compound_returns_mismatch(signals, trading_dates, date_to_pos, lookup)

    base_portfolio_rows = []
    turnover_rows = []
    max_date_pos = len(trading_dates) - 1

    for keys, group in signals.groupby(GROUP_COLS, sort=True):
        exp_name, model_name, universe_group = keys
        group_start = perf_counter()
        log(
            "Portfolio group start: "
            f"{exp_name}/{model_name}/{universe_group}, signal_rows={len(group):,}"
        )
        signals_by_pos, max_end_pos = build_signal_cohorts(group, date_to_pos, max_date_pos, lookup)
        if max_end_pos < 0:
            log(
                "Portfolio group skipped: "
                f"{exp_name}/{model_name}/{universe_group}, no valid cohorts"
            )
            continue

        min_start_pos = min(signals_by_pos)
        total_days = max_date_pos - min_start_pos + 1
        active = {decile: [] for decile in range(1, N_DECILES + 1)}
        forced_active = {decile: [] for decile in range(1, N_DECILES + 1)}
        sell_due = {decile: defaultdict(list) for decile in range(1, N_DECILES + 1)}
        ideal_active = {decile: [] for decile in range(1, N_DECILES + 1)}
        buy_constrained_active = {decile: [] for decile in range(1, N_DECILES + 1)}
        prev_weights = {decile: empty_weight_state() for decile in range(1, N_DECILES + 1)}
        group_horizon = int(group["horizon"].dropna().astype(int).max())
        last_progress = perf_counter()

        for return_pos in range(min_start_pos, max_date_pos + 1):
            date = trading_dates[return_pos]
            daily_info = get_daily_info(lookup, trading_dates, return_pos)
            day_i = return_pos - min_start_pos + 1
            now = perf_counter()
            if day_i == 1 or day_i == total_days or day_i % 250 == 0 or now - last_progress >= 60:
                active_cohorts = sum(len(active[d]) for d in range(1, N_DECILES + 1))
                forced_cohorts = sum(len(forced_active[d]) for d in range(1, N_DECILES + 1))
                active_holdings = sum(
                    int(cohort.get("n_codes", 0))
                    for d in range(1, N_DECILES + 1)
                    for cohort in active[d]
                ) + sum(
                    int(cohort.get("n_codes", 0))
                    for d in range(1, N_DECILES + 1)
                    for cohort in forced_active[d]
                )
                log(
                    "Portfolio progress: "
                    f"{exp_name}/{model_name}/{universe_group}, "
                    f"day={day_i:,}/{total_days:,}, date={pd.Timestamp(date).date()}, "
                    f"active_cohorts={active_cohorts:,}, forced_cohorts={forced_cohorts:,}, "
                    f"active_code_slots={active_holdings:,}"
                )
                last_progress = now
            blocked_buys_by_decile = defaultdict(int)
            buy_missing_by_decile = defaultdict(int)

            for cohort in signals_by_pos.get(return_pos, []):
                decile = cohort["decile"]
                ideal_active[decile].append(cohort.copy())
                buyable_code_ids, blocked_buys, buy_missing = filter_buyable_code_ids(
                    cohort["code_ids"],
                    daily_info,
                )
                blocked_buys_by_decile[decile] += blocked_buys
                buy_missing_by_decile[decile] += buy_missing
                if len(buyable_code_ids):
                    new_cohort = cohort.copy()
                    new_cohort["code_ids"] = buyable_code_ids
                    new_cohort["n_codes"] = int(len(buyable_code_ids))
                    buy_constrained_active[decile].append(new_cohort.copy())
                    active[decile].append(new_cohort)
                    sell_due[decile][new_cohort["target_end_pos"]].append(new_cohort)

            for decile in range(1, N_DECILES + 1):
                ideal_start = [
                    cohort
                    for cohort in ideal_active[decile]
                    if int(cohort.get("n_codes", 0)) > 0
                ]
                buy_constrained_start = [
                    cohort
                    for cohort in buy_constrained_active[decile]
                    if int(cohort.get("n_codes", 0)) > 0
                ]
                active_start = [
                    cohort
                    for cohort in active[decile]
                    if int(cohort.get("n_codes", 0)) > 0
                ]
                forced_start = [
                    cohort
                    for cohort in forced_active[decile]
                    if int(cohort.get("n_codes", 0)) > 0
                ]
                ideal_active[decile] = ideal_start
                buy_constrained_active[decile] = buy_constrained_start
                active[decile] = active_start
                forced_active[decile] = forced_start
                active_start = active_start + forced_start

                if not active_start and not ideal_start and not buy_constrained_start:
                    prev_weights[decile] = empty_weight_state()
                    continue

                weight_state = make_weight_state_from_cohorts(active_start)
                start_turnover = calc_weight_turnover_states(prev_weights[decile], weight_state)
                actual_return_info = (
                    calc_weighted_return_state(
                        weight_state,
                        daily_info,
                        start_ids_from_cohorts(active_start, return_pos),
                    )
                    if active_start
                    else empty_return_info(0.0)
                )
                ideal_weight_state = make_weight_state_from_cohorts(ideal_start)
                ideal_return_info = (
                    calc_weighted_return_state(
                        ideal_weight_state,
                        daily_info,
                        start_ids_from_cohorts(ideal_start, return_pos),
                    )
                    if ideal_start
                    else empty_return_info(0.0)
                )
                buy_constrained_weight_state = make_weight_state_from_cohorts(buy_constrained_start)
                buy_constrained_return_info = (
                    calc_weighted_return_state(
                        buy_constrained_weight_state,
                        daily_info,
                        start_ids_from_cohorts(buy_constrained_start, return_pos),
                    )
                    if buy_constrained_start
                    else empty_return_info(0.0)
                )
                gross_return = actual_return_info["return"]
                signal_gross_alpha = ideal_return_info["return"]
                buy_constrained_return = buy_constrained_return_info["return"]
                buy_blocked_loss = (
                    finite_or_zero(signal_gross_alpha)
                    - finite_or_zero(buy_constrained_return)
                )
                sell_blocked_forced_hold_loss = (
                    finite_or_zero(buy_constrained_return)
                    - finite_or_zero(gross_return)
                )

                due_normal = [
                    cohort
                    for cohort in sell_due[decile].pop(return_pos, [])
                    if int(cohort.get("n_codes", 0)) > 0
                ]
                due_ids = {id(cohort) for cohort in due_normal}
                due_for_sell = due_normal + forced_start
                (
                    forced_end,
                    blocked_sells,
                    forced_holds,
                    sell_missing,
                ) = process_sell_attempts(due_for_sell, daily_info, return_pos)
                ideal_end = retire_expired_cohorts(ideal_start, return_pos)
                buy_constrained_end = retire_expired_cohorts(buy_constrained_start, return_pos)
                normal_end = [
                    cohort
                    for cohort in active[decile]
                    if id(cohort) not in due_ids
                ]
                active_end = normal_end + forced_end
                end_weight_state = make_weight_state_from_cohorts(active_end)
                end_turnover = calc_weight_turnover_states(weight_state, end_weight_state)
                turnover = start_turnover + end_turnover
                portfolio_name = f"D{decile}"
                data_missing = (
                    int(actual_return_info["data_missing"])
                    + int(buy_missing_by_decile[decile])
                    + int(sell_missing)
                )
                blocked_buys = int(blocked_buys_by_decile[decile])
                holding_day_index = day_i
                is_warmup = holding_day_index < group_horizon

                turnover_rows.append({
                    "date": date,
                    "experiment_name": exp_name,
                    "model_name": model_name,
                    "universe_group": universe_group,
                    "decile": decile,
                    "portfolio_name": portfolio_name,
                    "turnover": turnover,
                    "start_turnover": start_turnover,
                    "end_turnover": end_turnover,
                    "active_cohorts": len(active_start),
                    "num_holdings": len(weight_state["ids"]),
                    "num_blocked_buys": blocked_buys,
                    "num_blocked_sells": blocked_sells,
                    "num_forced_holds": forced_holds,
                    "num_data_missing_returns": data_missing,
                    "valid_weight": actual_return_info["valid_weight"],
                    "missing_weight": actual_return_info["missing_weight"],
                    "is_warmup": is_warmup,
                })

                base_portfolio_rows.append({
                    "date": date,
                    "experiment_name": exp_name,
                    "model_name": model_name,
                    "universe_group": universe_group,
                    "decile": decile,
                    "portfolio_name": portfolio_name,
                    "gross_return": gross_return,
                    "turnover": turnover,
                    "start_turnover": start_turnover,
                    "end_turnover": end_turnover,
                    "active_cohorts": len(active_start),
                    "num_holdings": len(weight_state["ids"]),
                    "num_blocked_buys": blocked_buys,
                    "num_blocked_sells": blocked_sells,
                    "num_forced_holds": forced_holds,
                    "num_data_missing_returns": data_missing,
                    "valid_weight": actual_return_info["valid_weight"],
                    "missing_weight": actual_return_info["missing_weight"],
                    "signal_valid_weight": ideal_return_info["valid_weight"],
                    "signal_missing_weight": ideal_return_info["missing_weight"],
                    "signal_gross_alpha": signal_gross_alpha,
                    "buy_constrained_gross_return": buy_constrained_return,
                    "buy_blocked_loss": buy_blocked_loss,
                    "sell_blocked_forced_hold_loss": sell_blocked_forced_hold_loss,
                    "is_warmup": is_warmup,
                })

                active[decile] = normal_end
                forced_active[decile] = forced_end
                ideal_active[decile] = ideal_end
                buy_constrained_active[decile] = buy_constrained_end
                prev_weights[decile] = end_weight_state

            if (
                return_pos >= max_end_pos
                and all(not active[d] for d in range(1, N_DECILES + 1))
                and all(not forced_active[d] for d in range(1, N_DECILES + 1))
                and all(not ideal_active[d] for d in range(1, N_DECILES + 1))
                and all(not buy_constrained_active[d] for d in range(1, N_DECILES + 1))
            ):
                break

        log(
            "Portfolio group done: "
            f"{exp_name}/{model_name}/{universe_group}, "
            f"elapsed_sec={perf_counter() - group_start:.1f}"
        )

    base_portfolio = pd.DataFrame(base_portfolio_rows)
    del base_portfolio_rows
    portfolio_returns = expand_portfolio_cost_rows(base_portfolio)
    portfolio_returns = add_long_short_return_rows(portfolio_returns)
    portfolio_turnover = add_long_short_turnover_rows(pd.DataFrame(turnover_rows))
    return portfolio_returns, portfolio_turnover


def summarize_portfolios(portfolio_returns):
    """
    Annualized performance for decile and D10-D1 portfolios.
    """
    rows = []

    if portfolio_returns.empty:
        return pd.DataFrame()

    group_cols = GROUP_COLS + ["decile", "portfolio_name", "cost_bps"]

    for keys, group in portfolio_returns.groupby(group_cols, sort=True):
        exp_name, model_name, universe_group, decile, portfolio_name, cost_bps = keys
        group = group.sort_values("date")
        n_total_days = len(group)
        n_warmup_days = int(group["is_warmup"].sum()) if "is_warmup" in group.columns else 0
        eval_group = group[~group["is_warmup"].astype(bool)].copy() if "is_warmup" in group.columns else group
        if eval_group.empty:
            continue

        gross_ret = eval_group["gross_return"].dropna()
        net_ret = eval_group["net_return"].dropna()
        if len(net_ret) == 0:
            continue

        ann_return = net_ret.mean() * TRADING_DAYS_PER_YEAR
        ann_vol = net_ret.std(ddof=1) * np.sqrt(TRADING_DAYS_PER_YEAR)
        sharpe = ann_return / ann_vol if ann_vol > 0 else np.nan

        gross_nav = (1.0 + gross_ret).cumprod()
        net_nav = (1.0 + net_ret).cumprod()

        final_gross_nav = gross_nav.iloc[-1] if len(gross_nav) else np.nan
        final_net_nav = net_nav.iloc[-1] if len(net_nav) else np.nan

        rows.append({
            "experiment_name": exp_name,
            "model_name": model_name,
            "universe_group": universe_group,
            "decile": decile,
            "portfolio_name": portfolio_name,
            "cost_bps": cost_bps,
            "n_periods": len(net_ret),
            "n_total_days": n_total_days,
            "n_warmup_days": n_warmup_days,
            "annualized_return": ann_return,
            "annualized_volatility": ann_vol,
            "sharpe": sharpe,
            "max_drawdown": max_drawdown(net_nav),
            "win_rate": (net_ret > 0).mean(),
            "avg_turnover": eval_group["turnover"].mean(),
            "valid_day_ratio": eval_group["net_return"].notna().mean(),
            "avg_valid_weight": eval_group["valid_weight"].mean() if "valid_weight" in eval_group.columns else np.nan,
            "low_coverage_day_ratio": (
                (eval_group["valid_weight"] < LOW_COVERAGE_VALID_WEIGHT).mean()
                if "valid_weight" in eval_group.columns
                else np.nan
            ),
            "gross_cumulative_return": final_gross_nav - 1.0,
            "net_cumulative_return": final_net_nav - 1.0,
            "final_gross_nav": final_gross_nav,
            "final_net_nav": final_net_nav,
        })

    return pd.DataFrame(rows)


def calc_portfolio_nav(portfolio_returns):
    """
    Build daily gross/net NAV and net drawdown curves after warmup filtering.
    """
    if portfolio_returns.empty:
        return pd.DataFrame()

    group_cols = GROUP_COLS + ["decile", "portfolio_name", "cost_bps"]
    output_cols = [
        "date",
        "experiment_name",
        "model_name",
        "universe_group",
        "portfolio_name",
        "cost_bps",
        "gross_nav",
        "net_nav",
        "drawdown",
    ]
    frames = []

    for keys, group in portfolio_returns.groupby(group_cols, sort=True):
        exp_name, model_name, universe_group, _decile, portfolio_name, cost_bps = keys
        group = group.sort_values("date")
        eval_group = group[~group["is_warmup"].astype(bool)].copy() if "is_warmup" in group.columns else group.copy()
        eval_group = eval_group.dropna(subset=["gross_return", "net_return"])
        if eval_group.empty:
            continue

        gross_nav = (1.0 + eval_group["gross_return"].astype(float)).cumprod()
        net_nav = (1.0 + eval_group["net_return"].astype(float)).cumprod()
        running_max_net_nav = net_nav.cummax()

        frame = pd.DataFrame({
            "date": eval_group["date"].to_numpy(),
            "experiment_name": exp_name,
            "model_name": model_name,
            "universe_group": universe_group,
            "portfolio_name": portfolio_name,
            "cost_bps": cost_bps,
            "gross_nav": gross_nav.to_numpy(),
            "net_nav": net_nav.to_numpy(),
            "drawdown": (net_nav / running_max_net_nav - 1.0).to_numpy(),
        })
        frames.append(frame[output_cols])

    if not frames:
        return pd.DataFrame(columns=output_cols)
    return pd.concat(frames, ignore_index=True)


def calc_return_attribution(portfolio_returns):
    """
    Summarize daily return attribution after removing the warmup period.
    """
    if portfolio_returns.empty:
        return pd.DataFrame()

    component_cols = [
        "signal_gross_alpha",
        "buy_blocked_loss",
        "sell_blocked_forced_hold_loss",
        "missing_return_data_issue",
        "turnover_cost",
        "attributed_net_return",
        "net_return",
    ]
    group_cols = GROUP_COLS + ["decile", "portfolio_name", "cost_bps"]
    rows = []

    for keys, group in portfolio_returns.groupby(group_cols, sort=True):
        exp_name, model_name, universe_group, decile, portfolio_name, cost_bps = keys
        group = group.sort_values("date")
        eval_group = group[~group["is_warmup"].astype(bool)].copy() if "is_warmup" in group.columns else group
        if eval_group.empty:
            continue

        net_ret = eval_group["net_return"].dropna()
        final_net_nav = (1.0 + net_ret).cumprod().iloc[-1] if len(net_ret) else np.nan
        row = {
            "experiment_name": exp_name,
            "model_name": model_name,
            "universe_group": universe_group,
            "decile": decile,
            "portfolio_name": portfolio_name,
            "cost_bps": cost_bps,
            "n_periods": len(net_ret),
            "final_net_nav": final_net_nav,
            "net_cumulative_return": final_net_nav - 1.0 if not pd.isna(final_net_nav) else np.nan,
        }
        for col in component_cols:
            values = eval_group[col].dropna() if col in eval_group.columns else pd.Series(dtype=float)
            row[f"mean_daily_{col}"] = values.mean() if len(values) else np.nan
            row[f"annualized_{col}"] = values.mean() * TRADING_DAYS_PER_YEAR if len(values) else np.nan
            row[f"cumulative_linear_{col}"] = values.sum() if len(values) else np.nan

        row["annualized_attribution_residual"] = (
            row["annualized_signal_gross_alpha"]
            - row["annualized_buy_blocked_loss"]
            - row["annualized_sell_blocked_forced_hold_loss"]
            - row["annualized_missing_return_data_issue"]
            - row["annualized_turnover_cost"]
            - row["annualized_attributed_net_return"]
        )
        rows.append(row)

    return pd.DataFrame(rows)


def calc_cost_sensitivity(performance_summary):
    """
    Cost sensitivity table derived from portfolio summaries.
    """
    if performance_summary.empty:
        return pd.DataFrame()

    columns = [
        "experiment_name",
        "model_name",
        "universe_group",
        "decile",
        "portfolio_name",
        "cost_bps",
        "annualized_return",
        "sharpe",
        "max_drawdown",
        "avg_turnover",
        "valid_day_ratio",
        "avg_valid_weight",
        "low_coverage_day_ratio",
        "gross_cumulative_return",
        "net_cumulative_return",
        "final_gross_nav",
        "final_net_nav",
    ]
    return performance_summary[columns].copy()


def run_one_prediction_file(path, daily_context):
    """
    Run all evaluation layers for one prediction file.
    """
    file_start = perf_counter()
    pred = load_prediction_file(path)
    log(f"Prediction rows loaded: {Path(path).name}, rows={len(pred):,}")
    pred = add_universe_groups(pred)
    log(f"Universe rows expanded: {Path(path).name}, rows={len(pred):,}")

    step_start = perf_counter()
    ic_by_period = calc_ic_by_period(pred)
    ic_summary = calc_ic_summary(ic_by_period)
    cumulative_ic = calc_cumulative_ic(ic_by_period)
    log(f"IC done: {Path(path).name}, elapsed_sec={perf_counter() - step_start:.1f}")

    step_start = perf_counter()
    pred = assign_deciles(pred)
    decile_returns = calc_decile_returns(pred)
    decile_summary = calc_decile_summary(decile_returns)
    decile_monotonicity = calc_decile_monotonicity(decile_summary)
    log(f"Decile done: {Path(path).name}, elapsed_sec={perf_counter() - step_start:.1f}")

    step_start = perf_counter()
    portfolio_returns, portfolio_turnover = calc_overlapping_portfolios(pred, daily_context)
    performance_summary = summarize_portfolios(portfolio_returns)
    portfolio_nav = calc_portfolio_nav(portfolio_returns)
    return_attribution = calc_return_attribution(portfolio_returns)
    cost_sensitivity = calc_cost_sensitivity(performance_summary)
    log(
        f"Portfolio done: {Path(path).name}, "
        f"elapsed_sec={perf_counter() - step_start:.1f}, "
        f"portfolio_rows={len(portfolio_returns):,}"
    )
    log(f"Prediction file done: {Path(path).name}, elapsed_sec={perf_counter() - file_start:.1f}")

    return {
        "ic_by_period": ic_by_period,
        "ic_summary": ic_summary,
        "cumulative_ic": cumulative_ic,
        "decile_returns": decile_returns,
        "decile_summary": decile_summary,
        "decile_monotonicity": decile_monotonicity,
        "portfolio_returns": portfolio_returns,
        "portfolio_turnover": portfolio_turnover,
        "portfolio_nav": portfolio_nav,
        "performance_summary": performance_summary,
        "return_attribution": return_attribution,
        "cost_sensitivity": cost_sensitivity,
    }


def concat_frames(frames):
    """
    Concatenate non-empty DataFrames.
    """
    frames = [frame for frame in frames if frame is not None and not frame.empty]
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def save_table(df, filename):
    """
    Save one output table.
    """
    out_path = TABLE_DIR / filename
    df.to_csv(out_path, index=False, encoding="utf-8-sig")
    log(f"Saved {filename}: {out_path}, rows={len(df):,}")


def append_table(df, filename, initialized):
    """
    Append one prediction file's rows without rewriting prior completed files.
    """
    if df is None or df.empty:
        return initialized

    out_path = TABLE_DIR / filename
    mode = "a" if initialized else "w"
    encoding = "utf-8" if initialized else "utf-8-sig"
    df.to_csv(
        out_path,
        mode=mode,
        header=not initialized,
        index=False,
        encoding=encoding,
    )
    log(
        f"{'Appended' if initialized else 'Started'} {filename}: "
        f"{out_path}, added_rows={len(df):,}"
    )
    return True


def parse_args():
    """
    Parse CLI options used mainly for cluster debugging and partial reruns.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pred-pattern",
        default="pred_*.parquet",
        help="Prediction parquet glob under outputs/predictions.",
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=None,
        help="Optional cap on the number of prediction files to process.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    pred_files = sorted(glob.glob(str(PRED_DIR / args.pred_pattern)))
    if args.max_files is not None:
        pred_files = pred_files[: args.max_files]
    if not pred_files:
        raise RuntimeError(f"No prediction files found in {PRED_DIR} matching {args.pred_pattern}")

    earliest_prediction_date = load_earliest_prediction_date(pred_files)
    if earliest_prediction_date is not None:
        log(f"Earliest prediction date: {earliest_prediction_date.date()}")

    daily_returns = load_daily_returns(min_date=earliest_prediction_date)
    log(f"Daily return rows loaded: {len(daily_returns):,}")
    lookup_start = perf_counter()
    daily_context = prepare_daily_return_lookup(daily_returns)
    del daily_returns
    log(f"Daily return lookup ready: elapsed_sec={perf_counter() - lookup_start:.1f}")

    output_names = [
        "ic_by_period",
        "ic_summary",
        "cumulative_ic",
        "decile_returns",
        "decile_summary",
        "decile_monotonicity",
        "portfolio_returns",
        "portfolio_turnover",
        "portfolio_nav",
        "performance_summary",
        "return_attribution",
        "cost_sensitivity",
    ]
    initialized_outputs = set()
    performance_frames = []

    for idx, path in enumerate(pred_files, start=1):
        log(f"Start prediction file {idx}/{len(pred_files)}: {Path(path).name}")
        result = run_one_prediction_file(path, daily_context)
        performance_frames.append(result["performance_summary"])

        log(f"Appending partial tables after {Path(path).name}")
        for name in output_names:
            filename = f"{name}.csv"
            initialized = append_table(
                result[name],
                filename,
                name in initialized_outputs,
            )
            if initialized:
                initialized_outputs.add(name)

        del result

    for name in output_names:
        if name not in initialized_outputs:
            save_table(pd.DataFrame(), f"{name}.csv")

    perf = concat_frames(performance_frames)
    if not perf.empty:
        print("\nPerformance summary:")
        print(perf)


if __name__ == "__main__":
    main()
