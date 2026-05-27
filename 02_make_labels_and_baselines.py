# -*- coding: utf-8 -*-
"""
02_make_labels_and_baselines.py

Purpose
-------
Generate future-return labels, traditional baseline features, and execution
constraint fields from the code-partitioned panel dataset.

Input
-----
data/processed/panel_by_code/code=*/part.parquet

Outputs
-------
data/features/features_by_code_bucket/bucket={bucket}/part-*.parquet
data/features/features_by_year/year={year}/part-*.parquet

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

from concurrent.futures import ProcessPoolExecutor, as_completed
import argparse
from pathlib import Path
import shutil

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

try:
    from tqdm.auto import tqdm
except ImportError:
    tqdm = None

from config import (
    PANEL_BY_CODE_DIR,
    FEATURE_DIR,
    FEATURE_BY_CODE_BUCKET_DIR,
    FEATURE_BY_YEAR_DIR,
    MIN_AMOUNT,
    MIN_LIST_DAYS,
    EXPERIMENTS,
    LOW_VOLUME_LIMIT_RATIO,
    N_WORKERS,
    FEATURE_MAX_BUFFER_ROWS,
    FEATURE_ROW_GROUP_SIZE,
    FEATURE_CODE_BUCKETS,
)


BASE_PANEL_COLUMNS = [
    "date", "code", "name", "industry",
    "open_adj", "high_adj", "low_adj", "close_adj",
    "open_raw", "high_raw", "low_raw", "close_raw", "prev_close_raw",
    "volume", "amount", "turnover_rate", "pct_chg", "amplitude",
    "is_st", "volume_ratio",
    "ret_3d_raw", "ret_6d_raw", "ret_10d_raw", "ret_25d_raw",
    "is_limit_up",
    "total_shares", "float_shares", "total_mktcap", "float_mktcap",
    "pe_ttm", "pb", "ps_ttm",
    "list_date", "delist_date",
    "used_raw_as_adj", "is_suspended", "days_since_list", "is_tradable_basic",
]

BASELINE_FEATURE_COLUMNS = [
    "reversal_5d",
    "reversal_10d",
    "momentum_20d",
    "momentum_60d",
    "position_5d",
    "position_20d",
    "position_60d",
    "vol_20d",
    "vol_60d",
    "amount_mean_20d",
    "amount_change_20d",
    "turnover_mean_20d",
    "turnover_change_20d",
    "log_total_mktcap",
    "log_float_mktcap",
]

TRADING_CONSTRAINT_COLUMNS = [
    "limit_pct",
    "volume_mean_20d_prev",
    "volume_ratio_to_20d_prev",
    "is_low_volume_limit_up",
    "is_low_volume_limit_down",
    "is_tradable",
]

STRING_COLS = ["code", "name", "industry"]
DATETIME_COLS = ["date", "list_date", "delist_date"]
INT8_COLS = [
    "is_st",
    "is_limit_up",
    "used_raw_as_adj",
    "is_suspended",
    "is_tradable_basic",
    "is_low_volume_limit_up",
    "is_low_volume_limit_down",
    "is_tradable",
]
FLOAT64_COLS = [
    "volume",
    "amount",
    "total_shares",
    "float_shares",
    "total_mktcap",
    "float_mktcap",
    "days_since_list",
]


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


def return_windows():
    return sorted({1, 3, 5, 10, 20, 25, 60} | set(configured_windows()))


def future_return_columns():
    cols = []
    for h in configured_horizons():
        cols.extend([f"future_ret_{h}d", f"label_{h}d"])
    return cols


def ma_columns():
    cols = []
    for n in configured_ma_windows():
        cols.extend([f"ma{n}_adj", f"ma{n}_gap"])
    return cols


def feature_output_columns():
    cols = (
        BASE_PANEL_COLUMNS
        + [f"ret_{n}d" for n in return_windows()]
        + ["open_to_close_ret_1d"]
        + future_return_columns()
        + TRADING_CONSTRAINT_COLUMNS
        + ma_columns()
        + BASELINE_FEATURE_COLUMNS
    )
    return list(dict.fromkeys(cols))


FEATURE_OUTPUT_COLUMNS = feature_output_columns()


def feature_schema(include_code=True):
    fields = []
    for col in FEATURE_OUTPUT_COLUMNS:
        if not include_code and col == "code":
            continue
        if col in DATETIME_COLS:
            fields.append((col, pa.timestamp("ns")))
        elif col in STRING_COLS:
            fields.append((col, pa.string()))
        elif col in INT8_COLS:
            fields.append((col, pa.int8()))
        elif col in FLOAT64_COLS:
            fields.append((col, pa.float64()))
        else:
            fields.append((col, pa.float32()))
    return pa.schema(fields)


FEATURE_SCHEMA = feature_schema(include_code=True)


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

    Future-return convention:
    A signal formed at date t buys at the next trading day's adjusted open
    and exits at the horizon date's adjusted close:

        future_ret_h = close_adj[t + h] / open_adj[t + 1] - 1

    Therefore h=5 holds trading days t+1 through t+5.
    """
    g = df.groupby("code", group_keys=False)

    for n in return_windows():
        df[f"ret_{n}d"] = g["close_adj"].pct_change(n)

    df["open_to_close_ret_1d"] = safe_divide(
        df["close_adj"],
        df["open_adj"].where(df["open_adj"] > 0),
    ) - 1.0

    buy_open = g["open_adj"].shift(-1).where(lambda x: x > 0)
    for h in configured_horizons():
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

    df["reversal_5d"] = -df["ret_5d"]
    df["reversal_10d"] = -df["ret_10d"]
    df["momentum_20d"] = df["ret_20d"]
    df["momentum_60d"] = df["ret_60d"]

    for n in configured_ma_windows():
        ma_col = f"ma{n}_adj"
        gap_col = f"ma{n}_gap"

        if ma_col in df.columns:
            df[gap_col] = safe_divide(df["close_adj"], df[ma_col]) - 1.0
        else:
            ma = g["close_adj"].transform(lambda x: x.rolling(n, min_periods=n).mean())
            df[gap_col] = safe_divide(df["close_adj"], ma) - 1.0
            df[ma_col] = ma

    for n in [5, 20, 60]:
        rolling_high = g["high_adj"].transform(lambda x: x.rolling(n, min_periods=n).max())
        rolling_low = g["low_adj"].transform(lambda x: x.rolling(n, min_periods=n).min())
        denom = rolling_high - rolling_low
        df[f"position_{n}d"] = safe_divide(df["close_adj"] - rolling_low, denom)

    for n in [20, 60]:
        df[f"vol_{n}d"] = g["ret_1d"].transform(lambda x: x.rolling(n, min_periods=n).std())

    df["amount_mean_20d"] = g["amount"].transform(lambda x: x.rolling(20, min_periods=20).mean())
    df["amount_change_20d"] = safe_divide(df["amount"], df["amount_mean_20d"]) - 1.0

    if "turnover_rate" in df.columns:
        df["turnover_mean_20d"] = g["turnover_rate"].transform(
            lambda x: x.rolling(20, min_periods=20).mean()
        )
        df["turnover_change_20d"] = safe_divide(df["turnover_rate"], df["turnover_mean_20d"]) - 1.0

    if "total_mktcap" in df.columns:
        df["log_total_mktcap"] = np.log(df["total_mktcap"].where(df["total_mktcap"] > 0))
    if "float_mktcap" in df.columns:
        df["log_float_mktcap"] = np.log(df["float_mktcap"].where(df["float_mktcap"] > 0))

    return df


def add_final_tradable_flag(df):
    """
    Define the main tradable flag for model training and backtest.
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
        mask_new = df["days_since_list"].notna() & (df["days_since_list"] < MIN_LIST_DAYS)
        df.loc[mask_new, "is_tradable"] = 0

    return df


def prepare_feature_output_frame(df):
    """
    Normalize one stock feature frame before writing parquet datasets.
    """
    df = df.dropna(subset=["date", "code"]).copy()
    if df.empty:
        return df

    df = df.sort_values(["code", "date"]).reset_index(drop=True)

    for col in FEATURE_OUTPUT_COLUMNS:
        if col not in df.columns:
            df[col] = np.nan

    for col in DATETIME_COLS:
        df[col] = pd.to_datetime(df[col], errors="coerce")

    for col in STRING_COLS:
        df[col] = df[col].astype("string")

    for col in INT8_COLS:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype("int8")

    for col in FLOAT64_COLS:
        df[col] = pd.to_numeric(df[col], errors="coerce").astype("float64")

    float32_cols = [
        col for col in FEATURE_OUTPUT_COLUMNS
        if col not in set(DATETIME_COLS) | set(STRING_COLS) | set(INT8_COLS) | set(FLOAT64_COLS)
    ]
    for col in float32_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce").astype("float32")

    return df[FEATURE_OUTPUT_COLUMNS]


def list_panel_code_files(limit_codes=None):
    files = sorted(PANEL_BY_CODE_DIR.glob("code=*/part.parquet"))
    if limit_codes is not None:
        limit_codes = {str(code).zfill(6) for code in limit_codes}
        files = [
            path for path in files
            if path.parent.name.split("=", 1)[1].zfill(6) in limit_codes
        ]
    if not files:
        raise RuntimeError(
            f"No code-partitioned panel files found in {PANEL_BY_CODE_DIR}. "
            "Run 01_build_panel.py first."
        )
    return files


def progress_iter(iterable, total, desc):
    """
    Use tqdm when it is installed; otherwise fall back to the regular iterator.
    """
    if tqdm is None:
        return iterable
    return tqdm(iterable, total=total, desc=desc, unit="stock")


def clear_output_dirs():
    """
    Remove existing partitioned feature outputs before rebuilding.
    """
    paths = [
        FEATURE_BY_CODE_BUCKET_DIR,
        FEATURE_BY_YEAR_DIR,
        FEATURE_DIR / "features_by_code",
    ]
    for path in paths:
        if path.exists():
            shutil.rmtree(path)

    for path in [FEATURE_BY_CODE_BUCKET_DIR, FEATURE_BY_YEAR_DIR]:
        path.mkdir(parents=True, exist_ok=True)


def read_panel_code_partition(path):
    """
    Read one panel code partition and restore the partition code column.
    """
    code = path.parent.name.split("=", 1)[1].zfill(6)
    one = pq.ParquetFile(path).read().to_pandas()
    one["code"] = code
    one["date"] = pd.to_datetime(one["date"])
    one["code"] = one["code"].astype(str).str.zfill(6)
    return one.sort_values(["code", "date"]).reset_index(drop=True)


def build_features_for_one_stock(path):
    """
    Worker entrypoint for one code partition.
    """
    path = Path(path)
    code = path.parent.name.split("=", 1)[1].zfill(6)
    df = read_panel_code_partition(path)
    if df.empty:
        return code, None

    df = add_returns_and_labels(df)
    df = add_trading_constraint_features(df)
    df = add_baseline_features(df)
    df = add_final_tradable_flag(df)
    df = prepare_feature_output_frame(df)
    if df.empty:
        return code, None
    return code, df


def iter_feature_frames(files, n_workers):
    """
    Yield processed feature frames from single-process or multiprocess mode.
    """
    if n_workers <= 1:
        file_iter = progress_iter(files, total=len(files), desc="Building features")
        for i, path in enumerate(file_iter, start=1):
            if tqdm is None and (i == 1 or i % 500 == 0 or i == len(files)):
                print(f"Processing {i}/{len(files)}: {path.parent.name}")
            try:
                yield build_features_for_one_stock(path)
            except Exception as exc:
                code = path.parent.name.split("=", 1)[1].zfill(6)
                yield code, exc
        return

    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = {executor.submit(build_features_for_one_stock, str(path)): path for path in files}
        future_iter = progress_iter(
            as_completed(futures),
            total=len(futures),
            desc=f"Building features ({n_workers} workers)",
        )
        for i, future in enumerate(future_iter, start=1):
            path = futures[future]
            code = path.parent.name.split("=", 1)[1].zfill(6)
            if tqdm is None and (i == 1 or i % 500 == 0 or i == len(futures)):
                print(f"Processing {i}/{len(futures)}")
            try:
                yield future.result()
            except Exception as exc:
                yield code, exc


def code_bucket_id(code):
    """
    Map stock code to a stable bucket id.
    """
    digits = "".join(ch for ch in str(code) if ch.isdigit())
    if not digits:
        return 0
    return int(digits) % int(FEATURE_CODE_BUCKETS)


def write_code_bucket_batches(buffer, part_counter):
    """
    Flush buffered stock frames into code-bucketed feature parquet files.
    """
    if not buffer:
        return 0

    batch = pd.concat(buffer, ignore_index=True)
    bucket_ids = batch["code"].map(code_bucket_id).astype("int32")
    written_rows = len(batch)

    for bucket, group in batch.groupby(bucket_ids, sort=True):
        bucket = int(bucket)
        out_dir = FEATURE_BY_CODE_BUCKET_DIR / f"bucket={bucket:03d}"
        out_dir.mkdir(parents=True, exist_ok=True)
        part_id = part_counter.get(bucket, 0)
        part_counter[bucket] = part_id + 1

        table = pa.Table.from_pandas(
            group[FEATURE_OUTPUT_COLUMNS],
            schema=FEATURE_SCHEMA,
            preserve_index=False,
        )
        pq.write_table(
            table,
            out_dir / f"part-{part_id:05d}.parquet",
            compression="zstd",
            row_group_size=FEATURE_ROW_GROUP_SIZE,
        )

    return written_rows


def write_year_batches(buffer, part_counter):
    """
    Flush buffered stock frames into year-partitioned feature parquet files.
    """
    if not buffer:
        return 0

    batch = pd.concat(buffer, ignore_index=True)
    batch["year"] = pd.to_datetime(batch["date"]).dt.year.astype("int32")
    written_rows = len(batch)

    for year, group in batch.groupby("year", sort=True):
        out_dir = FEATURE_BY_YEAR_DIR / f"year={int(year)}"
        out_dir.mkdir(parents=True, exist_ok=True)
        part_id = part_counter.get(int(year), 0)
        part_counter[int(year)] = part_id + 1

        group = group.drop(columns=["year"])
        table = pa.Table.from_pandas(
            group[FEATURE_OUTPUT_COLUMNS],
            schema=FEATURE_SCHEMA,
            preserve_index=False,
        )
        pq.write_table(
            table,
            out_dir / f"part-{part_id:05d}.parquet",
            compression="zstd",
            row_group_size=FEATURE_ROW_GROUP_SIZE,
        )

    return written_rows


def update_stats(stats, one):
    """
    Update aggregate conversion statistics from one stock feature frame.
    """
    stats["rows"] += len(one)
    stats["stocks"] += 1
    one_min = one["date"].min()
    one_max = one["date"].max()
    stats["min_date"] = one_min if stats["min_date"] is None else min(stats["min_date"], one_min)
    stats["max_date"] = one_max if stats["max_date"] is None else max(stats["max_date"], one_max)


def write_feature_datasets(limit_codes=None, n_workers=N_WORKERS):
    """
    Build code-bucketed and year-partitioned feature datasets.
    """
    files = list_panel_code_files(limit_codes=limit_codes)
    print(f"Panel code partitions: {len(files)} from {PANEL_BY_CODE_DIR}")
    print(f"Workers: {n_workers}")

    clear_output_dirs()

    failed = []
    buffer = []
    buffer_rows = 0
    year_part_counter = {}
    bucket_part_counter = {}
    stats = {
        "rows": 0,
        "stocks": 0,
        "min_date": None,
        "max_date": None,
    }

    for code, result in iter_feature_frames(files, int(n_workers)):
        if isinstance(result, Exception):
            failed.append((code, str(result)))
            continue
        if result is None or result.empty:
            continue

        buffer.append(result)
        buffer_rows += len(result)
        update_stats(stats, result)

        if buffer_rows >= FEATURE_MAX_BUFFER_ROWS:
            write_code_bucket_batches(buffer, bucket_part_counter)
            write_year_batches(buffer, year_part_counter)
            buffer = []
            buffer_rows = 0

    if buffer:
        write_code_bucket_batches(buffer, bucket_part_counter)
        write_year_batches(buffer, year_part_counter)

    if stats["rows"] == 0:
        raise RuntimeError("No feature rows were built. Check panel dataset and date filters.")

    if failed:
        print("[Warning] Failed feature partitions:")
        for code, msg in failed[:20]:
            print(f"  - {code}: {msg}")
        if len(failed) > 20:
            print(f"  ... {len(failed) - 20} more")

    return {
        "rows": stats["rows"],
        "stocks": stats["stocks"],
        "min_date": stats["min_date"],
        "max_date": stats["max_date"],
        "failed": len(failed),
    }


def main():
    parser = argparse.ArgumentParser(description="Build partitioned baseline feature datasets.")
    parser.add_argument(
        "--workers",
        type=int,
        default=N_WORKERS,
        help=f"Number of worker processes. Default: {N_WORKERS}",
    )
    parser.add_argument(
        "--limit-codes",
        default=None,
        help="Optional comma-separated stock codes for smoke tests, e.g. 000001,000002.",
    )
    args = parser.parse_args()
    limit_codes = None
    if args.limit_codes:
        limit_codes = [code.strip() for code in args.limit_codes.split(",") if code.strip()]

    print("Building feature datasets from code-partitioned panel...")
    stats = write_feature_datasets(limit_codes=limit_codes, n_workers=args.workers)

    print("Done.")
    print(f"Saved code-bucketed features to: {FEATURE_BY_CODE_BUCKET_DIR}")
    print(f"Saved year-partitioned features to: {FEATURE_BY_YEAR_DIR}")
    print("Feature rows:", stats["rows"])
    print("Date range:", stats["min_date"], "to", stats["max_date"])
    print("Number of stocks:", stats["stocks"])
    print("Failed stocks:", stats["failed"])


if __name__ == "__main__":
    main()
