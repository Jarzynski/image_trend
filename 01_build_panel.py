# -*- coding: utf-8 -*-
"""
01_build_panel.py

Build the daily A-share panel from per-stock CSV files.

Input
-----
../daily_OHLVC/不复权/*.csv
../daily_OHLVC/后复权/*.csv

Outputs
-------
data/processed/panel_by_code/code={code}/part.parquet
data/processed/panel_by_year/year={year}/part-*.parquet

Design
------
The image model should use adjusted prices for pattern construction and
future-return labels, while trading filters and backtest constraints should
use raw prices, volume, amount, ST, and limit-up fields.
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
    PANEL_BY_YEAR_DIR,
    PROJECT_DIR,
    START_DATE,
    END_DATE,
    N_WORKERS,
    PANEL_MAX_BUFFER_ROWS,
    PANEL_ROW_GROUP_SIZE,
)


RAW_DATA_ROOT = PROJECT_DIR.parent / "daily_OHLVC"
RAW_PRICE_DIR = RAW_DATA_ROOT / "不复权"
ADJ_PRICE_DIR = RAW_DATA_ROOT / "后复权"

CSV_ENCODING = "gbk"
CSV_FALLBACK_ENCODINGS = ("utf-8-sig", "utf-8", "gb18030")

RAW_COLUMN_MAP = {
    "日期": "date",
    "代码": "code",
    "名称": "name",
    "所属行业": "industry",
    "开盘价": "open_raw",
    "最高价": "high_raw",
    "最低价": "low_raw",
    "收盘价": "close_raw",
    "前收盘价": "prev_close_raw",
    "成交量（股）": "volume",
    "成交额（元）": "amount",
    "换手率": "turnover_rate",
    "涨幅%": "pct_chg",
    "振幅%": "amplitude",
    "是否ST": "is_st",
    "量比": "volume_ratio",
    "3日涨幅%": "ret_3d_raw",
    "6日涨幅%": "ret_6d_raw",
    "10日涨幅%": "ret_10d_raw",
    "25日涨幅%": "ret_25d_raw",
    "是否涨停": "is_limit_up",
    "总股本（股）": "total_shares",
    "流通股本（股）": "float_shares",
    "总市值（元）": "total_mktcap",
    "流通市值（元）": "float_mktcap",
    "滚动市盈率": "pe_ttm",
    "市净率": "pb",
    "滚动市销率": "ps_ttm",
    "上市时间": "list_date",
    "退市时间": "delist_date",
}

ADJ_COLUMN_MAP = {
    "日期": "date",
    "代码": "code",
    "开盘价": "open_adj",
    "最高价": "high_adj",
    "最低价": "low_adj",
    "收盘价": "close_adj",
}

NUMERIC_COLS = [
    "open_adj", "high_adj", "low_adj", "close_adj",
    "open_raw", "high_raw", "low_raw", "close_raw", "prev_close_raw",
    "volume", "amount", "turnover_rate", "pct_chg", "amplitude",
    "volume_ratio",
    "ret_3d_raw", "ret_6d_raw", "ret_10d_raw", "ret_25d_raw",
    "total_shares", "float_shares", "total_mktcap", "float_mktcap",
    "pe_ttm", "pb", "ps_ttm",
]

OUTPUT_COLUMNS = [
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
CODE_OUTPUT_COLUMNS = [c for c in OUTPUT_COLUMNS if c != "code"]

STRING_COLS = ["code", "name", "industry"]
DATETIME_COLS = ["date", "list_date", "delist_date"]
INT8_COLS = ["is_st", "is_limit_up", "used_raw_as_adj", "is_suspended", "is_tradable_basic"]
FLOAT32_COLS = [
    "open_adj", "high_adj", "low_adj", "close_adj",
    "open_raw", "high_raw", "low_raw", "close_raw", "prev_close_raw",
    "turnover_rate", "pct_chg", "amplitude", "volume_ratio",
    "ret_3d_raw", "ret_6d_raw", "ret_10d_raw", "ret_25d_raw",
    "pe_ttm", "pb", "ps_ttm",
]
FLOAT64_COLS = [
    "volume", "amount",
    "total_shares", "float_shares",
    "total_mktcap", "float_mktcap",
    "days_since_list",
]

PANEL_SCHEMA = pa.schema(
    [(c, pa.timestamp("ns")) for c in DATETIME_COLS]
    + [(c, pa.string()) for c in STRING_COLS]
    + [(c, pa.float32()) for c in FLOAT32_COLS]
    + [(c, pa.float64()) for c in FLOAT64_COLS]
    + [(c, pa.int8()) for c in INT8_COLS]
)
CODE_PANEL_SCHEMA = pa.schema([field for field in PANEL_SCHEMA if field.name != "code"])


def read_csv_auto(path, needed_columns):
    """
    Read one CSV with gbk first, then a small encoding fallback list.
    """
    encodings = (CSV_ENCODING,) + CSV_FALLBACK_ENCODINGS
    last_error = None
    for encoding in encodings:
        try:
            return pd.read_csv(
                path,
                encoding=encoding,
                usecols=lambda c: c in needed_columns,
                low_memory=False,
            )
        except UnicodeDecodeError as exc:
            last_error = exc

    raise UnicodeDecodeError(
        "unknown",
        b"",
        0,
        1,
        f"Could not read {path} with encodings {encodings}: {last_error}",
    )


def fast_to_numeric(s):
    """
    Convert only string-like columns with string cleanup.
    """
    if pd.api.types.is_numeric_dtype(s):
        return pd.to_numeric(s, errors="coerce")

    s = s.replace({"--": np.nan, "-": np.nan, "": np.nan, "nan": np.nan, "None": np.nan})
    s = s.astype("string").str.replace(",", "", regex=False)
    return pd.to_numeric(s, errors="coerce")


def normalize_columns(df, column_map):
    """
    Keep and rename the columns used by downstream scripts.
    """
    keep = [c for c in column_map if c in df.columns]
    out = df.loc[:, keep].rename(columns=column_map)

    if "code" in out.columns:
        out["code"] = out["code"].astype(str).str.extract(r"(\d+)", expand=False).str.zfill(6)

    if "date" in out.columns:
        out["date"] = pd.to_datetime(out["date"], errors="coerce")

    return out


def clean_types(df):
    """
    Convert dates, numbers, and Chinese yes/no flags.
    """
    for c in ["list_date", "delist_date"]:
        if c in df.columns:
            df[c] = df[c].replace({"-": np.nan, "--": np.nan, "": np.nan})
            df[c] = pd.to_datetime(df[c], errors="coerce")

    for c in NUMERIC_COLS:
        if c in df.columns:
            df[c] = fast_to_numeric(df[c])

    for c in ["is_st", "is_limit_up"]:
        if c in df.columns:
            df[c] = df[c].replace({
                "是": 1,
                "否": 0,
                "Y": 1,
                "N": 0,
                "True": 1,
                "False": 0,
                True: 1,
                False: 0,
            })
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).astype(np.int8)

    return df


def filter_date_range(df):
    """
    Apply project-level date range early to reduce memory use.
    """
    if "date" not in df.columns:
        return df

    start = pd.Timestamp(START_DATE)
    end = pd.Timestamp(END_DATE)
    return df[(df["date"] >= start) & (df["date"] <= end)].copy()


def read_raw_file(path):
    df = read_csv_auto(path, set(RAW_COLUMN_MAP))
    df = normalize_columns(df, RAW_COLUMN_MAP)
    df = filter_date_range(df)
    return clean_types(df)


def read_adjusted_file(path):
    df = read_csv_auto(path, set(ADJ_COLUMN_MAP))
    df = normalize_columns(df, ADJ_COLUMN_MAP)
    df = filter_date_range(df)
    return clean_types(df)


def merge_one_stock(code, raw_path, adj_path):
    """
    Merge one stock's raw and adjusted price histories.
    """
    raw = read_raw_file(raw_path) if raw_path is not None else None
    adj = read_adjusted_file(adj_path) if adj_path is not None else None

    if raw is None and adj is None:
        return None
    if raw is None:
        out = adj.copy()
    elif adj is None:
        out = raw.copy()
    else:
        out = raw.merge(adj, on=["date", "code"], how="outer", validate="one_to_one")

    out["code"] = out["code"].fillna(code).astype(str).str.zfill(6)

    for adj_col, raw_col in [
        ("open_adj", "open_raw"),
        ("high_adj", "high_raw"),
        ("low_adj", "low_raw"),
        ("close_adj", "close_raw"),
    ]:
        if adj_col not in out.columns:
            out[adj_col] = np.nan

    missing_adjusted_price = out[["open_adj", "high_adj", "low_adj", "close_adj"]].isna().any(axis=1)

    # If adjusted data is unavailable for a stock/date, fall back to raw OHLC so
    # delisted or missing-adjustment stocks are not silently dropped here.
    for adj_col, raw_col in [
        ("open_adj", "open_raw"),
        ("high_adj", "high_raw"),
        ("low_adj", "low_raw"),
        ("close_adj", "close_raw"),
    ]:
        if raw_col in out.columns:
            out[adj_col] = out[adj_col].fillna(out[raw_col])

    out["used_raw_as_adj"] = missing_adjusted_price.astype(np.int8)

    return out


def add_basic_flags(df):
    """
    Add suspension, listing-age, and first-pass tradability flags.
    """
    required_price_cols = ["open_adj", "high_adj", "low_adj", "close_adj"]

    df["is_suspended"] = 0
    if "amount" in df.columns:
        df.loc[df["amount"].fillna(0) <= 0, "is_suspended"] = 1
    if "volume" in df.columns:
        df.loc[df["volume"].fillna(0) <= 0, "is_suspended"] = 1

    for c in required_price_cols:
        if c in df.columns:
            df.loc[df[c].isna(), "is_suspended"] = 1

    if "list_date" in df.columns:
        df["days_since_list"] = (df["date"] - df["list_date"]).dt.days
    else:
        df["days_since_list"] = np.nan

    df["is_tradable_basic"] = 1
    if "is_st" in df.columns:
        df.loc[df["is_st"] == 1, "is_tradable_basic"] = 0
    df.loc[df["is_suspended"] == 1, "is_tradable_basic"] = 0

    return df


def prepare_output_frame(df):
    """
    Normalize one stock frame before writing a parquet row group.
    """
    df = df.dropna(subset=["date", "code"]).copy()
    if df.empty:
        return df

    df = df.sort_values(["code", "date"]).reset_index(drop=True)
    df = add_basic_flags(df)

    for c in OUTPUT_COLUMNS:
        if c not in df.columns:
            df[c] = np.nan

    for c in DATETIME_COLS:
        df[c] = pd.to_datetime(df[c], errors="coerce")

    for c in STRING_COLS:
        df[c] = df[c].astype("string")

    for c in FLOAT32_COLS:
        df[c] = pd.to_numeric(df[c], errors="coerce").astype("float32")

    for c in FLOAT64_COLS:
        df[c] = pd.to_numeric(df[c], errors="coerce").astype("float64")

    for c in INT8_COLS:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).astype("int8")

    return df[OUTPUT_COLUMNS]


def list_stock_files(folder):
    if not folder.exists():
        raise FileNotFoundError(f"Raw data folder not found: {folder}")
    return {p.stem: p for p in folder.glob("*.csv")}


def progress_iter(iterable, total, desc):
    """
    Use tqdm when it is installed; otherwise fall back to the regular iterator.
    """
    if tqdm is None:
        return iterable
    return tqdm(iterable, total=total, desc=desc, unit="stock")


def clear_output_dirs():
    """
    Remove existing partitioned panel outputs before rebuilding.
    """
    for path in [PANEL_BY_CODE_DIR, PANEL_BY_YEAR_DIR]:
        if path.exists():
            shutil.rmtree(path)
        path.mkdir(parents=True, exist_ok=True)


def process_one_stock(task):
    """
    Worker entrypoint for reading, merging, and cleaning one stock.
    """
    code, raw_path, adj_path = task
    one = merge_one_stock(
        code=code,
        raw_path=Path(raw_path) if raw_path is not None else None,
        adj_path=Path(adj_path) if adj_path is not None else None,
    )
    if one is None:
        return code, None
    one = prepare_output_frame(one)
    if one is None or one.empty:
        return code, None
    return code, one


def write_code_partition(one):
    """
    Write one stock to the code-partitioned dataset.
    """
    code = str(one["code"].iloc[0]).zfill(6)
    out_dir = PANEL_BY_CODE_DIR / f"code={code}"
    out_dir.mkdir(parents=True, exist_ok=True)

    table = pa.Table.from_pandas(
        one[CODE_OUTPUT_COLUMNS],
        schema=CODE_PANEL_SCHEMA,
        preserve_index=False,
    )
    pq.write_table(
        table,
        out_dir / "part.parquet",
        compression="zstd",
        row_group_size=PANEL_ROW_GROUP_SIZE,
    )


def write_year_batches(buffer, part_counter):
    """
    Flush buffered stock frames into year-partitioned parquet files.
    """
    if not buffer:
        return 0

    batch = pd.concat(buffer, ignore_index=True)
    batch["year"] = pd.to_datetime(batch["date"]).dt.year.astype("int32")
    written_rows = len(batch)

    for year, group in batch.groupby("year", sort=True):
        out_dir = PANEL_BY_YEAR_DIR / f"year={int(year)}"
        out_dir.mkdir(parents=True, exist_ok=True)
        part_id = part_counter.get(int(year), 0)
        part_counter[int(year)] = part_id + 1

        group = group.drop(columns=["year"])
        table = pa.Table.from_pandas(group[OUTPUT_COLUMNS], schema=PANEL_SCHEMA, preserve_index=False)
        pq.write_table(
            table,
            out_dir / f"part-{part_id:05d}.parquet",
            compression="zstd",
            row_group_size=PANEL_ROW_GROUP_SIZE,
        )

    return written_rows


def update_stats(stats, one):
    """
    Update aggregate conversion statistics from one stock frame.
    """
    stats["rows"] += len(one)
    stats["stocks"] += 1
    one_min = one["date"].min()
    one_max = one["date"].max()
    stats["min_date"] = one_min if stats["min_date"] is None else min(stats["min_date"], one_min)
    stats["max_date"] = one_max if stats["max_date"] is None else max(stats["max_date"], one_max)


def iter_processed_stocks(tasks, n_workers):
    """
    Yield processed stock frames from either single-process or multiprocess mode.
    """
    if n_workers <= 1:
        task_iter = progress_iter(tasks, total=len(tasks), desc="Building panel")
        for i, task in enumerate(task_iter, start=1):
            if tqdm is None and (i == 1 or i % 500 == 0 or i == len(tasks)):
                print(f"Processing {i}/{len(tasks)}: {task[0]}")
            try:
                yield process_one_stock(task)
            except Exception as exc:
                yield task[0], exc
        return

    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = {executor.submit(process_one_stock, task): task[0] for task in tasks}
        future_iter = progress_iter(
            as_completed(futures),
            total=len(futures),
            desc=f"Building panel ({n_workers} workers)",
        )
        for i, future in enumerate(future_iter, start=1):
            code = futures[future]
            if tqdm is None and (i == 1 or i % 500 == 0 or i == len(futures)):
                print(f"Processing {i}/{len(futures)}")
            try:
                yield future.result()
            except Exception as exc:
                yield code, exc


def write_panel_datasets(limit_codes=None, n_workers=N_WORKERS):
    """
    Build code-partitioned and year-partitioned panel datasets.
    """
    raw_files = list_stock_files(RAW_PRICE_DIR)
    adj_files = list_stock_files(ADJ_PRICE_DIR)
    all_codes = sorted(set(raw_files) | set(adj_files))
    if limit_codes is not None:
        limit_codes = {str(code).zfill(6) for code in limit_codes}
        all_codes = [code for code in all_codes if code in limit_codes]

    print(f"Raw price files: {len(raw_files)} from {RAW_PRICE_DIR}")
    print(f"Adjusted price files: {len(adj_files)} from {ADJ_PRICE_DIR}")
    print(f"Total stock codes to process: {len(all_codes)}")
    print(f"Workers: {n_workers}")

    clear_output_dirs()

    tasks = [
        (
            code,
            str(raw_files[code]) if code in raw_files else None,
            str(adj_files[code]) if code in adj_files else None,
        )
        for code in all_codes
    ]

    failed = []
    buffer = []
    buffer_rows = 0
    part_counter = {}
    stats = {
        "rows": 0,
        "stocks": 0,
        "min_date": None,
        "max_date": None,
    }

    for code, result in iter_processed_stocks(tasks, int(n_workers)):
        if isinstance(result, Exception):
            failed.append((code, str(result)))
            continue
        if result is None or result.empty:
            continue

        write_code_partition(result)
        buffer.append(result)
        buffer_rows += len(result)
        update_stats(stats, result)

        if buffer_rows >= PANEL_MAX_BUFFER_ROWS:
            write_year_batches(buffer, part_counter)
            buffer = []
            buffer_rows = 0

    if buffer:
        write_year_batches(buffer, part_counter)

    if stats["rows"] == 0:
        raise RuntimeError("No panel rows were built. Check raw data paths and date filters.")

    if failed:
        print("[Warning] Failed files:")
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
    parser = argparse.ArgumentParser(description="Build partitioned A-share daily panel datasets.")
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

    print("Building daily panel datasets from per-stock CSV files...")
    stats = write_panel_datasets(limit_codes=limit_codes, n_workers=args.workers)

    print("Done.")
    print(f"Saved code-partitioned panel to: {PANEL_BY_CODE_DIR}")
    print(f"Saved year-partitioned panel to: {PANEL_BY_YEAR_DIR}")
    print("Panel rows:", stats["rows"])
    print("Date range:", stats["min_date"], "to", stats["max_date"])
    print("Number of stocks:", stats["stocks"])
    print("Failed stocks:", stats["failed"])


if __name__ == "__main__":
    main()
