# -*- coding: utf-8 -*-
"""
01_build_panel.py

Build the daily A-share panel from per-stock CSV files.

Input
-----
../daily_OHLVC/不复权/*.csv
../daily_OHLVC/后复权/*.csv

Output
------
data/processed/panel_daily.parquet

Design
------
The image model should use adjusted prices for pattern construction and
future-return labels, while trading filters and backtest constraints should
use raw prices, volume, amount, ST, and limit-up fields.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from config import (
    PANEL_PATH,
    PROJECT_DIR,
    START_DATE,
    END_DATE,
)


RAW_DATA_ROOT = PROJECT_DIR.parent / "daily_OHLVC"
RAW_PRICE_DIR = RAW_DATA_ROOT / "不复权"
ADJ_PRICE_DIR = RAW_DATA_ROOT / "后复权"

CSV_ENCODINGS = ("utf-8-sig", "utf-8", "gbk", "gb18030")

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

STRING_COLS = ["code", "name", "industry"]
DATETIME_COLS = ["date", "list_date", "delist_date"]
INT8_COLS = ["is_st", "is_limit_up", "used_raw_as_adj", "is_suspended", "is_tradable_basic"]
FLOAT_COLS = [c for c in OUTPUT_COLUMNS if c not in STRING_COLS + DATETIME_COLS + INT8_COLS]

PANEL_SCHEMA = pa.schema(
    [(c, pa.timestamp("ns")) for c in DATETIME_COLS]
    + [(c, pa.string()) for c in STRING_COLS]
    + [(c, pa.float64()) for c in FLOAT_COLS]
    + [(c, pa.int8()) for c in INT8_COLS]
)


def read_csv_auto(path, needed_columns):
    """
    Read one CSV with a small encoding fallback list.
    """
    last_error = None
    for encoding in CSV_ENCODINGS:
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
        f"Could not read {path} with encodings {CSV_ENCODINGS}: {last_error}",
    )


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
            df[c] = (
                df[c]
                .astype(str)
                .str.replace(",", "", regex=False)
                .replace({"nan": np.nan, "None": np.nan, "--": np.nan, "-": np.nan, "": np.nan})
            )
            df[c] = pd.to_numeric(df[c], errors="coerce")

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

    for c in FLOAT_COLS:
        df[c] = pd.to_numeric(df[c], errors="coerce").astype("float64")

    for c in INT8_COLS:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).astype("int8")

    return df[OUTPUT_COLUMNS]


def list_stock_files(folder):
    if not folder.exists():
        raise FileNotFoundError(f"Raw data folder not found: {folder}")
    return {p.stem: p for p in folder.glob("*.csv")}


def write_panel_streaming():
    """
    Build and write the panel one stock at a time.

    Each stock becomes one parquet row group, avoiding a full in-memory concat
    of the complete A-share history.
    """
    raw_files = list_stock_files(RAW_PRICE_DIR)
    adj_files = list_stock_files(ADJ_PRICE_DIR)
    all_codes = sorted(set(raw_files) | set(adj_files))

    print(f"Raw price files: {len(raw_files)} from {RAW_PRICE_DIR}")
    print(f"Adjusted price files: {len(adj_files)} from {ADJ_PRICE_DIR}")
    print(f"Total stock codes to process: {len(all_codes)}")

    PANEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    writer = None
    failed = []
    total_rows = 0
    written_stocks = 0
    min_date = None
    max_date = None

    try:
        for i, code in enumerate(all_codes, start=1):
            if i == 1 or i % 500 == 0 or i == len(all_codes):
                print(f"Processing {i}/{len(all_codes)}: {code}")

            try:
                one = merge_one_stock(
                    code=code,
                    raw_path=raw_files.get(code),
                    adj_path=adj_files.get(code),
                )
                one = prepare_output_frame(one)
            except Exception as exc:
                failed.append((code, str(exc)))
                continue

            if one is None or one.empty:
                continue

            table = pa.Table.from_pandas(one, schema=PANEL_SCHEMA, preserve_index=False)
            if writer is None:
                writer = pq.ParquetWriter(PANEL_PATH, PANEL_SCHEMA, compression="zstd")
            writer.write_table(table)

            total_rows += len(one)
            written_stocks += 1
            one_min = one["date"].min()
            one_max = one["date"].max()
            min_date = one_min if min_date is None else min(min_date, one_min)
            max_date = one_max if max_date is None else max(max_date, one_max)
    finally:
        if writer is not None:
            writer.close()

    if total_rows == 0:
        raise RuntimeError("No panel rows were built. Check raw data paths and date filters.")

    if failed:
        print("[Warning] Failed files:")
        for code, msg in failed[:20]:
            print(f"  - {code}: {msg}")
        if len(failed) > 20:
            print(f"  ... {len(failed) - 20} more")

    return {
        "rows": total_rows,
        "stocks": written_stocks,
        "min_date": min_date,
        "max_date": max_date,
        "failed": len(failed),
    }


def main():
    print("Building daily panel from per-stock CSV files with streaming parquet writes...")
    stats = write_panel_streaming()

    print("Done.")
    print(f"Saved panel to: {PANEL_PATH}")
    print("Panel rows:", stats["rows"])
    print("Date range:", stats["min_date"], "to", stats["max_date"])
    print("Number of stocks:", stats["stocks"])
    print("Failed stocks:", stats["failed"])


if __name__ == "__main__":
    main()
