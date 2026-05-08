# -*- coding: utf-8 -*-
"""
Project config for A-share price-image trend prediction.

This file centralizes all paths, column mappings, sample filters,
train/validation/test split, and experiment definitions.

You should edit this file first before running other scripts.
"""

from pathlib import Path


# ============================================================
# 1. Project paths
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parent

DATA_DIR = PROJECT_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
FEATURE_DIR = DATA_DIR / "features"
IMAGE_DIR = DATA_DIR / "images"

OUTPUT_DIR = PROJECT_DIR / "outputs"
PRED_DIR = OUTPUT_DIR / "predictions"
TABLE_DIR = OUTPUT_DIR / "tables"
MODEL_DIR = OUTPUT_DIR / "models"

for p in [
    DATA_DIR,
    RAW_DIR,
    PROCESSED_DIR,
    FEATURE_DIR,
    IMAGE_DIR,
    OUTPUT_DIR,
    PRED_DIR,
    TABLE_DIR,
    MODEL_DIR,
]:
    p.mkdir(parents=True, exist_ok=True)


# ============================================================
# 2. Raw data paths
# ============================================================
# You can place one merged CSV here, or modify 01_build_panel.py
# to read a folder of per-stock CSV files.

RAW_CSV_PATH = RAW_DIR / "a_share_daily.csv"


# ============================================================
# 3. Standard output files
# ============================================================

PANEL_PATH = PROCESSED_DIR / "panel_daily.parquet"
BASELINE_FEATURE_PATH = FEATURE_DIR / "baseline_features.parquet"

def image_path_for_experiment(exp_name):
    return IMAGE_DIR / f"images_{exp_name.lower()}.npy"


def meta_path_for_experiment(exp_name):
    return IMAGE_DIR / f"meta_{exp_name.lower()}.parquet"


IMAGE_I5_PATH = image_path_for_experiment("I5R5")
META_I5_PATH = meta_path_for_experiment("I5R5")

IMAGE_I20_PATH = image_path_for_experiment("I20R20")
META_I20_PATH = meta_path_for_experiment("I20R20")


# ============================================================
# 4. Column mapping
# ============================================================
# Left side: expected standard column name used by scripts.
# Right side: raw Chinese column name in your data.
#
# You should adjust the right side according to your actual CSV.
# If your data already uses English names, map them directly.

COLUMN_MAP = {
    "date": "日期",
    "code": "代码",
    "name": "名称",
    "industry": "所属行业",

    # Adjusted OHLC. If your raw file contains only one price system,
    # temporarily map them here and treat them as adjusted prices.
    "open_adj": "开盘价",
    "high_adj": "最高价",
    "low_adj": "最低价",
    "close_adj": "收盘价",

    # Raw OHLC. If you have separate non-adjusted data, merge it before this step.
    # For MVP, we temporarily set raw = adjusted if raw fields do not exist.
    "open_raw": "开盘价",
    "high_raw": "最高价",
    "low_raw": "最低价",
    "close_raw": "收盘价",
    "prev_close_raw": "前收盘价",

    "volume": "成交量（额）",
    "amount": "成交额（元）",
    "turnover_rate": "换手率",
    "pct_chg": "涨幅%",
    "amplitude": "振幅%",

    "is_st": "是否ST",
    "volume_ratio": "量比",

    "ret_3d_raw": "三日涨幅%",
    "ret_6d_raw": "六日涨幅%",
    "ret_10d_raw": "十日涨幅%",
    "ret_25d_raw": "25日涨幅%",

    "is_limit_up": "是否涨停",

    "total_shares": "总股本（股）",
    "float_shares": "流通股本（股）",
    "total_mktcap": "总市值（元）",
    "float_mktcap": "流通市值（元）",

    "pe_ttm": "滚动市盈率",
    "pb": "市净率",
    "ps_ttm": "滚动市销率",

    "ma5_adj": "5日线",
    "ma10_adj": "10日线",
    "ma20_adj": "20日线",
    "ma30_adj": "30日线",
    "ma120_adj": "120日线",
    "ma250_adj": "250日线",

    "list_date": "上市时间",
    "delist_date": "退市时间",
}


# ============================================================
# 5. Sample filters
# ============================================================

START_DATE = "2014-01-01"
END_DATE = "2024-12-31"

TRAIN_END = "2019-12-31"
VALID_START = "2020-01-01"
VALID_END = "2020-12-31"
TEST_START = "2021-01-01"

MIN_LIST_DAYS = 120

# Main liquidity threshold.
# You can later test 10m, 50m, 100m as robustness checks.
MIN_AMOUNT = 20_000_000


# ============================================================
# 6. Experiments
# ============================================================

DAY_WIDTH = 3
WHITE_PIXEL = 255

IMAGE_HEIGHT_BY_WINDOW = {
    5: 32,
    20: 64,
    60: 96,
}


def image_width_for_window(window):
    return window * DAY_WIDTH


def image_height_for_window(window):
    return IMAGE_HEIGHT_BY_WINDOW[window]


def price_height_for_window(window):
    return int(round(image_height_for_window(window) * 4 / 5))


def volume_height_for_window(window):
    return image_height_for_window(window) - price_height_for_window(window)


EXPERIMENTS = {
    "I5R5": {
        "window": 5,
        "horizon": 5,
        "ma_col": "ma5_adj",
        "image_height": image_height_for_window(5),
        "image_width": image_width_for_window(5),
        "price_height": price_height_for_window(5),
        "volume_height": volume_height_for_window(5),
    },
    "I20R5": {
        "window": 20,
        "horizon": 5,
        "ma_col": "ma20_adj",
        "image_height": image_height_for_window(20),
        "image_width": image_width_for_window(20),
        "price_height": price_height_for_window(20),
        "volume_height": volume_height_for_window(20),
    },
    "I60R5": {
        "window": 60,
        "horizon": 5,
        "ma_col": "ma60_adj",
        "image_height": image_height_for_window(60),
        "image_width": image_width_for_window(60),
        "price_height": price_height_for_window(60),
        "volume_height": volume_height_for_window(60),
    },
    "I20R20": {
        "window": 20,
        "horizon": 20,
        "ma_col": "ma20_adj",
        "image_height": image_height_for_window(20),
        "image_width": image_width_for_window(20),
        "price_height": price_height_for_window(20),
        "volume_height": volume_height_for_window(20),
    },
    "I60R20": {
        "window": 60,
        "horizon": 20,
        "ma_col": "ma60_adj",
        "image_height": image_height_for_window(60),
        "image_width": image_width_for_window(60),
        "price_height": price_height_for_window(60),
        "volume_height": volume_height_for_window(60),
    },
}


# ============================================================
# 7. Backtest config
# ============================================================

N_DECILES = 10
ONE_WAY_COST_BPS = 10
TRADING_DAYS_PER_YEAR = 252
