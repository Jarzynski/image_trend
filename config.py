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
# 2. Standard output files
# ============================================================

PANEL_BY_CODE_DIR = PROCESSED_DIR / "panel_by_code"
PANEL_BY_YEAR_DIR = PROCESSED_DIR / "panel_by_year"
FEATURE_BY_CODE_BUCKET_DIR = FEATURE_DIR / "features_by_code_bucket"
FEATURE_BY_YEAR_DIR = FEATURE_DIR / "features_by_year"

IMAGE_SHARD_SIZE = 10_000


def image_dir_for_window(window):
    return IMAGE_DIR / f"window_{int(window)}"


def shard_dir_for_window(window, shard_id):
    return image_dir_for_window(window) / f"shard_{shard_id:05d}"


def shard_image_path_for_window(window, shard_id):
    return shard_dir_for_window(window, shard_id) / "images.npy"


def shard_meta_path_for_window(window, shard_id):
    return shard_dir_for_window(window, shard_id) / "meta.parquet"


def image_dir_for_experiment(exp_name):
    return IMAGE_DIR / exp_name.lower()


def shard_dir_for_experiment(exp_name, shard_id):
    return image_dir_for_experiment(exp_name) / f"shard_{shard_id:05d}"


def shard_image_path(exp_name, shard_id):
    return shard_dir_for_experiment(exp_name, shard_id) / "images.npy"


def shard_meta_path(exp_name, shard_id):
    return shard_dir_for_experiment(exp_name, shard_id) / "meta.parquet"


# ============================================================
# 3. Sample filters
# ============================================================

START_DATE = "2014-01-01"
END_DATE = "2024-12-31"

TRAIN_END = "2019-12-31"
VALID_START = "2020-01-01"
VALID_END = "2020-12-31"
TEST_START = "2021-01-01"

MIN_LIST_DAYS = 120
EMBARGO_DAYS_BY_HORIZON = {
    5: 5,
    20: 20,
}

RANDOM_SEED = 42
CNN_WEIGHT_DECAY = 1e-4

N_WORKERS = 12
PANEL_MAX_BUFFER_ROWS = 500_000
PANEL_ROW_GROUP_SIZE = 128_000
FEATURE_MAX_BUFFER_ROWS = 500_000
FEATURE_ROW_GROUP_SIZE = 128_000
FEATURE_CODE_BUCKETS = 128

# Main liquidity threshold.
# You can later test 10m, 50m, 100m as robustness checks.
MIN_AMOUNT = 20_000_000
LOW_VOLUME_LIMIT_RATIO = 0.10


# ============================================================
# 4. Experiments
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
# 5. Backtest config
# ============================================================

N_DECILES = 10
COST_BPS_GRID = [0, 10, 25, 50, 100]
UNIVERSE_SPLIT_METHOD = "daily_tercile"
TRADING_DAYS_PER_YEAR = 252
