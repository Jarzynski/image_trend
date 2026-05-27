# -*- coding: utf-8 -*-
"""
03_make_images.py

Generate binary price images for all configured image/return experiments.

Input
-----
data/features/features_by_code_bucket/bucket=*/part-*.parquet

Outputs
-------
For each EXPERIMENTS entry in config.py:
data/images/{experiment_name}/shard_00000/images.npy
data/images/{experiment_name}/shard_00000/meta.parquet

Image design
------------
- Black background: 0
- White line/bar: 255
- Upper region: OHLC + MA
- Lower region: amount bar, about one-fifth of image height
- Each trading day occupies 3 horizontal pixels:
    x0: open tick
    x1: high-low vertical line
    x2: close tick
"""

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import shutil

try:
    from tqdm.auto import tqdm
except ImportError:
    tqdm = None

from config import (
    FEATURE_BY_CODE_BUCKET_DIR,
    DAY_WIDTH,
    WHITE_PIXEL,
    EXPERIMENTS,
    IMAGE_SHARD_SIZE,
    image_dir_for_experiment,
    shard_image_path,
    shard_meta_path,
)


def scale_price_to_y(price, p_min, p_max, price_height):
    """
    Map price to image y coordinate.
    """
    if p_max <= p_min:
        return None

    y = int(round((p_max - price) / (p_max - p_min) * (price_height - 1)))
    y = max(0, min(price_height - 1, y))
    return y


def draw_line(img, x0, y0, x1, y1):
    """
    Draw a simple line on a binary image using linear interpolation.
    """
    n = max(abs(x1 - x0), abs(y1 - y0)) + 1
    xs = np.linspace(x0, x1, n).round().astype(int)
    ys = np.linspace(y0, y1, n).round().astype(int)

    h, w = img.shape
    valid = (xs >= 0) & (xs < w) & (ys >= 0) & (ys < h)
    img[ys[valid], xs[valid]] = WHITE_PIXEL


def window_is_valid(window_df, ma_col):
    """
    Fast validity check shared by count and write passes.
    """
    required_cols = ["open_adj", "high_adj", "low_adj", "close_adj", "amount", ma_col]
    if window_df[required_cols].isna().any().any():
        return False

    p_max = max(window_df["high_adj"].max(), window_df[ma_col].max())
    p_min = min(window_df["low_adj"].min(), window_df[ma_col].min())
    if not np.isfinite(p_max) or not np.isfinite(p_min) or p_max <= p_min:
        return False

    amount_max = window_df["amount"].max()
    if not np.isfinite(amount_max) or amount_max <= 0:
        return False

    return True


def make_single_image(window_df, ma_col, image_height, price_height, volume_height):
    """
    Convert one stock-date window into a binary image.

    Returns
    -------
    img : np.ndarray, shape [image_height, window*3], dtype uint8
    """
    window = len(window_df)
    width = window * DAY_WIDTH

    img = np.zeros((image_height, width), dtype=np.uint8)

    if not window_is_valid(window_df, ma_col):
        return None

    p_max = max(window_df["high_adj"].max(), window_df[ma_col].max())
    p_min = min(window_df["low_adj"].min(), window_df[ma_col].min())
    amount_max = window_df["amount"].max()

    for d, row in enumerate(window_df.itertuples(index=False)):
        x0 = d * DAY_WIDTH
        x1 = x0 + 1
        x2 = x0 + 2

        open_y = scale_price_to_y(getattr(row, "open_adj"), p_min, p_max, price_height)
        high_y = scale_price_to_y(getattr(row, "high_adj"), p_min, p_max, price_height)
        low_y = scale_price_to_y(getattr(row, "low_adj"), p_min, p_max, price_height)
        close_y = scale_price_to_y(getattr(row, "close_adj"), p_min, p_max, price_height)

        if None in [open_y, high_y, low_y, close_y]:
            return None

        y_start, y_end = sorted([high_y, low_y])
        img[y_start:y_end + 1, x1] = WHITE_PIXEL
        img[open_y, x0] = WHITE_PIXEL
        img[close_y, x2] = WHITE_PIXEL

    ma_points = []
    for d, row in enumerate(window_df.itertuples(index=False)):
        x = d * DAY_WIDTH + 1
        y = scale_price_to_y(getattr(row, ma_col), p_min, p_max, price_height)
        if y is None:
            return None
        ma_points.append((x, y))

    for (x0, y0), (x1, y1) in zip(ma_points[:-1], ma_points[1:]):
        draw_line(img, x0, y0, x1, y1)

    volume_bottom = image_height - 1
    for d, row in enumerate(window_df.itertuples(index=False)):
        scaled = getattr(row, "amount") / amount_max
        bar_h = int(round(scaled * (volume_height - 1)))
        bar_h = max(0, min(volume_height - 1, bar_h))

        x0 = d * DAY_WIDTH
        x2 = x0 + 2
        y0 = volume_bottom - bar_h
        y1 = volume_bottom

        img[y0:y1 + 1, x0:x2 + 1] = WHITE_PIXEL

    return img


def needed_feature_columns():
    cols = {
        "date", "code", "industry", "is_tradable",
        "open_adj", "high_adj", "low_adj", "close_adj",
        "amount", "float_mktcap",
        "is_low_volume_limit_up", "is_low_volume_limit_down",
    }

    for cfg in EXPERIMENTS.values():
        horizon = cfg["horizon"]
        cols.add(f"future_ret_{horizon}d")
        cols.add(f"label_{horizon}d")
        cols.add(cfg["ma_col"])

    return sorted(cols)


def list_feature_bucket_files():
    """
    List code-bucketed feature files produced by 02_make_labels_and_baselines.py.
    """
    files = sorted(FEATURE_BY_CODE_BUCKET_DIR.glob("bucket=*/part-*.parquet"))
    if not files:
        raise RuntimeError(
            f"No code-bucketed feature files found in {FEATURE_BY_CODE_BUCKET_DIR}. "
            "Run 02_make_labels_and_baselines.py first."
        )
    return files


def progress_iter(iterable, total, desc):
    """
    Use tqdm when it is installed; otherwise fall back to the regular iterator.
    """
    if tqdm is None:
        return iterable
    return tqdm(iterable, total=total, desc=desc, unit="part")


def validate_feature_schema(path, required):
    """
    Check one feature partition before streaming image generation.
    """
    available = set(pq.read_schema(path).names)
    missing = [c for c in required if c not in available]

    if missing:
        raise RuntimeError(
            f"Missing required columns in feature partition {path}: "
            + ", ".join(missing)
        )


def read_feature_bucket_part(path, required):
    """
    Read one code-bucket feature part.
    """
    df = pd.read_parquet(path, columns=required)
    df["date"] = pd.to_datetime(df["date"])
    df["code"] = df["code"].astype(str).str.zfill(6)
    return df.sort_values(["code", "date"]).reset_index(drop=True)


def iter_valid_samples(df, exp_name, cfg, with_image):
    """
    Yield valid image samples and metadata for one experiment.
    """
    window = cfg["window"]
    horizon = cfg["horizon"]
    ma_col = cfg["ma_col"]
    image_height = cfg["image_height"]
    price_height = cfg["price_height"]
    volume_height = cfg["volume_height"]

    label_col = f"label_{horizon}d"
    future_ret_col = f"future_ret_{horizon}d"

    for code, g in df.groupby("code", sort=False):
        g = g.sort_values("date").reset_index(drop=True)
        if len(g) < window + horizon:
            continue

        for idx in range(window - 1, len(g)):
            row = g.iloc[idx]

            if pd.isna(row.get(label_col)) or pd.isna(row.get(future_ret_col)):
                continue
            if row.get("is_tradable", 0) != 1:
                continue

            window_df = g.iloc[idx - window + 1: idx + 1]
            if not window_is_valid(window_df, ma_col):
                continue

            meta = {
                "date": row["date"],
                "code": row["code"],
                "industry": row.get("industry", None),
                "experiment_name": exp_name,
                "window": window,
                "horizon": horizon,
                "image_height": image_height,
                "image_width": cfg["image_width"],
                "price_height": price_height,
                "volume_height": volume_height,
                "future_ret": row[future_ret_col],
                "label": row[label_col],
                "amount": row.get("amount", np.nan),
                "float_mktcap": row.get("float_mktcap", np.nan),
                "is_low_volume_limit_up": row.get("is_low_volume_limit_up", 0),
                "is_low_volume_limit_down": row.get("is_low_volume_limit_down", 0),
            }

            if with_image:
                img = make_single_image(
                    window_df,
                    ma_col=ma_col,
                    image_height=image_height,
                    price_height=price_height,
                    volume_height=volume_height,
                )
                if img is None:
                    continue
                yield img[:, :, None], meta
            else:
                yield None, meta


def clear_experiment_image_dir(exp_name):
    image_dir = image_dir_for_experiment(exp_name)
    if image_dir.exists():
        shutil.rmtree(image_dir)
    image_dir.mkdir(parents=True, exist_ok=True)


def write_image_shard(exp_name, shard_id, image_rows, meta_rows):
    if not image_rows:
        return 0

    shard_dir = shard_image_path(exp_name, shard_id).parent
    shard_dir.mkdir(parents=True, exist_ok=True)

    images = np.stack(image_rows, axis=0).astype(np.uint8, copy=False)
    np.save(shard_image_path(exp_name, shard_id), images)

    meta = pd.DataFrame(meta_rows)
    meta["shard_id"] = shard_id
    meta["local_index"] = np.arange(len(meta), dtype=np.int64)
    meta.to_parquet(shard_meta_path(exp_name, shard_id), index=False)

    return len(meta)


def generate_images_for_experiment(feature_files, exp_name, cfg):
    """
    Generate one experiment's image shards and metadata shards.
    """
    clear_experiment_image_dir(exp_name)

    required = needed_feature_columns()
    validate_feature_schema(feature_files[0], required)

    image_rows = []
    meta_rows = []
    shard_id = 0
    written = 0

    print(f"Writing sharded images to: {image_dir_for_experiment(exp_name)}")
    file_iter = progress_iter(feature_files, total=len(feature_files), desc=f"Images {exp_name}")
    for i, path in enumerate(file_iter, start=1):
        if tqdm is None and (i == 1 or i % 500 == 0 or i == len(feature_files)):
            print(f"{exp_name} processing {i}/{len(feature_files)}: {path.parent.name}")

        df = read_feature_bucket_part(path, required)
        for img, meta in iter_valid_samples(df, exp_name, cfg, with_image=True):
            image_rows.append(img)
            meta_rows.append(meta)

            if len(image_rows) >= IMAGE_SHARD_SIZE:
                written += write_image_shard(exp_name, shard_id, image_rows, meta_rows)
                print(f"{exp_name} wrote shard {shard_id:05d}: {len(image_rows)} samples")
                shard_id += 1
                image_rows = []
                meta_rows = []

    if image_rows:
        written += write_image_shard(exp_name, shard_id, image_rows, meta_rows)
        print(f"{exp_name} wrote shard {shard_id:05d}: {len(image_rows)} samples")

    if written == 0:
        raise RuntimeError(f"No images generated for {exp_name}. Check filters and columns.")

    print(exp_name, "total image rows:", written)


def main():
    feature_files = list_feature_bucket_files()
    print(f"Feature code-bucket parts: {len(feature_files)} from {FEATURE_BY_CODE_BUCKET_DIR}")

    for exp_name, cfg in EXPERIMENTS.items():
        print(f"Generating images for {exp_name}...")
        generate_images_for_experiment(feature_files, exp_name, cfg)

    print("Done.")


if __name__ == "__main__":
    main()
