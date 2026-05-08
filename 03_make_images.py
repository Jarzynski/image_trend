# -*- coding: utf-8 -*-
"""
03_make_images.py

Generate binary price images for all configured image/return experiments.

Input
-----
data/features/baseline_features.parquet

Outputs
-------
For each EXPERIMENTS entry in config.py:
data/images/images_{experiment_name}.npy
data/images/meta_{experiment_name}.parquet

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
import pyarrow as pa
import pyarrow.parquet as pq

from config import (
    BASELINE_FEATURE_PATH,
    DAY_WIDTH,
    WHITE_PIXEL,
    EXPERIMENTS,
    image_path_for_experiment,
    meta_path_for_experiment,
)


META_CHUNK_SIZE = 50_000


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
        "amount", "float_mktcap", "is_limit_up",
    }

    for cfg in EXPERIMENTS.values():
        horizon = cfg["horizon"]
        cols.add(f"future_ret_{horizon}d")
        cols.add(f"label_{horizon}d")
        cols.add(cfg["ma_col"])

    return sorted(cols)


def load_features():
    """
    Load only columns required by image generation.
    """
    required = needed_feature_columns()
    available = set(pq.read_schema(BASELINE_FEATURE_PATH).names)
    missing = [c for c in required if c not in available]

    if missing:
        raise RuntimeError(
            "Missing required columns in baseline feature file: "
            + ", ".join(missing)
        )

    print(f"Reading features: {BASELINE_FEATURE_PATH}")
    df = pd.read_parquet(BASELINE_FEATURE_PATH, columns=required)
    df["date"] = pd.to_datetime(df["date"])
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
                "is_limit_up": row.get("is_limit_up", 0),
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


def count_samples(df, exp_name, cfg):
    return sum(1 for _img, _meta in iter_valid_samples(df, exp_name, cfg, with_image=False))


def write_meta_chunk(meta_rows, writer, meta_path):
    table = pa.Table.from_pandas(pd.DataFrame(meta_rows), preserve_index=False)
    if writer is None:
        writer = pq.ParquetWriter(meta_path, table.schema, compression="zstd")
    writer.write_table(table)
    return writer


def generate_images_for_experiment(df, exp_name, cfg):
    """
    Generate one experiment's image .npy and metadata parquet.
    """
    image_height = cfg["image_height"]
    width = cfg["image_width"]

    image_path = image_path_for_experiment(exp_name)
    meta_path = meta_path_for_experiment(exp_name)

    print(f"Counting samples for {exp_name}...")
    n_samples = count_samples(df, exp_name, cfg)
    if n_samples == 0:
        raise RuntimeError(f"No images generated for {exp_name}. Check filters and columns.")

    print(f"{exp_name} sample count: {n_samples}")
    print(f"Writing images to: {image_path}")
    images = np.lib.format.open_memmap(
        image_path,
        mode="w+",
        dtype=np.uint8,
        shape=(n_samples, image_height, width, 1),
    )

    writer = None
    meta_rows = []
    written = 0

    try:
        for img, meta in iter_valid_samples(df, exp_name, cfg, with_image=True):
            if written >= n_samples:
                raise RuntimeError(f"{exp_name} generated more samples than counted.")

            images[written] = img
            written += 1
            meta_rows.append(meta)

            if len(meta_rows) >= META_CHUNK_SIZE:
                writer = write_meta_chunk(meta_rows, writer, meta_path)
                meta_rows = []

        if meta_rows:
            writer = write_meta_chunk(meta_rows, writer, meta_path)
    finally:
        images.flush()
        if writer is not None:
            writer.close()

    if written != n_samples:
        raise RuntimeError(f"{exp_name} count mismatch: counted {n_samples}, wrote {written}.")

    print(f"Saved metadata to: {meta_path}")
    print(exp_name, "images shape:", images.shape)
    print(exp_name, "meta rows:", written)


def main():
    df = load_features()

    for exp_name, cfg in EXPERIMENTS.items():
        print(f"Generating images for {exp_name}...")
        generate_images_for_experiment(df, exp_name, cfg)

    print("Done.")


if __name__ == "__main__":
    main()
