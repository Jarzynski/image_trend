# -*- coding: utf-8 -*-
"""
Generate sharded binary price images by image window.

This script deliberately decouples image generation from label horizon:
the same 20-day image is written once under data/images/window_20 and its
metadata carries every requested horizon label, for example label_5d and
label_20d. Downstream experiment scripts choose the label matching their
horizon without duplicating the physical image.

Inputs
------
data/features/features_by_code_bucket/bucket=*/part-*.parquet

Outputs
-------
data/images/window_{window}/shard_*/images.npy
data/images/window_{window}/shard_*/meta.parquet
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
import argparse
import os
from pathlib import Path
import shutil
from typing import Dict, Iterable, Iterator, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover
    tqdm = None

from config import (
    FEATURE_BY_CODE_BUCKET_DIR,
    DAY_WIDTH,
    WHITE_PIXEL,
    EXPERIMENTS,
    IMAGE_SHARD_SIZE,
    image_dir_for_window,
    shard_dir_for_window,
)


def default_worker_count() -> int:
    """Use all CPUs visible to this process by default."""
    try:
        return max(1, len(os.sched_getaffinity(0)))
    except AttributeError:
        return max(1, os.cpu_count() or 1)


def progress_iter(iterable: Iterable, total: int, desc: str, unit: str = "part", disable: bool = False):
    if disable or tqdm is None:
        return iterable
    return tqdm(iterable, total=total, desc=desc, unit=unit)


def parse_experiment_filter(experiments_arg: Optional[str]) -> List[str]:
    if not experiments_arg:
        return list(EXPERIMENTS)

    requested = [name.strip().upper() for name in experiments_arg.split(",") if name.strip()]
    unknown = [name for name in requested if name not in EXPERIMENTS]
    if unknown:
        raise RuntimeError(f"Unknown experiments: {unknown}. Available: {sorted(EXPERIMENTS)}")
    return requested


def window_specs_from_experiments(experiment_names: Sequence[str]) -> List[dict]:
    """
    Collapse experiments to unique image windows.

    I20R5 and I20R20 share window=20 and therefore share one physical image
    dataset, while metadata carries both requested horizons.
    """
    by_window: Dict[int, dict] = {}
    for exp_name in experiment_names:
        cfg = EXPERIMENTS[exp_name]
        window = int(cfg["window"])
        horizon = int(cfg["horizon"])
        if window not in by_window:
            by_window[window] = {
                "window": window,
                "cfg": dict(cfg),
                "horizons": set(),
                "experiments": [],
            }
        by_window[window]["horizons"].add(horizon)
        by_window[window]["experiments"].append(exp_name)

    specs = []
    for spec in sorted(by_window.values(), key=lambda x: x["window"]):
        specs.append(
            {
                "window": int(spec["window"]),
                "cfg": spec["cfg"],
                "horizons": sorted(int(h) for h in spec["horizons"]),
                "experiments": sorted(spec["experiments"]),
            }
        )
    return specs


def list_feature_bucket_files() -> List[Path]:
    files = sorted(FEATURE_BY_CODE_BUCKET_DIR.glob("bucket=*/part-*.parquet"))
    if not files:
        raise RuntimeError(
            f"No code-bucketed feature files found in {FEATURE_BY_CODE_BUCKET_DIR}. "
            "Run 02_make_labels_and_baselines.py first."
        )
    return files


def needed_feature_columns(window_specs: Sequence[Mapping]) -> List[str]:
    cols = {
        "date",
        "code",
        "industry",
        "is_tradable",
        "open_adj",
        "high_adj",
        "low_adj",
        "close_adj",
        "amount",
        "float_mktcap",
        "is_low_volume_limit_up",
        "is_low_volume_limit_down",
    }
    for spec in window_specs:
        cols.add(spec["cfg"]["ma_col"])
        for horizon in spec["horizons"]:
            cols.add(f"future_ret_{horizon}d")
            cols.add(f"label_{horizon}d")
    return sorted(cols)


def validate_feature_schema(path: Path, required: Sequence[str]) -> None:
    available = set(pq.read_schema(path).names)
    missing = [c for c in required if c not in available]
    if missing:
        raise RuntimeError(
            f"Missing required columns in feature partition {path}: " + ", ".join(missing)
        )


def read_feature_bucket_part(path: Path, required: Sequence[str]) -> pd.DataFrame:
    df = pd.read_parquet(path, columns=list(required))
    df["date"] = pd.to_datetime(df["date"])
    df["code"] = df["code"].astype(str).str.zfill(6)
    return df.sort_values(["code", "date"]).reset_index(drop=True)


def clear_output_dirs(window_specs: Sequence[Mapping]) -> None:
    for spec in window_specs:
        image_dir = image_dir_for_window(spec["window"])
        if image_dir.exists():
            shutil.rmtree(image_dir)
        image_dir.mkdir(parents=True, exist_ok=True)


def rolling_max(a: np.ndarray, window: int) -> np.ndarray:
    return pd.Series(a, copy=False).rolling(window=window, min_periods=window).max().to_numpy()


def rolling_min(a: np.ndarray, window: int) -> np.ndarray:
    return pd.Series(a, copy=False).rolling(window=window, min_periods=window).min().to_numpy()


def rolling_sum_int(a: np.ndarray, window: int) -> np.ndarray:
    return pd.Series(a.astype(np.int16, copy=False), copy=False).rolling(
        window=window, min_periods=window
    ).sum().to_numpy()


def as_float_array(g: pd.DataFrame, col: str) -> np.ndarray:
    return pd.to_numeric(g[col], errors="coerce").to_numpy(dtype=np.float64, copy=True)


def as_int_array(g: pd.DataFrame, col: str, default: int = 0) -> np.ndarray:
    if col not in g.columns:
        return np.full(len(g), default, dtype=np.int8)
    return pd.to_numeric(g[col], errors="coerce").fillna(default).to_numpy(dtype=np.int8, copy=True)


def scale_prices_to_y(prices: np.ndarray, p_min: float, p_max: float, price_height: int) -> np.ndarray:
    y = np.rint((p_max - prices) / (p_max - p_min) * (price_height - 1)).astype(np.int32)
    return np.clip(y, 0, price_height - 1)


def draw_line(img: np.ndarray, x0: int, y0: int, x1: int, y1: int) -> None:
    n = max(abs(x1 - x0), abs(y1 - y0)) + 1
    xs = np.rint(np.linspace(x0, x1, n)).astype(np.int32)
    ys = np.rint(np.linspace(y0, y1, n)).astype(np.int32)
    h, w = img.shape
    valid = (xs >= 0) & (xs < w) & (ys >= 0) & (ys < h)
    img[ys[valid], xs[valid]] = WHITE_PIXEL


def make_single_image_from_arrays(
    open_w: np.ndarray,
    high_w: np.ndarray,
    low_w: np.ndarray,
    close_w: np.ndarray,
    amount_w: np.ndarray,
    ma_w: np.ndarray,
    *,
    p_min: float,
    p_max: float,
    amount_max: float,
    image_height: int,
    price_height: int,
    volume_height: int,
) -> np.ndarray:
    window = len(open_w)
    width = window * DAY_WIDTH
    img = np.zeros((image_height, width), dtype=np.uint8)

    open_y = scale_prices_to_y(open_w, p_min, p_max, price_height)
    high_y = scale_prices_to_y(high_w, p_min, p_max, price_height)
    low_y = scale_prices_to_y(low_w, p_min, p_max, price_height)
    close_y = scale_prices_to_y(close_w, p_min, p_max, price_height)
    ma_y = scale_prices_to_y(ma_w, p_min, p_max, price_height)

    for d in range(window):
        x0 = d * DAY_WIDTH
        x1 = x0 + 1
        x2 = x0 + 2
        y_start = min(high_y[d], low_y[d])
        y_end = max(high_y[d], low_y[d])
        img[y_start : y_end + 1, x1] = WHITE_PIXEL
        img[open_y[d], x0] = WHITE_PIXEL
        img[close_y[d], x2] = WHITE_PIXEL

    ma_x = np.arange(window, dtype=np.int32) * DAY_WIDTH + 1
    for d in range(window - 1):
        draw_line(img, int(ma_x[d]), int(ma_y[d]), int(ma_x[d + 1]), int(ma_y[d + 1]))

    volume_bottom = image_height - 1
    bar_h = np.rint((amount_w / amount_max) * (volume_height - 1)).astype(np.int32)
    bar_h = np.clip(bar_h, 0, volume_height - 1)
    for d in range(window):
        x0 = d * DAY_WIDTH
        x2 = x0 + 2
        y0 = volume_bottom - int(bar_h[d])
        img[y0 : volume_bottom + 1, x0 : x2 + 1] = WHITE_PIXEL

    return img[:, :, None]


def iter_valid_window_samples(
    df: pd.DataFrame,
    spec: Mapping,
) -> Iterator[Tuple[np.ndarray, dict]]:
    window = int(spec["window"])
    cfg = spec["cfg"]
    horizons = [int(h) for h in spec["horizons"]]
    ma_col = cfg["ma_col"]
    image_height = int(cfg["image_height"])
    price_height = int(cfg["price_height"])
    volume_height = int(cfg["volume_height"])
    image_width = int(cfg.get("image_width", window * DAY_WIDTH))

    if DAY_WIDTH < 3:
        raise RuntimeError("This image layout requires DAY_WIDTH >= 3.")
    if image_width != window * DAY_WIDTH:
        raise RuntimeError(
            f"window={window}: cfg image_width={image_width}, "
            f"but window*DAY_WIDTH={window * DAY_WIDTH}."
        )

    for code, g in df.groupby("code", sort=False):
        n = len(g)
        if n < window:
            continue

        open_arr = as_float_array(g, "open_adj")
        high_arr = as_float_array(g, "high_adj")
        low_arr = as_float_array(g, "low_adj")
        close_arr = as_float_array(g, "close_adj")
        amount_arr = as_float_array(g, "amount")
        ma_arr = as_float_array(g, ma_col)

        label_arrays = {h: as_float_array(g, f"label_{h}d") for h in horizons}
        future_ret_arrays = {h: as_float_array(g, f"future_ret_{h}d") for h in horizons}
        target_valid = np.zeros(n, dtype=bool)
        for h in horizons:
            target_valid |= np.isfinite(label_arrays[h]) & np.isfinite(future_ret_arrays[h])

        tradable_arr = as_int_array(g, "is_tradable", default=0)
        low_vol_up_arr = as_int_array(g, "is_low_volume_limit_up", default=0)
        low_vol_down_arr = as_int_array(g, "is_low_volume_limit_down", default=0)
        float_mktcap_arr = as_float_array(g, "float_mktcap") if "float_mktcap" in g.columns else np.full(n, np.nan)
        date_arr = g["date"].to_numpy(copy=False)
        industry_arr = g["industry"].to_numpy(copy=False) if "industry" in g.columns else np.full(n, None, dtype=object)

        finite_row = (
            np.isfinite(open_arr)
            & np.isfinite(high_arr)
            & np.isfinite(low_arr)
            & np.isfinite(close_arr)
            & np.isfinite(amount_arr)
            & np.isfinite(ma_arr)
        )
        valid_window = rolling_sum_int(finite_row, window) == window
        price_max_arr = np.maximum(rolling_max(high_arr, window), rolling_max(ma_arr, window))
        price_min_arr = np.minimum(rolling_min(low_arr, window), rolling_min(ma_arr, window))
        amount_max_arr = rolling_max(amount_arr, window)

        idx_all = np.arange(n, dtype=np.int64)
        candidate_mask = (
            (idx_all >= window - 1)
            & target_valid
            & (tradable_arr == 1)
            & valid_window
            & np.isfinite(price_max_arr)
            & np.isfinite(price_min_arr)
            & (price_max_arr > price_min_arr)
            & np.isfinite(amount_max_arr)
            & (amount_max_arr > 0)
        )

        for idx in np.flatnonzero(candidate_mask):
            start = int(idx) - window + 1
            end = int(idx) + 1
            img = make_single_image_from_arrays(
                open_arr[start:end],
                high_arr[start:end],
                low_arr[start:end],
                close_arr[start:end],
                amount_arr[start:end],
                ma_arr[start:end],
                p_min=float(price_min_arr[idx]),
                p_max=float(price_max_arr[idx]),
                amount_max=float(amount_max_arr[idx]),
                image_height=image_height,
                price_height=price_height,
                volume_height=volume_height,
            )

            meta = {
                "date": date_arr[idx],
                "code": code,
                "industry": industry_arr[idx],
                "window": window,
                "image_height": image_height,
                "image_width": image_width,
                "price_height": price_height,
                "volume_height": volume_height,
                "amount": amount_arr[idx],
                "float_mktcap": float_mktcap_arr[idx],
                "is_low_volume_limit_up": low_vol_up_arr[idx],
                "is_low_volume_limit_down": low_vol_down_arr[idx],
            }
            for h in horizons:
                meta[f"future_ret_{h}d"] = future_ret_arrays[h][idx]
                meta[f"label_{h}d"] = label_arrays[h][idx]
            yield img, meta


class TempShardWriter:
    def __init__(self, task_dir: Path, shard_size: int = IMAGE_SHARD_SIZE):
        self.task_dir = task_dir
        self.shard_size = int(shard_size)
        self.image_rows: List[np.ndarray] = []
        self.meta_rows: List[dict] = []
        self.local_shard_id = 0
        self.written_rows = 0
        self.written_shards = 0
        self.task_dir.mkdir(parents=True, exist_ok=True)

    def append(self, img: np.ndarray, meta: dict) -> None:
        self.image_rows.append(img)
        self.meta_rows.append(meta)
        if len(self.image_rows) >= self.shard_size:
            self.flush()

    def flush(self) -> None:
        if not self.image_rows:
            return

        shard_dir = self.task_dir / f"shard_{self.local_shard_id:05d}"
        shard_dir.mkdir(parents=True, exist_ok=True)
        np.save(shard_dir / "images.npy", np.stack(self.image_rows, axis=0).astype(np.uint8, copy=False))

        meta = pd.DataFrame(self.meta_rows)
        meta["shard_id"] = self.local_shard_id
        meta["local_index"] = np.arange(len(meta), dtype=np.int64)
        meta.to_parquet(shard_dir / "meta.parquet", index=False)

        n = len(self.meta_rows)
        self.written_rows += n
        self.written_shards += 1
        self.local_shard_id += 1
        self.image_rows.clear()
        self.meta_rows.clear()

    def close(self) -> Tuple[int, int]:
        self.flush()
        return self.written_rows, self.written_shards


def temp_root_for_window(window: int) -> Path:
    return image_dir_for_window(window) / "_tmp_parts"


def process_part_all_windows_task(task: Tuple[int, str, Sequence[dict], Sequence[str]]) -> dict:
    part_id, path_str, window_specs, required = task
    df = read_feature_bucket_part(Path(path_str), required)

    rows_by_window: Dict[int, int] = {}
    shards_by_window: Dict[int, int] = {}

    for spec in window_specs:
        window = int(spec["window"])
        writer = TempShardWriter(temp_root_for_window(window) / f"part_{part_id:05d}")
        for img, meta in iter_valid_window_samples(df, spec):
            writer.append(img, meta)
        rows, shards = writer.close()
        rows_by_window[window] = rows
        shards_by_window[window] = shards

    return {
        "part_id": part_id,
        "rows_by_window": rows_by_window,
        "shards_by_window": shards_by_window,
    }


def rewrite_meta_shard_id(meta_path: Path, global_shard_id: int) -> None:
    meta = pd.read_parquet(meta_path)
    meta["shard_id"] = int(global_shard_id)
    meta["local_index"] = np.arange(len(meta), dtype=np.int64)
    meta.to_parquet(meta_path, index=False)


def move_temp_shards_to_final(window: int) -> int:
    temp_root = temp_root_for_window(window)
    global_shard_id = 0
    if not temp_root.exists():
        return 0

    for part_dir in sorted(temp_root.glob("part_*")):
        for temp_shard_dir in sorted(part_dir.glob("shard_*")):
            final_dir = shard_dir_for_window(window, global_shard_id)
            if final_dir.exists():
                shutil.rmtree(final_dir)
            final_dir.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(temp_shard_dir), str(final_dir))
            rewrite_meta_shard_id(final_dir / "meta.parquet", global_shard_id)
            global_shard_id += 1

    shutil.rmtree(temp_root)
    return global_shard_id


def cleanup_temp_roots(window_specs: Sequence[Mapping]) -> None:
    for spec in window_specs:
        temp_root = temp_root_for_window(int(spec["window"]))
        if temp_root.exists():
            shutil.rmtree(temp_root)


def run_generation(
    feature_files: Sequence[Path],
    window_specs: Sequence[dict],
    required: Sequence[str],
    n_workers: int,
) -> List[dict]:
    for spec in window_specs:
        temp_root_for_window(int(spec["window"])).mkdir(parents=True, exist_ok=True)

    tasks = [
        (part_id, str(path), list(window_specs), list(required))
        for part_id, path in enumerate(feature_files)
    ]
    n_workers = max(1, min(int(n_workers), len(tasks)))
    part_results: List[dict] = []

    if n_workers == 1:
        task_iter = progress_iter(tasks, total=len(tasks), desc="Image parts", unit="part")
        for task in task_iter:
            part_results.append(process_part_all_windows_task(task))
    else:
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            futures = {executor.submit(process_part_all_windows_task, task): task[0] for task in tasks}
            future_iter = progress_iter(
                as_completed(futures),
                total=len(futures),
                desc=f"Image parts ({n_workers} workers)",
                unit="part",
            )
            for future in future_iter:
                part_id = futures[future]
                try:
                    part_results.append(future.result())
                except Exception as exc:
                    cleanup_temp_roots(window_specs)
                    raise RuntimeError(f"Image part {part_id} failed: {exc}") from exc

    rows_by_window = {int(spec["window"]): 0 for spec in window_specs}
    local_shards_by_window = {int(spec["window"]): 0 for spec in window_specs}
    for result in part_results:
        for spec in window_specs:
            window = int(spec["window"])
            rows_by_window[window] += int(result["rows_by_window"].get(window, 0))
            local_shards_by_window[window] += int(result["shards_by_window"].get(window, 0))

    final_results: List[dict] = []
    for spec in window_specs:
        window = int(spec["window"])
        if rows_by_window[window] == 0:
            cleanup_temp_roots(window_specs)
            raise RuntimeError(f"No images generated for window={window}. Check filters and columns.")

        final_shards = move_temp_shards_to_final(window)
        result = {
            "window": window,
            "horizons": ",".join(str(h) for h in spec["horizons"]),
            "experiments": ",".join(spec["experiments"]),
            "rows": rows_by_window[window],
            "shards": final_shards,
            "local_temp_shards": local_shards_by_window[window],
        }
        final_results.append(result)
        print(
            f"window={window}: rows={rows_by_window[window]}, "
            f"final shards={final_shards}, horizons={result['horizons']}"
        )

    return final_results


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate shared sharded price images by window.")
    parser.add_argument(
        "--workers",
        type=int,
        default=default_worker_count(),
        help="Number of worker processes. Default: all CPUs visible to this process.",
    )
    parser.add_argument(
        "--experiments",
        default=None,
        help="Optional comma-separated experiments, e.g. I5R5,I20R5.",
    )
    parser.add_argument(
        "--limit-parts",
        type=int,
        default=None,
        help="Optional number of feature parts for smoke tests.",
    )
    args = parser.parse_args()

    experiment_names = parse_experiment_filter(args.experiments)
    window_specs = window_specs_from_experiments(experiment_names)
    required = needed_feature_columns(window_specs)

    feature_files = list_feature_bucket_files()
    if args.limit_parts is not None:
        feature_files = feature_files[: int(args.limit_parts)]
    if not feature_files:
        raise RuntimeError("No feature parts selected.")

    validate_feature_schema(feature_files[0], required)
    clear_output_dirs(window_specs)

    print(f"Feature code-bucket parts: {len(feature_files)} from {FEATURE_BY_CODE_BUCKET_DIR}")
    print(f"Experiments: {', '.join(experiment_names)}")
    print(
        "Shared image windows: "
        + ", ".join(
            f"{spec['window']}d(h={','.join(str(h) for h in spec['horizons'])})"
            for spec in window_specs
        )
    )
    print(f"Required columns: {len(required)}")
    print(f"Requested workers: {args.workers}")
    print("Mode: read each feature part once, generate each image window once, attach all selected horizons.")

    results = run_generation(
        feature_files=feature_files,
        window_specs=window_specs,
        required=required,
        n_workers=args.workers,
    )

    print("Done.")
    for result in sorted(results, key=lambda x: x["window"]):
        print(
            f"window={result['window']}: rows={result['rows']}, "
            f"shards={result['shards']}, horizons={result['horizons']}, "
            f"experiments={result['experiments']}"
        )


if __name__ == "__main__":
    main()
