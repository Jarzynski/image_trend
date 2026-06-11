# -*- coding: utf-8 -*-
"""
05_train_cnn2d.py

Official optimized training script for Jiang, Kelly, and Xiu-style 2D CNNs
on binary stock price images.

Target environment
------------------
- NVIDIA RTX 4080 / 4090 class GPU
- Around 75 GB system RAM
- SSD / NVMe storage strongly recommended

Main optimizations versus the baseline 05_train_cnn2d.py
--------------------------------------------------------
1. Uses np.load(..., mmap_mode="r") for image shards, same as baseline.
2. Keeps images as uint8 in Dataset/DataLoader and converts to float32/0-1
   on GPU per batch. This reduces CPU work and host-to-device traffic.
3. Enables CUDA AMP mixed precision by default.
4. Enables TF32 on Ampere/Ada GPUs for faster conv/matmul where applicable.
5. Uses configurable DataLoader workers, pin_memory, persistent_workers,
   prefetch_factor, and non_blocking CUDA transfer.
6. Loads each image window once and reuses the memmaps/metadata for all
   experiments sharing that window.
7. Compresses metadata dtypes to reduce RAM pressure.
8. Adds command-line controls for experiments, windows, epochs, batch size,
   workers, learning rate, early stopping, and validation metric interval.

Inputs
------
data/images/window_{window}/shard_*/images.npy
data/images/window_{window}/shard_*/meta.parquet

Outputs
-------
outputs/models/jiang_cnn2d_{experiment_name}.pt
outputs/models/jiang_cnn2d_{experiment_name}_run{run_id}.pt
outputs/predictions/ensemble_runs/{experiment_name}/run{run_id}.parquet
outputs/predictions/pred_{experiment_name}_jiang_cnn2d.parquet

Notes
-----
The default trains one model per experiment for backward-compatible runtime.
Use --ensemble-runs 5 to match the paper-style protocol of independently
training five models and arithmetically averaging their predicted
probabilities into the final signal.
"""

from __future__ import annotations

import argparse
import gc
import os
import time
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from sklearn.metrics import roc_auc_score, accuracy_score, brier_score_loss

from config import (
    PRED_DIR,
    MODEL_DIR,
    TABLE_DIR,
    TRAIN_END,
    VALID_START,
    VALID_END,
    TEST_START,
    EMBARGO_DAYS_BY_HORIZON,
    RANDOM_SEED,
    EXPERIMENTS,
    image_dir_for_window,
)


# ============================================================
# 0. Runtime configuration
# ============================================================

@dataclass
class TrainOptions:
    epochs: int = 50
    batch_size: int = 256
    lr: float = 1e-4
    num_workers: int = 4
    prefetch_factor: int = 2
    patience: int = 8
    min_epochs: int = 8
    min_delta: float = 1e-4
    amp: bool = True
    tf32: bool = True
    compile_model: bool = False
    valid_metric_interval: int = 1
    max_valid_batches: Optional[int] = None
    max_test_batches: Optional[int] = None
    channels_last: bool = False
    pin_memory: bool = True
    persistent_workers: bool = True
    drop_last_train: bool = False
    shard_cache_size: int = 32
    optimizer: str = "adamw"
    weight_decay: float = 3e-5
    scheduler: str = "cosine"
    warmup_epochs: int = 1
    fc_dropout: float = 0.20
    spatial_dropout: float = 0.0
    arch: str = "jiang"
    log_interval: int = 200
    profile_batches: int = 0
    ensemble_runs: int = 1
    ensemble_run_id: Optional[int] = None
    ensemble_aggregate_only: bool = False


def default_num_workers() -> int:
    """
    Conservative default for a 4080/4090 workstation/server.

    For image memmap workloads, too many DataLoader workers can overwhelm
    disk I/O. Four workers is usually a good first point on NVMe.
    """
    cpu_count = os.cpu_count() or 4
    return max(2, min(4, cpu_count // 2))


def configure_torch(seed: int, use_tf32: bool = True) -> None:
    """
    Configure reproducibility and GPU math modes.

    The original script used cudnn.benchmark=False. For fixed-size images,
    benchmark=True often improves CNN speed. This is not bitwise deterministic,
    but is appropriate for throughput-oriented research runs.
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False

    if use_tf32 and torch.cuda.is_available():
        # Useful on Ampere/Ada GPUs, including RTX 4080/4090.
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def cuda_sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()


def cache_stats(dataset: Dataset) -> dict:
    if not isinstance(dataset, ImageDataset):
        return {}
    stats = dataset.cache_stats()
    if stats["hits"] + stats["misses"] == 0:
        stats["hit_rate"] = np.nan
    else:
        stats["hit_rate"] = stats["hits"] / (stats["hits"] + stats["misses"])
    return stats


# ============================================================
# 1. Dataset
# ============================================================

def close_memmap(array: np.ndarray) -> None:
    """
    Close the mmap handle behind a numpy memmap when possible.
    """
    mmap_obj = getattr(array, "_mmap", None)
    if mmap_obj is not None:
        mmap_obj.close()


class ImageDataset(Dataset):
    """
    Dataset backed by memory-mapped uint8 image shards.

    Important performance choice:
    - __getitem__ returns uint8 CHW tensors.
    - float conversion and division by 255 happen once per batch on GPU.

    This avoids per-sample float32 copies on CPU and reduces pinned-memory
    transfer volume by roughly 4x versus returning float32 images.
    """

    def __init__(
        self,
        image_paths: Sequence[Path],
        labels: np.ndarray,
        shard_ids: np.ndarray,
        local_indices: np.ndarray,
        indices: np.ndarray,
        shard_cache_size: int = 32,
    ) -> None:
        self.image_paths = [Path(p) for p in image_paths]
        self.labels = np.asarray(labels, dtype=np.float32)
        self.shard_ids = np.asarray(shard_ids, dtype=np.int32)
        self.local_indices = np.asarray(local_indices, dtype=np.int32)
        self.indices = np.asarray(indices, dtype=np.int64)
        self.shard_cache_size = max(1, int(shard_cache_size))
        self._shard_cache: OrderedDict[int, np.ndarray] = OrderedDict()
        self._cache_hits = 0
        self._cache_misses = 0
        self._np_load_count = 0

    def __len__(self) -> int:
        return int(len(self.indices))

    def _get_shard(self, shard_id: int) -> np.ndarray:
        """
        Lazily open shard memmaps inside each DataLoader worker.

        Opening all 2,500+ shards at once can exceed Linux ulimit -n. A small
        LRU cache keeps file descriptors bounded while preserving mmap reads.
        """
        shard_id = int(shard_id)
        if shard_id in self._shard_cache:
            self._shard_cache.move_to_end(shard_id)
            self._cache_hits += 1
            return self._shard_cache[shard_id]

        self._cache_misses += 1
        self._np_load_count += 1
        shard = np.load(self.image_paths[shard_id], mmap_mode="r")
        self._shard_cache[shard_id] = shard

        while len(self._shard_cache) > self.shard_cache_size:
            _, old_shard = self._shard_cache.popitem(last=False)
            close_memmap(old_shard)

        return shard

    def __getitem__(self, idx: int):
        real_idx = int(self.indices[idx])
        shard_id = int(self.shard_ids[real_idx])
        local_idx = int(self.local_indices[real_idx])

        # Stored shape: H, W, C. PyTorch expects C, H, W.
        # The source is a read-only memmap. Make an explicit writable uint8 copy
        # before torch.from_numpy; passing a non-writable array can crash worker
        # processes in some PyTorch/Numpy builds.
        arr = self._get_shard(shard_id)[local_idx]
        arr = np.array(np.transpose(arr, (2, 0, 1)), dtype=np.uint8, order="C", copy=True)
        x = torch.from_numpy(arr)  # uint8 tensor
        y = torch.tensor(self.labels[real_idx], dtype=torch.float32)
        return x, y

    def cache_stats(self) -> dict:
        return {
            "hits": int(self._cache_hits),
            "misses": int(self._cache_misses),
            "np_load_count": int(self._np_load_count),
            "open_cached_shards": int(len(self._shard_cache)),
        }


# ============================================================
# 2. Model
# ============================================================

class JiangCNNBlock(nn.Module):
    """
    One Jiang et al. CNN building block:
    Conv -> BatchNorm -> LeakyReLU -> MaxPool.
    """

    def __init__(self, in_channels: int, out_channels: int, stride=(1, 1), dilation=(1, 1)):
        super().__init__()
        kernel_size = (5, 3)
        padding = (
            ((kernel_size[0] - 1) * dilation[0]) // 2,
            ((kernel_size[1] - 1) * dilation[1]) // 2,
        )

        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                dilation=dilation,
                padding=padding,
            ),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(negative_slope=0.01),
            nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1), ceil_mode=True),
        )

    def forward(self, x):
        return self.block(x)


class ResLiteCNNBlock(nn.Module):
    """
    Lightweight residual variant for ablation.

    It keeps the Jiang block's pooling schedule but adds a two-convolution
    residual path before pooling.
    """

    def __init__(self, in_channels: int, out_channels: int, stride=(1, 1), dilation=(1, 1)):
        super().__init__()
        kernel_size = (5, 3)
        padding = (
            ((kernel_size[0] - 1) * dilation[0]) // 2,
            ((kernel_size[1] - 1) * dilation[1]) // 2,
        )
        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            dilation=dilation,
            padding=padding,
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.act = nn.LeakyReLU(negative_slope=0.01)
        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=(1, 1),
            padding=(kernel_size[0] // 2, kernel_size[1] // 2),
        )
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.skip = (
            nn.Identity()
            if in_channels == out_channels and stride == (1, 1)
            else nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels),
            )
        )
        self.pool = nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1), ceil_mode=True)

    def forward(self, x):
        residual = self.skip(x)
        out = self.act(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.act(out + residual)
        return self.pool(out)


class JiangCNN2D(nn.Module):
    """
    Window-dependent CNN architecture from Jiang, Kelly, and Xiu (2023).

    The 5/20/60-day image models use 2/3/4 convolutional blocks,
    first-layer vertical stride/dilation of 1/1, 3/2, and 3/3,
    channels 64, 128, 256, 512, and a dropout-regularized final FC layer.
    """

    WINDOW_CONFIG = {
        5: {"num_blocks": 2, "first_stride_v": 1, "first_dilation_v": 1},
        20: {"num_blocks": 3, "first_stride_v": 3, "first_dilation_v": 2},
        60: {"num_blocks": 4, "first_stride_v": 3, "first_dilation_v": 3},
    }

    def __init__(
        self,
        window: int,
        image_height: int,
        image_width: int,
        in_channels: int = 1,
        fc_dropout: float = 0.20,
        spatial_dropout: float = 0.0,
        arch: str = "jiang",
    ):
        super().__init__()
        if int(window) not in self.WINDOW_CONFIG:
            raise ValueError(f"Unsupported CNN image window: {window}")
        if arch not in {"jiang", "reslite"}:
            raise ValueError(f"Unsupported CNN arch: {arch}")

        cfg = self.WINDOW_CONFIG[int(window)]
        blocks: List[nn.Module] = []
        channels = [64 * (2 ** i) for i in range(cfg["num_blocks"])]
        block_cls = JiangCNNBlock if arch == "jiang" else ResLiteCNNBlock

        prev_channels = in_channels
        for i, out_channels in enumerate(channels):
            if i == 0:
                stride = (cfg["first_stride_v"], 1)
                dilation = (cfg["first_dilation_v"], 1)
            else:
                stride = (1, 1)
                dilation = (1, 1)

            blocks.append(block_cls(prev_channels, out_channels, stride=stride, dilation=dilation))
            prev_channels = out_channels

        self.features = nn.Sequential(*blocks)
        self.spatial_dropout = (
            nn.Dropout2d(float(spatial_dropout))
            if float(spatial_dropout) > 0
            else nn.Identity()
        )

        with torch.no_grad():
            dummy = torch.zeros(1, in_channels, image_height, image_width)
            feature_dim = self.spatial_dropout(self.features(dummy)).flatten(1).shape[1]

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(float(fc_dropout)),
            nn.Linear(feature_dim, 1),
        )

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, x):
        x = self.features(x)
        x = self.spatial_dropout(x)
        return self.classifier(x).squeeze(-1)


# ============================================================
# 3. Metadata and split utilities
# ============================================================

def parse_csv_arg(value: Optional[str]) -> Optional[List[str]]:
    if value is None or str(value).strip() == "":
        return None
    return [x.strip().upper() for x in value.split(",") if x.strip()]


def selected_experiments(experiments_arg: Optional[str], windows_arg: Optional[str]) -> Dict[str, dict]:
    experiments = dict(EXPERIMENTS)

    requested = parse_csv_arg(experiments_arg)
    if requested is not None:
        unknown = [name for name in requested if name not in experiments]
        if unknown:
            raise RuntimeError(f"Unknown experiments: {unknown}. Available: {sorted(experiments)}")
        experiments = {name: experiments[name] for name in requested}

    if windows_arg:
        requested_windows = {int(x.strip()) for x in windows_arg.split(",") if x.strip()}
        experiments = {
            name: cfg for name, cfg in experiments.items()
            if int(cfg["window"]) in requested_windows
        }
        if not experiments:
            raise RuntimeError(f"No experiments selected for windows={sorted(requested_windows)}")

    return experiments


def group_experiments_by_window(experiments: Dict[str, dict]) -> Dict[int, Dict[str, dict]]:
    grouped: Dict[int, Dict[str, dict]] = {}
    for exp_name, cfg in experiments.items():
        grouped.setdefault(int(cfg["window"]), {})[exp_name] = cfg
    return grouped


def needed_meta_columns_for_window(window_experiments: Dict[str, dict]) -> Optional[List[str]]:
    """
    Read only useful columns from meta.parquet when possible.

    If a column is missing from a shard, pandas will raise; in that case the
    caller falls back to reading the full parquet for robustness.
    """
    cols = {
        "date", "code", "industry", "shard_id", "local_index",
        "amount", "float_mktcap",
        "is_low_volume_limit_up", "is_low_volume_limit_down",
    }
    for cfg in window_experiments.values():
        horizon = int(cfg["horizon"])
        cols.add(f"label_{horizon}d")
        cols.add(f"future_ret_{horizon}d")
    return sorted(cols)


def compress_meta_dtypes(meta: pd.DataFrame) -> pd.DataFrame:
    """
    Reduce pandas RAM footprint for large all-A-share metadata tables.
    """
    if "date" in meta.columns:
        meta["date"] = pd.to_datetime(meta["date"])

    for col in ["code", "industry", "experiment_name"]:
        if col in meta.columns:
            # category is much smaller than object for repeated code/industry.
            meta[col] = meta[col].astype("category")

    for col in ["shard_id", "local_index", "window", "horizon"]:
        if col in meta.columns:
            meta[col] = pd.to_numeric(meta[col], errors="coerce").fillna(0).astype(np.int32)

    for col in meta.columns:
        if col.startswith("label_") or col.startswith("future_ret_"):
            meta[col] = pd.to_numeric(meta[col], errors="coerce").astype(np.float32)

    for col in ["amount", "float_mktcap"]:
        if col in meta.columns:
            meta[col] = pd.to_numeric(meta[col], errors="coerce").astype(np.float32)

    for col in ["is_low_volume_limit_up", "is_low_volume_limit_down"]:
        if col in meta.columns:
            meta[col] = pd.to_numeric(meta[col], errors="coerce").fillna(0).astype(np.int8)

    return meta


def load_image_window(window: int, window_experiments: Dict[str, dict]):
    """
    Load image shard paths and metadata once for all experiments sharing a window.

    Image memmaps are opened lazily by ImageDataset workers. This avoids
    opening thousands of .npy files in the main process and hitting ulimit -n.
    """
    image_dir = image_dir_for_window(window)
    shard_dirs = sorted(image_dir.glob("shard_*"))
    if not shard_dirs:
        raise RuntimeError(f"No image shards found for window={window}: {image_dir}")

    image_paths: List[Path] = []
    meta_frames: List[pd.DataFrame] = []
    wanted_cols = needed_meta_columns_for_window(window_experiments)
    image_shape: Optional[Tuple[int, int, int, int]] = None

    print(f"Loading window={window} shards from: {image_dir}")
    print(f"Shard count: {len(shard_dirs)}")

    for expected_shard_id, shard_dir in enumerate(shard_dirs):
        image_path = shard_dir / "images.npy"
        meta_path = shard_dir / "meta.parquet"
        if not image_path.exists() or not meta_path.exists():
            raise RuntimeError(f"Incomplete image shard: {shard_dir}")

        images = np.load(image_path, mmap_mode="r")
        current_shape = tuple(int(x) for x in images.shape)
        if image_shape is None:
            image_shape = current_shape
        elif current_shape[1:] != image_shape[1:]:
            close_memmap(images)
            raise RuntimeError(
                f"Image shape mismatch in {image_path}: "
                f"shape={current_shape}, expected trailing shape={image_shape[1:]}"
            )
        n_images = int(current_shape[0])
        close_memmap(images)

        try:
            meta = pd.read_parquet(meta_path, columns=wanted_cols)
        except Exception:
            # Older shards may not have the exact selected columns.
            meta = pd.read_parquet(meta_path)

        if "shard_id" not in meta.columns:
            meta["shard_id"] = expected_shard_id
        if "local_index" not in meta.columns:
            meta["local_index"] = np.arange(len(meta), dtype=np.int32)

        if len(meta) != n_images:
            raise RuntimeError(
                f"Shard row mismatch in {shard_dir}: images={n_images}, meta={len(meta)}"
            )

        # Force current in-memory shard numbering to match image_paths list.
        meta["shard_id"] = np.int32(expected_shard_id)
        meta["local_index"] = np.asarray(meta["local_index"], dtype=np.int32)

        image_paths.append(image_path)
        meta_frames.append(meta)

    meta = pd.concat(meta_frames, ignore_index=True)
    meta = compress_meta_dtypes(meta)

    if image_shape is None:
        raise RuntimeError(f"No readable image shards found for window={window}: {image_dir}")

    image_height, image_width, channels = int(image_shape[1]), int(image_shape[2]), int(image_shape[3])
    print(
        f"Loaded window={window}: rows={len(meta):,}, "
        f"image_shape=({image_height}, {image_width}, {channels})"
    )
    return image_paths, meta, image_shape


def select_experiment_label_view(meta: pd.DataFrame, exp_name: str, cfg: dict) -> pd.DataFrame:
    horizon = int(cfg["horizon"])
    label_col = f"label_{horizon}d"
    future_ret_col = f"future_ret_{horizon}d"
    missing = [c for c in [label_col, future_ret_col] if c not in meta.columns]
    if missing:
        raise RuntimeError(
            f"Image metadata for {exp_name} is missing columns: {missing}. "
            "Rerun image generation with this experiment/horizon selected."
        )

    valid = meta[label_col].notna().to_numpy() & meta[future_ret_col].notna().to_numpy()
    view = meta.loc[valid].copy().reset_index(drop=True)
    if view.empty:
        raise RuntimeError(f"{exp_name} has no rows with finite {label_col}/{future_ret_col}.")

    view["experiment_name"] = exp_name
    view["horizon"] = np.int32(horizon)
    view["label"] = view[label_col].astype(np.float32)
    view["future_ret"] = view[future_ret_col].astype(np.float32)
    view = compress_meta_dtypes(view)
    return view


def cutoff_before_boundary(dates, boundary, gap_days: int) -> pd.Timestamp:
    unique_dates = pd.DatetimeIndex(sorted(pd.to_datetime(dates).dropna().unique()))
    boundary = pd.Timestamp(boundary)
    before = unique_dates[unique_dates < boundary]
    if gap_days <= 0:
        return boundary
    if len(before) <= gap_days:
        return pd.Timestamp.min
    return before[-gap_days]


def get_split_indices(meta: pd.DataFrame, horizon: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    date = pd.to_datetime(meta["date"])
    gap_days = max(int(horizon), int(EMBARGO_DAYS_BY_HORIZON.get(int(horizon), horizon)))
    train_purge_start = cutoff_before_boundary(date, VALID_START, gap_days)
    valid_purge_start = cutoff_before_boundary(date, TEST_START, gap_days)

    train_mask = (date <= pd.Timestamp(TRAIN_END)) & (date < train_purge_start)
    valid_mask = (
        (date >= pd.Timestamp(VALID_START))
        & (date <= pd.Timestamp(VALID_END))
        & (date < valid_purge_start)
    )
    test_mask = date >= pd.Timestamp(TEST_START)

    return (
        np.flatnonzero(train_mask.to_numpy()),
        np.flatnonzero(valid_mask.to_numpy()),
        np.flatnonzero(test_mask.to_numpy()),
    )


def safe_corr(a: pd.Series, b: pd.Series) -> float:
    if len(a) < 2 or a.nunique(dropna=True) < 2 or b.nunique(dropna=True) < 2:
        return np.nan
    return float(a.corr(b))


def signal_metrics(meta: pd.DataFrame, indices: np.ndarray, probs: Optional[np.ndarray]) -> dict:
    if probs is None or len(indices) == 0:
        return {
            "rankic_mean": np.nan,
            "rankic_positive_rate": np.nan,
            "ic_mean": np.nan,
            "decile_spearman": np.nan,
            "decile_violations": np.nan,
        }

    frame = meta.iloc[indices][["date", "future_ret"]].copy()
    frame["pred_prob"] = np.asarray(probs, dtype=np.float32)
    frame = frame.dropna(subset=["date", "future_ret", "pred_prob"])
    if frame.empty:
        return {
            "rankic_mean": np.nan,
            "rankic_positive_rate": np.nan,
            "ic_mean": np.nan,
            "decile_spearman": np.nan,
            "decile_violations": np.nan,
        }

    period_rows = []
    for _, group in frame.groupby("date", sort=False):
        valid = group[["pred_prob", "future_ret"]].dropna()
        if len(valid) < 10:
            continue
        ic = safe_corr(valid["pred_prob"], valid["future_ret"])
        rankic = safe_corr(
            valid["pred_prob"].rank(method="average"),
            valid["future_ret"].rank(method="average"),
        )
        period_rows.append((ic, rankic))

    if period_rows:
        ic_arr = np.asarray([x[0] for x in period_rows], dtype=np.float64)
        rankic_arr = np.asarray([x[1] for x in period_rows], dtype=np.float64)
        ic_mean = float(np.nanmean(ic_arr))
        rankic_mean = float(np.nanmean(rankic_arr))
        rankic_positive_rate = float(np.nanmean(rankic_arr > 0))
    else:
        ic_mean = rankic_mean = rankic_positive_rate = np.nan

    rank = frame.groupby("date")["pred_prob"].rank(method="first", pct=True)
    frame["decile"] = np.ceil(rank * 10).clip(1, 10).astype(np.int8)
    decile_ret = frame.groupby("decile")["future_ret"].mean()
    decile_ret = decile_ret.reindex(range(1, 11))
    if decile_ret.notna().sum() >= 3:
        decile_spearman = safe_corr(
            pd.Series(decile_ret.index.astype(float), index=decile_ret.index),
            decile_ret.astype(float),
        )
        diffs = decile_ret.diff().iloc[1:]
        decile_violations = int((diffs < 0).sum())
    else:
        decile_spearman = np.nan
        decile_violations = np.nan

    return {
        "rankic_mean": rankic_mean,
        "rankic_positive_rate": rankic_positive_rate,
        "ic_mean": ic_mean,
        "decile_spearman": decile_spearman,
        "decile_violations": decile_violations,
    }


def probability_summary(probs: Optional[np.ndarray], labels: Optional[np.ndarray]) -> dict:
    if probs is None or labels is None or len(probs) == 0:
        return {
            "prob_mean": np.nan,
            "prob_std": np.nan,
            "prob_p01": np.nan,
            "prob_p50": np.nan,
            "prob_p99": np.nan,
            "label_positive_rate": np.nan,
        }
    probs = np.asarray(probs, dtype=np.float32)
    labels = np.asarray(labels, dtype=np.float32)
    return {
        "prob_mean": float(np.mean(probs)),
        "prob_std": float(np.std(probs)),
        "prob_p01": float(np.quantile(probs, 0.01)),
        "prob_p50": float(np.quantile(probs, 0.50)),
        "prob_p99": float(np.quantile(probs, 0.99)),
        "label_positive_rate": float(np.mean(labels)),
    }


def build_optimizer(model: nn.Module, options: TrainOptions) -> torch.optim.Optimizer:
    if options.optimizer == "adam":
        return torch.optim.Adam(
            model.parameters(),
            lr=float(options.lr),
            weight_decay=float(options.weight_decay),
        )
    if options.optimizer == "adamw":
        return torch.optim.AdamW(
            model.parameters(),
            lr=float(options.lr),
            weight_decay=float(options.weight_decay),
        )
    raise ValueError(f"Unsupported optimizer: {options.optimizer}")


def set_warmup_lr(optimizer: torch.optim.Optimizer, base_lr: float, epoch: int, warmup_epochs: int) -> None:
    if warmup_epochs <= 0:
        return
    if epoch <= warmup_epochs:
        lr = float(base_lr) * float(epoch) / float(warmup_epochs)
        for group in optimizer.param_groups:
            group["lr"] = lr


def build_scheduler(optimizer: torch.optim.Optimizer, options: TrainOptions):
    if options.scheduler == "none":
        return None
    if options.scheduler == "cosine":
        t_max = max(1, int(options.epochs) - max(0, int(options.warmup_epochs)))
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=t_max)
    if options.scheduler == "plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="max",
            factor=0.5,
            patience=2,
            min_lr=float(options.lr) * 0.01,
        )
    raise ValueError(f"Unsupported scheduler: {options.scheduler}")


def checkpoint_score(valid_metrics: dict, valid_auc: float, valid_loss: float) -> Tuple[float, str]:
    rankic = valid_metrics.get("rankic_mean", np.nan)
    if np.isfinite(rankic):
        return float(rankic), "valid_rankic_mean"
    if np.isfinite(valid_auc):
        return float(valid_auc), "valid_auc"
    return -float(valid_loss), "negative_valid_loss"


def model_name_for_arch(arch: str) -> str:
    return "JiangCNN2D" if arch == "jiang" else f"JiangCNN2D_{arch}"


def output_stem(exp_name: str, arch: str) -> str:
    base = exp_name.lower()
    return base if arch == "jiang" else f"{base}_{arch}"


def ensemble_run_dir(exp_name: str, arch: str) -> Path:
    return PRED_DIR / "ensemble_runs" / output_stem(exp_name, arch)


def ensemble_run_prediction_path(exp_name: str, arch: str, run_id: int) -> Path:
    return ensemble_run_dir(exp_name, arch) / f"run{int(run_id):02d}.parquet"


def final_prediction_path(exp_name: str, arch: str) -> Path:
    return PRED_DIR / f"pred_{output_stem(exp_name, arch)}_jiang_cnn2d.parquet"


# ============================================================
# 4. Training and evaluation
# ============================================================

def make_loader(
    dataset: Dataset,
    batch_size: int,
    shuffle: bool,
    options: TrainOptions,
    generator: Optional[torch.Generator] = None,
    drop_last: bool = False,
) -> DataLoader:
    kwargs = dict(
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=max(0, int(options.num_workers)),
        pin_memory=bool(options.pin_memory and torch.cuda.is_available()),
        generator=generator,
        drop_last=drop_last,
    )
    if int(options.num_workers) > 0:
        kwargs["persistent_workers"] = bool(options.persistent_workers)
        kwargs["prefetch_factor"] = int(options.prefetch_factor)
    return DataLoader(dataset, **kwargs)


def prepare_batch(x: torch.Tensor, y: torch.Tensor, device: torch.device, channels_last: bool = False):
    """
    Move batch to GPU and convert binary uint8 images to float32 [0, 1].
    """
    x = x.to(device, non_blocking=True)
    y = y.to(device, non_blocking=True)

    # Convert on GPU instead of per sample on CPU.
    x = x.float().div_(255.0)
    if channels_last:
        x = x.contiguous(memory_format=torch.channels_last)
    return x, y


def evaluate_model(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    criterion: nn.Module,
    options: TrainOptions,
    max_batches: Optional[int] = None,
    return_probs: bool = True,
    desc: str = "Eval",
):
    model.eval()
    probs: List[np.ndarray] = []
    labels: List[np.ndarray] = []
    total_loss = 0.0
    n_obs = 0

    use_amp = bool(options.amp and device.type == "cuda")
    eval_start = time.perf_counter()

    with torch.inference_mode():
        total_batches = len(loader)
        for batch_i, (x, y) in enumerate(loader):
            if max_batches is not None and batch_i >= int(max_batches):
                break
            x, y = prepare_batch(x, y, device, channels_last=options.channels_last)
            with torch.amp.autocast(device_type="cuda", enabled=use_amp):
                logit = model(x)
                loss = criterion(logit, y)
            prob = torch.sigmoid(logit).detach().float().cpu().numpy().astype(np.float32)
            y_cpu = y.detach().float().cpu().numpy().astype(np.float32)

            if return_probs:
                probs.append(prob)
                labels.append(y_cpu)
            total_loss += float(loss.item()) * len(y_cpu)
            n_obs += len(y_cpu)

            if options.log_interval > 0 and (batch_i + 1) % int(options.log_interval) == 0:
                elapsed = time.perf_counter() - eval_start
                print(
                    f"{desc} batch {batch_i + 1}/{total_batches} | "
                    f"samples/s={n_obs / max(elapsed, 1e-9):.1f}"
                )

    cuda_sync(device)
    eval_seconds = time.perf_counter() - eval_start
    avg_loss = total_loss / max(n_obs, 1)

    if not return_probs:
        return {
            "probs": None,
            "labels": None,
            "auc": np.nan,
            "acc": np.nan,
            "brier": np.nan,
            "loss": avg_loss,
            "seconds": eval_seconds,
            "samples_per_sec": n_obs / max(eval_seconds, 1e-9),
            "metric_seconds": 0.0,
        }

    if n_obs == 0:
        return {
            "probs": np.array([], dtype=np.float32),
            "labels": np.array([], dtype=np.float32),
            "auc": np.nan,
            "acc": np.nan,
            "brier": np.nan,
            "loss": np.nan,
            "seconds": eval_seconds,
            "samples_per_sec": 0.0,
            "metric_seconds": 0.0,
        }

    probs_arr = np.concatenate(probs).astype(np.float32, copy=False)
    labels_arr = np.concatenate(labels).astype(np.float32, copy=False)

    metric_start = time.perf_counter()
    if len(np.unique(labels_arr)) < 2:
        auc = np.nan
    else:
        auc = roc_auc_score(labels_arr, probs_arr)
    acc = accuracy_score(labels_arr, probs_arr > 0.5)
    brier = brier_score_loss(labels_arr, probs_arr)
    metric_seconds = time.perf_counter() - metric_start
    return {
        "probs": probs_arr,
        "labels": labels_arr,
        "auc": auc,
        "acc": acc,
        "brier": brier,
        "loss": avg_loss,
        "seconds": eval_seconds,
        "samples_per_sec": n_obs / max(eval_seconds, 1e-9),
        "metric_seconds": metric_seconds,
    }


def fit_one_experiment_run(
    exp_name: str,
    cfg: dict,
    image_paths: Sequence[Path],
    image_shape: Tuple[int, int, int, int],
    meta_window: pd.DataFrame,
    options: TrainOptions,
    run_id: int,
    run_count: int,
    seed: int,
):
    """
    Train one independently initialized model run and return test predictions.
    """
    configure_torch(seed, use_tf32=options.tf32)

    meta = select_experiment_label_view(meta_window, exp_name, cfg)
    image_height, image_width = int(image_shape[1]), int(image_shape[2])

    labels = meta["label"].to_numpy(dtype=np.float32, copy=False)
    shard_ids = meta["shard_id"].to_numpy(dtype=np.int32, copy=False)
    local_indices = meta["local_index"].to_numpy(dtype=np.int32, copy=False)

    train_idx, valid_idx, test_idx = get_split_indices(meta, int(cfg["horizon"]))
    print(f"{exp_name} train/valid/test sizes: {len(train_idx):,}/{len(valid_idx):,}/{len(test_idx):,}")

    if len(train_idx) == 0 or len(valid_idx) == 0 or len(test_idx) == 0:
        raise RuntimeError(
            f"{exp_name} has an empty split: "
            f"train={len(train_idx)}, valid={len(valid_idx)}, test={len(test_idx)}"
        )

    device = get_device()
    train_generator = torch.Generator()
    train_generator.manual_seed(seed)

    train_ds = ImageDataset(
        image_paths,
        labels,
        shard_ids,
        local_indices,
        train_idx,
        shard_cache_size=options.shard_cache_size,
    )
    valid_ds = ImageDataset(
        image_paths,
        labels,
        shard_ids,
        local_indices,
        valid_idx,
        shard_cache_size=options.shard_cache_size,
    )
    test_ds = ImageDataset(
        image_paths,
        labels,
        shard_ids,
        local_indices,
        test_idx,
        shard_cache_size=options.shard_cache_size,
    )

    train_loader = make_loader(
        train_ds,
        batch_size=options.batch_size,
        shuffle=True,
        options=options,
        generator=train_generator,
        drop_last=options.drop_last_train,
    )
    valid_loader = make_loader(valid_ds, options.batch_size, False, options)
    test_loader = make_loader(test_ds, options.batch_size, False, options)

    model = JiangCNN2D(
        window=int(cfg["window"]),
        image_height=image_height,
        image_width=image_width,
        fc_dropout=float(options.fc_dropout),
        spatial_dropout=float(options.spatial_dropout),
        arch=options.arch,
    ).to(device)

    if options.channels_last and device.type == "cuda":
        model = model.to(memory_format=torch.channels_last)

    if options.compile_model and hasattr(torch, "compile"):
        print("Compiling model with torch.compile...")
        model = torch.compile(model)

    optimizer = build_optimizer(model, options)
    scheduler = build_scheduler(optimizer, options)
    criterion = nn.BCEWithLogitsLoss()
    scaler = torch.amp.GradScaler("cuda", enabled=bool(options.amp and device.type == "cuda"))

    best_score = -np.inf
    best_state = None
    best_metric_name = None
    bad_epochs = 0
    epoch_logs: List[dict] = []
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    stem = output_stem(exp_name, options.arch)
    log_suffix = f"_run{run_id:02d}" if run_count > 1 else ""
    log_path = TABLE_DIR / f"cnn_training_log_{stem}{log_suffix}.csv"

    print(
        f"Training {exp_name} run {run_id}/{run_count} on {device} | "
        f"batch={options.batch_size}, workers={options.num_workers}, "
        f"amp={options.amp and device.type == 'cuda'}, tf32={options.tf32}, "
        f"optimizer={options.optimizer}, lr={options.lr}, wd={options.weight_decay}, "
        f"arch={options.arch}, fc_dropout={options.fc_dropout}, "
        f"spatial_dropout={options.spatial_dropout}, seed={seed}"
    )

    use_amp = bool(options.amp and device.type == "cuda")

    for epoch in range(1, int(options.epochs) + 1):
        epoch_start = time.perf_counter()
        set_warmup_lr(optimizer, float(options.lr), epoch, int(options.warmup_epochs))
        model.train()
        total_loss = 0.0
        n_obs = 0
        data_wait_seconds = 0.0
        h2d_seconds = 0.0
        gpu_step_seconds = 0.0
        profiled_batches = 0
        train_start = time.perf_counter()
        last_batch_end = train_start

        for batch_i, (x, y) in enumerate(train_loader, start=1):
            batch_ready = time.perf_counter()
            data_wait_seconds += batch_ready - last_batch_end

            profile_limit = int(options.profile_batches)
            profile_this_batch = profile_limit > 0 and profiled_batches < profile_limit
            if profile_this_batch:
                cuda_sync(device)
            h2d_start = time.perf_counter()
            x, y = prepare_batch(x, y, device, channels_last=options.channels_last)
            if profile_this_batch:
                cuda_sync(device)
            h2d_seconds += time.perf_counter() - h2d_start

            optimizer.zero_grad(set_to_none=True)
            if profile_this_batch:
                cuda_sync(device)
            gpu_start = time.perf_counter()
            with torch.amp.autocast(device_type="cuda", enabled=use_amp):
                logit = model(x)
                loss = criterion(logit, y)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            if profile_this_batch:
                cuda_sync(device)
                profiled_batches += 1
            gpu_step_seconds += time.perf_counter() - gpu_start

            total_loss += float(loss.item()) * len(y)
            n_obs += len(y)
            last_batch_end = time.perf_counter()

            if options.log_interval > 0 and batch_i % int(options.log_interval) == 0:
                elapsed = last_batch_end - train_start
                print(
                    f"{exp_name} epoch {epoch:03d} batch {batch_i}/{len(train_loader)} | "
                    f"loss={total_loss / max(n_obs, 1):.5f} | "
                    f"samples/s={n_obs / max(elapsed, 1e-9):.1f} | "
                    f"data_wait_ms={data_wait_seconds / max(batch_i, 1) * 1000:.1f} | "
                    f"h2d_ms={h2d_seconds / max(batch_i, 1) * 1000:.1f} | "
                    f"gpu_step_ms={gpu_step_seconds / max(batch_i, 1) * 1000:.1f}"
                )

        cuda_sync(device)
        train_seconds = time.perf_counter() - train_start
        train_loss = total_loss / max(n_obs, 1)

        # For early stopping, always compute validation loss. Metrics can be
        # computed less frequently to save time on huge validation sets.
        do_full_metric = (epoch % max(1, int(options.valid_metric_interval)) == 0)
        valid_eval = evaluate_model(
            model,
            valid_loader,
            device,
            criterion,
            options,
            max_batches=options.max_valid_batches,
            return_probs=do_full_metric,
            desc=f"{exp_name} epoch {epoch:03d} valid",
        )
        valid_prob = valid_eval["probs"]
        valid_y = valid_eval["labels"]
        valid_auc = valid_eval["auc"]
        valid_acc = valid_eval["acc"]
        valid_brier = valid_eval["brier"]
        valid_loss = valid_eval["loss"]
        valid_signal = signal_metrics(meta, valid_idx, valid_prob) if do_full_metric else signal_metrics(meta, np.array([], dtype=np.int64), None)
        valid_prob_summary = probability_summary(valid_prob, valid_y)
        score, metric_name = checkpoint_score(valid_signal, valid_auc, valid_loss)

        if do_full_metric:
            metric_msg = (
                f"valid loss={valid_loss:.5f} | "
                f"valid AUC={valid_auc:.4f} | valid ACC={valid_acc:.4f} | "
                f"valid Brier={valid_brier:.4f} | "
                f"valid RankIC={valid_signal['rankic_mean']:.5f}"
            )
        else:
            metric_msg = f"valid loss={valid_loss:.5f} | metrics skipped"

        epoch_seconds = time.perf_counter() - epoch_start
        avg_data_wait_ms = data_wait_seconds / max(len(train_loader), 1) * 1000
        avg_h2d_ms = h2d_seconds / max(len(train_loader), 1) * 1000
        avg_gpu_step_ms = gpu_step_seconds / max(len(train_loader), 1) * 1000
        train_samples_per_sec = n_obs / max(train_seconds, 1e-9)
        cache_info = cache_stats(train_ds) if int(options.num_workers) == 0 else {}
        log_row = {
            "experiment_name": exp_name,
            "model_name": model_name_for_arch(options.arch),
            "arch": options.arch,
            "run_id": run_id,
            "run_count": run_count,
            "seed": seed,
            "epoch": epoch,
            "lr": optimizer.param_groups[0]["lr"],
            "epoch_seconds": epoch_seconds,
            "train_seconds": train_seconds,
            "valid_seconds": valid_eval["seconds"],
            "train_samples_per_sec": train_samples_per_sec,
            "valid_samples_per_sec": valid_eval["samples_per_sec"],
            "avg_data_wait_ms": avg_data_wait_ms,
            "avg_h2d_ms": avg_h2d_ms,
            "avg_gpu_step_ms": avg_gpu_step_ms,
            "avg_loss": train_loss,
            "train_loss": train_loss,
            "valid_loss": valid_loss,
            "valid_auc": valid_auc,
            "valid_acc": valid_acc,
            "valid_brier": valid_brier,
            "valid_rankic_mean": valid_signal["rankic_mean"],
            "valid_rankic_positive_rate": valid_signal["rankic_positive_rate"],
            "valid_ic_mean": valid_signal["ic_mean"],
            "valid_decile_spearman": valid_signal["decile_spearman"],
            "valid_decile_violations": valid_signal["decile_violations"],
            "checkpoint_score": score,
            "checkpoint_metric": metric_name,
            **{f"valid_{k}": v for k, v in valid_prob_summary.items()},
            **{f"cache_{k}": v for k, v in cache_info.items()},
        }
        epoch_logs.append(log_row)
        pd.DataFrame(epoch_logs).to_csv(log_path, index=False)

        print(
            f"Epoch {epoch:03d} | loss={train_loss:.5f} | {metric_msg} | "
            f"score={score:.5f}({metric_name}) | "
            f"train_s={train_seconds:.1f} | valid_s={valid_eval['seconds']:.1f} | "
            f"samples/s={train_samples_per_sec:.1f} | "
            f"data_wait_ms={avg_data_wait_ms:.1f} | "
            f"h2d_ms={avg_h2d_ms:.1f} | gpu_step_ms={avg_gpu_step_ms:.1f}"
        )

        if score > best_score + float(options.min_delta):
            best_score = score
            best_metric_name = metric_name
            # Save on CPU to release GPU activation/optimizer pressure.
            raw_model = model._orig_mod if hasattr(model, "_orig_mod") else model
            best_state = {k: v.detach().cpu().clone() for k, v in raw_model.state_dict().items()}
            bad_epochs = 0
        else:
            if epoch >= int(options.min_epochs):
                bad_epochs += 1

        if options.scheduler == "cosine" and scheduler is not None and epoch > int(options.warmup_epochs):
            scheduler.step()
        elif options.scheduler == "plateau" and scheduler is not None:
            scheduler.step(score)

        if epoch >= int(options.min_epochs) and bad_epochs >= int(options.patience):
            print("Early stopping triggered.")
            break

    if best_state is None:
        raise RuntimeError(f"{exp_name} did not produce a valid checkpoint.")

    raw_model = model._orig_mod if hasattr(model, "_orig_mod") else model
    raw_model.load_state_dict(best_state)
    model = raw_model.to(device)
    if options.channels_last and device.type == "cuda":
        model = model.to(memory_format=torch.channels_last)

    print(f"Restored best checkpoint: score={best_score:.5f} metric={best_metric_name}")

    test_eval = evaluate_model(
        model,
        test_loader,
        device,
        criterion,
        options,
        max_batches=options.max_test_batches,
        return_probs=True,
        desc=f"{exp_name} test",
    )
    test_prob = test_eval["probs"]
    test_auc = test_eval["auc"]
    test_acc = test_eval["acc"]
    test_brier = test_eval["brier"]
    test_loss = test_eval["loss"]

    print(
        f"{exp_name} TEST | loss={test_loss:.5f} | "
        f"AUC={test_auc:.4f} | ACC={test_acc:.4f} | Brier={test_brier:.4f} | "
        f"seconds={test_eval['seconds']:.1f} | samples/s={test_eval['samples_per_sec']:.1f}"
    )

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    PRED_DIR.mkdir(parents=True, exist_ok=True)

    model_suffix = f"_run{run_id:02d}" if run_count > 1 else ""
    model_path = MODEL_DIR / f"jiang_cnn2d_{stem}{model_suffix}.pt"
    torch.save(model.state_dict(), model_path)
    print(f"Saved model to: {model_path}")

    pred_cols = [
        "date", "code", "industry",
        "future_ret", "label",
        "amount", "float_mktcap",
        "is_low_volume_limit_up", "is_low_volume_limit_down",
    ]
    pred_cols = [c for c in pred_cols if c in meta.columns]
    pred_idx = test_idx[: len(test_prob)]
    pred = meta.iloc[pred_idx][pred_cols].copy()

    # category columns can sometimes serialize awkwardly across environments;
    # convert key identifiers back to string in prediction output.
    for c in ["code", "industry"]:
        if c in pred.columns:
            pred[c] = pred[c].astype(str)

    pred["experiment_name"] = exp_name
    pred["window"] = int(cfg["window"])
    pred["horizon"] = int(cfg["horizon"])
    pred[f"pred_prob_run_{run_id:02d}"] = test_prob.astype(np.float32, copy=False)

    # Explicit cleanup between experiments helps when running many configs.
    del train_loader, valid_loader, test_loader, train_ds, valid_ds, test_ds
    del model, optimizer, scaler, criterion, meta, labels, shard_ids, local_indices
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    return {
        "pred": pred,
        "test_prob": test_prob.astype(np.float32, copy=False),
        "test_auc": test_auc,
        "test_acc": test_acc,
        "test_brier": test_brier,
        "test_loss": test_loss,
        "model_path": model_path,
        "seed": seed,
    }


def model_name_for_ensemble(arch: str, ensemble_runs: int) -> str:
    base = model_name_for_arch(arch)
    return base if int(ensemble_runs) == 1 else f"{base}_ens{int(ensemble_runs)}"


def validate_same_prediction_rows(first: pd.DataFrame, current: pd.DataFrame, run_id: int) -> None:
    key_cols = [c for c in ["date", "code", "future_ret", "label"] if c in first.columns and c in current.columns]
    if len(first) != len(current):
        raise RuntimeError(
            f"Ensemble run {run_id} produced a different prediction row count: "
            f"{len(current)} vs {len(first)}."
        )
    for col in key_cols:
        left = first[col].reset_index(drop=True)
        right = current[col].reset_index(drop=True)
        if not left.equals(right):
            raise RuntimeError(f"Ensemble run {run_id} prediction rows differ in column {col}.")


def save_run_prediction(exp_name: str, arch: str, run_id: int, pred: pd.DataFrame) -> Path:
    out_dir = ensemble_run_dir(exp_name, arch)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = ensemble_run_prediction_path(exp_name, arch, run_id)
    pred.to_parquet(out_path, index=False)
    print(f"Saved ensemble run prediction to: {out_path}")
    return out_path


def aggregate_ensemble_predictions(
    exp_name: str,
    cfg: dict,
    options: TrainOptions,
    run_frames: Optional[Sequence[pd.DataFrame]] = None,
) -> pd.DataFrame:
    """
    Average independently trained run probabilities into the final prediction file.
    """
    run_count = max(1, int(options.ensemble_runs))
    pred_base = None
    prob_rows = []

    if run_frames is None:
        run_frames = []
        for run_id in range(1, run_count + 1):
            run_path = ensemble_run_prediction_path(exp_name, options.arch, run_id)
            if not run_path.exists():
                raise RuntimeError(
                    f"Missing ensemble run prediction for {exp_name} run {run_id}: {run_path}"
                )
            run_frames.append(pd.read_parquet(run_path))

    if len(run_frames) != run_count:
        raise RuntimeError(
            f"{exp_name} expected {run_count} ensemble runs, got {len(run_frames)}."
        )

    for run_id, pred_run in enumerate(run_frames, start=1):
        prob_col = f"pred_prob_run_{run_id:02d}"
        if prob_col not in pred_run.columns:
            raise RuntimeError(f"{exp_name} run {run_id} is missing {prob_col}.")
        if pred_base is None:
            pred_base = pred_run.copy()
        else:
            validate_same_prediction_rows(pred_base, pred_run, run_id)
            pred_base[prob_col] = pred_run[prob_col].to_numpy()
        prob_rows.append(pred_run[prob_col].to_numpy(dtype=np.float32, copy=False))

    if pred_base is None:
        raise RuntimeError(f"{exp_name} has no ensemble run predictions to aggregate.")

    probs = np.vstack(prob_rows).astype(np.float32, copy=False)
    pred_base["model_name"] = model_name_for_ensemble(options.arch, run_count)
    pred_base["ensemble_runs"] = run_count
    pred_base["pred_prob"] = probs.mean(axis=0).astype(np.float32, copy=False)
    pred_base["window"] = int(cfg["window"])
    pred_base["horizon"] = int(cfg["horizon"])
    pred_base["experiment_name"] = exp_name

    out_path = final_prediction_path(exp_name, options.arch)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pred_base.to_parquet(out_path, index=False)
    print(f"Saved averaged predictions to: {out_path}")

    if run_count > 1:
        TABLE_DIR.mkdir(parents=True, exist_ok=True)
        summary = pd.DataFrame([
            {
                "experiment_name": exp_name,
                "model_name": model_name_for_ensemble(options.arch, run_count),
                "arch": options.arch,
                "run_id": run_id,
                "seed": int(RANDOM_SEED) + run_id - 1,
                "prediction_path": str(ensemble_run_prediction_path(exp_name, options.arch, run_id)),
            }
            for run_id in range(1, run_count + 1)
        ])
        summary_path = TABLE_DIR / f"cnn_ensemble_summary_{output_stem(exp_name, options.arch)}.csv"
        summary.to_csv(summary_path, index=False)
        print(f"Saved ensemble aggregation summary to: {summary_path}")

    return pred_base


def fit_one_experiment(
    exp_name: str,
    cfg: dict,
    image_paths: Sequence[Path],
    image_shape: Tuple[int, int, int, int],
    meta_window: pd.DataFrame,
    options: TrainOptions,
) -> None:
    """
    Train one or more independent runs and save averaged test predictions.
    """
    run_count = max(1, int(options.ensemble_runs))
    selected_run_id = options.ensemble_run_id
    if selected_run_id is not None:
        selected_run_id = int(selected_run_id)
        if selected_run_id < 1 or selected_run_id > run_count:
            raise RuntimeError(
                f"--ensemble-run-id must be between 1 and {run_count}, got {selected_run_id}."
            )

    run_results = []
    run_frames = []

    run_ids = [selected_run_id] if selected_run_id is not None else list(range(1, run_count + 1))
    for run_id in run_ids:
        seed = int(RANDOM_SEED) + run_id - 1
        result = fit_one_experiment_run(
            exp_name=exp_name,
            cfg=cfg,
            image_paths=image_paths,
            image_shape=image_shape,
            meta_window=meta_window,
            options=options,
            run_id=run_id,
            run_count=run_count,
            seed=seed,
        )
        pred_run = result["pred"]
        save_run_prediction(exp_name, options.arch, run_id, pred_run)
        run_frames.append(pred_run)
        run_results.append(result)

    if selected_run_id is not None:
        print(
            f"{exp_name} run {selected_run_id}/{run_count} finished. "
            "Final averaged prediction will be created by --ensemble-aggregate-only."
        )
        return

    aggregate_ensemble_predictions(exp_name, cfg, options, run_frames=run_frames)

    if run_count > 1:
        stem = output_stem(exp_name, options.arch)
        summary = pd.DataFrame([
            {
                "experiment_name": exp_name,
                "model_name": model_name_for_ensemble(options.arch, run_count),
                "arch": options.arch,
                "run_id": i + 1,
                "seed": result["seed"],
                "test_loss": result["test_loss"],
                "test_auc": result["test_auc"],
                "test_acc": result["test_acc"],
                "test_brier": result["test_brier"],
                "model_path": str(result["model_path"]),
            }
            for i, result in enumerate(run_results)
        ])
        summary_path = TABLE_DIR / f"cnn_ensemble_summary_{stem}.csv"
        summary.to_csv(summary_path, index=False)
        print(f"Saved ensemble summary to: {summary_path}")


# ============================================================
# 5. CLI
# ============================================================

def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train optimized Jiang-style 2D CNNs on price image shards."
    )
    parser.add_argument("--experiments", default=None, help="Comma-separated experiments, e.g. I5R5,I20R5")
    parser.add_argument("--windows", default=None, help="Comma-separated windows, e.g. 5,20,60")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--workers", type=int, default=default_num_workers())
    parser.add_argument("--prefetch-factor", type=int, default=2)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--min-epochs", type=int, default=8)
    parser.add_argument("--min-delta", type=float, default=1e-4)
    parser.add_argument("--optimizer", choices=["adamw", "adam"], default="adamw")
    parser.add_argument("--weight-decay", type=float, default=3e-5)
    parser.add_argument("--scheduler", choices=["none", "cosine", "plateau"], default="cosine")
    parser.add_argument("--warmup-epochs", type=int, default=1)
    parser.add_argument("--fc-dropout", type=float, default=0.20)
    parser.add_argument("--spatial-dropout", type=float, default=0.0)
    parser.add_argument("--arch", choices=["jiang", "reslite"], default="jiang")
    parser.add_argument(
        "--ensemble-runs",
        type=int,
        default=1,
        help=(
            "Number of independent training runs per experiment. "
            "Use 5 to match the Jiang et al. paper-style probability averaging protocol."
        ),
    )
    parser.add_argument(
        "--ensemble-run-id",
        type=int,
        default=None,
        help=(
            "Train only one ensemble member, 1-based. Intended for Slurm GPU arrays. "
            "Use together with --ensemble-runs N."
        ),
    )
    parser.add_argument(
        "--ensemble-aggregate-only",
        action="store_true",
        help=(
            "Do not train. Read outputs/predictions/ensemble_runs/*/runXX.parquet "
            "and write final averaged pred_*_jiang_cnn2d.parquet files."
        ),
    )
    parser.add_argument("--valid-metric-interval", type=int, default=1)
    parser.add_argument("--max-valid-batches", type=int, default=None,
                        help="Optional validation batch cap for smoke tests or fast monitoring.")
    parser.add_argument("--max-test-batches", type=int, default=None,
                        help="Optional test batch cap for smoke tests only. Do not use for final results.")

    parser.add_argument("--no-amp", action="store_true", help="Disable CUDA AMP mixed precision.")
    parser.add_argument("--no-tf32", action="store_true", help="Disable TF32 on supported GPUs.")
    parser.add_argument("--compile", action="store_true", help="Use torch.compile if available. First epoch may be slower.")
    parser.add_argument("--channels-last", action="store_true", help="Use channels_last memory format on GPU.")
    parser.add_argument("--no-pin-memory", action="store_true", help="Disable pin_memory in DataLoader.")
    parser.add_argument("--no-persistent-workers", action="store_true", help="Disable persistent DataLoader workers.")
    parser.add_argument("--drop-last-train", action="store_true", help="Drop last incomplete train batch.")
    parser.add_argument("--log-interval", type=int, default=200)
    parser.add_argument(
        "--profile-batches",
        type=int,
        default=0,
        help="Synchronize CUDA timing for the first N train batches per epoch. 0 disables exact profiling.",
    )
    parser.add_argument(
        "--shard-cache-size",
        type=int,
        default=32,
        help="Max open image shard memmaps per DataLoader worker. Keeps file descriptors bounded.",
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()

    experiments = selected_experiments(args.experiments, args.windows)
    grouped = group_experiments_by_window(experiments)

    options = TrainOptions(
        epochs=int(args.epochs),
        batch_size=int(args.batch_size),
        lr=float(args.lr),
        num_workers=int(args.workers),
        prefetch_factor=int(args.prefetch_factor),
        patience=int(args.patience),
        min_epochs=int(args.min_epochs),
        min_delta=float(args.min_delta),
        amp=not bool(args.no_amp),
        tf32=not bool(args.no_tf32),
        compile_model=bool(args.compile),
        valid_metric_interval=max(1, int(args.valid_metric_interval)),
        max_valid_batches=args.max_valid_batches,
        max_test_batches=args.max_test_batches,
        channels_last=bool(args.channels_last),
        pin_memory=not bool(args.no_pin_memory),
        persistent_workers=not bool(args.no_persistent_workers),
        drop_last_train=bool(args.drop_last_train),
        shard_cache_size=int(args.shard_cache_size),
        optimizer=args.optimizer,
        weight_decay=float(args.weight_decay),
        scheduler=args.scheduler,
        warmup_epochs=int(args.warmup_epochs),
        fc_dropout=float(args.fc_dropout),
        spatial_dropout=float(args.spatial_dropout),
        arch=args.arch,
        log_interval=int(args.log_interval),
        profile_batches=int(args.profile_batches),
        ensemble_runs=max(1, int(args.ensemble_runs)),
        ensemble_run_id=args.ensemble_run_id,
        ensemble_aggregate_only=bool(args.ensemble_aggregate_only),
    )

    if options.ensemble_aggregate_only:
        print("Mode: ensemble aggregation only")
        print("Selected experiments:", ", ".join(experiments.keys()))
        print(f"Ensemble runs: {options.ensemble_runs}")
        for exp_name, cfg in experiments.items():
            aggregate_ensemble_predictions(exp_name, cfg, options)
        print("Done.")
        return

    configure_torch(RANDOM_SEED, use_tf32=options.tf32)
    device = get_device()
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        props = torch.cuda.get_device_properties(0)
        print(f"GPU memory: {props.total_memory / 1024**3:.2f} GB")

    print("Selected experiments:", ", ".join(experiments.keys()))
    print("Windows:", ", ".join(str(w) for w in sorted(grouped)))
    print("Options:", options)

    for window in sorted(grouped):
        window_experiments = grouped[window]
        image_paths, meta_window, image_shape = load_image_window(window, window_experiments)

        for exp_name, cfg in window_experiments.items():
            fit_one_experiment(
                exp_name=exp_name,
                cfg=cfg,
                image_paths=image_paths,
                image_shape=image_shape,
                meta_window=meta_window,
                options=options,
            )

        # Release current window before loading the next one.
        del image_paths, meta_window, image_shape
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

    print("Done.")


if __name__ == "__main__":
    main()
