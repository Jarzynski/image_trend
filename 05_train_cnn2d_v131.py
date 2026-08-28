# -*- coding: utf-8 -*-
"""
Version 1.3.1 CNN training entry point.

This entry point intentionally lives beside the historical
``05_train_cnn2d.py`` implementation.  It provides a reproducible
2009--2024 purged five-fold ablation without changing the v1.2.x artifacts.

The logical training unit is one complete trading-day cross-section.  A
physical micro-batch is used inside the logical batch to keep I60 runs within
GPU memory.  BCE and Huber accumulate the exact date-level mean gradient;
Huber+IC uses a replayed second pass so Pearson IC is calculated over the full
cross-section while only one physical micro-batch of activations is retained.
"""

from __future__ import annotations

import argparse
import gc
import importlib.util
import json
import math
import os
import random
import sys
import time
from collections import OrderedDict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset, Sampler

import config as CONFIG
from config import (
    EXPERIMENTS,
    FEATURE_BY_YEAR_DIR,
    MODEL_DIR,
    OUTPUT_DIR,
    PRED_DIR,
    TABLE_DIR,
    TEST_START,
)


LOSS_CHOICES = ("bce", "huber", "huber_ic")
DEFAULT_SEEDS = (42, 43, 44, 45)
DEFAULT_PURGE_DAYS = 20
DEFAULT_MICRO_BATCH_SIZE = 256
DEFAULT_EPOCHS = 25
DEFAULT_WARMUP_EPOCHS = 2
DEFAULT_PATIENCE = 4
DEFAULT_MIN_DELTA = 1e-3
DEFAULT_IC_WEIGHT = 1.0
DEFAULT_HUBER_BETA = 1.0
CANDIDATE_END = pd.Timestamp("2020-12-31")
TARGET_WINSOR_LOW = 0.01
TARGET_WINSOR_HIGH = 0.99
IC_EPS = 1e-8


def load_legacy_module() -> Any:
    """Load reusable v1.2.x CNN/model/image utilities from the numeric file."""
    path = Path(__file__).with_name("05_train_cnn2d.py")
    spec = importlib.util.spec_from_file_location("image_trend_cnn_legacy", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load legacy training module: {path}")
    module = importlib.util.module_from_spec(spec)
    # dataclasses resolves postponed annotations through sys.modules while the
    # legacy module is being executed; spec_from_file_location alone does not
    # register the temporary module.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


LEGACY = load_legacy_module()


def apply_version_roots(data_dir: Optional[str], output_dir: Optional[str]) -> None:
    """Apply CLI directory overrides to both v1.3.1 and legacy image helpers."""
    global FEATURE_BY_YEAR_DIR, OUTPUT_DIR, PRED_DIR, TABLE_DIR, MODEL_DIR
    if data_dir:
        root = Path(data_dir).expanduser().resolve()
        CONFIG.DATA_DIR = root
        CONFIG.RAW_DIR = root / "raw"
        CONFIG.PROCESSED_DIR = root / "processed"
        CONFIG.FEATURE_DIR = root / "features"
        CONFIG.IMAGE_DIR = root / "images"
        CONFIG.FEATURE_BY_YEAR_DIR = CONFIG.FEATURE_DIR / "features_by_year"
        CONFIG.FEATURE_BY_CODE_BUCKET_DIR = CONFIG.FEATURE_DIR / "features_by_code_bucket"
        FEATURE_BY_YEAR_DIR = CONFIG.FEATURE_BY_YEAR_DIR
        # legacy.load_image_window calls config.image_dir_for_window, whose
        # function globals point at CONFIG.IMAGE_DIR.
    if output_dir:
        root = Path(output_dir).expanduser().resolve()
        CONFIG.OUTPUT_DIR = root
        CONFIG.PRED_DIR = root / "predictions"
        CONFIG.TABLE_DIR = root / "tables"
        CONFIG.MODEL_DIR = root / "models"
        OUTPUT_DIR = CONFIG.OUTPUT_DIR
        PRED_DIR = CONFIG.PRED_DIR
        TABLE_DIR = CONFIG.TABLE_DIR
        MODEL_DIR = CONFIG.MODEL_DIR
    for path in (OUTPUT_DIR, PRED_DIR, TABLE_DIR, MODEL_DIR):
        path.mkdir(parents=True, exist_ok=True)


@dataclass(frozen=True)
class TrainOptions:
    loss: str
    fold_id: int
    seed: int
    purge_days: int = DEFAULT_PURGE_DAYS
    micro_batch_size: int = DEFAULT_MICRO_BATCH_SIZE
    epochs: int = DEFAULT_EPOCHS
    warmup_epochs: int = DEFAULT_WARMUP_EPOCHS
    patience: int = DEFAULT_PATIENCE
    min_delta: float = DEFAULT_MIN_DELTA
    ic_weight: float = DEFAULT_IC_WEIGHT
    huber_beta: float = DEFAULT_HUBER_BETA
    batch_workers: int = 2
    prefetch_factor: int = 2
    shard_cache_size: int = 32
    lr: float = 1e-4
    weight_decay: float = 3e-5
    fc_dropout: float = 0.20
    spatial_dropout: float = 0.0
    amp: bool = True
    tf32: bool = True
    pin_memory: bool = True
    persistent_workers: bool = True
    channels_last: bool = False
    log_interval: int = 200
    smoke_dates: Optional[int] = None


def configure_torch(seed: int, tf32: bool = True) -> None:
    """Configure deterministic seeds while retaining cuDNN performance."""
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))
        torch.backends.cuda.matmul.allow_tf32 = bool(tf32)
        torch.backends.cudnn.allow_tf32 = bool(tf32)
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass


def close_memmap(array: np.ndarray) -> None:
    mmap_obj = getattr(array, "_mmap", None)
    if mmap_obj is not None:
        mmap_obj.close()


class V131ImageDataset(Dataset):
    """Memory-mapped image dataset returning binary, regression and row keys."""

    def __init__(
        self,
        image_paths: Sequence[Path],
        labels: np.ndarray,
        regression_targets: np.ndarray,
        shard_ids: np.ndarray,
        local_indices: np.ndarray,
        indices: np.ndarray,
        shard_cache_size: int = 32,
    ) -> None:
        self.image_paths = [Path(path) for path in image_paths]
        self.labels = np.asarray(labels, dtype=np.float32)
        self.regression_targets = np.asarray(regression_targets, dtype=np.float32)
        self.shard_ids = np.asarray(shard_ids, dtype=np.int32)
        self.local_indices = np.asarray(local_indices, dtype=np.int32)
        self.indices = np.asarray(indices, dtype=np.int64)
        self.shard_cache_size = max(1, int(shard_cache_size))
        self._shard_cache: OrderedDict[int, np.ndarray] = OrderedDict()

    def __len__(self) -> int:
        return int(len(self.indices))

    def _get_shard(self, shard_id: int) -> np.ndarray:
        shard_id = int(shard_id)
        cached = self._shard_cache.get(shard_id)
        if cached is not None:
            self._shard_cache.move_to_end(shard_id)
            return cached

        shard = np.load(self.image_paths[shard_id], mmap_mode="r")
        self._shard_cache[shard_id] = shard
        while len(self._shard_cache) > self.shard_cache_size:
            _, old = self._shard_cache.popitem(last=False)
            close_memmap(old)
        return shard

    def __getitem__(self, position: int):
        real_idx = int(self.indices[int(position)])
        shard_id = int(self.shard_ids[real_idx])
        local_idx = int(self.local_indices[real_idx])
        arr = self._get_shard(shard_id)[local_idx]
        arr = np.array(
            np.transpose(arr, (2, 0, 1)), dtype=np.uint8, order="C", copy=True
        )
        return (
            torch.from_numpy(arr),
            torch.tensor(self.labels[real_idx], dtype=torch.float32),
            torch.tensor(self.regression_targets[real_idx], dtype=torch.float32),
            torch.tensor(real_idx, dtype=torch.int64),
        )


class DateBatchSampler(Sampler[List[int]]):
    """Yield one chronologically ordered list of global rows per trading day."""

    def __init__(self, meta: pd.DataFrame, indices: np.ndarray) -> None:
        indices = np.asarray(indices, dtype=np.int64)
        # Dataset.__getitem__ receives positions in the split-local
        # ``dataset.indices`` array.  Group rows using their global metadata
        # indices for chronological ordering, but emit the corresponding local
        # positions so a fold cannot index past the compact split array.
        local_position = {
            int(global_idx): int(local_idx)
            for local_idx, global_idx in enumerate(indices.tolist())
        }
        frame = meta.iloc[indices][["date", "code"]].copy()
        frame["global_index"] = indices
        frame["date"] = pd.to_datetime(frame["date"])
        frame["code_sort"] = frame["code"].astype(str)
        frame = frame.sort_values(
            ["date", "code_sort", "global_index"], kind="mergesort"
        )
        self.batches: List[List[int]] = [
            [local_position[int(global_idx)] for global_idx in group["global_index"]]
            for _, group in frame.groupby("date", sort=True)
        ]

    def __iter__(self) -> Iterator[List[int]]:
        yield from self.batches

    def __len__(self) -> int:
        return len(self.batches)


class DeviceMicrobatchPool:
    """Double-buffer pinned CPU to GPU copies on a dedicated CUDA stream."""

    def __init__(
        self,
        device: torch.device,
        micro_batch_size: int,
        channels_last: bool = False,
    ) -> None:
        self.device = device
        self.micro_batch_size = max(1, int(micro_batch_size))
        self.channels_last = bool(channels_last)
        self.copy_stream: Optional[torch.cuda.Stream] = None
        self.events: List[Optional[torch.cuda.Event]] = [None, None]
        self.buffers: List[Dict[str, Tensor]] = [{}, {}]
        self.busy = [False, False]
        self.shape: Optional[Tuple[int, int, int]] = None
        if device.type == "cuda":
            self.copy_stream = torch.cuda.Stream(device=device)

    def _ensure(self, shape: Tuple[int, int, int]) -> None:
        if self.shape == shape and all("x" in buf for buf in self.buffers):
            return
        self.shape = shape
        channels, height, width = shape
        memory_format = (
            torch.channels_last if self.channels_last else torch.contiguous_format
        )
        for i in range(2):
            self.buffers[i] = {
                "x": torch.empty(
                    self.micro_batch_size,
                    channels,
                    height,
                    width,
                    dtype=torch.float32,
                    device=self.device,
                ).to(memory_format=memory_format),
                "label": torch.empty(
                    self.micro_batch_size, dtype=torch.float32, device=self.device
                ),
                "target": torch.empty(
                    self.micro_batch_size, dtype=torch.float32, device=self.device
                ),
            }
            self.events[i] = torch.cuda.Event() if self.device.type == "cuda" else None
            self.busy[i] = False

    def _preload(
        self,
        slot: int,
        x_cpu: Tensor,
        label_cpu: Tensor,
        target_cpu: Tensor,
        n: int,
    ) -> None:
        buffer = self.buffers[slot]
        if self.device.type != "cuda":
            # ``Tensor.float()`` is a view when the loader already returns
            # float32.  Keep the fallback path side-effect free by using an
            # out-of-place division rather than mutating the caller's batch.
            buffer["x"][:n].copy_(x_cpu[:n].float().div(255.0))
            buffer["label"][:n].copy_(label_cpu[:n])
            buffer["target"][:n].copy_(target_cpu[:n])
            return

        assert self.copy_stream is not None and self.events[slot] is not None
        current = torch.cuda.current_stream(self.device)
        if self.busy[slot]:
            self.copy_stream.wait_stream(current)
        with torch.cuda.stream(self.copy_stream):
            buffer["x"][:n].copy_(x_cpu[:n], non_blocking=True)
            buffer["x"][:n].div_(255.0)
            buffer["label"][:n].copy_(label_cpu[:n], non_blocking=True)
            buffer["target"][:n].copy_(target_cpu[:n], non_blocking=True)
            self.events[slot].record(self.copy_stream)
        self.busy[slot] = True

    def iterate(
        self,
        x_cpu: Tensor,
        label_cpu: Tensor,
        target_cpu: Tensor,
    ) -> Iterator[Tuple[Tensor, Tensor, Tensor, int]]:
        if x_cpu.ndim != 4:
            raise ValueError(f"Expected CHW image batch, got shape={tuple(x_cpu.shape)}")
        n = int(x_cpu.shape[0])
        self._ensure((int(x_cpu.shape[1]), int(x_cpu.shape[2]), int(x_cpu.shape[3])))

        starts = list(range(0, n, self.micro_batch_size))
        if not starts:
            return
        self._preload(
            0,
            x_cpu[starts[0] : starts[0] + self.micro_batch_size],
            label_cpu[starts[0] : starts[0] + self.micro_batch_size],
            target_cpu[starts[0] : starts[0] + self.micro_batch_size],
            min(self.micro_batch_size, n),
        )
        for batch_no, start in enumerate(starts):
            slot = batch_no % 2
            end = min(start + self.micro_batch_size, n)
            size = end - start
            next_no = batch_no + 1
            if next_no < len(starts):
                next_start = starts[next_no]
                next_slot = next_no % 2
                next_end = min(next_start + self.micro_batch_size, n)
                self._preload(
                    next_slot,
                    x_cpu[next_start:next_end],
                    label_cpu[next_start:next_end],
                    target_cpu[next_start:next_end],
                    next_end - next_start,
                )
            if self.device.type == "cuda":
                assert self.events[slot] is not None
                torch.cuda.current_stream(self.device).wait_event(self.events[slot])
            yield (
                self.buffers[slot]["x"][:size],
                self.buffers[slot]["label"][:size],
                self.buffers[slot]["target"][:size],
                size,
            )


def make_loader(
    dataset: Dataset,
    sampler: DateBatchSampler,
    options: TrainOptions,
) -> DataLoader:
    kwargs: Dict[str, Any] = {
        "batch_sampler": sampler,
        "num_workers": max(0, int(options.batch_workers)),
        "pin_memory": bool(options.pin_memory and torch.cuda.is_available()),
    }
    if int(options.batch_workers) > 0:
        kwargs["persistent_workers"] = bool(options.persistent_workers)
        kwargs["prefetch_factor"] = int(options.prefetch_factor)
    return DataLoader(dataset, **kwargs)


def load_master_calendar() -> pd.DatetimeIndex:
    """Read only date columns to make fold boundaries independent of experiment."""
    date_parts: List[pd.Series] = []
    for path in sorted(FEATURE_BY_YEAR_DIR.glob("year=*/*.parquet")):
        year_name = path.parent.name.split("=", 1)[-1]
        try:
            year = int(year_name)
        except ValueError:
            continue
        if 2009 <= year <= 2020:
            date_parts.append(pd.read_parquet(path, columns=["date"])["date"])
    if not date_parts:
        raise RuntimeError(f"No 2009-2020 feature partitions found under {FEATURE_BY_YEAR_DIR}")
    dates = pd.DatetimeIndex(
        sorted(pd.to_datetime(pd.concat(date_parts, ignore_index=True)).dropna().unique())
    )
    if len(dates) < 100:
        raise RuntimeError(f"Master calendar is unexpectedly short: {len(dates)} dates")
    return dates


def make_fold_ranges(calendar: pd.DatetimeIndex, n_folds: int = 5) -> List[Tuple[pd.Timestamp, pd.Timestamp]]:
    chunks = np.array_split(calendar.to_numpy(), int(n_folds))
    ranges: List[Tuple[pd.Timestamp, pd.Timestamp]] = []
    for chunk in chunks:
        if len(chunk) == 0:
            raise RuntimeError(f"Cannot construct {n_folds} non-empty folds from calendar")
        ranges.append((pd.Timestamp(chunk[0]), pd.Timestamp(chunk[-1])))
    return ranges


def split_indices(
    meta: pd.DataFrame,
    fold_range: Tuple[pd.Timestamp, pd.Timestamp],
    calendar: pd.DatetimeIndex,
    purge_days: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
    dates = pd.to_datetime(meta["date"])
    val_start, val_end = fold_range
    # v1.3.1 uses every complete pre-2021 trading day as the candidate pool;
    # the historical v1.2.x TRAIN_END/VALID_* split remains untouched.
    pre_test = dates <= CANDIDATE_END

    before = calendar[calendar < val_start]
    after = calendar[calendar > val_end]
    purge_before = before[-int(purge_days) :] if purge_days > 0 else pd.DatetimeIndex([])
    purge_after = after[: int(purge_days)] if purge_days > 0 else pd.DatetimeIndex([])
    purge_dates = set(purge_before.tolist()) | set(purge_after.tolist())

    val_mask = pre_test & (dates >= val_start) & (dates <= val_end)
    train_mask = pre_test & ~val_mask
    if purge_dates:
        train_mask &= ~dates.isin(purge_dates)
    test_mask = dates >= pd.Timestamp(TEST_START)

    metadata = {
        "valid_start": val_start.strftime("%Y-%m-%d"),
        "valid_end": val_end.strftime("%Y-%m-%d"),
        "purge_before_start": (
            pd.Timestamp(purge_before[0]).strftime("%Y-%m-%d") if len(purge_before) else None
        ),
        "purge_before_end": (
            pd.Timestamp(purge_before[-1]).strftime("%Y-%m-%d") if len(purge_before) else None
        ),
        "purge_after_start": (
            pd.Timestamp(purge_after[0]).strftime("%Y-%m-%d") if len(purge_after) else None
        ),
        "purge_after_end": (
            pd.Timestamp(purge_after[-1]).strftime("%Y-%m-%d") if len(purge_after) else None
        ),
        "candidate_start": "2009-01-01",
        "candidate_end": CANDIDATE_END.strftime("%Y-%m-%d"),
        "test_start": pd.Timestamp(TEST_START).strftime("%Y-%m-%d"),
    }
    return (
        np.flatnonzero(train_mask.to_numpy()),
        np.flatnonzero(val_mask.to_numpy()),
        np.flatnonzero(test_mask.to_numpy()),
        metadata,
    )


def winsorized_cross_section_target(meta: pd.DataFrame) -> np.ndarray:
    """Return per-date 1%-99% winsorized, population-z-scored future returns."""
    ret = pd.to_numeric(meta["future_ret"], errors="coerce").astype(np.float32)
    frame = pd.DataFrame({"date": pd.to_datetime(meta["date"]), "ret": ret})
    grouped = frame.groupby("date", sort=False)["ret"]
    low = grouped.transform(lambda x: x.quantile(TARGET_WINSOR_LOW))
    high = grouped.transform(lambda x: x.quantile(TARGET_WINSOR_HIGH))
    clipped = frame["ret"].clip(lower=low, upper=high)
    mean = clipped.groupby(frame["date"], sort=False).transform("mean")
    centered = clipped - mean
    std = centered.groupby(frame["date"], sort=False).transform(
        lambda x: float(np.sqrt(np.mean(np.square(x.to_numpy(dtype=np.float64)))))
    )
    target = centered / std.replace(0.0, np.nan)
    return target.fillna(0.0).to_numpy(dtype=np.float32, copy=False)


def pearson_ic_and_gradient(scores: Tensor, target: Tensor) -> Tuple[Tensor, Tensor]:
    """Calculate Pearson IC and its exact gradient with respect to scores."""
    scores = scores.float()
    target = target.float()
    centered_scores = scores - scores.mean()
    centered_target = target - target.mean()
    var_scores = torch.sum(centered_scores * centered_scores)
    var_target = torch.sum(centered_target * centered_target)
    if float(var_scores.detach().item()) <= IC_EPS or float(var_target.detach().item()) <= IC_EPS:
        return scores.new_zeros(()), torch.zeros_like(scores)
    denominator = torch.sqrt(var_scores * var_target)
    corr = torch.sum(centered_scores * centered_target) / denominator
    grad_corr = (
        centered_target / torch.sqrt(var_scores * var_target)
        - corr.detach() * centered_scores / var_scores
    )
    # Pearson correlation is invariant to a constant shift in every score,
    # therefore its score gradient must sum to zero.  Enforce that invariant
    # after the float32 arithmetic to avoid a spurious intercept gradient from
    # cancellation when a model's daily scores are nearly constant.
    grad_corr = grad_corr - grad_corr.mean()
    return corr, grad_corr


def huber_value_and_gradient(scores: Tensor, target: Tensor, beta: float) -> Tuple[Tensor, Tensor]:
    diff = scores.float() - target.float()
    abs_diff = diff.abs()
    beta_tensor = diff.new_tensor(float(beta))
    loss = torch.where(
        abs_diff <= beta_tensor,
        0.5 * diff.square() / beta_tensor,
        abs_diff - 0.5 * beta_tensor,
    ).mean()
    grad = torch.where(
        abs_diff <= beta_tensor,
        diff / beta_tensor,
        diff.sign(),
    ) / max(int(diff.numel()), 1)
    return loss, grad


def objective_from_scores(
    scores: Tensor,
    labels: Tensor,
    target: Tensor,
    options: TrainOptions,
) -> Tuple[Tensor, Dict[str, float]]:
    if options.loss == "bce":
        value = F.binary_cross_entropy_with_logits(scores, labels)
        return value, {"bce": float(value.detach().item()), "huber": np.nan, "ic": np.nan}
    huber, _ = huber_value_and_gradient(scores, target, options.huber_beta)
    ic, _ = pearson_ic_and_gradient(scores, target)
    if options.loss == "huber":
        return huber, {"bce": np.nan, "huber": float(huber.detach().item()), "ic": float(ic.detach().item())}
    value = huber + float(options.ic_weight) * (1.0 - ic)
    return value, {"bce": np.nan, "huber": float(huber.detach().item()), "ic": float(ic.detach().item())}


def snapshot_batchnorm_buffers(model: nn.Module) -> Dict[str, Tensor]:
    snapshot: Dict[str, Tensor] = {}
    for name, buffer in model.named_buffers():
        if name.endswith("running_mean") or name.endswith("running_var") or name.endswith("num_batches_tracked"):
            snapshot[name] = buffer.detach().clone()
    return snapshot


def restore_batchnorm_buffers(model: nn.Module, snapshot: Mapping[str, Tensor]) -> None:
    named_buffers = dict(model.named_buffers())
    for name, value in snapshot.items():
        if name in named_buffers:
            named_buffers[name].copy_(value)


def snapshot_rng_state(device: torch.device) -> Tuple[Tensor, Optional[List[Tensor]]]:
    cuda_state = torch.cuda.get_rng_state_all() if device.type == "cuda" else None
    return torch.random.get_rng_state(), cuda_state


def restore_rng_state(device: torch.device, state: Tuple[Tensor, Optional[List[Tensor]]]) -> None:
    cpu_state, cuda_state = state
    torch.random.set_rng_state(cpu_state)
    if device.type == "cuda" and cuda_state is not None:
        torch.cuda.set_rng_state_all(cuda_state)


def hybrid_date_backward(
    model: nn.Module,
    x_cpu: Tensor,
    labels_cpu: Tensor,
    target_cpu: Tensor,
    pool: DeviceMicrobatchPool,
    device: torch.device,
    options: TrainOptions,
    scaler: torch.amp.GradScaler,
) -> Dict[str, float]:
    """Replay a date so hybrid loss has exact full-cross-section IC gradients."""
    bn_state = snapshot_batchnorm_buffers(model)
    rng_state = snapshot_rng_state(device)
    first_scores: List[Tensor] = []
    first_labels: List[Tensor] = []
    first_targets: List[Tensor] = []

    with torch.no_grad():
        for x, labels, target, _ in pool.iterate(x_cpu, labels_cpu, target_cpu):
            with torch.amp.autocast(device_type="cuda", enabled=device.type == "cuda" and options.amp):
                logits = model(x)
            first_scores.append(logits.detach().float())
            first_labels.append(labels.detach().float())
            first_targets.append(target.detach().float())

    if not first_scores:
        raise RuntimeError("Encountered empty date batch in hybrid loss")
    scores = torch.cat(first_scores, dim=0)
    labels = torch.cat(first_labels, dim=0)
    target = torch.cat(first_targets, dim=0)
    huber_value, huber_grad = huber_value_and_gradient(scores, target, options.huber_beta)
    ic_value, grad_ic = pearson_ic_and_gradient(scores, target)
    total_grad = huber_grad - float(options.ic_weight) * grad_ic
    total_value = huber_value + float(options.ic_weight) * (1.0 - ic_value)

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    restore_batchnorm_buffers(model, bn_state)
    restore_rng_state(device, rng_state)

    offset = 0
    for x, _labels, _target, size in pool.iterate(x_cpu, labels_cpu, target_cpu):
        with torch.amp.autocast(
            device_type="cuda", enabled=device.type == "cuda" and options.amp
        ):
            logits = model(x)
        grad_slice = total_grad[offset : offset + size]
        if device.type == "cuda" and options.amp:
            # Keep the externally computed IC gradient in float32 during the
            # replay even though the autocast model output may be fp16.
            scaler.scale(logits.float()).backward(grad_slice)
        else:
            logits.backward(grad_slice.to(dtype=logits.dtype))
        offset += size
    if offset != int(scores.numel()):
        raise RuntimeError(f"Hybrid replay row mismatch: replayed={offset}, expected={scores.numel()}")
    return {
        "objective": float(total_value.detach().item()),
        "huber": float(huber_value.detach().item()),
        "ic": float(ic_value.detach().item()),
        "bce": np.nan,
    }


def prepare_batch_cpu(batch: Sequence[Tensor]) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    x_cpu, labels_cpu, target_cpu, row_idx = batch
    return x_cpu, labels_cpu, target_cpu, row_idx


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    pool: DeviceMicrobatchPool,
    device: torch.device,
    options: TrainOptions,
) -> Dict[str, float]:
    model.train()
    date_objectives: List[float] = []
    components: Dict[str, List[float]] = {"bce": [], "huber": [], "ic": []}
    start_time = time.perf_counter()
    n_dates = 0
    n_rows = 0
    wait_seconds = 0.0
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    loader_iter = iter(loader)
    fetch_started = time.perf_counter()
    while True:
        try:
            batch = next(loader_iter)
        except StopIteration:
            break
        wait_seconds += time.perf_counter() - fetch_started
        x_cpu, labels_cpu, target_cpu, _row_idx = prepare_batch_cpu(batch)
        n = int(x_cpu.shape[0])
        if n == 0:
            fetch_started = time.perf_counter()
            continue
        optimizer.zero_grad(set_to_none=True)
        if options.loss == "huber_ic":
            metrics = hybrid_date_backward(
                model, x_cpu, labels_cpu, target_cpu, pool, device, options, scaler
            )
            scaler.step(optimizer)
            scaler.update()
        else:
            date_objective = 0.0
            date_component = {"bce": 0.0, "huber": 0.0, "ic": 0.0}
            offset = 0
            for x, labels, target, size in pool.iterate(x_cpu, labels_cpu, target_cpu):
                with torch.amp.autocast(
                    device_type="cuda", enabled=device.type == "cuda" and options.amp
                ):
                    logits = model(x)
                    value, _ = objective_from_scores(logits, labels, target, options)
                weight = float(size) / float(n)
                scaler.scale(value * weight).backward()
                date_objective += float(value.detach().item()) * weight
                if options.loss == "bce":
                    date_component["bce"] += float(value.detach().item()) * weight
                else:
                    huber_value, _ = huber_value_and_gradient(logits.detach(), target, options.huber_beta)
                    ic_value, _ = pearson_ic_and_gradient(logits.detach(), target)
                    date_component["huber"] += float(huber_value.item()) * weight
                    date_component["ic"] += float(ic_value.item()) * weight
                offset += size
            if offset != n:
                raise RuntimeError(f"Physical microbatch row mismatch: {offset} != {n}")
            scaler.step(optimizer)
            scaler.update()
            metrics = {
                "objective": date_objective,
                **date_component,
            }

        date_objectives.append(float(metrics["objective"]))
        for key in components:
            if np.isfinite(metrics.get(key, np.nan)):
                components[key].append(float(metrics[key]))
        n_dates += 1
        n_rows += n
        if options.log_interval > 0 and n_dates % int(options.log_interval) == 0:
            elapsed = time.perf_counter() - start_time
            print(
                f"Train date {n_dates}/{len(loader)} | rows={n_rows:,} | "
                f"objective={np.mean(date_objectives):.6f} | rows/s={n_rows / max(elapsed, 1e-9):.1f}",
                flush=True,
            )
        fetch_started = time.perf_counter()

    if not date_objectives:
        raise RuntimeError("Training loader produced no date batches")
    return {
        "objective": float(np.mean(date_objectives)),
        "bce": float(np.mean(components["bce"])) if components["bce"] else np.nan,
        "huber": float(np.mean(components["huber"])) if components["huber"] else np.nan,
        "ic": float(np.mean(components["ic"])) if components["ic"] else np.nan,
        "n_dates": float(n_dates),
        "n_rows": float(n_rows),
        "seconds": time.perf_counter() - start_time,
        "wait_seconds": float(wait_seconds),
        "cuda_peak_bytes": int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0,
    }


@torch.no_grad()
def evaluate_dates(
    model: nn.Module,
    loader: DataLoader,
    pool: DeviceMicrobatchPool,
    device: torch.device,
    options: TrainOptions,
    return_predictions: bool = False,
) -> Dict[str, Any]:
    model.eval()
    objectives: List[float] = []
    components: Dict[str, List[float]] = {"bce": [], "huber": [], "ic": []}
    row_indices: List[np.ndarray] = []
    row_scores: List[np.ndarray] = []
    start_time = time.perf_counter()
    wait_seconds = 0.0
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    loader_iter = iter(loader)
    fetch_started = time.perf_counter()
    while True:
        try:
            batch = next(loader_iter)
        except StopIteration:
            break
        wait_seconds += time.perf_counter() - fetch_started
        x_cpu, labels_cpu, target_cpu, row_idx_cpu = prepare_batch_cpu(batch)
        scores_parts: List[Tensor] = []
        labels_parts: List[Tensor] = []
        target_parts: List[Tensor] = []
        for x, labels, target, _size in pool.iterate(x_cpu, labels_cpu, target_cpu):
            with torch.amp.autocast(
                device_type="cuda", enabled=device.type == "cuda" and options.amp
            ):
                logits = model(x)
            scores_parts.append(logits.float())
            labels_parts.append(labels.float())
            target_parts.append(target.float())
        scores = torch.cat(scores_parts, dim=0)
        labels = torch.cat(labels_parts, dim=0)
        target = torch.cat(target_parts, dim=0)
        value, metric = objective_from_scores(scores, labels, target, options)
        objectives.append(float(value.item()))
        for key in components:
            if np.isfinite(metric.get(key, np.nan)):
                components[key].append(float(metric[key]))
        if return_predictions:
            row_indices.append(row_idx_cpu.numpy().astype(np.int64, copy=False))
            row_scores.append(scores.detach().cpu().numpy().astype(np.float32, copy=False))
        fetch_started = time.perf_counter()

    if not objectives:
        raise RuntimeError("Evaluation loader produced no date batches")
    result: Dict[str, Any] = {
        "objective": float(np.mean(objectives)),
        "bce": float(np.mean(components["bce"])) if components["bce"] else np.nan,
        "huber": float(np.mean(components["huber"])) if components["huber"] else np.nan,
        "ic": float(np.mean(components["ic"])) if components["ic"] else np.nan,
        "n_dates": len(objectives),
        "seconds": time.perf_counter() - start_time,
        "wait_seconds": float(wait_seconds),
        "cuda_peak_bytes": int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0,
    }
    if return_predictions:
        result["row_indices"] = np.concatenate(row_indices) if row_indices else np.array([], dtype=np.int64)
        result["scores"] = np.concatenate(row_scores) if row_scores else np.array([], dtype=np.float32)
    return result


def init_kaiming(module: nn.Module) -> None:
    if isinstance(module, (nn.Conv2d, nn.Linear)):
        nn.init.kaiming_normal_(module.weight, a=0.01, mode="fan_out", nonlinearity="leaky_relu")
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d)):
        if module.weight is not None:
            nn.init.ones_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)


def set_warmup_lr(optimizer: torch.optim.Optimizer, base_lr: float, epoch: int, warmup_epochs: int) -> None:
    if int(warmup_epochs) <= 0 or int(epoch) > int(warmup_epochs):
        return
    lr = float(base_lr) * float(epoch) / float(warmup_epochs)
    for group in optimizer.param_groups:
        group["lr"] = lr


def objective_improved(current: float, best: float, min_delta: float) -> bool:
    """Return whether ``current`` beats ``best`` by the configured margin."""
    if not np.isfinite(best):
        return True
    return float(current) < float(best) - float(min_delta)


def atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(dict(payload), ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, path)


def atomic_parquet(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    frame.to_parquet(tmp, index=False)
    os.replace(tmp, path)


def output_paths(options: TrainOptions, exp_name: str) -> Dict[str, Path]:
    stem = exp_name.lower()
    return {
        "model": Path(MODEL_DIR) / options.loss / stem / f"fold{options.fold_id:02d}_seed{options.seed}.pt",
        "member": Path(PRED_DIR) / "members" / options.loss / stem / f"fold{options.fold_id:02d}_seed{options.seed}.parquet",
        "log": Path(TABLE_DIR) / "training" / options.loss / f"cnn_training_log_{stem}_fold{options.fold_id:02d}_seed{options.seed}.csv",
        "done": Path(OUTPUT_DIR) / "manifests" / options.loss / stem / f"fold{options.fold_id:02d}_seed{options.seed}.json",
    }


def filter_smoke_dates(indices: np.ndarray, meta: pd.DataFrame, limit: Optional[int]) -> np.ndarray:
    if limit is None:
        return indices
    dates = pd.DatetimeIndex(sorted(pd.to_datetime(meta.iloc[indices]["date"]).unique()))[: int(limit)]
    return indices[np.isin(pd.to_datetime(meta.iloc[indices]["date"]).to_numpy(), dates.to_numpy())]


def train_one_experiment(
    exp_name: str,
    cfg: Mapping[str, Any],
    image_paths: Sequence[Path],
    image_shape: Tuple[int, int, int, int],
    meta_window: pd.DataFrame,
    calendar: pd.DatetimeIndex,
    fold_range: Tuple[pd.Timestamp, pd.Timestamp],
    options: TrainOptions,
    device: torch.device,
) -> Dict[str, Any]:
    paths = output_paths(options, exp_name)
    if paths["done"].exists() and paths["member"].exists() and paths["model"].exists():
        print(f"Skip completed {options.loss}/{exp_name}/fold={options.fold_id}/seed={options.seed}", flush=True)
        return json.loads(paths["done"].read_text(encoding="utf-8"))

    # Reset the logical model seed for every experiment.  This keeps a resumed
    # array task bitwise reproducible even when an earlier experiment was
    # skipped from its completion manifest.
    configure_torch(options.seed, options.tf32)

    meta = LEGACY.select_experiment_label_view(meta_window, exp_name, cfg)
    meta = meta.sort_values(["date", "code"], kind="mergesort").reset_index(drop=True)
    meta["reg_target"] = winsorized_cross_section_target(meta)
    train_idx, valid_idx, test_idx, split_meta = split_indices(
        meta, fold_range, calendar, options.purge_days
    )
    train_idx = filter_smoke_dates(train_idx, meta, options.smoke_dates)
    valid_idx = filter_smoke_dates(valid_idx, meta, options.smoke_dates)
    test_idx = filter_smoke_dates(test_idx, meta, options.smoke_dates)
    if min(len(train_idx), len(valid_idx), len(test_idx)) == 0:
        raise RuntimeError(
            f"Empty split for {exp_name}: train={len(train_idx)}, valid={len(valid_idx)}, test={len(test_idx)}"
        )

    labels = meta["label"].to_numpy(dtype=np.float32, copy=False)
    reg_target = meta["reg_target"].to_numpy(dtype=np.float32, copy=False)
    shard_ids = meta["shard_id"].to_numpy(dtype=np.int32, copy=False)
    local_indices = meta["local_index"].to_numpy(dtype=np.int32, copy=False)
    train_ds = V131ImageDataset(image_paths, labels, reg_target, shard_ids, local_indices, train_idx, options.shard_cache_size)
    valid_ds = V131ImageDataset(image_paths, labels, reg_target, shard_ids, local_indices, valid_idx, options.shard_cache_size)
    test_ds = V131ImageDataset(image_paths, labels, reg_target, shard_ids, local_indices, test_idx, options.shard_cache_size)
    train_loader = make_loader(train_ds, DateBatchSampler(meta, train_idx), options)
    valid_loader = make_loader(valid_ds, DateBatchSampler(meta, valid_idx), options)
    test_loader = make_loader(test_ds, DateBatchSampler(meta, test_idx), options)

    model = LEGACY.JiangCNN2D(
        window=int(cfg["window"]),
        image_height=int(image_shape[1]),
        image_width=int(image_shape[2]),
        fc_dropout=float(options.fc_dropout),
        spatial_dropout=float(options.spatial_dropout),
        arch="jiang",
    ).to(device)
    model.apply(init_kaiming)
    if options.channels_last and device.type == "cuda":
        model = model.to(memory_format=torch.channels_last)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(options.lr), weight_decay=float(options.weight_decay))
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=1, threshold=1e-4, min_lr=1e-6
    )
    use_amp = bool(options.amp and device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    pool = DeviceMicrobatchPool(device, options.micro_batch_size, options.channels_last)

    best_objective = math.inf
    best_state: Optional[Dict[str, Tensor]] = None
    best_epoch = 0
    bad_epochs = 0
    epoch_logs: List[Dict[str, Any]] = []
    started = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    print(
        f"Start {options.loss}/{exp_name} fold={options.fold_id}/seed={options.seed} "
        f"train/valid/test={len(train_idx):,}/{len(valid_idx):,}/{len(test_idx):,} "
        f"device={device} logical_batch=date micro_batch={options.micro_batch_size}",
        flush=True,
    )

    for epoch in range(1, int(options.epochs) + 1):
        set_warmup_lr(optimizer, options.lr, epoch, options.warmup_epochs)
        train_metrics = train_one_epoch(model, train_loader, optimizer, scaler, pool, device, options)
        valid_metrics = evaluate_dates(model, valid_loader, pool, device, options, return_predictions=False)
        scheduler.step(float(valid_metrics["objective"]))
        valid_objective = float(valid_metrics["objective"])
        improved = objective_improved(valid_objective, best_objective, options.min_delta)
        if improved or best_state is None:
            best_objective = valid_objective
            best_epoch = epoch
            bad_epochs = 0
            best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
        else:
            bad_epochs += 1

        row = {
            "loss_name": options.loss,
            "experiment_name": exp_name,
            "fold_id": options.fold_id,
            "seed": options.seed,
            "epoch": epoch,
            "lr": optimizer.param_groups[0]["lr"],
            "train_objective": train_metrics["objective"],
            "valid_objective": valid_metrics["objective"],
            "train_bce": train_metrics["bce"],
            "valid_bce": valid_metrics["bce"],
            "train_huber": train_metrics["huber"],
            "valid_huber": valid_metrics["huber"],
            "train_ic": train_metrics["ic"],
            "valid_ic": valid_metrics["ic"],
            "train_dates": train_metrics["n_dates"],
            "valid_dates": valid_metrics["n_dates"],
            "train_rows": train_metrics["n_rows"],
            "train_seconds": train_metrics["seconds"],
            "train_wait_seconds": train_metrics["wait_seconds"],
            "train_cuda_peak_bytes": train_metrics["cuda_peak_bytes"],
            "valid_seconds": valid_metrics["seconds"],
            "valid_wait_seconds": valid_metrics["wait_seconds"],
            "valid_cuda_peak_bytes": valid_metrics["cuda_peak_bytes"],
            "best_valid_objective": best_objective,
            "best_epoch": best_epoch,
            "bad_epochs": bad_epochs,
            "early_stop_min_delta": options.min_delta,
        }
        epoch_logs.append(row)
        paths["log"].parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(epoch_logs).to_csv(paths["log"], index=False)
        print(
            f"Epoch {epoch:02d} | train={train_metrics['objective']:.6f} "
            f"valid={valid_objective:.6f} | lr={optimizer.param_groups[0]['lr']:.3g} "
            f"bad={bad_epochs}/{options.patience}",
            flush=True,
        )
        if bad_epochs >= int(options.patience):
            print(f"Early stopping at epoch {epoch}; best epoch={best_epoch}", flush=True)
            break

    if best_state is None:
        raise RuntimeError(f"No checkpoint produced for {options.loss}/{exp_name}")
    model.load_state_dict(best_state)
    model.to(device)
    test_metrics = evaluate_dates(model, test_loader, pool, device, options, return_predictions=True)
    prediction_indices = np.asarray(test_metrics["row_indices"], dtype=np.int64)
    prediction_scores = np.asarray(test_metrics["scores"], dtype=np.float32)
    if len(prediction_indices) != len(test_idx):
        raise RuntimeError(
            f"Test prediction row mismatch for {exp_name}: {len(prediction_indices)} != {len(test_idx)}"
        )

    pred_cols = [
        "date", "code", "industry", "future_ret", "label", "amount", "float_mktcap",
        "is_low_volume_limit_up", "is_low_volume_limit_down",
    ]
    pred_cols = [column for column in pred_cols if column in meta.columns]
    pred = meta.iloc[prediction_indices][pred_cols].copy()
    for column in ["code", "industry"]:
        if column in pred.columns:
            pred[column] = pred[column].astype(str)
    pred["experiment_name"] = exp_name
    pred["window"] = int(cfg["window"])
    pred["horizon"] = int(cfg["horizon"])
    pred["loss_name"] = options.loss
    pred["fold_id"] = int(options.fold_id)
    pred["seed"] = int(options.seed)
    pred["pred_score_raw"] = prediction_scores
    pred["pred_probability"] = 1.0 / (1.0 + np.exp(-np.clip(prediction_scores, -30.0, 30.0)))
    atomic_parquet(pred, paths["member"])

    paths["model"].parent.mkdir(parents=True, exist_ok=True)
    model_tmp = paths["model"].with_suffix(paths["model"].suffix + ".tmp")
    torch.save(
        {
            "state_dict": model.state_dict(),
            "metadata": {
                "version": "1.3.1",
                "loss_name": options.loss,
                "experiment_name": exp_name,
                "fold_id": options.fold_id,
                "seed": options.seed,
                "best_epoch": best_epoch,
                "best_valid_objective": best_objective,
                "options": asdict(options),
                "split": split_meta,
            },
        },
        model_tmp,
    )
    os.replace(model_tmp, paths["model"])
    done_payload = {
        "version": "1.3.1",
        "loss_name": options.loss,
        "experiment_name": exp_name,
        "fold_id": options.fold_id,
        "seed": options.seed,
        "best_epoch": best_epoch,
        "best_valid_objective": best_objective,
        "test_objective": test_metrics["objective"],
        "test_rows": len(prediction_indices),
        "test_wait_seconds": test_metrics["wait_seconds"],
        "test_cuda_peak_bytes": test_metrics["cuda_peak_bytes"],
        "train_rows": len(train_idx),
        "valid_rows": len(valid_idx),
        "started_at": started,
        "completed_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "model_path": str(paths["model"]),
        "member_prediction_path": str(paths["member"]),
        "training_log_path": str(paths["log"]),
        "split": split_meta,
        "options": asdict(options),
    }
    atomic_json(paths["done"], done_payload)

    del train_loader, valid_loader, test_loader, train_ds, valid_ds, test_ds, model, optimizer, scheduler, scaler
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return done_payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="v1.3.1 purged five-fold CNN training")
    parser.add_argument("--data-dir", default=None, help="Versioned data root (overrides IMAGE_TREND_DATA_DIR)")
    parser.add_argument("--output-dir", default=None, help="Versioned output root (overrides IMAGE_TREND_OUTPUT_DIR)")
    parser.add_argument("--loss", choices=LOSS_CHOICES, required=True)
    parser.add_argument("--fold-id", type=int, required=True, choices=range(1, 6))
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--experiments", default=None)
    parser.add_argument("--purge-days", type=int, default=DEFAULT_PURGE_DAYS)
    parser.add_argument("--micro-batch-size", type=int, default=DEFAULT_MICRO_BATCH_SIZE)
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--warmup-epochs", type=int, default=DEFAULT_WARMUP_EPOCHS)
    parser.add_argument("--patience", type=int, default=DEFAULT_PATIENCE)
    parser.add_argument("--min-delta", type=float, default=DEFAULT_MIN_DELTA)
    parser.add_argument("--ic-weight", type=float, default=DEFAULT_IC_WEIGHT)
    parser.add_argument("--huber-beta", type=float, default=DEFAULT_HUBER_BETA)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--prefetch-factor", type=int, default=2)
    parser.add_argument("--shard-cache-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=3e-5)
    parser.add_argument("--fc-dropout", type=float, default=0.20)
    parser.add_argument("--spatial-dropout", type=float, default=0.0)
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--no-tf32", action="store_true")
    parser.add_argument("--no-pin-memory", action="store_true")
    parser.add_argument("--no-persistent-workers", action="store_true")
    parser.add_argument("--channels-last", action="store_true")
    parser.add_argument("--log-interval", type=int, default=200)
    parser.add_argument("--smoke-dates", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    apply_version_roots(args.data_dir, args.output_dir)
    if args.epochs <= 0 or args.patience <= 0 or args.min_delta < 0:
        raise ValueError("epochs and patience must be positive; min_delta must be non-negative")
    options = TrainOptions(
        loss=args.loss,
        fold_id=int(args.fold_id),
        seed=int(args.seed),
        purge_days=int(args.purge_days),
        micro_batch_size=int(args.micro_batch_size),
        epochs=int(args.epochs),
        warmup_epochs=int(args.warmup_epochs),
        patience=int(args.patience),
        min_delta=float(args.min_delta),
        ic_weight=float(args.ic_weight),
        huber_beta=float(args.huber_beta),
        batch_workers=int(args.workers),
        prefetch_factor=int(args.prefetch_factor),
        shard_cache_size=int(args.shard_cache_size),
        lr=float(args.lr),
        weight_decay=float(args.weight_decay),
        fc_dropout=float(args.fc_dropout),
        spatial_dropout=float(args.spatial_dropout),
        amp=not bool(args.no_amp),
        tf32=not bool(args.no_tf32),
        pin_memory=not bool(args.no_pin_memory),
        persistent_workers=not bool(args.no_persistent_workers),
        channels_last=bool(args.channels_last),
        log_interval=int(args.log_interval),
        smoke_dates=args.smoke_dates,
    )
    configure_torch(options.seed, options.tf32)
    device = LEGACY.get_device()
    calendar = load_master_calendar()
    fold_ranges = make_fold_ranges(calendar, 5)
    fold_range = fold_ranges[int(options.fold_id) - 1]
    experiments = LEGACY.selected_experiments(args.experiments, None)
    grouped: Dict[int, Dict[str, Mapping[str, Any]]] = {}
    for name, cfg in experiments.items():
        grouped.setdefault(int(cfg["window"]), {})[name] = cfg

    results: List[Dict[str, Any]] = []
    for window, window_experiments in sorted(grouped.items()):
        image_paths, meta_window, image_shape = LEGACY.load_image_window(window, window_experiments)
        for exp_name, cfg in window_experiments.items():
            results.append(
                train_one_experiment(
                    exp_name,
                    cfg,
                    image_paths,
                    image_shape,
                    meta_window,
                    calendar,
                    fold_range,
                    options,
                    device,
                )
            )
        del meta_window, image_paths
        gc.collect()
    print(json.dumps(results, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
