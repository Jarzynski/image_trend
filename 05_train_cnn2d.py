# -*- coding: utf-8 -*-
"""
05_train_cnn2d.py

Purpose
-------
Train Jiang, Kelly, and Xiu-style 2D CNNs on binary price images.

Inputs
------
data/images/window_{window}/shard_*/images.npy
data/images/window_{window}/shard_*/meta.parquet

Outputs
-------
outputs/models/cnn2d_{experiment_name}.pt
outputs/predictions/pred_{experiment_name}_cnn2d.parquet

Notes
-----
The architecture follows the paper's window-dependent design:
2/3/4 CNN building blocks for 5/20/60-day images.
"""

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from sklearn.metrics import roc_auc_score, accuracy_score, brier_score_loss

from config import (
    PRED_DIR,
    MODEL_DIR,
    TRAIN_END,
    VALID_START,
    VALID_END,
    TEST_START,
    EMBARGO_DAYS_BY_HORIZON,
    RANDOM_SEED,
    CNN_WEIGHT_DECAY,
    EXPERIMENTS,
    image_dir_for_window,
)


# ============================================================
# 1. Dataset
# ============================================================

class ImageDataset(Dataset):
    """
    Simple torch Dataset for price images.

    image_shards: list of numpy arrays or memmaps [N, H, W, C]
    labels: numpy array [N]
    shard_ids: numpy array [N]
    local_indices: numpy array [N]
    indices: row indices for this split
    """
    def __init__(self, image_shards, labels, shard_ids, local_indices, indices):
        self.image_shards = image_shards
        self.labels = labels.astype(np.float32)
        self.shard_ids = np.asarray(shard_ids, dtype=np.int64)
        self.local_indices = np.asarray(local_indices, dtype=np.int64)
        self.indices = np.asarray(indices, dtype=np.int64)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        real_idx = self.indices[idx]
        shard_id = self.shard_ids[real_idx]
        local_idx = self.local_indices[real_idx]
        x = self.image_shards[shard_id][local_idx].astype(np.float32)

        # Convert NHWC to CHW for PyTorch.
        x = np.ascontiguousarray(np.transpose(x, (2, 0, 1)))

        y = self.labels[real_idx]
        return torch.from_numpy(x), torch.tensor(y)


# ============================================================
# 2. Model
# ============================================================

class JiangCNNBlock(nn.Module):
    """
    One Jiang et al. CNN building block:
    Conv -> BatchNorm -> LeakyReLU -> MaxPool.
    """
    def __init__(self, in_channels, out_channels, stride=(1, 1), dilation=(1, 1)):
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

    def __init__(self, window, image_height, image_width, in_channels=1):
        super().__init__()

        if window not in self.WINDOW_CONFIG:
            raise ValueError(f"Unsupported CNN image window: {window}")

        cfg = self.WINDOW_CONFIG[window]
        blocks = []
        channels = [64 * (2 ** i) for i in range(cfg["num_blocks"])]

        prev_channels = in_channels
        for i, out_channels in enumerate(channels):
            if i == 0:
                stride = (cfg["first_stride_v"], 1)
                dilation = (cfg["first_dilation_v"], 1)
            else:
                stride = (1, 1)
                dilation = (1, 1)

            blocks.append(
                JiangCNNBlock(
                    in_channels=prev_channels,
                    out_channels=out_channels,
                    stride=stride,
                    dilation=dilation,
                )
            )
            prev_channels = out_channels

        self.features = nn.Sequential(*blocks)

        with torch.no_grad():
            dummy = torch.zeros(1, in_channels, image_height, image_width)
            feature_dim = self.features(dummy).flatten(1).shape[1]

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.50),
            nn.Linear(feature_dim, 1),
        )

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module):
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, x):
        x = self.features(x)
        logit = self.classifier(x).squeeze(-1)
        return logit


# ============================================================
# 3. Utilities
# ============================================================

def get_device():
    """
    Use CUDA if available.
    """
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_random_seed(seed):
    """
    Set the main random seeds used by numpy and torch.
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False


def cutoff_before_boundary(dates, boundary, gap_days):
    """
    Return the first purged date before a split boundary.
    """
    unique_dates = pd.DatetimeIndex(sorted(pd.to_datetime(dates).dropna().unique()))
    boundary = pd.Timestamp(boundary)
    before = unique_dates[unique_dates < boundary]
    if gap_days <= 0:
        return boundary
    if len(before) <= gap_days:
        return pd.Timestamp.min
    return before[-gap_days]


def get_split_masks(meta, horizon):
    """
    Fixed time split with purge/embargo around validation and test boundaries.
    """
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

    return train_mask.values, valid_mask.values, test_mask.values


def evaluate_model(model, loader, device, criterion):
    """
    Return probabilities and labels for a dataloader.
    """
    model.eval()

    probs = []
    labels = []
    total_loss = 0.0
    n_obs = 0

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            logit = model(x)
            loss = criterion(logit, y)
            prob = torch.sigmoid(logit).cpu().numpy()

            probs.append(prob)
            labels.append(y.cpu().numpy())
            total_loss += loss.item() * len(y)
            n_obs += len(y)

    probs = np.concatenate(probs)
    labels = np.concatenate(labels)

    if len(np.unique(labels)) < 2:
        auc = np.nan
    else:
        auc = roc_auc_score(labels, probs)
    acc = accuracy_score(labels, probs > 0.5)
    brier = brier_score_loss(labels, probs)
    avg_loss = total_loss / max(n_obs, 1)

    return probs, labels, auc, acc, brier, avg_loss


def load_image_shards(exp_name, cfg):
    """
    Load all image shards and metadata shards for one experiment's image window.

    Multiple experiments can share the same physical image directory. The
    caller later selects label_{horizon}d/future_ret_{horizon}d for the target
    experiment.
    """
    image_dir = image_dir_for_window(cfg["window"])
    shard_dirs = sorted(image_dir.glob("shard_*"))
    if not shard_dirs:
        raise RuntimeError(f"No image shards found for {exp_name}: {image_dir}")

    image_shards = []
    meta_frames = []

    for expected_shard_id, shard_dir in enumerate(shard_dirs):
        image_path = shard_dir / "images.npy"
        meta_path = shard_dir / "meta.parquet"
        if not image_path.exists() or not meta_path.exists():
            raise RuntimeError(f"Incomplete image shard: {shard_dir}")

        images = np.load(image_path, mmap_mode="r")
        meta = pd.read_parquet(meta_path)

        if "shard_id" not in meta.columns:
            meta["shard_id"] = expected_shard_id
        if "local_index" not in meta.columns:
            meta["local_index"] = np.arange(len(meta), dtype=np.int64)

        if len(meta) != images.shape[0]:
            raise RuntimeError(
                f"Shard row mismatch in {shard_dir}: "
                f"images={images.shape[0]}, meta={len(meta)}"
            )

        meta["shard_id"] = expected_shard_id
        image_shards.append(images)
        meta_frames.append(meta)

    meta = pd.concat(meta_frames, ignore_index=True)
    meta["date"] = pd.to_datetime(meta["date"])
    return image_shards, meta


def select_experiment_label_view(meta, exp_name, cfg):
    """
    Return metadata rows with finite labels for the experiment horizon.
    """
    horizon = int(cfg["horizon"])
    label_col = f"label_{horizon}d"
    future_ret_col = f"future_ret_{horizon}d"
    missing = [c for c in [label_col, future_ret_col] if c not in meta.columns]
    if missing:
        raise RuntimeError(
            f"Image metadata for {exp_name} is missing columns: {missing}. "
            "Rerun 03_make_images.py with this experiment selected."
        )

    valid = meta[label_col].notna() & meta[future_ret_col].notna()
    view = meta.loc[valid].copy().reset_index(drop=True)
    if view.empty:
        raise RuntimeError(f"{exp_name} has no rows with finite {label_col}/{future_ret_col}.")

    view["experiment_name"] = exp_name
    view["horizon"] = horizon
    view["label"] = view[label_col].astype(np.float32)
    view["future_ret"] = view[future_ret_col].astype(np.float32)
    return view


def train_one_experiment(exp_name, cfg, n_epochs=50, batch_size=128, lr=1e-5):
    """
    Train CNN for one experiment.
    """
    set_random_seed(RANDOM_SEED)

    print(f"Loading image shards: {image_dir_for_window(cfg['window'])}")
    image_shards, meta = load_image_shards(exp_name, cfg)
    meta = select_experiment_label_view(meta, exp_name, cfg)
    image_height, image_width = image_shards[0].shape[1], image_shards[0].shape[2]

    labels = meta["label"].values.astype(np.float32)
    shard_ids = meta["shard_id"].values.astype(np.int64)
    local_indices = meta["local_index"].values.astype(np.int64)

    train_mask, valid_mask, test_mask = get_split_masks(meta, cfg["horizon"])

    train_idx = np.flatnonzero(train_mask)
    valid_idx = np.flatnonzero(valid_mask)
    test_idx = np.flatnonzero(test_mask)

    print(exp_name, "train/valid/test sizes:", len(train_idx), len(valid_idx), len(test_idx))

    if len(train_idx) == 0 or len(valid_idx) == 0 or len(test_idx) == 0:
        raise RuntimeError(
            f"{exp_name} has an empty split: "
            f"train={len(train_idx)}, valid={len(valid_idx)}, test={len(test_idx)}"
        )

    device = get_device()
    pin_memory = device.type == "cuda"
    train_generator = torch.Generator()
    train_generator.manual_seed(RANDOM_SEED)

    train_loader = DataLoader(
        ImageDataset(image_shards, labels, shard_ids, local_indices, train_idx),
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=pin_memory,
        generator=train_generator,
    )

    valid_loader = DataLoader(
        ImageDataset(image_shards, labels, shard_ids, local_indices, valid_idx),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=pin_memory,
    )

    test_loader = DataLoader(
        ImageDataset(image_shards, labels, shard_ids, local_indices, test_idx),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=pin_memory,
    )

    model = JiangCNN2D(
        window=cfg["window"],
        image_height=image_height,
        image_width=image_width,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=CNN_WEIGHT_DECAY)
    criterion = nn.BCEWithLogitsLoss()

    best_valid_loss = np.inf
    best_state = None
    patience = 2
    bad_epochs = 0

    print(f"Training JiangCNN2D for {exp_name} on {device}...")

    for epoch in range(1, n_epochs + 1):
        model.train()

        total_loss = 0.0
        n_obs = 0

        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad()
            logit = model(x)
            loss = criterion(logit, y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * len(y)
            n_obs += len(y)

        train_loss = total_loss / max(n_obs, 1)

        valid_prob, valid_y, valid_auc, valid_acc, valid_brier, valid_loss = evaluate_model(
            model, valid_loader, device, criterion
        )

        print(
            f"Epoch {epoch:03d} | "
            f"loss={train_loss:.5f} | "
            f"valid loss={valid_loss:.5f} | "
            f"valid AUC={valid_auc:.4f} | "
            f"valid ACC={valid_acc:.4f} | "
            f"valid Brier={valid_brier:.4f}"
        )

        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1

        if bad_epochs >= patience:
            print("Early stopping triggered.")
            break

    # Restore best model.
    if best_state is None:
        raise RuntimeError(f"{exp_name} did not produce a valid checkpoint.")

    model.load_state_dict(best_state)
    model = model.to(device)

    test_prob, test_y, test_auc, test_acc, test_brier, test_loss = evaluate_model(
        model, test_loader, device, criterion
    )

    print(
        f"{exp_name} TEST | "
        f"loss={test_loss:.5f} | "
        f"AUC={test_auc:.4f} | ACC={test_acc:.4f} | Brier={test_brier:.4f}"
    )

    # Save model.
    model_path = MODEL_DIR / f"jiang_cnn2d_{exp_name.lower()}.pt"
    torch.save(model.state_dict(), model_path)
    print(f"Saved model to: {model_path}")

    # Save test predictions.
    pred_cols = [
        "date", "code", "industry",
        "future_ret", "label",
        "amount", "float_mktcap",
        "is_low_volume_limit_up", "is_low_volume_limit_down",
    ]
    pred_cols = [c for c in pred_cols if c in meta.columns]
    pred = meta.loc[test_mask, pred_cols].copy()

    pred["experiment_name"] = exp_name
    pred["window"] = int(cfg["window"])
    pred["horizon"] = int(cfg["horizon"])
    pred["model_name"] = "JiangCNN2D"
    pred["pred_prob"] = test_prob

    out_path = PRED_DIR / f"pred_{exp_name.lower()}_jiang_cnn2d.parquet"
    pred.to_parquet(out_path, index=False)
    print(f"Saved predictions to: {out_path}")


def main():
    for exp_name, cfg in EXPERIMENTS.items():
        train_one_experiment(
            exp_name,
            cfg,
        )


if __name__ == "__main__":
    main()
