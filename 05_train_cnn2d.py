# -*- coding: utf-8 -*-
"""
05_train_cnn2d.py

Purpose
-------
Train Jiang, Kelly, and Xiu-style 2D CNNs on binary price images.

Inputs
------
data/images/images_{experiment_name}.npy
data/images/meta_{experiment_name}.parquet

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
    EXPERIMENTS,
    image_path_for_experiment,
    meta_path_for_experiment,
)


# ============================================================
# 1. Dataset
# ============================================================

class ImageDataset(Dataset):
    """
    Simple torch Dataset for price images.

    images: numpy array or memmap [N, H, W, C]
    labels: numpy array [N]
    indices: row indices for this split
    """
    def __init__(self, images, labels, indices):
        self.images = images
        self.labels = labels.astype(np.float32)
        self.indices = np.asarray(indices, dtype=np.int64)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        real_idx = self.indices[idx]
        x = self.images[real_idx].astype(np.float32)

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


def get_split_masks(meta):
    """
    Fixed time split.
    """
    date = pd.to_datetime(meta["date"])

    train_mask = date <= TRAIN_END
    valid_mask = (date >= VALID_START) & (date <= VALID_END)
    test_mask = date >= TEST_START

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


def train_one_experiment(exp_name, cfg, image_path, meta_path, n_epochs=50, batch_size=128, lr=1e-5):
    """
    Train CNN for one experiment.
    """
    print(f"Loading images: {image_path}")
    images = np.load(image_path, mmap_mode="r")
    image_height, image_width = images.shape[1], images.shape[2]

    print(f"Loading metadata: {meta_path}")
    meta = pd.read_parquet(meta_path)
    meta["date"] = pd.to_datetime(meta["date"])

    labels = meta["label"].values.astype(np.float32)

    train_mask, valid_mask, test_mask = get_split_masks(meta)

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

    train_loader = DataLoader(
        ImageDataset(images, labels, train_idx),
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=pin_memory,
    )

    valid_loader = DataLoader(
        ImageDataset(images, labels, valid_idx),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=pin_memory,
    )

    test_loader = DataLoader(
        ImageDataset(images, labels, test_idx),
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

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
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
    pred = meta.loc[test_mask, [
        "date", "code", "industry",
        "future_ret", "label",
        "amount", "float_mktcap", "is_limit_up",
    ]].copy()

    pred["experiment_name"] = exp_name
    if "window" in meta.columns:
        pred["window"] = meta.loc[test_mask, "window"].values
    if "horizon" in meta.columns:
        pred["horizon"] = meta.loc[test_mask, "horizon"].values
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
            image_path_for_experiment(exp_name),
            meta_path_for_experiment(exp_name),
        )


if __name__ == "__main__":
    main()
