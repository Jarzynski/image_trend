# -*- coding: utf-8 -*-
"""
04_train_logistic.py

Purpose
-------
Train logistic regression baselines using traditional price-volume features.

Input
-----
data/features/baseline_features.parquet

Output
------
outputs/predictions/pred_{experiment_name}_logistic.parquet

Notes
-----
This is a compact baseline. It is not meant to be the final optimized model.
It is mainly used to answer:

1. Are simple price-volume features predictive?
2. How much incremental value does CNN add?
"""

import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score, brier_score_loss

from config import (
    BASELINE_FEATURE_PATH,
    PRED_DIR,
    TRAIN_END,
    VALID_START,
    VALID_END,
    TEST_START,
    EXPERIMENTS,
)


FEATURE_COLS = [
    "ret_1d",
    "ret_3d",
    "ret_5d",
    "ret_10d",
    "ret_20d",
    "ret_60d",
    "reversal_5d",
    "reversal_10d",
    "momentum_20d",
    "momentum_60d",
    "ma5_gap",
    "ma10_gap",
    "ma20_gap",
    "ma30_gap",
    "ma60_gap",
    "position_5d",
    "position_20d",
    "position_60d",
    "vol_20d",
    "vol_60d",
    "amount_change_20d",
    "turnover_change_20d",
    "log_float_mktcap",
]


def get_split_masks(df):
    """
    Fixed time split.
    """
    train_mask = df["date"] <= TRAIN_END
    valid_mask = (df["date"] >= VALID_START) & (df["date"] <= VALID_END)
    test_mask = df["date"] >= TEST_START
    return train_mask, valid_mask, test_mask


def calc_binary_metrics(y, prob):
    if len(np.unique(y)) < 2:
        auc = np.nan
    else:
        auc = roc_auc_score(y, prob)

    acc = accuracy_score(y, prob > 0.5)
    brier = brier_score_loss(y, prob)
    return auc, acc, brier


def train_one_experiment(df, experiment_name, cfg):
    """
    Train logistic regression for one configured experiment.
    """
    horizon = cfg["horizon"]
    window = cfg["window"]
    ma_col = cfg.get("ma_col")

    label_col = f"label_{horizon}d"
    future_ret_col = f"future_ret_{horizon}d"

    use_cols = [c for c in FEATURE_COLS if c in df.columns]

    sample_mask = (
        (df["is_tradable"] == 1)
        & df[label_col].notna()
        & df[future_ret_col].notna()
    )

    # Make matrix baselines respect each experiment's minimum history roughly.
    # The exact image universe is enforced in 03 via rolling windows.
    if ma_col in df.columns:
        sample_mask &= df[ma_col].notna()

    sub = df[sample_mask].copy()

    train_mask, valid_mask, test_mask = get_split_masks(sub)

    X_train = sub.loc[train_mask, use_cols]
    y_train = sub.loc[train_mask, label_col].astype(int)

    X_valid = sub.loc[valid_mask, use_cols]
    y_valid = sub.loc[valid_mask, label_col].astype(int)

    X_test = sub.loc[test_mask, use_cols]
    y_test = sub.loc[test_mask, label_col].astype(int)

    if len(y_train) == 0 or len(y_valid) == 0 or len(y_test) == 0:
        raise RuntimeError(
            f"{experiment_name} has an empty split: "
            f"train={len(y_train)}, valid={len(y_valid)}, test={len(y_test)}"
        )

    model = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("logit", LogisticRegression(
            max_iter=1000,
            class_weight=None,
            n_jobs=None,
        )),
    ])

    print(f"Training logistic model for {experiment_name}...")
    model.fit(X_train, y_train)

    for split_name, X, y in [
        ("valid", X_valid, y_valid),
        ("test", X_test, y_test),
    ]:
        prob = model.predict_proba(X)[:, 1]
        auc, acc, brier = calc_binary_metrics(y, prob)
        print(f"{experiment_name} {split_name}: AUC={auc:.4f}, ACC={acc:.4f}, Brier={brier:.4f}")

    # Save test predictions.
    test_sub = sub.loc[test_mask, [
        "date", "code", "industry",
        future_ret_col, label_col,
        "is_tradable", "is_limit_up",
        "amount", "float_mktcap",
    ]].copy()

    test_prob = model.predict_proba(X_test)[:, 1]

    pred = test_sub.rename(columns={
        future_ret_col: "future_ret",
        label_col: "label",
    })
    pred["experiment_name"] = experiment_name
    pred["window"] = window
    pred["horizon"] = horizon
    pred["model_name"] = "LogisticRaw"
    pred["pred_prob"] = test_prob

    # Cross-sectional rank and decile are assigned later in backtest script.
    out_path = PRED_DIR / f"pred_{experiment_name.lower()}_logistic.parquet"
    pred.to_parquet(out_path, index=False)

    print(f"Saved predictions to: {out_path}")


def main():
    print(f"Reading features: {BASELINE_FEATURE_PATH}")
    df = pd.read_parquet(BASELINE_FEATURE_PATH)
    df["date"] = pd.to_datetime(df["date"])

    for experiment_name, cfg in EXPERIMENTS.items():
        train_one_experiment(df, experiment_name, cfg)


if __name__ == "__main__":
    main()
