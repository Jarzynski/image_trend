# -*- coding: utf-8 -*-
"""Aggregate v1.3.1 CNN fold/seed predictions into one daily signal."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

import config as CONFIG
from config import EXPERIMENTS, PRED_DIR, TABLE_DIR


EXPECTED_FOLDS = 5
EXPECTED_SEEDS = (42, 43, 44, 45)
KEY_COLUMNS = ["date", "code"]
MEMBER_COLUMNS = [
    "date",
    "code",
    "industry",
    "future_ret",
    "label",
    "amount",
    "float_mktcap",
    "is_low_volume_limit_up",
    "is_low_volume_limit_down",
    "experiment_name",
    "window",
    "horizon",
    "loss_name",
    "fold_id",
    "seed",
    "pred_score_raw",
    "pred_probability",
]


def atomic_parquet(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    frame.to_parquet(tmp, index=False)
    os.replace(tmp, path)


def member_paths(loss: str, exp_name: str) -> List[Path]:
    root = Path(PRED_DIR) / "members" / str(loss) / exp_name.lower()
    return sorted(root.glob("fold*_seed*.parquet"))


def expected_member_names(paths: Sequence[Path]) -> set[tuple[int, int]]:
    result: set[tuple[int, int]] = set()
    for path in paths:
        name = path.stem
        try:
            fold_part, seed_part = name.split("_", 1)
            fold = int(fold_part.removeprefix("fold"))
            seed = int(seed_part.removeprefix("seed"))
        except Exception as exc:
            raise RuntimeError(f"Invalid member prediction filename: {path.name}") from exc
        result.add((fold, seed))
    return result


def normalize_member_scores(frame: pd.DataFrame) -> pd.Series:
    score = pd.to_numeric(frame["pred_score_raw"], errors="coerce").astype(np.float64)
    grouped = frame.assign(_score=score).groupby("date", sort=False)["_score"]
    mean = grouped.transform("mean")
    centered = score - mean
    std = centered.groupby(frame["date"], sort=False).transform(
        lambda x: float(np.sqrt(np.mean(np.square(x.to_numpy(dtype=np.float64)))))
    )
    if bool((std <= 1e-12).any()):
        bad_dates = frame.loc[std <= 1e-12, "date"].drop_duplicates().head(5).tolist()
        raise RuntimeError(f"Member has constant score on one or more dates: {bad_dates}")
    result = centered / std
    if not bool(np.isfinite(result.to_numpy()).all()):
        raise RuntimeError("Member prediction contains non-finite standardized scores")
    return result.astype(np.float32)


def validate_keys(base: pd.DataFrame, current: pd.DataFrame, path: Path) -> None:
    base_keys = base[KEY_COLUMNS].reset_index(drop=True)
    current_keys = current[KEY_COLUMNS].reset_index(drop=True)
    if not base_keys.equals(current_keys):
        raise RuntimeError(f"Prediction keys/order differ from first member: {path}")


def aggregate_one(loss: str, exp_name: str, expected_members: int = 20) -> Dict[str, object]:
    paths = member_paths(loss, exp_name)
    if len(paths) != int(expected_members):
        raise RuntimeError(
            f"{loss}/{exp_name} expected {expected_members} members, found {len(paths)} under {Path(PRED_DIR) / 'members'}"
        )
    names = expected_member_names(paths)
    expected_names = {(fold, seed) for fold in range(1, EXPECTED_FOLDS + 1) for seed in EXPECTED_SEEDS}
    if names != expected_names:
        raise RuntimeError(f"{loss}/{exp_name} member set mismatch: {sorted(names)}")

    base: pd.DataFrame | None = None
    score_sum: Optional[np.ndarray] = None
    probability_sum: Optional[np.ndarray] = None
    for path in paths:
        frame = pd.read_parquet(path)
        missing = sorted(set(KEY_COLUMNS + ["pred_score_raw", "pred_probability"]) - set(frame.columns))
        if missing:
            raise RuntimeError(f"{path} missing columns: {missing}")
        frame["date"] = pd.to_datetime(frame["date"])
        frame = frame.sort_values(KEY_COLUMNS, kind="mergesort").reset_index(drop=True)
        if frame.duplicated(KEY_COLUMNS).any():
            raise RuntimeError(f"{path} contains duplicate date/code keys")
        if base is None:
            keep = [column for column in MEMBER_COLUMNS if column in frame.columns]
            base = frame[keep].copy()
            base["date"] = pd.to_datetime(base["date"])
            score_sum = np.zeros(len(base), dtype=np.float64)
            probability_sum = np.zeros(len(base), dtype=np.float64)
        else:
            validate_keys(base, frame, path)
        score_sum += normalize_member_scores(frame).to_numpy(dtype=np.float64, copy=False)
        probability_sum += pd.to_numeric(frame["pred_probability"], errors="coerce").to_numpy(dtype=np.float64)

    if base is None or score_sum is None or probability_sum is None:
        raise RuntimeError(f"No members found for {loss}/{exp_name}")
    if not np.isfinite(score_sum).all() or not np.isfinite(probability_sum).all():
        raise RuntimeError(f"Non-finite aggregate input for {loss}/{exp_name}")

    model_name = f"JiangCNN2D_v131_{loss}_k5s4"
    base["pred_score"] = (score_sum / float(expected_members)).astype(np.float32)
    # Keep the historical column so v1.2.x backtest code can consume the new
    # signal.  v1.3.1-aware backtests prefer pred_score explicitly.
    base["pred_prob"] = base["pred_score"]
    base["pred_probability_mean"] = (probability_sum / float(expected_members)).astype(np.float32)
    base["model_name"] = model_name
    base["ensemble_member_count"] = int(expected_members)
    base["ensemble_fold_count"] = EXPECTED_FOLDS
    base["ensemble_seed_count"] = len(EXPECTED_SEEDS)
    base["loss_name"] = str(loss)
    base = base.sort_values(KEY_COLUMNS, kind="mergesort").reset_index(drop=True)

    output_path = Path(PRED_DIR) / f"pred_{exp_name.lower()}_jiang_cnn2d_v131_{loss}_k5s4.parquet"
    atomic_parquet(base, output_path)
    return {
        "loss_name": loss,
        "experiment_name": exp_name,
        "model_name": model_name,
        "member_count": int(expected_members),
        "rows": int(len(base)),
        "n_dates": int(base["date"].nunique()),
        "date_start": str(base["date"].min().date()),
        "date_end": str(base["date"].max().date()),
        "score_mean": float(base["pred_score"].mean()),
        "score_std": float(base["pred_score"].std(ddof=0)),
        "prediction_path": str(output_path),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate v1.3.1 fold/seed predictions")
    parser.add_argument("--output-dir", default=None, help="Versioned output root (overrides IMAGE_TREND_OUTPUT_DIR)")
    parser.add_argument("--pred-dir", default=None, help="Prediction directory override")
    parser.add_argument("--table-dir", default=None, help="Ensemble summary table directory override")
    parser.add_argument("--loss", choices=("bce", "huber", "huber_ic"), required=True)
    parser.add_argument("--experiments", default=None, help="Comma-separated experiment names")
    parser.add_argument("--expected-members", type=int, default=20)
    return parser.parse_args()


def main() -> None:
    global PRED_DIR, TABLE_DIR
    args = parse_args()
    if args.output_dir:
        output_root = Path(args.output_dir).expanduser().resolve()
        PRED_DIR = output_root / "predictions"
        TABLE_DIR = output_root / "tables"
    if args.pred_dir:
        PRED_DIR = Path(args.pred_dir).expanduser().resolve()
    if args.table_dir:
        TABLE_DIR = Path(args.table_dir).expanduser().resolve()
    PRED_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    if args.experiments:
        names = [x.strip().upper() for x in args.experiments.split(",") if x.strip()]
        unknown = sorted(set(names) - set(EXPERIMENTS))
        if unknown:
            raise RuntimeError(f"Unknown experiments: {unknown}")
    else:
        names = list(EXPERIMENTS)
    rows = [aggregate_one(args.loss, name, args.expected_members) for name in names]
    summary_path = Path(TABLE_DIR) / "ensemble" / f"cnn_ensemble_summary_v131_{args.loss}.csv"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(summary_path, index=False, encoding="utf-8-sig")
    print(pd.DataFrame(rows).to_string(index=False), flush=True)
    print(f"Saved ensemble summary: {summary_path}", flush=True)


if __name__ == "__main__":
    main()
