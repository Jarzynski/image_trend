# -*- coding: utf-8 -*-
"""Data and artifact QA for the isolated Image Trend v1.3.1 rebuild.

The checks are intentionally read-only.  They validate year coverage, date/code
keys, feature schemas, image/meta alignment and the expected tail of incomplete
forward-return labels.  When ``--old-data-dir`` is supplied, a deterministic
sample of shared image keys is compared against the historical data directory.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from config import END_DATE, EXPERIMENTS, IMAGE_SHARD_SIZE, START_DATE, image_dir_for_window


EXPECTED_YEARS = tuple(range(2009, 2025))
KEY_COLUMNS = ["date", "code"]


def parquet_parts(root: Path, pattern: str = "year=*/*.parquet") -> List[Path]:
    return sorted(root.glob(pattern))


def _key_hashes(frame: pd.DataFrame) -> np.ndarray:
    return pd.util.hash_pandas_object(frame[KEY_COLUMNS], index=False).to_numpy(dtype=np.uint64)


def qa_year_dataset(root: Path, name: str, required: Sequence[str]) -> Dict[str, Any]:
    parts = parquet_parts(root)
    if not parts:
        raise RuntimeError(f"{name}: no parquet partitions under {root}")
    years: Dict[int, Dict[str, Any]] = {}
    seen_by_year: Dict[int, set[int]] = {}
    for path in parts:
        year_text = path.parent.name.split("=", 1)[-1]
        try:
            year = int(year_text)
        except ValueError as exc:
            raise RuntimeError(f"{name}: invalid year partition {path.parent}") from exc
        frame = pd.read_parquet(path)
        missing = sorted(set(required) - set(frame.columns))
        if missing:
            raise RuntimeError(f"{name}: {path} missing columns {missing}")
        frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
        if frame[KEY_COLUMNS].isna().any().any():
            raise RuntimeError(f"{name}: null date/code key in {path}")
        frame["code"] = frame["code"].astype(str).str.zfill(6)
        if frame.duplicated(KEY_COLUMNS).any():
            raise RuntimeError(f"{name}: duplicate date/code key within {path}")
        hashes = _key_hashes(frame)
        seen = seen_by_year.setdefault(year, set())
        overlap = set(hashes.tolist()).intersection(seen)
        if overlap:
            raise RuntimeError(f"{name}: duplicate date/code key across parts in year={year}")
        seen.update(hashes.tolist())
        if year not in years:
            years[year] = {"rows": 0, "parts": 0, "min_date": None, "max_date": None, "codes": set()}
        item = years[year]
        item["rows"] += int(len(frame))
        item["parts"] += 1
        item["codes"].update(frame["code"].unique().tolist())
        min_date = frame["date"].min()
        max_date = frame["date"].max()
        item["min_date"] = min_date if item["min_date"] is None else min(item["min_date"], min_date)
        item["max_date"] = max_date if item["max_date"] is None else max(item["max_date"], max_date)

    actual_years = tuple(sorted(years))
    missing_years = sorted(set(EXPECTED_YEARS) - set(actual_years))
    extra_years = sorted(set(actual_years) - set(EXPECTED_YEARS))
    if missing_years or extra_years:
        raise RuntimeError(f"{name}: year coverage mismatch; missing={missing_years}, extra={extra_years}")
    result_years = {}
    for year, item in sorted(years.items()):
        result_years[str(year)] = {
            "rows": item["rows"],
            "parts": item["parts"],
            "codes": len(item["codes"]),
            "min_date": pd.Timestamp(item["min_date"]).strftime("%Y-%m-%d"),
            "max_date": pd.Timestamp(item["max_date"]).strftime("%Y-%m-%d"),
        }
    return {"root": str(root), "parts": len(parts), "years": result_years}


def _read_image_metadata(window_dir: Path) -> pd.DataFrame:
    frames = []
    for shard_id, image_path in enumerate(sorted(window_dir.glob("shard_*/images.npy"))):
        meta_path = image_path.parent / "meta.parquet"
        meta = pd.read_parquet(meta_path)
        if "shard_id" not in meta.columns:
            meta["shard_id"] = shard_id
        if "local_index" not in meta.columns:
            meta["local_index"] = np.arange(len(meta), dtype=np.int32)
        meta["_image_path"] = str(image_path)
        frames.append(meta)
    if not frames:
        raise RuntimeError(f"No image shards found under {window_dir}")
    return pd.concat(frames, ignore_index=True)


def qa_images(data_root: Path, sample_size: int) -> Dict[str, Any]:
    image_root = data_root / "images"
    result: Dict[str, Any] = {"root": str(image_root), "windows": {}}
    for window in sorted({int(cfg["window"]) for cfg in EXPERIMENTS.values()}):
        window_dir = image_root / f"window_{window}"
        meta = _read_image_metadata(window_dir)
        row_count = 0
        shapes = set()
        pixel_values = set()
        for image_path in sorted(window_dir.glob("shard_*/images.npy")):
            images = np.load(image_path, mmap_mode="r")
            shape = tuple(int(x) for x in images.shape)
            if len(shape) != 4 or shape[3] != 1:
                raise RuntimeError(f"window={window} invalid image shape {shape} in {image_path}")
            shard_meta = pd.read_parquet(image_path.parent / "meta.parquet")
            if len(shard_meta) != shape[0]:
                raise RuntimeError(f"window={window} image/meta row mismatch in {image_path.parent}")
            shapes.add(shape[1:])
            row_count += shape[0]
            sample = np.asarray(images[: min(shape[0], max(1, sample_size))])
            pixel_values.update(np.unique(sample).astype(int).tolist())
            if not set(np.unique(sample).tolist()).issubset({0, 255}):
                raise RuntimeError(f"window={window} contains non-binary sampled pixels in {image_path}")
        if row_count != len(meta):
            raise RuntimeError(f"window={window} total image/meta row mismatch: {row_count} != {len(meta)}")
        if meta.duplicated(KEY_COLUMNS).any():
            raise RuntimeError(f"window={window} duplicate date/code metadata key")
        meta["date"] = pd.to_datetime(meta["date"], errors="coerce")
        if meta["date"].min() < pd.Timestamp(START_DATE) or meta["date"].max() > pd.Timestamp(END_DATE):
            raise RuntimeError(f"window={window} date range falls outside {START_DATE}..{END_DATE}")
        label_columns = sorted(c for c in meta.columns if c.startswith("label_") and c.endswith("d"))
        future_columns = sorted(c for c in meta.columns if c.startswith("future_ret_") and c.endswith("d"))
        if not label_columns or not future_columns:
            raise RuntimeError(f"window={window} missing label/future-return columns")
        label_stats = {}
        for column in label_columns:
            horizon = int(column.removeprefix("label_").removesuffix("d"))
            future = f"future_ret_{horizon}d"
            valid = meta[column].notna() & meta[future].notna()
            invalid_pair = meta[column].notna() ^ meta[future].notna()
            if invalid_pair.any():
                raise RuntimeError(f"window={window} mismatched {column}/{future} null pattern")
            # A complete forward label cannot exist after END_DATE-horizon; this
            # catches accidental end-of-sample fabrication.
            late_valid = valid & (meta["date"] > pd.Timestamp(END_DATE) - pd.Timedelta(days=horizon + 5))
            # Calendar gaps mean the exact date subtraction is only a guardrail;
            # report rather than fail if the source's last valid trading date is
            # near the calendar boundary.
            label_stats[column] = {"valid_rows": int(valid.sum()), "missing_rows": int((~valid).sum()), "late_valid_rows": int(late_valid.sum())}
        result["windows"][str(window)] = {
            "rows": int(row_count),
            "shards": len(list(window_dir.glob("shard_*/images.npy"))),
            "shapes": sorted([list(shape) for shape in shapes]),
            "sampled_pixel_values": sorted(pixel_values),
            "min_date": meta["date"].min().strftime("%Y-%m-%d"),
            "max_date": meta["date"].max().strftime("%Y-%m-%d"),
            "labels": label_stats,
        }
    return result


def compare_images(new_root: Path, old_root: Path, sample_size: int, seed: int = 20260828) -> Dict[str, Any]:
    rng = np.random.default_rng(seed)
    result: Dict[str, Any] = {"old_root": str(old_root), "new_root": str(new_root), "windows": {}}
    for window in sorted({int(cfg["window"]) for cfg in EXPERIMENTS.values()}):
        new_meta = _read_image_metadata(new_root / "images" / f"window_{window}")
        old_meta = _read_image_metadata(old_root / "images" / f"window_{window}")
        new_meta["date"] = pd.to_datetime(new_meta["date"])
        old_meta["date"] = pd.to_datetime(old_meta["date"])
        common = new_meta.merge(old_meta, on=KEY_COLUMNS, suffixes=("_new", "_old"), how="inner")
        if common.empty:
            raise RuntimeError(f"window={window} has no common keys for old/new image comparison")
        n = min(int(sample_size), len(common))
        sample = common.iloc[np.sort(rng.choice(len(common), size=n, replace=False))]
        pixel_mismatches = 0
        max_pixel_diff = 0
        label_mismatches = 0
        for _, row in sample.iterrows():
            new_arr = np.load(row["_image_path_new"], mmap_mode="r")[int(row["local_index_new"])]
            old_arr = np.load(row["_image_path_old"], mmap_mode="r")[int(row["local_index_old"])]
            diff = np.abs(new_arr.astype(np.int16) - old_arr.astype(np.int16))
            pixel_mismatches += int(np.count_nonzero(diff))
            max_pixel_diff = max(max_pixel_diff, int(diff.max(initial=0)))
            for horizon in sorted({int(cfg["horizon"]) for cfg in EXPERIMENTS.values() if int(cfg["window"]) == window}):
                for prefix in ("label", "future_ret"):
                    new_col = f"{prefix}_{horizon}d_new"
                    old_col = f"{prefix}_{horizon}d_old"
                    if new_col in row.index and old_col in row.index:
                        a, b = row[new_col], row[old_col]
                        if (pd.isna(a) != pd.isna(b)) or (pd.notna(a) and not np.isclose(float(a), float(b), atol=1e-6, rtol=1e-6)):
                            label_mismatches += 1
        result["windows"][str(window)] = {
            "common_keys": int(len(common)),
            "sampled_keys": int(n),
            "pixel_mismatches": int(pixel_mismatches),
            "max_pixel_abs_diff": int(max_pixel_diff),
            "label_or_return_mismatches": int(label_mismatches),
        }
    return result


def qa_predictions(output_root: Path) -> Dict[str, Any]:
    pred_root = output_root / "predictions"
    paths = sorted(pred_root.glob("pred_*_jiang_cnn2d_v131_*_k5s4.parquet"))
    if len(paths) != 15:
        raise RuntimeError(f"Expected 15 v1.3.1 aggregate predictions, found {len(paths)} under {pred_root}")
    rows = []
    for path in paths:
        frame = pd.read_parquet(path)
        for required in ("date", "code", "pred_score", "pred_prob", "ensemble_member_count"):
            if required not in frame.columns:
                raise RuntimeError(f"{path} missing {required}")
        if frame.duplicated(KEY_COLUMNS).any():
            raise RuntimeError(f"{path} contains duplicate prediction keys")
        if not np.isfinite(pd.to_numeric(frame["pred_score"], errors="coerce")).all():
            raise RuntimeError(f"{path} contains non-finite pred_score")
        if set(frame["ensemble_member_count"].unique()) != {20}:
            raise RuntimeError(f"{path} does not report exactly 20 ensemble members")
        rows.append({"path": str(path), "rows": int(len(frame)), "dates": int(frame["date"].nunique())})
    return {"files": rows, "count": len(rows)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="QA the isolated Image Trend v1.3.1 artifacts")
    parser.add_argument("--data-root", default=None, help="Versioned data root; defaults to config.DATA_DIR")
    parser.add_argument("--output-root", default=None, help="Versioned output root; defaults to config.OUTPUT_DIR")
    parser.add_argument("--old-data-dir", default=None, help="Historical data root for sampled image comparison")
    parser.add_argument("--sample-size", type=int, default=32)
    parser.add_argument("--check-predictions", action="store_true")
    parser.add_argument("--output", default=None, help="JSON report path")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    from config import DATA_DIR, OUTPUT_DIR

    data_root = Path(args.data_root).expanduser() if args.data_root else Path(DATA_DIR)
    output_root = Path(args.output_root).expanduser() if args.output_root else Path(OUTPUT_DIR)
    panel_required = ["date", "code", "open_raw", "high_raw", "low_raw", "close_raw", "volume"]
    feature_required = ["date", "code", "future_ret_5d", "future_ret_20d", "label_5d", "label_20d"]
    report: Dict[str, Any] = {
        "version": "1.3.1",
        "data_root": str(data_root),
        "output_root": str(output_root),
        "panel_by_year": qa_year_dataset(data_root / "processed" / "panel_by_year", "panel_by_year", panel_required),
        "features_by_year": qa_year_dataset(data_root / "features" / "features_by_year", "features_by_year", feature_required),
        "images": qa_images(data_root, int(args.sample_size)),
    }
    if args.old_data_dir:
        report["old_comparison"] = compare_images(data_root, Path(args.old_data_dir).expanduser(), int(args.sample_size))
    if args.check_predictions:
        report["predictions"] = qa_predictions(output_root)
    output_path = Path(args.output).expanduser() if args.output else output_root / "tables" / "qa" / "data_qa_v131.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2, default=str), flush=True)
    print(f"Saved QA report: {output_path}", flush=True)


if __name__ == "__main__":
    main()
