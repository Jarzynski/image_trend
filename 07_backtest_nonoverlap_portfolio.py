# -*- coding: utf-8 -*-
"""
07_backtest_nonoverlap_portfolio.py

Evaluate model predictions with strict non-overlapping decile portfolios and
a self-financing D10-D1 long-short portfolio.

Execution convention
--------------------
A signal formed at date t buys at the next trading day's adjusted open and
holds through the horizon date's adjusted close. For horizon H, this script
only rebalances on signal dates t0, t0 + H, t0 + 2H, ... within each
experiment/model/universe group. It does not create daily overlapping
subportfolios and does not scale target weights by 1/H.

Inputs
------
outputs/predictions/pred_*.parquet
data/features/features_by_year/year=*/part-*.parquet

Outputs
-------
outputs/tables/nonoverlap_portfolio_returns.csv
outputs/tables/nonoverlap_portfolio_turnover.csv
outputs/tables/nonoverlap_portfolio_nav.csv
outputs/tables/nonoverlap_performance_summary.csv
outputs/tables/nonoverlap_cost_sensitivity.csv
outputs/tables/nonoverlap_return_attribution.csv
outputs/tables/nonoverlap_d10_long_only_returns.csv
outputs/tables/nonoverlap_d10_long_only_nav.csv
outputs/tables/nonoverlap_d10_long_only_performance.csv
outputs/tables/nonoverlap_portfolio_holdings.csv
outputs/tables/nonoverlap_holding_overlap.csv
outputs/tables/nonoverlap_holding_overlap_summary.csv
"""

import argparse
import glob
import importlib.util
from collections import defaultdict
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd


BACKTEST_06_PATH = Path(__file__).resolve().with_name("06_backtest_decile.py")
SPEC = importlib.util.spec_from_file_location("backtest_decile_06", BACKTEST_06_PATH)
bt = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(bt)


N_DECILES = bt.N_DECILES
PRED_DIR = bt.PRED_DIR
TABLE_DIR = bt.TABLE_DIR
GROUP_COLS = bt.GROUP_COLS
LONG_SHORT_DECILE = bt.LONG_SHORT_DECILE
LONG_SHORT_PORTFOLIO_NAME = bt.LONG_SHORT_PORTFOLIO_NAME
D10_PORTFOLIO_NAME = f"D{N_DECILES}"


def log(message):
    """
    Print progress immediately in Slurm logs.
    """
    print(f"[07] {message}", flush=True)


bt.log = log


def empty_weight_state():
    return {
        "ids": np.empty(0, dtype=np.int32),
        "weights": np.empty(0, dtype=np.float64),
        "static_missing_weight": 0.0,
        "static_missing_count": 0,
    }


def make_weighted_cohort(code_ids, weights, start_pos, target_end_pos):
    code_ids = np.asarray(code_ids, dtype=np.int32)
    weights = np.asarray(weights, dtype=np.float64)
    if len(code_ids) != len(weights):
        raise RuntimeError("code_ids and weights must have the same length.")
    return {
        "code_ids": code_ids,
        "weights": weights,
        "start_pos": int(start_pos),
        "target_end_pos": int(target_end_pos),
        "n_codes": int(len(code_ids)),
    }


def equal_weight_cohort(code_ids, start_pos, target_end_pos):
    code_ids = np.asarray(code_ids, dtype=np.int32)
    if len(code_ids) == 0:
        return make_weighted_cohort(code_ids, np.empty(0, dtype=np.float64), start_pos, target_end_pos)
    weights = np.full(len(code_ids), 1.0 / len(code_ids), dtype=np.float64)
    return make_weighted_cohort(code_ids, weights, start_pos, target_end_pos)


def make_weight_state_from_weighted_cohorts(cohorts):
    """
    Convert explicit-weight cohorts into a vectorized weight state.

    Unlike 06_backtest_decile.py, this function never divides by horizon. A
    rebalance cohort is full-weight: selected names are equal weighted to sum
    to 1 on each leg.
    """
    id_parts = []
    weight_parts = []
    static_missing_weight = 0.0
    static_missing_count = 0

    for cohort in cohorts:
        ids = np.asarray(cohort.get("code_ids", np.empty(0, dtype=np.int32)), dtype=np.int32)
        weights = np.asarray(cohort.get("weights", np.empty(0, dtype=np.float64)), dtype=np.float64)
        if len(ids) == 0:
            continue
        known = ids >= 0
        if known.any():
            id_parts.append(ids[known])
            weight_parts.append(weights[known])
        if (~known).any():
            static_missing_count += int((~known).sum())
            static_missing_weight += float(weights[~known].sum())

    if not id_parts:
        state = empty_weight_state()
        state["static_missing_weight"] = static_missing_weight
        state["static_missing_count"] = static_missing_count
        return state

    ids = np.concatenate(id_parts).astype(np.int32, copy=False)
    weights = np.concatenate(weight_parts).astype(np.float64, copy=False)
    unique_ids, inverse = np.unique(ids, return_inverse=True)
    summed_weights = np.bincount(inverse, weights=weights).astype(np.float64, copy=False)
    return {
        "ids": unique_ids.astype(np.int32, copy=False),
        "weights": summed_weights,
        "static_missing_weight": static_missing_weight,
        "static_missing_count": static_missing_count,
    }


def start_ids_from_weighted_cohorts(cohorts, return_pos):
    parts = [
        np.asarray(cohort.get("code_ids", np.empty(0, dtype=np.int32)), dtype=np.int32)
        for cohort in cohorts
        if int(cohort.get("start_pos", -1)) == int(return_pos)
    ]
    if not parts:
        return np.empty(0, dtype=np.int32)
    ids = np.concatenate(parts)
    ids = ids[ids >= 0]
    return np.unique(ids).astype(np.int32, copy=False)


def calc_leg_return(cohorts, daily_info, return_pos):
    if not cohorts:
        return bt.empty_return_info(0.0)
    state = make_weight_state_from_weighted_cohorts(cohorts)
    return bt.calc_weighted_return_state(
        state,
        daily_info,
        start_ids_from_weighted_cohorts(cohorts, return_pos),
    )


def signed_state(long_cohorts, short_cohorts):
    long_state = make_weight_state_from_weighted_cohorts(long_cohorts)
    short_state = make_weight_state_from_weighted_cohorts(short_cohorts)
    if len(long_state["ids"]) == 0 and len(short_state["ids"]) == 0:
        return empty_weight_state()

    ids = []
    weights = []
    if len(long_state["ids"]):
        ids.append(long_state["ids"])
        weights.append(long_state["weights"])
    if len(short_state["ids"]):
        ids.append(short_state["ids"])
        weights.append(-short_state["weights"])

    ids = np.concatenate(ids).astype(np.int32, copy=False)
    weights = np.concatenate(weights).astype(np.float64, copy=False)
    unique_ids, inverse = np.unique(ids, return_inverse=True)
    summed_weights = np.bincount(inverse, weights=weights).astype(np.float64, copy=False)
    return {
        "ids": unique_ids.astype(np.int32, copy=False),
        "weights": summed_weights,
        "static_missing_weight": 0.0,
        "static_missing_count": 0,
    }


def filter_shortable_code_ids(code_ids, daily_info):
    """
    Keep stocks that can be shorted/sold at the next open.

    This mirrors the long-entry filter but uses the low-volume limit-down flag
    because opening the short leg is a sell-side execution.
    """
    code_ids = np.asarray(code_ids, dtype=np.int32)
    if len(code_ids) == 0:
        return code_ids, 0, 0
    if daily_info is None:
        return np.empty(0, dtype=np.int32), int(len(code_ids)), int(len(code_ids))

    lookup, date_pos = daily_info
    valid_id = code_ids >= 0
    safe_ids = code_ids[valid_id]

    present = np.zeros(len(code_ids), dtype=bool)
    open_ret = np.full(len(code_ids), np.nan, dtype=lookup["open_to_close_ret_1d"].dtype)
    volume = np.full(len(code_ids), np.nan, dtype=lookup["volume"].dtype)
    suspended = np.ones(len(code_ids), dtype=bool)
    low_volume_limit_down = np.zeros(len(code_ids), dtype=bool)

    if len(safe_ids):
        present_valid = lookup["present"][date_pos, safe_ids]
        valid_positions = np.flatnonzero(valid_id)
        present[valid_positions] = present_valid
        open_ret[valid_positions] = lookup["open_to_close_ret_1d"][date_pos, safe_ids]
        volume[valid_positions] = lookup["volume"][date_pos, safe_ids]
        suspended[valid_positions] = lookup["is_suspended"][date_pos, safe_ids].astype(bool)
        low_volume_limit_down[valid_positions] = lookup["is_low_volume_limit_down"][date_pos, safe_ids].astype(bool)

    shortable = (
        valid_id
        & present
        & (~suspended)
        & (~low_volume_limit_down)
        & np.isfinite(open_ret)
        & np.isfinite(volume)
        & (volume > 0)
    )
    data_missing = (~valid_id) | (~present) | ~np.isfinite(open_ret)
    return (
        code_ids[shortable].astype(np.int32, copy=False),
        int((~shortable).sum()),
        int(data_missing.sum()),
    )


def process_exit_attempts(active_cohorts, daily_info, return_pos, exit_side):
    """
    Close normal or forced positions.

    exit_side="sell_long" blocks on low-volume limit-down. exit_side="cover_short"
    blocks on low-volume limit-up.
    """
    next_active = []
    blocked = 0
    forced = 0
    data_missing = 0

    for cohort in active_cohorts:
        code_ids = np.asarray(cohort.get("code_ids", np.empty(0, dtype=np.int32)), dtype=np.int32)
        weights = np.asarray(cohort.get("weights", np.empty(0, dtype=np.float64)), dtype=np.float64)
        if len(code_ids) == 0:
            continue

        if daily_info is None:
            keep_mask = np.ones(len(code_ids), dtype=bool)
            data_missing += int(len(code_ids))
        else:
            lookup, date_pos = daily_info
            valid_id = code_ids >= 0
            safe_ids = code_ids[valid_id]
            present = np.zeros(len(code_ids), dtype=bool)
            ret = np.full(len(code_ids), np.nan, dtype=lookup["ret_1d"].dtype)
            suspended = np.ones(len(code_ids), dtype=bool)
            low_block = np.zeros(len(code_ids), dtype=bool)
            if len(safe_ids):
                valid_positions = np.flatnonzero(valid_id)
                present[valid_positions] = lookup["present"][date_pos, safe_ids]
                ret[valid_positions] = lookup["ret_1d"][date_pos, safe_ids]
                suspended[valid_positions] = lookup["is_suspended"][date_pos, safe_ids].astype(bool)
                if exit_side == "sell_long":
                    low_block[valid_positions] = lookup["is_low_volume_limit_down"][date_pos, safe_ids].astype(bool)
                elif exit_side == "cover_short":
                    low_block[valid_positions] = lookup["is_low_volume_limit_up"][date_pos, safe_ids].astype(bool)
                else:
                    raise RuntimeError(f"Unsupported exit side: {exit_side}")

            valid_exit = valid_id & present & (~suspended) & (~low_block) & np.isfinite(ret)
            keep_mask = ~valid_exit
            data_missing += int(((~valid_id) | (~present) | ~np.isfinite(ret)).sum())

        blocked += int(keep_mask.sum())
        forced += int(keep_mask.sum())
        if keep_mask.any():
            next_active.append(
                make_weighted_cohort(
                    code_ids[keep_mask],
                    weights[keep_mask],
                    cohort.get("start_pos", return_pos),
                    cohort.get("target_end_pos", return_pos),
                )
            )

    return next_active, blocked, forced, data_missing


def unique_nonnull_codes(values):
    result = []
    seen = set()
    for value in values:
        if pd.isna(value) or value in seen:
            continue
        seen.add(value)
        result.append(value)
    return result


def build_id_to_code(lookup):
    id_to_code = [None] * len(lookup["code_to_id"])
    for code, code_id in lookup["code_to_id"].items():
        id_to_code[int(code_id)] = code
    return id_to_code


def codes_from_ids(code_ids, id_to_code):
    codes = []
    for code_id in np.asarray(code_ids, dtype=np.int32):
        if code_id < 0 or code_id >= len(id_to_code):
            continue
        code = id_to_code[int(code_id)]
        if code is not None:
            codes.append(str(code))
    return sorted(set(codes))


def append_holding_row(rows, cycle, trading_dates, keys, portfolio_name, leg, code_ids, id_to_code):
    codes = codes_from_ids(code_ids, id_to_code)
    rows.append({
        "signal_date": trading_dates[cycle["signal_pos"]],
        "rebalance_date": trading_dates[cycle["start_pos"]],
        "target_end_date": trading_dates[cycle["target_end_pos"]],
        "horizon": int(cycle["horizon"]),
        "experiment_name": keys[0],
        "model_name": keys[1],
        "universe_group": keys[2],
        "portfolio_name": portfolio_name,
        "leg": leg,
        "n_holdings": int(len(codes)),
        "codes": "|".join(codes),
    })


def split_codes(codes):
    if pd.isna(codes) or str(codes) == "":
        return set()
    return {code for code in str(codes).split("|") if code}


def model_family(model_name):
    name = str(model_name).lower()
    if "logistic" in name:
        return "Logistic"
    if "cnn" in name:
        return "CNN"
    return None


def calc_holding_overlap(portfolio_holdings):
    if portfolio_holdings.empty:
        return pd.DataFrame()

    holdings = portfolio_holdings.copy()
    holdings["model_family"] = holdings["model_name"].map(model_family)
    holdings = holdings[holdings["model_family"].isin(["Logistic", "CNN"])].copy()
    if holdings.empty:
        return pd.DataFrame()

    key_cols = [
        "experiment_name",
        "universe_group",
        "portfolio_name",
        "leg",
        "rebalance_date",
    ]
    rows = []
    for keys, group in holdings.groupby(key_cols, sort=True):
        logistic = group[group["model_family"] == "Logistic"]
        cnn = group[group["model_family"] == "CNN"]
        if logistic.empty or cnn.empty:
            continue

        logistic_row = logistic.iloc[0]
        cnn_row = cnn.iloc[0]
        logistic_codes = split_codes(logistic_row["codes"])
        cnn_codes = split_codes(cnn_row["codes"])
        intersection = logistic_codes & cnn_codes
        union = logistic_codes | cnn_codes
        logistic_n = len(logistic_codes)
        cnn_n = len(cnn_codes)
        union_n = len(union)

        row = dict(zip(key_cols, keys))
        row.update({
            "signal_date_logistic": logistic_row["signal_date"],
            "signal_date_cnn": cnn_row["signal_date"],
            "target_end_date_logistic": logistic_row["target_end_date"],
            "target_end_date_cnn": cnn_row["target_end_date"],
            "horizon_logistic": int(logistic_row["horizon"]),
            "horizon_cnn": int(cnn_row["horizon"]),
            "model_name_logistic": logistic_row["model_name"],
            "model_name_cnn": cnn_row["model_name"],
            "logistic_n": logistic_n,
            "cnn_n": cnn_n,
            "intersection_n": len(intersection),
            "union_n": union_n,
            "jaccard_overlap": len(intersection) / union_n if union_n > 0 else np.nan,
            "logistic_coverage": len(intersection) / logistic_n if logistic_n > 0 else np.nan,
            "cnn_coverage": len(intersection) / cnn_n if cnn_n > 0 else np.nan,
            "intersection_codes": "|".join(sorted(intersection)),
        })
        rows.append(row)

    return pd.DataFrame(rows)


def summarize_holding_overlap(holding_overlap):
    if holding_overlap.empty:
        return pd.DataFrame()

    group_cols = ["experiment_name", "universe_group", "portfolio_name", "leg"]
    rows = []
    for keys, group in holding_overlap.groupby(group_cols, sort=True):
        valid = group["jaccard_overlap"].dropna()
        rows.append({
            "experiment_name": keys[0],
            "universe_group": keys[1],
            "portfolio_name": keys[2],
            "leg": keys[3],
            "n_rebalances": int(len(group)),
            "jaccard_mean": valid.mean() if len(valid) else np.nan,
            "jaccard_median": valid.median() if len(valid) else np.nan,
            "jaccard_p25": valid.quantile(0.25) if len(valid) else np.nan,
            "jaccard_p75": valid.quantile(0.75) if len(valid) else np.nan,
            "avg_intersection_n": group["intersection_n"].mean(),
            "avg_logistic_n": group["logistic_n"].mean(),
            "avg_cnn_n": group["cnn_n"].mean(),
        })
    return pd.DataFrame(rows)


def filter_d10_long_only(portfolio_returns, portfolio_nav, performance_summary):
    returns = portfolio_returns[
        (portfolio_returns["decile"] == N_DECILES)
        & (portfolio_returns["portfolio_name"] == D10_PORTFOLIO_NAME)
    ].copy()
    nav = portfolio_nav[portfolio_nav["portfolio_name"] == D10_PORTFOLIO_NAME].copy()
    performance = performance_summary[
        (performance_summary["decile"] == N_DECILES)
        & (performance_summary["portfolio_name"] == D10_PORTFOLIO_NAME)
    ].copy()
    return returns, nav, performance


def selected_rebalance_positions(group, horizon):
    positions = sorted(pd.unique(group["signal_pos"].dropna().astype(int)))
    if not positions:
        return set()
    anchor = positions[0]
    return {pos for pos in positions if (pos - anchor) % horizon == 0}


def build_nonoverlap_cycles(group, date_to_pos, max_date_pos, lookup):
    """
    Build full-weight rebalance cohorts keyed by first holding date.
    """
    group = group.copy()
    group["signal_pos"] = group["date"].map(date_to_pos)
    group = group[group["signal_pos"].notna()].copy()
    if group.empty:
        return {}, -1

    group["signal_pos"] = group["signal_pos"].astype(int)
    group["decile"] = group["decile"].astype(int)
    group["horizon"] = group["horizon"].astype(int)

    cycles_by_start = {}
    max_end_pos = -1
    for horizon, horizon_group in group.groupby("horizon", sort=True):
        horizon = int(horizon)
        if horizon <= 0:
            continue
        selected_positions = selected_rebalance_positions(horizon_group, horizon)
        selected = horizon_group[horizon_group["signal_pos"].isin(selected_positions)].copy()
        for signal_pos, signal_group in selected.groupby("signal_pos", sort=True):
            signal_pos = int(signal_pos)
            start_pos = signal_pos + 1
            target_end_pos = signal_pos + horizon
            if start_pos > max_date_pos or target_end_pos > max_date_pos:
                continue

            cycle = {
                "signal_pos": signal_pos,
                "start_pos": start_pos,
                "target_end_pos": target_end_pos,
                "horizon": horizon,
                "deciles": {},
            }
            for decile, decile_group in signal_group.groupby("decile", sort=True):
                decile = int(decile)
                codes = unique_nonnull_codes(decile_group["code"])
                if not codes:
                    continue
                code_ids = np.asarray([lookup["code_to_id"].get(code, -1) for code in codes], dtype=np.int32)
                cycle["deciles"][decile] = code_ids

            if cycle["deciles"]:
                cycles_by_start[start_pos] = cycle
                max_end_pos = max(max_end_pos, target_end_pos)

    return cycles_by_start, max_end_pos


def finite_or_nan_diff(left, right):
    if pd.isna(left) or pd.isna(right):
        return np.nan
    return float(left) - float(right)


def combine_long_short_info(long_info, short_info):
    if long_info["valid_weight"] <= 0 or short_info["valid_weight"] <= 0:
        gross_return = np.nan
    else:
        gross_return = float(long_info["return"] - short_info["return"])
    return {
        "return": gross_return,
        "valid_weight": min(float(long_info["valid_weight"]), float(short_info["valid_weight"])),
        "missing_weight": float(long_info["missing_weight"]) + float(short_info["missing_weight"]),
        "data_missing": int(long_info["data_missing"]) + int(short_info["data_missing"]),
    }


def append_base_row(rows, date, keys, decile, portfolio_name, gross_return, turnover, start_turnover,
                    end_turnover, active_cohorts, num_holdings, blocked_buys, blocked_sells,
                    forced_holds, data_missing, valid_weight, missing_weight, signal_info,
                    constrained_info, is_warmup=False):
    signal_return = signal_info["return"]
    constrained_return = constrained_info["return"]
    rows.append({
        "date": date,
        "experiment_name": keys[0],
        "model_name": keys[1],
        "universe_group": keys[2],
        "decile": decile,
        "portfolio_name": portfolio_name,
        "gross_return": gross_return,
        "turnover": turnover,
        "start_turnover": start_turnover,
        "end_turnover": end_turnover,
        "active_cohorts": active_cohorts,
        "num_holdings": num_holdings,
        "num_blocked_buys": int(blocked_buys),
        "num_blocked_sells": int(blocked_sells),
        "num_forced_holds": int(forced_holds),
        "num_data_missing_returns": int(data_missing),
        "valid_weight": valid_weight,
        "missing_weight": missing_weight,
        "signal_valid_weight": signal_info["valid_weight"],
        "signal_missing_weight": signal_info["missing_weight"],
        "signal_gross_alpha": signal_return,
        "buy_constrained_gross_return": constrained_return,
        "buy_blocked_loss": finite_or_nan_diff(signal_return, constrained_return),
        "sell_blocked_forced_hold_loss": finite_or_nan_diff(constrained_return, gross_return),
        "is_warmup": bool(is_warmup),
    })


def calc_nonoverlap_portfolios(pred, daily_context):
    """
    Build strict non-overlapping D1-D10 long-only and D10-D1 portfolios.
    """
    signals = pred.dropna(subset=["decile", "horizon"]).copy()
    if signals.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    trading_dates, date_to_pos, lookup = daily_context
    if len(trading_dates) == 0:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    bt.warn_if_compound_returns_mismatch(signals, trading_dates, date_to_pos, lookup)

    base_rows = []
    turnover_rows = []
    holding_rows = []
    id_to_code = build_id_to_code(lookup)
    max_date_pos = len(trading_dates) - 1

    for keys, group in signals.groupby(GROUP_COLS, sort=True):
        exp_name, model_name, universe_group = keys
        group_start = perf_counter()
        log(
            "Nonoverlap group start: "
            f"{exp_name}/{model_name}/{universe_group}, signal_rows={len(group):,}"
        )
        cycles_by_start, max_end_pos = build_nonoverlap_cycles(group, date_to_pos, max_date_pos, lookup)
        if max_end_pos < 0:
            log(
                "Nonoverlap group skipped: "
                f"{exp_name}/{model_name}/{universe_group}, no valid rebalance cycles"
            )
            continue

        min_start_pos = min(cycles_by_start)
        total_days = max_date_pos - min_start_pos + 1
        active = {decile: [] for decile in range(1, N_DECILES + 1)}
        forced_active = {decile: [] for decile in range(1, N_DECILES + 1)}
        ideal = {decile: [] for decile in range(1, N_DECILES + 1)}
        constrained = {decile: [] for decile in range(1, N_DECILES + 1)}
        prev_weights = {decile: empty_weight_state() for decile in range(1, N_DECILES + 1)}

        short_active = []
        short_forced = []
        short_ideal = []
        short_constrained = []
        prev_ls_weights = empty_weight_state()
        last_progress = perf_counter()

        for return_pos in range(min_start_pos, max_date_pos + 1):
            date = trading_dates[return_pos]
            daily_info = bt.get_daily_info(lookup, trading_dates, return_pos)
            day_i = return_pos - min_start_pos + 1
            now = perf_counter()
            if day_i == 1 or day_i == total_days or day_i % 250 == 0 or now - last_progress >= 60:
                active_slots = sum(
                    int(cohort.get("n_codes", 0))
                    for decile in range(1, N_DECILES + 1)
                    for cohort in active[decile] + forced_active[decile]
                )
                log(
                    "Nonoverlap progress: "
                    f"{exp_name}/{model_name}/{universe_group}, "
                    f"day={day_i:,}/{total_days:,}, date={pd.Timestamp(date).date()}, "
                    f"active_code_slots={active_slots:,}, short_slots={sum(c['n_codes'] for c in short_active + short_forced):,}"
                )
                last_progress = now

            blocked_buys = defaultdict(int)
            buy_missing = defaultdict(int)
            short_blocked = 0
            short_missing = 0

            cycle = cycles_by_start.get(return_pos)
            if cycle is not None:
                start_pos = cycle["start_pos"]
                target_end_pos = cycle["target_end_pos"]
                d10_actual_ids = np.empty(0, dtype=np.int32)
                for decile in range(1, N_DECILES + 1):
                    code_ids = cycle["deciles"].get(decile, np.empty(0, dtype=np.int32))
                    if len(code_ids) == 0:
                        ideal[decile] = []
                        constrained[decile] = []
                        active[decile] = []
                        if decile == N_DECILES:
                            d10_actual_ids = np.empty(0, dtype=np.int32)
                        continue

                    ideal[decile] = [equal_weight_cohort(code_ids, start_pos, target_end_pos)]
                    buyable_ids, blocked, missing = bt.filter_buyable_code_ids(code_ids, daily_info)
                    blocked_buys[decile] += blocked
                    buy_missing[decile] += missing
                    if decile == N_DECILES:
                        d10_actual_ids = buyable_ids
                    if len(buyable_ids):
                        new_cohort = equal_weight_cohort(buyable_ids, start_pos, target_end_pos)
                        constrained[decile] = [new_cohort]
                        active[decile] = [new_cohort]
                    else:
                        constrained[decile] = []
                        active[decile] = []

                short_ids = cycle["deciles"].get(1, np.empty(0, dtype=np.int32))
                short_ideal = [equal_weight_cohort(short_ids, start_pos, target_end_pos)] if len(short_ids) else []
                shortable_ids, short_blocked, short_missing = filter_shortable_code_ids(short_ids, daily_info)
                if len(shortable_ids):
                    new_short = equal_weight_cohort(shortable_ids, start_pos, target_end_pos)
                    short_constrained = [new_short]
                    short_active = [new_short]
                else:
                    short_constrained = []
                    short_active = []

                append_holding_row(
                    holding_rows,
                    cycle,
                    trading_dates,
                    keys,
                    D10_PORTFOLIO_NAME,
                    "long",
                    d10_actual_ids,
                    id_to_code,
                )
                append_holding_row(
                    holding_rows,
                    cycle,
                    trading_dates,
                    keys,
                    LONG_SHORT_PORTFOLIO_NAME,
                    "long",
                    d10_actual_ids,
                    id_to_code,
                )
                append_holding_row(
                    holding_rows,
                    cycle,
                    trading_dates,
                    keys,
                    LONG_SHORT_PORTFOLIO_NAME,
                    "short",
                    shortable_ids,
                    id_to_code,
                )

            ls_long_start_snapshot = list(active[N_DECILES] + forced_active[N_DECILES])
            ls_long_ideal_snapshot = list(ideal[N_DECILES])
            ls_long_constrained_snapshot = list(constrained[N_DECILES])
            long_exit_stats = {}
            for decile in range(1, N_DECILES + 1):
                active_start = active[decile] + forced_active[decile]
                ideal_start = ideal[decile]
                constrained_start = constrained[decile]

                if not active_start and not ideal_start and not constrained_start:
                    prev_weights[decile] = empty_weight_state()
                    long_exit_stats[decile] = (0, 0, 0)
                    continue

                weight_state = make_weight_state_from_weighted_cohorts(active_start)
                start_turnover = bt.calc_weight_turnover_states(prev_weights[decile], weight_state)
                actual_info = calc_leg_return(active_start, daily_info, return_pos)
                signal_info = calc_leg_return(ideal_start, daily_info, return_pos)
                constrained_info = calc_leg_return(constrained_start, daily_info, return_pos)

                due_normal = [cohort for cohort in active[decile] if return_pos >= cohort["target_end_pos"]]
                still_normal = [cohort for cohort in active[decile] if return_pos < cohort["target_end_pos"]]
                forced_due = forced_active[decile]
                forced_end, blocked_sells, forced_holds, sell_missing = process_exit_attempts(
                    due_normal + forced_due,
                    daily_info,
                    return_pos,
                    "sell_long",
                )
                if ideal[decile] and return_pos >= ideal[decile][0]["target_end_pos"]:
                    ideal[decile] = []
                if constrained[decile] and return_pos >= constrained[decile][0]["target_end_pos"]:
                    constrained[decile] = []

                active[decile] = still_normal
                forced_active[decile] = forced_end
                end_weight_state = make_weight_state_from_weighted_cohorts(still_normal + forced_end)
                end_turnover = bt.calc_weight_turnover_states(weight_state, end_weight_state)
                turnover = start_turnover + end_turnover
                data_missing = int(actual_info["data_missing"]) + int(buy_missing[decile]) + int(sell_missing)
                long_exit_stats[decile] = (blocked_sells, forced_holds, sell_missing)

                append_base_row(
                    base_rows,
                    date,
                    keys,
                    decile,
                    f"D{decile}",
                    actual_info["return"],
                    turnover,
                    start_turnover,
                    end_turnover,
                    len(active_start),
                    len(weight_state["ids"]),
                    blocked_buys[decile],
                    blocked_sells,
                    forced_holds,
                    data_missing,
                    actual_info["valid_weight"],
                    actual_info["missing_weight"],
                    signal_info,
                    constrained_info,
                )

                turnover_rows.append({
                    "date": date,
                    "experiment_name": exp_name,
                    "model_name": model_name,
                    "universe_group": universe_group,
                    "decile": decile,
                    "portfolio_name": f"D{decile}",
                    "turnover": turnover,
                    "start_turnover": start_turnover,
                    "end_turnover": end_turnover,
                    "active_cohorts": len(active_start),
                    "num_holdings": len(weight_state["ids"]),
                    "num_blocked_buys": int(blocked_buys[decile]),
                    "num_blocked_sells": int(blocked_sells),
                    "num_forced_holds": int(forced_holds),
                    "num_data_missing_returns": data_missing,
                    "valid_weight": actual_info["valid_weight"],
                    "missing_weight": actual_info["missing_weight"],
                    "is_warmup": False,
                })
                prev_weights[decile] = end_weight_state

            ls_long_start = ls_long_start_snapshot
            ls_short_start = short_active + short_forced
            if ls_long_start or ls_short_start or short_ideal or short_constrained:
                ls_weight_state = signed_state(ls_long_start, ls_short_start)
                ls_start_turnover = bt.calc_weight_turnover_states(prev_ls_weights, ls_weight_state)

                long_actual = calc_leg_return(ls_long_start, daily_info, return_pos)
                short_actual = calc_leg_return(ls_short_start, daily_info, return_pos)
                actual_info = combine_long_short_info(long_actual, short_actual)

                long_signal = calc_leg_return(ls_long_ideal_snapshot, daily_info, return_pos)
                short_signal = calc_leg_return(short_ideal, daily_info, return_pos)
                signal_info = combine_long_short_info(long_signal, short_signal)

                long_constrained = calc_leg_return(ls_long_constrained_snapshot, daily_info, return_pos)
                short_constrained_info = calc_leg_return(short_constrained, daily_info, return_pos)
                constrained_info = combine_long_short_info(long_constrained, short_constrained_info)

                due_short = [cohort for cohort in short_active if return_pos >= cohort["target_end_pos"]]
                still_short = [cohort for cohort in short_active if return_pos < cohort["target_end_pos"]]
                short_forced_end, short_blocked_sells, short_forced_holds, short_sell_missing = process_exit_attempts(
                    due_short + short_forced,
                    daily_info,
                    return_pos,
                    "cover_short",
                )
                if short_ideal and return_pos >= short_ideal[0]["target_end_pos"]:
                    short_ideal = []
                if short_constrained and return_pos >= short_constrained[0]["target_end_pos"]:
                    short_constrained = []
                short_active = still_short
                short_forced = short_forced_end

                ls_end_state = signed_state(active[N_DECILES] + forced_active[N_DECILES], short_active + short_forced)
                ls_end_turnover = bt.calc_weight_turnover_states(ls_weight_state, ls_end_state)
                ls_turnover = ls_start_turnover + ls_end_turnover
                d10_blocked_sells, d10_forced_holds, d10_sell_missing = long_exit_stats.get(N_DECILES, (0, 0, 0))
                data_missing = (
                    int(actual_info["data_missing"])
                    + int(buy_missing[N_DECILES])
                    + int(short_missing)
                    + int(d10_sell_missing)
                    + int(short_sell_missing)
                )

                append_base_row(
                    base_rows,
                    date,
                    keys,
                    LONG_SHORT_DECILE,
                    LONG_SHORT_PORTFOLIO_NAME,
                    actual_info["return"],
                    ls_turnover,
                    ls_start_turnover,
                    ls_end_turnover,
                    len(ls_long_start) + len(ls_short_start),
                    len(ls_weight_state["ids"]),
                    int(blocked_buys[N_DECILES]) + int(short_blocked),
                    int(d10_blocked_sells) + int(short_blocked_sells),
                    int(d10_forced_holds) + int(short_forced_holds),
                    data_missing,
                    actual_info["valid_weight"],
                    actual_info["missing_weight"],
                    signal_info,
                    constrained_info,
                )

                turnover_rows.append({
                    "date": date,
                    "experiment_name": exp_name,
                    "model_name": model_name,
                    "universe_group": universe_group,
                    "decile": LONG_SHORT_DECILE,
                    "portfolio_name": LONG_SHORT_PORTFOLIO_NAME,
                    "turnover": ls_turnover,
                    "start_turnover": ls_start_turnover,
                    "end_turnover": ls_end_turnover,
                    "active_cohorts": len(ls_long_start) + len(ls_short_start),
                    "num_holdings": len(ls_weight_state["ids"]),
                    "num_blocked_buys": int(blocked_buys[N_DECILES]) + int(short_blocked),
                    "num_blocked_sells": int(d10_blocked_sells) + int(short_blocked_sells),
                    "num_forced_holds": int(d10_forced_holds) + int(short_forced_holds),
                    "num_data_missing_returns": data_missing,
                    "valid_weight": actual_info["valid_weight"],
                    "missing_weight": actual_info["missing_weight"],
                    "is_warmup": False,
                })
                prev_ls_weights = ls_end_state
            else:
                prev_ls_weights = empty_weight_state()

            if (
                return_pos >= max_end_pos
                and all(not active[d] and not forced_active[d] and not ideal[d] and not constrained[d] for d in range(1, N_DECILES + 1))
                and not short_active
                and not short_forced
                and not short_ideal
                and not short_constrained
            ):
                break

        log(
            "Nonoverlap group done: "
            f"{exp_name}/{model_name}/{universe_group}, "
            f"elapsed_sec={perf_counter() - group_start:.1f}"
        )

    base = pd.DataFrame(base_rows)
    portfolio_returns = bt.expand_portfolio_cost_rows(base)
    portfolio_turnover = pd.DataFrame(turnover_rows)
    portfolio_holdings = pd.DataFrame(holding_rows)
    return portfolio_returns, portfolio_turnover, portfolio_holdings


def run_one_prediction_file(path, daily_context):
    file_start = perf_counter()
    pred = bt.load_prediction_file(path)
    log(f"Prediction rows loaded: {Path(path).name}, rows={len(pred):,}")
    pred = bt.add_universe_groups(pred)
    log(f"Universe rows expanded: {Path(path).name}, rows={len(pred):,}")
    pred = bt.assign_deciles(pred)

    step_start = perf_counter()
    portfolio_returns, portfolio_turnover, portfolio_holdings = calc_nonoverlap_portfolios(pred, daily_context)
    performance_summary = bt.summarize_portfolios(portfolio_returns)
    portfolio_nav = bt.calc_portfolio_nav(portfolio_returns)
    return_attribution = bt.calc_return_attribution(portfolio_returns)
    cost_sensitivity = bt.calc_cost_sensitivity(performance_summary)
    d10_returns, d10_nav, d10_performance = filter_d10_long_only(
        portfolio_returns,
        portfolio_nav,
        performance_summary,
    )
    log(
        f"Nonoverlap portfolio done: {Path(path).name}, "
        f"elapsed_sec={perf_counter() - step_start:.1f}, "
        f"portfolio_rows={len(portfolio_returns):,}"
    )
    log(f"Prediction file done: {Path(path).name}, elapsed_sec={perf_counter() - file_start:.1f}")

    return {
        "portfolio_returns": portfolio_returns,
        "portfolio_turnover": portfolio_turnover,
        "portfolio_nav": portfolio_nav,
        "performance_summary": performance_summary,
        "return_attribution": return_attribution,
        "cost_sensitivity": cost_sensitivity,
        "d10_long_only_returns": d10_returns,
        "d10_long_only_nav": d10_nav,
        "d10_long_only_performance": d10_performance,
        "portfolio_holdings": portfolio_holdings,
    }


def prefixed_name(output_prefix, name):
    return f"{output_prefix}{name}.csv"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pred-pattern",
        default="pred_*.parquet",
        help="Prediction parquet glob under outputs/predictions.",
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=None,
        help="Optional cap on the number of prediction files to process.",
    )
    parser.add_argument(
        "--output-prefix",
        default="nonoverlap_",
        help="Prefix for output CSV files under outputs/tables.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    pred_files = sorted(glob.glob(str(PRED_DIR / args.pred_pattern)))
    if args.max_files is not None:
        pred_files = pred_files[: args.max_files]
    if not pred_files:
        raise RuntimeError(f"No prediction files found in {PRED_DIR} matching {args.pred_pattern}")

    earliest_prediction_date = bt.load_earliest_prediction_date(pred_files)
    if earliest_prediction_date is not None:
        log(f"Earliest prediction date: {earliest_prediction_date.date()}")

    daily_returns = bt.load_daily_returns(min_date=earliest_prediction_date)
    log(f"Daily return rows loaded: {len(daily_returns):,}")
    lookup_start = perf_counter()
    daily_context = bt.prepare_daily_return_lookup(daily_returns)
    del daily_returns
    log(f"Daily return lookup ready: elapsed_sec={perf_counter() - lookup_start:.1f}")

    output_names = [
        "portfolio_returns",
        "portfolio_turnover",
        "portfolio_nav",
        "performance_summary",
        "return_attribution",
        "cost_sensitivity",
        "d10_long_only_returns",
        "d10_long_only_nav",
        "d10_long_only_performance",
        "portfolio_holdings",
    ]
    initialized_outputs = set()
    performance_frames = []
    holding_frames = []

    for idx, path in enumerate(pred_files, start=1):
        log(f"Start prediction file {idx}/{len(pred_files)}: {Path(path).name}")
        result = run_one_prediction_file(path, daily_context)
        performance_frames.append(result["performance_summary"])
        holding_frames.append(result["portfolio_holdings"])

        log(f"Appending partial nonoverlap tables after {Path(path).name}")
        for name in output_names:
            filename = prefixed_name(args.output_prefix, name)
            initialized = bt.append_table(
                result[name],
                filename,
                name in initialized_outputs,
            )
            if initialized:
                initialized_outputs.add(name)

        del result

    for name in output_names:
        if name not in initialized_outputs:
            bt.save_table(pd.DataFrame(), prefixed_name(args.output_prefix, name))

    portfolio_holdings = bt.concat_frames(holding_frames)
    holding_overlap = calc_holding_overlap(portfolio_holdings)
    holding_overlap_summary = summarize_holding_overlap(holding_overlap)
    bt.save_table(holding_overlap, prefixed_name(args.output_prefix, "holding_overlap"))
    bt.save_table(holding_overlap_summary, prefixed_name(args.output_prefix, "holding_overlap_summary"))

    perf = bt.concat_frames(performance_frames)
    if not perf.empty:
        print("\nNonoverlap performance summary:")
        print(perf)


if __name__ == "__main__":
    main()
