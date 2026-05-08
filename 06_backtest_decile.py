# -*- coding: utf-8 -*-
"""
06_backtest_decile.py

Evaluate model predictions with IC, decile returns, and long-only overlapping
decile portfolios.

Execution convention
--------------------
A signal formed at date t buys at the next trading day's adjusted open and
holds through the horizon date's adjusted close. For the first holding day,
portfolio returns use open-to-close returns. Later holding days use standard
close-to-close returns.

Inputs
------
outputs/predictions/pred_*.parquet
data/features/baseline_features.parquet

Outputs
-------
outputs/tables/ic_by_period.csv
outputs/tables/ic_summary.csv
outputs/tables/cumulative_ic.csv
outputs/tables/decile_returns.csv
outputs/tables/decile_summary.csv
outputs/tables/decile_monotonicity.csv
outputs/tables/portfolio_returns.csv
outputs/tables/portfolio_turnover.csv
outputs/tables/performance_summary.csv
outputs/tables/cost_sensitivity.csv
"""

from collections import defaultdict
import glob

import numpy as np
import pandas as pd

from config import (
    BASELINE_FEATURE_PATH,
    PRED_DIR,
    TABLE_DIR,
    N_DECILES,
    COST_BPS_GRID,
    UNIVERSE_SPLIT_METHOD,
    TRADING_DAYS_PER_YEAR,
)


PRED_REQUIRED_COLS = {
    "date",
    "code",
    "pred_prob",
    "future_ret",
    "float_mktcap",
    "horizon",
    "experiment_name",
    "model_name",
}

DAILY_RETURN_REQUIRED_COLS = {"date", "code", "ret_1d", "open_to_close_ret_1d"}

GROUP_COLS = ["experiment_name", "model_name", "universe_group"]
PERIOD_GROUP_COLS = ["date"] + GROUP_COLS


def require_columns(df, required_cols, context):
    """
    Fail early with a clear missing-column error.
    """
    missing = sorted(set(required_cols) - set(df.columns))
    if missing:
        raise RuntimeError(f"{context} missing required columns: {missing}")


def safe_corr(x, y):
    """
    Correlation that returns NaN for too-small or constant samples.
    """
    valid = pd.DataFrame({"x": x, "y": y}).dropna()
    if len(valid) < 2:
        return np.nan
    if valid["x"].nunique() < 2 or valid["y"].nunique() < 2:
        return np.nan
    return valid["x"].corr(valid["y"])


def max_drawdown(nav):
    """
    Compute maximum drawdown from a NAV series.
    """
    nav = nav.dropna()
    if len(nav) == 0:
        return np.nan
    running_max = nav.cummax()
    drawdown = nav / running_max - 1.0
    return drawdown.min()


def load_prediction_file(path):
    """
    Read and validate one prediction parquet file.
    """
    print(f"Reading predictions: {path}")
    pred = pd.read_parquet(path)
    require_columns(pred, PRED_REQUIRED_COLS, path)

    pred = pred.copy()
    pred["date"] = pd.to_datetime(pred["date"])
    pred["pred_prob"] = pd.to_numeric(pred["pred_prob"], errors="coerce")
    pred["future_ret"] = pd.to_numeric(pred["future_ret"], errors="coerce")
    pred["float_mktcap"] = pd.to_numeric(pred["float_mktcap"], errors="coerce")
    pred["horizon"] = pd.to_numeric(pred["horizon"], errors="coerce")

    pred = pred[
        pred["date"].notna()
        & pred["code"].notna()
        & pred["pred_prob"].notna()
        & pred["future_ret"].notna()
        & pred["horizon"].notna()
    ].copy()

    if pred.empty:
        raise RuntimeError(f"{path} has no valid prediction rows.")

    return pred


def load_daily_returns():
    """
    Read daily realized returns used by overlapping portfolio evaluation.
    """
    print(f"Reading daily returns: {BASELINE_FEATURE_PATH}")
    try:
        daily_ret = pd.read_parquet(
            BASELINE_FEATURE_PATH,
            columns=sorted(DAILY_RETURN_REQUIRED_COLS),
        )
    except Exception as exc:
        raise RuntimeError(
            f"{BASELINE_FEATURE_PATH} must include {sorted(DAILY_RETURN_REQUIRED_COLS)}. "
            "Rerun 02_make_labels_and_baselines.py after this execution-return update."
        ) from exc
    require_columns(daily_ret, DAILY_RETURN_REQUIRED_COLS, str(BASELINE_FEATURE_PATH))

    daily_ret = daily_ret.copy()
    daily_ret["date"] = pd.to_datetime(daily_ret["date"])
    daily_ret["ret_1d"] = pd.to_numeric(daily_ret["ret_1d"], errors="coerce")
    daily_ret["open_to_close_ret_1d"] = pd.to_numeric(
        daily_ret["open_to_close_ret_1d"],
        errors="coerce",
    )
    daily_ret = daily_ret.dropna(subset=["date", "code"])

    if daily_ret.empty:
        raise RuntimeError(f"{BASELINE_FEATURE_PATH} has no valid daily return rows.")

    return daily_ret


def add_universe_groups(pred):
    """
    Add all / large_cap / small_mid_cap rows.

    The daily tercile split assigns the top one-third by float_mktcap to
    large_cap and the bottom two-thirds to small_mid_cap.
    """
    if UNIVERSE_SPLIT_METHOD != "daily_tercile":
        raise RuntimeError(f"Unsupported universe split method: {UNIVERSE_SPLIT_METHOD}")

    all_pred = pred.copy()
    all_pred["universe_group"] = "all"

    cap_pred = pred[pred["float_mktcap"].notna()].copy()
    if cap_pred.empty:
        return all_pred

    rank_pct = cap_pred.groupby("date")["float_mktcap"].rank(method="first", pct=True)
    cap_pred["universe_group"] = np.where(
        rank_pct > (2.0 / 3.0),
        "large_cap",
        "small_mid_cap",
    )

    return pd.concat([all_pred, cap_pred], ignore_index=True)


def calc_ic_by_period(pred):
    """
    Calculate IC and RankIC for each date and universe group.
    """
    rows = []

    for keys, group in pred.groupby(PERIOD_GROUP_COLS, sort=True):
        date, exp_name, model_name, universe_group = keys
        valid = group[["pred_prob", "future_ret"]].dropna()

        ic = safe_corr(valid["pred_prob"], valid["future_ret"])
        rankic = safe_corr(valid["pred_prob"].rank(method="average"), valid["future_ret"].rank(method="average"))

        rows.append({
            "date": date,
            "experiment_name": exp_name,
            "model_name": model_name,
            "universe_group": universe_group,
            "num_stocks": len(valid),
            "ic": ic,
            "rankic": rankic,
        })

    return pd.DataFrame(rows)


def summarize_metric(series):
    """
    Summary statistics for an IC-like time series.
    """
    valid = series.dropna()
    n = len(valid)
    if n == 0:
        return {
            "n": 0,
            "mean": np.nan,
            "std": np.nan,
            "ir": np.nan,
            "t_stat": np.nan,
            "positive_rate": np.nan,
        }

    mean = valid.mean()
    std = valid.std(ddof=1)
    ir = mean / std if std > 0 else np.nan
    t_stat = mean / (std / np.sqrt(n)) if std > 0 else np.nan

    return {
        "n": n,
        "mean": mean,
        "std": std,
        "ir": ir,
        "t_stat": t_stat,
        "positive_rate": (valid > 0).mean(),
    }


def calc_ic_summary(ic_by_period):
    """
    Summarize IC and RankIC across time.
    """
    rows = []

    for keys, group in ic_by_period.groupby(GROUP_COLS, sort=True):
        exp_name, model_name, universe_group = keys
        ic_stats = summarize_metric(group["ic"])
        rankic_stats = summarize_metric(group["rankic"])

        rows.append({
            "experiment_name": exp_name,
            "model_name": model_name,
            "universe_group": universe_group,
            "n_periods": len(group),
            "n_ic": ic_stats["n"],
            "ic_mean": ic_stats["mean"],
            "ic_std": ic_stats["std"],
            "icir": ic_stats["ir"],
            "ic_t_stat": ic_stats["t_stat"],
            "ic_positive_rate": ic_stats["positive_rate"],
            "n_rankic": rankic_stats["n"],
            "rankic_mean": rankic_stats["mean"],
            "rankic_std": rankic_stats["std"],
            "rankicir": rankic_stats["ir"],
            "rankic_t_stat": rankic_stats["t_stat"],
            "rankic_positive_rate": rankic_stats["positive_rate"],
        })

    return pd.DataFrame(rows)


def calc_cumulative_ic(ic_by_period):
    """
    Cumulative sum of IC and RankIC over time.
    """
    out = ic_by_period.sort_values(PERIOD_GROUP_COLS).copy()
    group = out.groupby(GROUP_COLS, sort=False)
    out["cumulative_ic"] = group["ic"].transform(lambda x: x.fillna(0.0).cumsum())
    out["cumulative_rankic"] = group["rankic"].transform(lambda x: x.fillna(0.0).cumsum())
    return out


def assign_deciles(pred):
    """
    Assign cross-sectional deciles within date/model/universe group.

    D10 has the highest predicted probability.
    """
    pred = pred.copy()
    pred["decile"] = np.nan

    for _, index in pred.groupby(PERIOD_GROUP_COLS, sort=False).groups.items():
        if len(index) < N_DECILES:
            continue

        rank = pred.loc[index, "pred_prob"].rank(method="first")
        decile = pd.qcut(rank, N_DECILES, labels=False) + 1
        pred.loc[index, "decile"] = decile.astype(float).values

    return pred


def calc_decile_returns(pred):
    """
    Average future return for D1-D10 on each date.
    """
    valid = pred.dropna(subset=["decile"]).copy()
    if valid.empty:
        return pd.DataFrame()

    valid["decile"] = valid["decile"].astype(int)

    decile_ret = (
        valid
        .groupby(PERIOD_GROUP_COLS + ["decile"], sort=True)
        .agg(
            mean_future_ret=("future_ret", "mean"),
            median_future_ret=("future_ret", "median"),
            avg_pred_prob=("pred_prob", "mean"),
            num_stocks=("code", "count"),
        )
        .reset_index()
    )

    decile_ret["portfolio_name"] = "D" + decile_ret["decile"].astype(str)
    return decile_ret


def calc_decile_summary(decile_returns):
    """
    Full-period D1-D10 average future return summary.
    """
    if decile_returns.empty:
        return pd.DataFrame()

    return (
        decile_returns
        .groupby(GROUP_COLS + ["decile", "portfolio_name"], sort=True)
        .agg(
            n_periods=("date", "count"),
            mean_future_ret=("mean_future_ret", "mean"),
            std_period_future_ret=("mean_future_ret", "std"),
            median_period_future_ret=("median_future_ret", "median"),
            avg_pred_prob=("avg_pred_prob", "mean"),
            avg_num_stocks=("num_stocks", "mean"),
        )
        .reset_index()
    )


def calc_decile_monotonicity(decile_summary):
    """
    Check whether average future return rises from D1 to D10.
    """
    rows = []

    if decile_summary.empty:
        return pd.DataFrame()

    for keys, group in decile_summary.groupby(GROUP_COLS, sort=True):
        exp_name, model_name, universe_group = keys
        group = group.sort_values("decile").dropna(subset=["mean_future_ret"])

        deciles = group["decile"].astype(float)
        returns = group["mean_future_ret"].astype(float)
        diffs = returns.diff().dropna()

        if len(group) >= 2:
            decile_return_spearman = safe_corr(
                deciles.rank(method="average"),
                returns.rank(method="average"),
            )
            strict = bool((diffs > 0).all())
            nonstrict = bool((diffs >= 0).all())
            violations = int((diffs < 0).sum())
        else:
            decile_return_spearman = np.nan
            strict = False
            nonstrict = False
            violations = np.nan

        rows.append({
            "experiment_name": exp_name,
            "model_name": model_name,
            "universe_group": universe_group,
            "valid_decile_count": len(group),
            "decile_return_spearman": decile_return_spearman,
            "strict_monotonic_increasing": strict,
            "nonstrict_monotonic_increasing": nonstrict,
            "adjacent_violation_count": violations,
        })

    return pd.DataFrame(rows)


def prepare_daily_return_lookup(daily_returns):
    """
    Build trading calendar and date -> stock return lookups.
    """
    trading_dates = pd.DatetimeIndex(sorted(daily_returns["date"].dropna().unique()))
    date_to_pos = {date: i for i, date in enumerate(trading_dates)}

    close_to_close_by_date = {}
    open_to_close_by_date = {}

    for date, group in daily_returns.groupby("date", sort=True):
        close_to_close_by_date[pd.Timestamp(date)] = (
            group
            .groupby("code")["ret_1d"]
            .mean()
        )
        open_to_close_by_date[pd.Timestamp(date)] = (
            group
            .groupby("code")["open_to_close_ret_1d"]
            .mean()
        )

    return trading_dates, date_to_pos, close_to_close_by_date, open_to_close_by_date


def unique_nonnull(values):
    """
    Keep first occurrence of each non-null code.
    """
    return [x for x in pd.unique(pd.Series(values).dropna())]


def make_weights_from_cohorts(cohorts):
    """
    Average active equal-weight cohorts into one long-only target weight vector.
    """
    weights = defaultdict(float)
    if not cohorts:
        return {}

    cohort_weight = 1.0 / len(cohorts)

    for cohort in cohorts:
        codes = cohort["codes"]
        if not codes:
            continue
        stock_weight = cohort_weight / len(codes)
        for code in codes:
            weights[code] += stock_weight

    return dict(weights)


def calc_weight_turnover(prev_weights, next_weights):
    """
    Sum absolute target-weight changes.
    """
    codes = set(prev_weights) | set(next_weights)
    return float(sum(abs(next_weights.get(code, 0.0) - prev_weights.get(code, 0.0)) for code in codes))


def calc_weighted_return(weights, daily_return):
    """
    Long-only weighted return. Missing stock returns are treated as zero.
    """
    if not weights:
        return np.nan, 0

    codes = list(weights)
    weight_array = np.array([weights[code] for code in codes], dtype=float)
    ret = daily_return.reindex(codes)
    missing_returns = int(ret.isna().sum())
    ret_array = ret.fillna(0.0).astype(float).values

    return float(np.dot(weight_array, ret_array)), missing_returns


def calc_cohort_return(cohort, close_to_close_return, open_to_close_return, return_pos):
    """
    Return for one equal-weight signal cohort on one holding day.
    """
    if return_pos == cohort["start_pos"]:
        daily_return = open_to_close_return
    else:
        daily_return = close_to_close_return

    codes = cohort["codes"]
    if not codes:
        return np.nan, 0

    weights = {code: 1.0 / len(codes) for code in codes}
    return calc_weighted_return(weights, daily_return)


def calc_active_cohort_return(active_cohorts, close_to_close_return, open_to_close_return, return_pos):
    """
    Average active cohort returns into one long-only portfolio return.
    """
    returns = []
    missing_returns = 0

    for cohort in active_cohorts:
        cohort_ret, cohort_missing = calc_cohort_return(
            cohort,
            close_to_close_return,
            open_to_close_return,
            return_pos,
        )
        if not pd.isna(cohort_ret):
            returns.append(cohort_ret)
        missing_returns += cohort_missing

    if not returns:
        return np.nan, missing_returns

    return float(np.mean(returns)), missing_returns


def build_signal_cohorts(group, date_to_pos, max_date_pos):
    """
    Convert signal rows into cohorts keyed by first holding date.
    """
    group = group.copy()
    group["signal_pos"] = group["date"].map(date_to_pos)
    group = group[group["signal_pos"].notna()].copy()

    signals_by_pos = defaultdict(list)
    max_end_pos = -1

    if group.empty:
        return signals_by_pos, max_end_pos

    group["signal_pos"] = group["signal_pos"].astype(int)
    group["decile"] = group["decile"].astype(int)
    group["horizon"] = group["horizon"].astype(int)

    for (signal_pos, decile, horizon), cohort in group.groupby(["signal_pos", "decile", "horizon"]):
        signal_pos = int(signal_pos)
        decile = int(decile)
        horizon = int(horizon)
        if horizon <= 0:
            continue

        codes = unique_nonnull(cohort["code"])
        if not codes:
            continue

        start_pos = signal_pos + 1
        end_pos = min(signal_pos + horizon, max_date_pos)
        if start_pos > end_pos:
            continue

        signals_by_pos[start_pos].append({
            "decile": decile,
            "start_pos": start_pos,
            "end_pos": end_pos,
            "codes": codes,
        })
        max_end_pos = max(max_end_pos, end_pos)

    return signals_by_pos, max_end_pos


def calc_overlapping_portfolios(pred, daily_returns):
    """
    Build D1-D10 long-only daily portfolio returns with overlapping cohorts.
    """
    signals = pred.dropna(subset=["decile", "horizon"]).copy()
    if signals.empty:
        return pd.DataFrame(), pd.DataFrame()

    (
        trading_dates,
        date_to_pos,
        close_to_close_by_date,
        open_to_close_by_date,
    ) = prepare_daily_return_lookup(daily_returns)
    if len(trading_dates) == 0:
        return pd.DataFrame(), pd.DataFrame()

    empty_return = pd.Series(dtype=float)
    portfolio_rows = []
    turnover_rows = []
    max_date_pos = len(trading_dates) - 1

    for keys, group in signals.groupby(GROUP_COLS, sort=True):
        exp_name, model_name, universe_group = keys
        signals_by_pos, max_end_pos = build_signal_cohorts(group, date_to_pos, max_date_pos)
        if max_end_pos < 0:
            continue

        min_start_pos = min(signals_by_pos)
        active = {decile: [] for decile in range(1, N_DECILES + 1)}
        prev_weights = {decile: {} for decile in range(1, N_DECILES + 1)}

        for return_pos in range(min_start_pos, max_end_pos + 1):
            for cohort in signals_by_pos.get(return_pos, []):
                active[cohort["decile"]].append(cohort)

            date = trading_dates[return_pos]
            close_to_close_return = close_to_close_by_date.get(pd.Timestamp(date), empty_return)
            open_to_close_return = open_to_close_by_date.get(pd.Timestamp(date), empty_return)

            for decile in range(1, N_DECILES + 1):
                active_start = [
                    cohort
                    for cohort in active[decile]
                    if cohort["end_pos"] >= return_pos
                ]
                active[decile] = active_start

                if not active_start:
                    prev_weights[decile] = {}
                    continue

                weights = make_weights_from_cohorts(active_start)
                start_turnover = calc_weight_turnover(prev_weights[decile], weights)
                gross_return, missing_returns = calc_active_cohort_return(
                    active_start,
                    close_to_close_return,
                    open_to_close_return,
                    return_pos,
                )

                active_end = [
                    cohort
                    for cohort in active_start
                    if cohort["end_pos"] > return_pos
                ]
                end_weights = make_weights_from_cohorts(active_end)
                end_turnover = calc_weight_turnover(weights, end_weights)
                turnover = start_turnover + end_turnover
                portfolio_name = f"D{decile}"

                turnover_rows.append({
                    "date": date,
                    "experiment_name": exp_name,
                    "model_name": model_name,
                    "universe_group": universe_group,
                    "decile": decile,
                    "portfolio_name": portfolio_name,
                    "turnover": turnover,
                    "start_turnover": start_turnover,
                    "end_turnover": end_turnover,
                    "active_cohorts": len(active_start),
                    "num_holdings": len(weights),
                    "num_missing_returns": missing_returns,
                })

                for cost_bps in COST_BPS_GRID:
                    cost_rate = float(cost_bps) / 10_000.0
                    portfolio_rows.append({
                        "date": date,
                        "experiment_name": exp_name,
                        "model_name": model_name,
                        "universe_group": universe_group,
                        "decile": decile,
                        "portfolio_name": portfolio_name,
                        "cost_bps": cost_bps,
                        "gross_return": gross_return,
                        "turnover": turnover,
                        "start_turnover": start_turnover,
                        "end_turnover": end_turnover,
                        "net_return": gross_return - turnover * cost_rate,
                        "active_cohorts": len(active_start),
                        "num_holdings": len(weights),
                        "num_missing_returns": missing_returns,
                    })

                active[decile] = active_end
                prev_weights[decile] = end_weights

    return pd.DataFrame(portfolio_rows), pd.DataFrame(turnover_rows)


def summarize_portfolios(portfolio_returns):
    """
    Annualized performance for long-only decile portfolios.
    """
    rows = []

    if portfolio_returns.empty:
        return pd.DataFrame()

    group_cols = GROUP_COLS + ["decile", "portfolio_name", "cost_bps"]

    for keys, group in portfolio_returns.groupby(group_cols, sort=True):
        exp_name, model_name, universe_group, decile, portfolio_name, cost_bps = keys
        group = group.sort_values("date")

        gross_ret = group["gross_return"].dropna()
        net_ret = group["net_return"].dropna()
        if len(net_ret) == 0:
            continue

        ann_return = net_ret.mean() * TRADING_DAYS_PER_YEAR
        ann_vol = net_ret.std(ddof=1) * np.sqrt(TRADING_DAYS_PER_YEAR)
        sharpe = ann_return / ann_vol if ann_vol > 0 else np.nan

        gross_nav = (1.0 + gross_ret).cumprod()
        net_nav = (1.0 + net_ret).cumprod()

        final_gross_nav = gross_nav.iloc[-1] if len(gross_nav) else np.nan
        final_net_nav = net_nav.iloc[-1] if len(net_nav) else np.nan

        rows.append({
            "experiment_name": exp_name,
            "model_name": model_name,
            "universe_group": universe_group,
            "decile": decile,
            "portfolio_name": portfolio_name,
            "cost_bps": cost_bps,
            "n_periods": len(net_ret),
            "annualized_return": ann_return,
            "annualized_volatility": ann_vol,
            "sharpe": sharpe,
            "max_drawdown": max_drawdown(net_nav),
            "win_rate": (net_ret > 0).mean(),
            "avg_turnover": group["turnover"].mean(),
            "gross_cumulative_return": final_gross_nav - 1.0,
            "net_cumulative_return": final_net_nav - 1.0,
            "final_gross_nav": final_gross_nav,
            "final_net_nav": final_net_nav,
        })

    return pd.DataFrame(rows)


def calc_cost_sensitivity(performance_summary):
    """
    Cost sensitivity table derived from long-only portfolio summaries.
    """
    if performance_summary.empty:
        return pd.DataFrame()

    columns = [
        "experiment_name",
        "model_name",
        "universe_group",
        "decile",
        "portfolio_name",
        "cost_bps",
        "annualized_return",
        "sharpe",
        "max_drawdown",
        "avg_turnover",
        "gross_cumulative_return",
        "net_cumulative_return",
        "final_net_nav",
    ]
    return performance_summary[columns].copy()


def run_one_prediction_file(path, daily_returns):
    """
    Run all evaluation layers for one prediction file.
    """
    pred = load_prediction_file(path)
    pred = add_universe_groups(pred)

    ic_by_period = calc_ic_by_period(pred)
    ic_summary = calc_ic_summary(ic_by_period)
    cumulative_ic = calc_cumulative_ic(ic_by_period)

    pred = assign_deciles(pred)
    decile_returns = calc_decile_returns(pred)
    decile_summary = calc_decile_summary(decile_returns)
    decile_monotonicity = calc_decile_monotonicity(decile_summary)

    portfolio_returns, portfolio_turnover = calc_overlapping_portfolios(pred, daily_returns)
    performance_summary = summarize_portfolios(portfolio_returns)
    cost_sensitivity = calc_cost_sensitivity(performance_summary)

    return {
        "ic_by_period": ic_by_period,
        "ic_summary": ic_summary,
        "cumulative_ic": cumulative_ic,
        "decile_returns": decile_returns,
        "decile_summary": decile_summary,
        "decile_monotonicity": decile_monotonicity,
        "portfolio_returns": portfolio_returns,
        "portfolio_turnover": portfolio_turnover,
        "performance_summary": performance_summary,
        "cost_sensitivity": cost_sensitivity,
    }


def concat_frames(frames):
    """
    Concatenate non-empty DataFrames.
    """
    frames = [frame for frame in frames if frame is not None and not frame.empty]
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def save_table(df, filename):
    """
    Save one output table.
    """
    out_path = TABLE_DIR / filename
    df.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"Saved {filename}: {out_path}")


def main():
    pred_files = sorted(glob.glob(str(PRED_DIR / "pred_*.parquet")))
    if not pred_files:
        raise RuntimeError(f"No prediction files found in {PRED_DIR}")

    daily_returns = load_daily_returns()

    output_names = [
        "ic_by_period",
        "ic_summary",
        "cumulative_ic",
        "decile_returns",
        "decile_summary",
        "decile_monotonicity",
        "portfolio_returns",
        "portfolio_turnover",
        "performance_summary",
        "cost_sensitivity",
    ]
    outputs = {name: [] for name in output_names}

    for path in pred_files:
        result = run_one_prediction_file(path, daily_returns)
        for name in output_names:
            outputs[name].append(result[name])

    for name in output_names:
        save_table(concat_frames(outputs[name]), f"{name}.csv")

    perf = concat_frames(outputs["performance_summary"])
    if not perf.empty:
        print("\nPerformance summary:")
        print(perf)


if __name__ == "__main__":
    main()
