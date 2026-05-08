# -*- coding: utf-8 -*-
"""
06_backtest_decile.py

Purpose
-------
Run simple cross-sectional decile backtest for prediction files.

Inputs
------
outputs/predictions/pred_*.parquet

Outputs
-------
outputs/tables/decile_returns.csv
outputs/tables/performance_summary.csv

Backtest logic
--------------
1. On each date, sort stocks by pred_prob.
2. Assign deciles D1-D10.
3. Compute equal-weight future return for each decile.
4. Compute D10-D1 long-short return.
5. Apply simple turnover-based transaction cost.
6. Summarize performance.

Important simplification
------------------------
This MVP uses non-overlapping prediction dates as portfolio return dates.
For more realistic implementation, you can later convert to daily overlapping
holding portfolios.
"""

import glob
import numpy as np
import pandas as pd

from config import (
    PRED_DIR,
    TABLE_DIR,
    N_DECILES,
    ONE_WAY_COST_BPS,
    TRADING_DAYS_PER_YEAR,
)


def assign_deciles(pred):
    """
    Assign cross-sectional deciles by date.

    D10 means highest predicted probability.
    D1 means lowest predicted probability.
    """
    pred = pred.copy()

    def _assign_one_day(x):
        if len(x) < N_DECILES:
            x["decile"] = np.nan
            return x

        # rank(method="first") avoids duplicate bin-edge issues.
        rank = x["pred_prob"].rank(method="first")
        x["decile"] = pd.qcut(rank, N_DECILES, labels=False) + 1
        return x

    pred = pred.groupby("date", group_keys=False).apply(_assign_one_day)
    pred["decile"] = pred["decile"].astype("float")
    return pred


def calc_decile_returns(pred):
    """
    Equal-weight return for each decile on each date.
    """
    decile_ret = (
        pred.dropna(subset=["decile"])
        .groupby(["date", "experiment_name", "model_name", "decile"])
        .agg(
            gross_return=("future_ret", "mean"),
            num_stocks=("code", "count"),
        )
        .reset_index()
    )

    return decile_ret


def pivot_long_short(decile_ret):
    """
    Construct D10-D1 long-short return from decile return table.
    """
    pivot = decile_ret.pivot_table(
        index=["date", "experiment_name", "model_name"],
        columns="decile",
        values="gross_return",
    )

    # D10 highest score, D1 lowest score.
    pivot["D10_minus_D1"] = pivot[10.0] - pivot[1.0]
    pivot["D10"] = pivot[10.0]
    pivot["D1"] = pivot[1.0]

    out = pivot[["D10", "D1", "D10_minus_D1"]].reset_index()
    return out


def calc_turnover(pred):
    """
    Very simple equal-weight turnover for D10 and D1.

    For each date, construct set of stocks in D10 and D1.
    Turnover is approximated as:
        1 - overlap ratio with previous rebalance

    This is a simplified estimate.
    More exact portfolio-weight turnover can be implemented later.
    """
    pred = pred.dropna(subset=["decile"]).copy()
    dates = sorted(pred["date"].unique())

    rows = []
    prev_d10 = set()
    prev_d1 = set()

    for date in dates:
        day = pred[pred["date"] == date]

        exp_name = day["experiment_name"].iloc[0]
        model_name = day["model_name"].iloc[0]

        d10 = set(day.loc[day["decile"] == 10, "code"])
        d1 = set(day.loc[day["decile"] == 1, "code"])

        if len(prev_d10) == 0:
            turnover_d10 = 1.0
        else:
            turnover_d10 = 1.0 - len(d10 & prev_d10) / max(len(d10), 1)

        if len(prev_d1) == 0:
            turnover_d1 = 1.0
        else:
            turnover_d1 = 1.0 - len(d1 & prev_d1) / max(len(d1), 1)

        rows.append({
            "date": date,
            "experiment_name": exp_name,
            "model_name": model_name,
            "turnover_D10": turnover_d10,
            "turnover_D1": turnover_d1,
            "turnover_long_short": 0.5 * (turnover_d10 + turnover_d1),
        })

        prev_d10 = d10
        prev_d1 = d1

    return pd.DataFrame(rows)


def add_transaction_cost(ls_ret, turnover):
    """
    Apply simple one-way transaction cost.

    cost per period = turnover * one_way_cost
    where one_way_cost = bps / 10000.
    """
    cost = ONE_WAY_COST_BPS / 10_000.0

    out = ls_ret.merge(
        turnover,
        on=["date", "experiment_name", "model_name"],
        how="left",
    )

    out["D10_net"] = out["D10"] - out["turnover_D10"].fillna(0) * cost
    out["D1_net"] = out["D1"] - out["turnover_D1"].fillna(0) * cost

    # Long-short requires trading both legs.
    out["D10_minus_D1_net"] = (
        out["D10_minus_D1"]
        - out["turnover_long_short"].fillna(0) * cost
    )

    return out


def max_drawdown(nav):
    """
    Compute maximum drawdown from a NAV series.
    """
    running_max = nav.cummax()
    dd = nav / running_max - 1.0
    return dd.min()


def summarize_return_series(ret, name, exp_name, model_name):
    """
    Annualized performance summary.

    This assumes each row is one rebalance period.
    For I5R5 and I20R20, annualization should ideally depend on holding period.
    Here we use a rough 252-day annualization based on observed frequency.

    Codex can refine this by reading experiment horizon:
    I5R5 -> approx 252/5 periods per year
    I20R20 -> approx 252/20 periods per year
    """
    ret = ret.dropna()
    if len(ret) == 0:
        return None

    # Infer period-per-year from median date spacing.
    # If dates are every trading day, this will be close to 252.
    # If dates are every 5 days, close to 50.
    # If dates are every 20 days, close to 12.
    # For MVP, use count per year directly from sample length.
    ann_factor = TRADING_DAYS_PER_YEAR

    mean = ret.mean()
    vol = ret.std(ddof=1)

    ann_return = mean * ann_factor
    ann_vol = vol * np.sqrt(ann_factor)
    sharpe = ann_return / ann_vol if ann_vol > 0 else np.nan

    nav = (1.0 + ret).cumprod()
    mdd = max_drawdown(nav)

    win_rate = (ret > 0).mean()

    return {
        "experiment_name": exp_name,
        "model_name": model_name,
        "portfolio_name": name,
        "n_periods": len(ret),
        "annual_return": ann_return,
        "annual_volatility": ann_vol,
        "sharpe": sharpe,
        "max_drawdown": mdd,
        "win_rate": win_rate,
        "final_nav": nav.iloc[-1],
    }


def run_one_prediction_file(path):
    """
    Run decile backtest for one prediction file.
    """
    print(f"Backtesting: {path}")
    pred = pd.read_parquet(path)
    pred["date"] = pd.to_datetime(pred["date"])

    # Basic valid rows.
    pred = pred[
        pred["pred_prob"].notna()
        & pred["future_ret"].notna()
    ].copy()

    pred = assign_deciles(pred)
    decile_ret = calc_decile_returns(pred)
    ls_ret = pivot_long_short(decile_ret)
    turnover = calc_turnover(pred)
    ls_ret_net = add_transaction_cost(ls_ret, turnover)

    exp_name = pred["experiment_name"].iloc[0]
    model_name = pred["model_name"].iloc[0]

    summaries = []

    for col, pname in [
        ("D10", "D10_gross"),
        ("D1", "D1_gross"),
        ("D10_minus_D1", "D10_minus_D1_gross"),
        ("D10_net", "D10_net"),
        ("D10_minus_D1_net", "D10_minus_D1_net"),
    ]:
        summary = summarize_return_series(
            ls_ret_net[col],
            name=pname,
            exp_name=exp_name,
            model_name=model_name,
        )
        if summary is not None:
            summaries.append(summary)

    perf = pd.DataFrame(summaries)

    return decile_ret, ls_ret_net, perf


def main():
    pred_files = sorted(glob.glob(str(PRED_DIR / "pred_*.parquet")))
    if not pred_files:
        raise RuntimeError(f"No prediction files found in {PRED_DIR}")

    all_decile = []
    all_ls = []
    all_perf = []

    for path in pred_files:
        decile_ret, ls_ret_net, perf = run_one_prediction_file(path)
        all_decile.append(decile_ret)
        all_ls.append(ls_ret_net)
        all_perf.append(perf)

    decile_out = pd.concat(all_decile, ignore_index=True)
    ls_out = pd.concat(all_ls, ignore_index=True)
    perf_out = pd.concat(all_perf, ignore_index=True)

    decile_path = TABLE_DIR / "decile_returns.csv"
    ls_path = TABLE_DIR / "long_short_returns.csv"
    perf_path = TABLE_DIR / "performance_summary.csv"

    decile_out.to_csv(decile_path, index=False, encoding="utf-8-sig")
    ls_out.to_csv(ls_path, index=False, encoding="utf-8-sig")
    perf_out.to_csv(perf_path, index=False, encoding="utf-8-sig")

    print(f"Saved decile returns to: {decile_path}")
    print(f"Saved long-short returns to: {ls_path}")
    print(f"Saved performance summary to: {perf_path}")

    print("\nPerformance summary:")
    print(perf_out)


if __name__ == "__main__":
    main()