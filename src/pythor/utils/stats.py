import joblib, os, glob, copy
import pandas as pd
import numpy as np


import scipy.stats

from .linalg import numpybackward


"""
Calculating Rolling Stats from a list of numpy arrays 

"""

## current_data is np array of size (1,no_features)
def update_cumulative_stats_step(rolling_stats_dict, current_data):
    ## Reshape Input
    current_data = np.nan_to_num(
        np.array(current_data).reshape(1, -1),
    )
    no_features = current_data.shape[1]
    ## For
    for k in [
        "cumulative_mean",
        "cumulative_SSD",
        "cumulative_variance",
        "pos_cumulative_sum",
        "pos_cumulative_sum_max",
        "pos_max_drawdown",
        "neg_cumulative_sum",
        "neg_cumulative_sum_max",
        "neg_max_drawdown",
        "max_drawdown",
        "num_reversals",
        "reversal_freq",
        "current_run_length",
        "average_run_length",
        "num_wins",
        "consistent_rate",
    ]:
        if not k in rolling_stats_dict.keys():
            rolling_stats_dict[k] = np.zeros(
                (
                    1,
                    no_features,
                ),
            )

    ## Welford's online algorithm
    ## Rolling Mean Update
    old_mean = rolling_stats_dict["cumulative_mean"].copy()
    rolling_stats_dict["num_samples"] = rolling_stats_dict.get("num_samples", 0) + 1
    rolling_stats_dict["cumulative_mean"] = (
        old_mean + (current_data - old_mean) / rolling_stats_dict["num_samples"]
    )
    new_mean = rolling_stats_dict["cumulative_mean"].copy()
    ## Rolling Variance Update
    correction = (current_data - old_mean) * (current_data - new_mean)
    rolling_stats_dict["cumulative_SSD"] = (
        rolling_stats_dict["cumulative_SSD"] + correction
    )
    rolling_stats_dict["cumulative_variance"] = (
        rolling_stats_dict["cumulative_SSD"] / rolling_stats_dict["num_samples"]
    )
    ## Pos/Neg Max DrawDown Calculation
    rolling_stats_dict["pos_cumulative_sum"] = (
        rolling_stats_dict["pos_cumulative_sum"] + current_data
    )
    rolling_stats_dict["neg_cumulative_sum"] = (
        rolling_stats_dict["pos_cumulative_sum"] * -1
    )
    rolling_stats_dict["pos_cumulative_sum_max"] = np.maximum(
        rolling_stats_dict["pos_cumulative_sum"],
        rolling_stats_dict["pos_cumulative_sum_max"],
    )
    rolling_stats_dict["neg_cumulative_sum_max"] = np.minimum(
        rolling_stats_dict["neg_cumulative_sum"],
        rolling_stats_dict["neg_cumulative_sum_max"],
    )

    pos_drawdown = (
        rolling_stats_dict["pos_cumulative_sum_max"]
        - rolling_stats_dict["pos_cumulative_sum"]
    )
    rolling_stats_dict["pos_max_drawdown"] = np.maximum(
        rolling_stats_dict["pos_max_drawdown"], pos_drawdown
    )
    neg_drawdown = (
        rolling_stats_dict["neg_cumulative_sum"]
        - rolling_stats_dict["neg_cumulative_sum_max"]
    )
    rolling_stats_dict["neg_max_drawdown"] = np.maximum(
        rolling_stats_dict["neg_max_drawdown"], neg_drawdown
    )
    rolling_stats_dict["max_drawdown"] = np.minimum(
        rolling_stats_dict["pos_max_drawdown"], rolling_stats_dict["neg_max_drawdown"]
    )
    ## Reversal/Current Run Length/Average Run Length
    reversal = np.where(old_mean * new_mean < 0, 1, 0)
    rolling_stats_dict["num_reversals"] += reversal
    rolling_stats_dict["current_run_length"] = np.where(
        reversal == 1, 1, rolling_stats_dict["current_run_length"] + 1
    )
    temp = rolling_stats_dict["num_reversals"] + 1
    rolling_stats_dict["average_run_length"] = (
        1 + temp * rolling_stats_dict["average_run_length"]
    ) / temp
    rolling_stats_dict["reversal_freq"] = (
        rolling_stats_dict["num_reversals"] / rolling_stats_dict["num_samples"]
    )
    ## Win Rate/Lose Rate
    rolling_stats_dict["num_wins"] = rolling_stats_dict["num_wins"] + np.where(
        new_mean > 0, 1, 0
    )
    win_rate = rolling_stats_dict["num_wins"] / rolling_stats_dict["num_samples"]
    rolling_stats_dict["consistent_rate"] = np.maximum(win_rate, 1 - win_rate)
    return rolling_stats_dict


def update_cumulative_stats(data):
    output = dict()
    data = np.array(data)
    for i in range(data.shape[0]):
        output = update_cumulative_stats_step(output, data[i, :])
    return output


"""
Strategy Metrics

Default to set interval to be 13 for presenting annual return?

"""
import scipy.stats


def strategy_metrics(
    strategy_raw, interval=1, compound=False, accuracy=4, no_days=20, payout_ratio=0.06
):
    strategy = np.array(strategy_raw)
    if len(strategy.shape) < 2:
        strategy = strategy.reshape(-1, 1)
    epsilon = 1e-6
    results = dict()
    results["mean"] = np.mean(strategy, axis=0) * interval
    results["volatility"] = np.clip(
        np.std(strategy, axis=0) * np.sqrt(interval), epsilon, np.inf
    )
    results["skew"] = scipy.stats.skew(strategy, axis=0)
    results["kurtosis"] = scipy.stats.kurtosis(strategy, axis=0)
    if not compound:
        portfolio = np.cumsum(strategy, axis=0)
        temp = pd.DataFrame(portfolio).cummax(axis=0).values
        dd = portfolio - temp
    else:
        portfolio = np.cumprod(1 + strategy * payout_ratio * 5, axis=0)
        temp = pd.DataFrame(portfolio).cummax(axis=0).values
        dd = (portfolio - temp) / temp
    if compound:
        portfolio_ts = pd.DataFrame(portfolio)
        log_returns = np.log(portfolio_ts) - np.log(portfolio_ts.shift(1))
        results["mean"] = np.mean(log_returns) * 52
        results["volatility"] = np.std(log_returns) * np.sqrt(52)
    results["max_drawdown"] = np.clip(-1 * dd.min(axis=0), epsilon, np.inf)
    results["sharpe"] = results["mean"] / results["volatility"]
    results["calmar"] = results["mean"] / results["max_drawdown"]
    df = pd.DataFrame(results)
    if isinstance(strategy_raw, pd.DataFrame):
        df.index = strategy_raw.columns
        return df.round(accuracy)
    else:
        ## For Backward Comptability with Optuna Optimisation
        return df.round(accuracy).loc[0].to_dict()


"""
Dynamic paired t-test

"""


from scipy.stats import norm


def dynamic_t_test(obs_a, obs_b, duration_cutoffs=[0.2, 0.4, 0.6, 0.8, 1]):
    assert obs_a.shape[0] == obs_b.shape[0]
    t_stats = list()
    p_values = list()
    for cutoff in duration_cutoffs:
        no_samples = int(cutoff * obs_a.shape[0])
        delta_series = (obs_a - obs_b)[:no_samples]
        variance = np.std(delta_series) * np.sqrt(no_samples)
        ## T-distribution or Gaussian distribution
        t_stat = np.sum(delta_series) / variance
        p_value = (1 - norm.cdf(np.abs(t_stat))) / 2
        t_stats.append(t_stat)
        p_values.append(p_value)
    ## Sum over pvalues as Bonferroni correction
    return np.sum(p_values), t_stats


def dynamic_t_mtx(B):
    output = np.ones((B.shape[1], B.shape[1]))
    for i in range(B.shape[1]):
        for j in range(i):
            sample1 = B.iloc[:, i].values
            sample2 = B.iloc[:, j].values
            output[i, j] = dynamic_t_test(sample1, sample2)[0]
            output[j, i] = output[i, j]
    return output


"""


Bond Yield


"""

import scipy
import datetime


def years_to_maturity(maturity_date):
    end = datetime.datetime.strptime(maturity_date, "%Y-%m-%d")
    start = datetime.datetime.now()
    return (end - start).days / 365


def bond_yield(current, coupon_rate=0.125, freq=2, maturity_date="2026-01-30", par=100):

    maturity = years_to_maturity(maturity_date)

    def bond_formula(d, current, par, coupon_rate=0.125, freq=2, maturity=2.75):
        coupon = coupon_rate / freq
        cashflow_count = int(np.floor(maturity / (1 / freq)))
        first_cashflow_day = maturity - cashflow_count / freq
        bond_price = par * np.power(d, maturity) + coupon * (
            np.power(d, maturity + 1 / freq) - np.power(d, first_cashflow_day)
        ) / (np.power(d, 1 / freq) - 1)
        accured_int = coupon_rate * (1 / freq - first_cashflow_day)
        return bond_price - current - accured_int

    coupon_rate_guess = (
        coupon_rate + np.exp(np.log(par / current) / maturity) - 1
    ) / 100
    x0 = 1 / (1 + coupon_rate_guess)
    ans = scipy.optimize.fsolve(
        bond_formula, x0=x0, args=(current, par, coupon_rate, freq, maturity)
    )
    return 1 / ans - 1
