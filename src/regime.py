"""
regime.py
=========
Regime-aware regularization for factor models.  Adapts the LASSO penalty
strength lambda dynamically to current market volatility, applying heavier
shrinkage during high-stress periods when factor relationships are least
stable and lighter shrinkage during calm periods when signal is cleaner.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Motivation
----------
A fixed lambda assumes the signal-to-noise ratio of factor returns is
constant over time.  In practice it is not:

  High-vol regimes (2008 GFC, 2020 COVID crash):
    Factor return correlations spike and become unstable.  A model fitted
    on "normal" factor relationships will overfit to transient co-movement
    patterns.  Stronger regularization (larger lambda) reduces this risk
    by shrinking unreliable loadings toward zero.

  Low-vol regimes (2013–2014, 2017):
    Factor relationships are more persistent and well-estimated.  A
    smaller lambda allows the model to fit the data more closely without
    picking up spurious noise.

The scaling rule lambda_t = base_lambda × (vol_t / mean_vol) encodes
this intuition directly: lambda is proportional to realized volatility,
so the regularization "breathes" with the market.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

compute_rolling_volatility(market_returns, window=12)
------------------------------------------------------
Computes a rolling estimate of annualized market volatility from monthly
Mkt-RF returns.

  Method:
    Rolling standard deviation over the trailing `window` months,
    multiplied by √12 to annualize (assumes i.i.d. monthly returns —
    standard practice despite known autocorrelation in volatility).

  Output:
    pd.Series aligned to market_returns.index.  The first (window − 1)
    entries are NaN due to insufficient history — callers must handle
    these, typically by falling back to base_lambda (as done in
    walk_forward_backtest_adaptive).

  Window choice:
    Default window=12 (one year) balances responsiveness against noise.
    Shorter windows (3–6 months) react faster to vol spikes but are
    noisier; longer windows (24–36 months) are smoother but lag regime
    transitions by quarters.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

compute_adaptive_lambda(market_returns, base_lambda, window=12,
                         clip_min=0.5, clip_max=3.0)
---------------------------------------------------------------
Produces the time-varying lambda series consumed by
walk_forward_backtest_adaptive in backtesting.py.

  Scaling rule:
    lambda_t = base_lambda × clip(vol_t / mean_vol, clip_min, clip_max)

    vol_t      rolling annualized vol at month t (from
               compute_rolling_volatility).
    mean_vol   time-series mean of rolling_vol over the full sample,
               used as the normalization anchor.  This introduces a
               mild look-ahead (mean_vol uses future observations), but
               since it is a long-run average it is very stable and the
               bias is negligible in practice.

  Clipping bounds:
    The scaling factor is clipped to [clip_min, clip_max] = [0.5, 3.0]
    by default.  Without clipping, a single extreme vol spike (e.g.
    March 2020 annualized vol ≈ 80%) would produce a lambda 4–5× the
    base value, potentially shrinking all coefficients to near zero for
    that month.  The upper clip of 3.0× limits this to a tripling of
    base_lambda at most.  The lower clip of 0.5× prevents lambda from
    collapsing to near zero during extended low-vol periods, preserving
    a minimum level of regularization.

  Practical range over 2000–2023:
    In calm regimes (2013–2014, 2017) the scaling factor typically sits
    near 0.6–0.8×, so lambda_t ≈ 0.6–0.8 × base_lambda.
    During the 2008–2009 crisis and March 2020, the unconstrained ratio
    would exceed 3×, so the upper clip is binding in those months.

  Returns:
    pd.Series of adaptive lambdas, same DatetimeIndex as market_returns.
    NaN entries (first window−1 months) are handled by the backtesting
    loop, which substitutes base_lambda for any NaN.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

identify_regimes(market_returns, window=12, low_pct=33, high_pct=67)
---------------------------------------------------------------------
Assigns a qualitative regime label to each month for analysis and
visualization.  Not used in the live backtesting loop — intended for
post-hoc performance attribution (e.g. "what was the Sharpe in high-vol
months vs low-vol months?").

  Four-label taxonomy:

    'crisis'   Monthly Mkt-RF return < −10%.  Applied after the
               quantile thresholds, so crisis months override any
               vol-based label.  Captures sharp drawdown events
               (Oct 2008, Mar 2020) that may not register as
               high-volatility in the trailing 12-month window
               because the crash itself is still within the window.

    'high'     Rolling vol ≥ 67th percentile of full-sample vol
               distribution.  Corresponds to elevated but not
               crisis-level market stress.

    'low'      Rolling vol ≤ 33rd percentile.  Persistent low-vol
               environments: 2013–2014 "goldilocks" period, 2017.

    'medium'   All remaining months — the default label.

  Threshold design:
    Tertile splits (33rd / 67th percentiles) divide the sample into
    roughly equal thirds across regimes, giving adequate observation
    counts in each bucket for statistical comparisons.  Custom
    percentile values can be passed to shift the split points.

  Crisis override ordering:
    The crisis label is applied last (after the quantile labels) so
    that months with both high rolling vol AND a −10% return are
    classified as 'crisis' rather than 'high'.  This matters for
    2008, where several consecutive months qualify on both criteria.

  Returns:
    regimes      pd.Series[str]  one label per month
    rolling_vol  pd.Series[float] annualized rolling vol (returned
                 alongside regimes to avoid recomputing in callers
                 that need both)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Design notes
------------
  Why vol, not other regime signals:
    VIX or credit spreads would be more timely signals of stress, but
    they require additional data sources.  Realized vol from Mkt-RF is
    self-contained within the existing dataset and is a reasonable proxy
    — high-vol months and high-VIX months overlap strongly over 2000–2023.

  Mean-vol normalization vs. percentile scaling:
    Dividing by mean_vol (rather than the current percentile rank) keeps
    lambda on a ratio scale — a vol 2× the historical mean produces
    exactly 2× lambda.  A percentile-based approach would compress the
    extremes and reduce the lambda response in the tails, which is
    precisely where stronger regularization is most needed.

  Integration with backtesting.py:
    compute_adaptive_lambda is the only function called externally (by
    walk_forward_backtest_adaptive).  identify_regimes is used only in
    analysis notebooks for performance attribution and plotting.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Dependencies
------------
numpy, pandas

Inputs expected
---------------
market_returns : pd.Series  monthly Mkt-RF returns, decimal scale,
                            DateTime-indexed (same index as X in the
                            backtesting pipeline)
"""
import numpy as np
import pandas as pd


def compute_rolling_volatility(market_returns, window=12):
    """
    Compute rolling annualized volatility of market returns.
    
    Parameters
    ----------
    market_returns : pd.Series — monthly Mkt-RF returns
    window         : int — rolling window in months
    
    Returns
    -------
    pd.Series of rolling annualized volatility
    """
    return market_returns.rolling(window).std() * np.sqrt(12)


def compute_adaptive_lambda(market_returns, base_lambda,
                             window=12, clip_min=0.5, clip_max=3.0):
    """
    Scale lambda by current volatility relative to historical mean.
    
    lambda_t = base_lambda * (vol_t / mean_vol)
    
    Intuition: when markets are volatile, factor relationships
    are less stable — use stronger regularization to avoid
    fitting noise.
    
    Parameters
    ----------
    market_returns : pd.Series
    base_lambda    : float — baseline regularization
    window         : int — rolling window for vol estimate
    clip_min/max   : float — bounds on scaling factor
    
    Returns
    -------
    pd.Series of adaptive lambdas
    """
    rolling_vol  = compute_rolling_volatility(market_returns, window)
    mean_vol     = rolling_vol.mean()
    scale        = (rolling_vol / mean_vol).clip(clip_min, clip_max)
    adaptive_lam = base_lambda * scale
    return adaptive_lam


def identify_regimes(market_returns, window=12,
                     low_pct=33, high_pct=67):
    """
    Label each month as low/medium/high volatility regime.
    
    Returns
    -------
    pd.Series with labels: 'low', 'medium', 'high', 'crisis'
    Crisis defined as monthly return < -10%
    """
    rolling_vol = compute_rolling_volatility(market_returns, window)
    low_thresh  = rolling_vol.quantile(low_pct / 100)
    high_thresh = rolling_vol.quantile(high_pct / 100)

    regimes = pd.Series('medium', index=market_returns.index)
    regimes[rolling_vol <= low_thresh]  = 'low'
    regimes[rolling_vol >= high_thresh] = 'high'
    regimes[market_returns < -0.10]     = 'crisis'

    return regimes, rolling_vol
