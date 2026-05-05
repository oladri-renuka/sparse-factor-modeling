"""
signal_discovery.py
===================
Bottom-up factor discovery pipeline.  Rather than starting from
pre-constructed Fama-French factors, this module builds 50+ candidate
predictive signals from raw price and volume data, then uses LASSO to
select which signals actually predict next-month returns cross-sectionally.
The result is a data-driven answer to the question: which technical and
statistical regularities in OHLCV data contain genuine alpha, and which
are noise?

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Universe
--------
50 large-cap US equities across 10 GICS sectors (5 stocks per sector):
Technology, Financials, Healthcare, Energy, Consumer, Industrials,
Utilities, Materials, Real Estate, Communication Services.  The 10-sector
spread ensures the cross-sectional signal regressions are not dominated
by a single industry's return pattern, and that sector-specific effects
do not masquerade as market-wide signals.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

download_price_data(tickers, start, end)
-----------------------------------------
Downloads monthly OHLCV data for all tickers via yfinance using the
'1mo' interval with auto_adjust=True (prices are split- and
dividend-adjusted).  Returns a multi-level DataFrame with
(Open/High/Low/Close/Volume) on the first axis and tickers on the second.

  Date range default: 2000-01-01 to 2023-12-31 — a 24-year panel chosen
  to span multiple full market cycles (dot-com bust, GFC, COVID crash,
  2022 rate-hike drawdown) while keeping the download size manageable.

  Note: yfinance data quality for monthly bars degrades before ~1993 and
  is unreliable for delisted tickers.  The 50-ticker universe was selected
  for survivorship stability over the chosen window, but survivorship bias
  is present — all 50 stocks survived to 2024 by construction.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

construct_signals(data)
------------------------
Builds the full signal library from the downloaded data.  All signals
are computed per-ticker and stored as (T × 50) DataFrames, then
concatenated into a single (T × (signals × tickers)) MultiIndex panel.

  Signal taxonomy (17 signals × 50 tickers = 850 columns total):

  1. Price momentum  — pct_change(k) for k ∈ {1, 3, 6, 12} months.
     Captures the well-documented cross-sectional momentum effect: past
     winners tend to outperform past losers over 3–12 month horizons
     (Jegadeesh & Titman 1993).

  2. Skip-month momentum  — mom_2_12 = pct_change(12) − pct_change(1).
     Omits the most recent month to avoid contamination by short-term
     reversal, which is mechanically negative at the one-month horizon
     due to bid-ask bounce and microstructure effects.

  3. Short-term reversal  — reversal_1m = −pct_change(1).
     Negated one-month return.  At monthly frequency, last month's losers
     tend to outperform next month (De Bondt & Thaler 1985 at long
     horizons; Jegadeesh 1990 at short horizons).

  4. Long-term reversal  — reversal_36m = −pct_change(36).
     Negated 36-month return.  Long-term losers (3-year horizon) tend to
     mean-revert, capturing the value-like contrarian effect.

  5. Realized volatility  — rolling(k).std() of monthly returns for
     k ∈ {3, 6, 12}.  Low-volatility stocks are known to earn higher
     risk-adjusted returns than high-volatility stocks (the low-vol
     anomaly), so vol itself is a candidate predictor in addition to
     being a risk control variable.

  6. Volatility ratio  — vol_ratio_3_12 = rolling(3).std() /
     rolling(12).std().  A ratio > 1 indicates vol is expanding
     (recent stress); < 1 indicates vol is contracting (calming).
     Captures the mean-reversion tendency of volatility itself.

  7. Price-to-moving-average ratio  — close / rolling(k).mean() − 1
     for k ∈ {3, 6, 12}.  Positive values indicate price is above its
     own recent trend (trend-following signal); negative values indicate
     below (mean-reversion signal).  These are discretized versions of
     the moving-average crossover rules used in technical analysis.

  8. 52-week high ratio  — close / rolling(12).max().  George & Hwang
     (2004) show that proximity to the 52-week high predicts positive
     future returns — investors use it as a reference point and
     underreact when prices approach it, creating a predictable drift.

  9. Volume trend  — volume / rolling(k).mean() − 1 for k ∈ {3, 6}.
     Abnormally high volume relative to the recent trend signals
     increased attention and can predict both continuation and reversal
     depending on price direction.  Only computed when Volume is present
     in the downloaded data.

  NaN handling:
    Small epsilon (1e-8) denominators prevent division-by-zero in ratio
    signals.  NaN values from insufficient history at the start of the
    sample propagate naturally and are handled by the 24-month burn-in
    in build_cross_sectional_dataset.

  Output:
    signal_df       MultiIndex DataFrame (T × (signal × ticker))
    signal_names    list of 17 signal name strings (outer index labels)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

build_cross_sectional_dataset(data, signal_names, forward_months=1)
--------------------------------------------------------------------
Reshapes the per-ticker signal panel into a stacked cross-sectional
dataset where each row is one (stock, month) observation, suitable for
fitting a single LASSO model that pools information across all stocks.

  Structure at each date t:
    X_t  (n_valid_stocks × n_signals)  signal matrix for month t
    y_t  (n_valid_stocks,)             forward return at t + forward_months

  Forward return construction:
    fwd_returns = close.pct_change(forward_months).shift(−forward_months)
    The negative shift aligns each row's target with the return realized
    over the next forward_months months — strictly future relative to the
    signal date.

  Burn-in:
    The loop begins at close.index[24], skipping the first 24 months to
    allow rolling signals with up to 12-month windows to be fully
    populated.  Months within forward_months of the end of the sample are
    also excluded (no future return available).

  Validity filter:
    A date is included only if at least 20 stocks have non-NaN forward
    returns.  Dates with fewer valid stocks (e.g., early in the sample
    when some tickers have not yet listed) are silently dropped to ensure
    the LASSO always has a well-conditioned cross-section to fit.

  Signal retrieval note:
    The inner loop uses a simplified lookup based on string matching
    ('1m', '3m', etc.) to re-derive signal values from raw price
    pct_change rather than from the signal_df panel.  This is an
    approximate reconstruction — it does not faithfully reproduce all 17
    signals (e.g. vol signals, ratio signals, reversal signals are mapped
    to their closest momentum analog).  For production use, signals should
    be read directly from signal_df rather than re-derived.

  Returns:
    X_list      list of (n_stocks × n_signals) arrays, one per date
    y_list      list of (n_stocks,) arrays, one per date
    dates_list  list of timestamps corresponding to each entry

  Downstream usage:
    X_list and y_list are passed to a LASSO solver month-by-month in a
    rolling cross-sectional regression — the same walk-forward structure
    as backtesting.py but operating on individual stocks rather than
    pre-built factor portfolios.  Coefficient frequency across months
    reveals which signals are persistently selected (genuine factors)
    vs. sporadically selected (noise or regime-specific).

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Known limitations
-----------------
  Survivorship bias:
    All 50 tickers are chosen because they exist and are liquid as of
    2024.  Stocks that were delisted, merged, or went bankrupt over
    2000–2023 are absent, upward-biasing all return statistics.

  Signal reconstruction approximation:
    The inner loop in build_cross_sectional_dataset re-derives signals
    via approximate string matching rather than reading from the fully
    constructed signal_df.  Vol signals, ratio signals, and reversal
    signals are silently replaced by their nearest momentum equivalent,
    so the actual features fed to LASSO differ from the documented 17.

  No cross-sectional standardization:
    Signal values are passed to LASSO without cross-sectional
    rank-normalization or winsorization.  Outlier returns in individual
    stocks (earnings surprises, M&A events) can dominate the regression
    and produce unstable coefficients.  Rank-normalizing signals within
    each cross-section before fitting is standard practice in production
    quantitative equity models.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Dependencies
------------
numpy, pandas, yfinance, itertools.product (imported but unused)

References
----------
Jegadeesh & Titman (1993) — cross-sectional momentum.
De Bondt & Thaler (1985), Jegadeesh (1990) — short and long-term reversal.
George & Hwang (2004) — 52-week high as return predictor.
"""
import numpy as np
import pandas as pd
import yfinance as yf
from itertools import product


# S&P 500 representative tickers across sectors
TICKERS = [
    # Tech
    'AAPL', 'MSFT', 'GOOGL', 'META', 'NVDA',
    # Finance
    'JPM', 'BAC', 'GS', 'MS', 'WFC',
    # Healthcare
    'JNJ', 'PFE', 'UNH', 'ABBV', 'MRK',
    # Energy
    'XOM', 'CVX', 'COP', 'SLB', 'EOG',
    # Consumer
    'AMZN', 'WMT', 'HD', 'MCD', 'NKE',
    # Industrials
    'BA', 'CAT', 'GE', 'MMM', 'HON',
    # Utilities
    'NEE', 'DUK', 'SO', 'AEP', 'EXC',
    # Materials
    'LIN', 'APD', 'ECL', 'DD', 'NEM',
    # Real Estate
    'AMT', 'PLD', 'CCI', 'EQIX', 'PSA',
    # Communication
    'VZ', 'T', 'CMCSA', 'NFLX', 'DIS'
]


def download_price_data(tickers=TICKERS,
                         start='2000-01-01',
                         end='2023-12-31'):
    """Download monthly OHLCV data via yfinance."""
    print(f'Downloading price data for {len(tickers)} stocks...')
    data = yf.download(tickers, start=start, end=end,
                        interval='1mo', auto_adjust=True,
                        progress=False)
    return data


def construct_signals(data):
    """
    Construct 50+ candidate predictive signals from price/volume data.

    Signal categories:
    1. Momentum (1,3,6,12 month)
    2. Short-term reversal (1 week via monthly proxy)
    3. Volatility (realized vol, vol ratio)
    4. Volume signals (turnover, volume trend)
    5. Price ratios (price to moving average)
    6. Trend signals (52-week high ratio)
    """
    close  = data['Close']
    volume = data['Volume'] if 'Volume' in data else None

    signals = {}

    # ── Momentum signals ──────────────────────────────────────
    for months in [1, 3, 6, 12]:
        ret = close.pct_change(months)
        signals[f'mom_{months}m'] = ret

    # ── Skip-month momentum (months 2-12, skip month 1) ──────
    signals['mom_2_12'] = close.pct_change(12) - close.pct_change(1)

    # ── Short-term reversal ───────────────────────────────────
    signals['reversal_1m'] = -close.pct_change(1)

    # ── Long-term reversal (36-month) ─────────────────────────
    signals['reversal_36m'] = -close.pct_change(36)

    # ── Realized volatility ───────────────────────────────────
    monthly_ret = close.pct_change(1)
    for window in [3, 6, 12]:
        signals[f'vol_{window}m'] = monthly_ret.rolling(window).std()

    # ── Volatility ratio (short / long) ───────────────────────
    signals['vol_ratio_3_12'] = (
        monthly_ret.rolling(3).std() /
        (monthly_ret.rolling(12).std() + 1e-8)
    )

    # ── Price to moving average ratio ─────────────────────────
    for window in [3, 6, 12]:
        ma = close.rolling(window).mean()
        signals[f'price_ma_{window}m'] = close / (ma + 1e-8) - 1

    # ── 52-week high ratio ─────────────────────────────────────
    high_12m = close.rolling(12).max()
    signals['high_52w_ratio'] = close / (high_12m + 1e-8)

    # ── Volume signals (if available) ─────────────────────────
    if volume is not None:
        for window in [3, 6]:
            vol_ma = volume.rolling(window).mean()
            signals[f'vol_trend_{window}m'] = (
                volume / (vol_ma + 1e-8) - 1
            )

    # Stack into panel: (date x ticker x signal)
    signal_df = pd.concat(signals, axis=1)
    signal_df.columns = pd.MultiIndex.from_tuples(
        [(sig, ticker)
         for sig in signals.keys()
         for ticker in close.columns],
        names=['signal', 'ticker']
    )

    return signal_df, list(signals.keys())


def build_cross_sectional_dataset(data, signal_names,
                                   forward_months=1):
    """
    Build cross-sectional dataset for return prediction.

    For each month t:
    - X: signal values at month t for all stocks
    - y: forward return at month t+1 for all stocks

    Returns stacked (observations x signals) dataset
    suitable for LASSO.
    """
    close       = data['Close']
    fwd_returns = close.pct_change(forward_months).shift(-forward_months)

    X_list, y_list, dates_list = [], [], []

    for date in close.index[24:-forward_months]:
        x_row, y_row = [], []
        valid        = True

        for ticker in close.columns:
            # Get signals for this stock at this date
            sig_vals = []
            for sig in signal_names:
                try:
                    val = data['Close'][ticker].pct_change(
                        1 if '1m' in sig else
                        3 if '3m' in sig else
                        6 if '6m' in sig else 12
                    ).loc[date]
                    sig_vals.append(val if not np.isnan(val) else 0)
                except Exception:
                    sig_vals.append(0)

            fwd_ret = fwd_returns[ticker].loc[date] \
                      if date in fwd_returns.index else np.nan

            if not np.isnan(fwd_ret):
                x_row.append(sig_vals)
                y_row.append(fwd_ret)

        if len(x_row) >= 20:
            X_list.append(np.array(x_row))
            y_list.append(np.array(y_row))
            dates_list.append(date)

    return X_list, y_list, dates_list
