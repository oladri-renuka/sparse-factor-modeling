"""
04_dollar_neutral_sharpe.py
===================
Decomposes the walk-forward LASSO strategy's Sharpe ratio across three
portfolio construction variants, isolating the contribution of short
selling and dollar-neutralization to overall risk-adjusted performance.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Motivation
----------
A long-short strategy's Sharpe ratio reflects two distinct sources of value:

  Signal quality: how well the LASSO predictions rank portfolios by
    future return — captured by the long-only strategy's performance.

  Short-selling alpha: the additional return from shorting predicted
    losers, net of the risk those short positions introduce —
    captured by the difference between long-short and long-only Sharpe.

Separating these allows attribution of the overall strategy performance
to prediction quality alone vs. the structural benefit of shorting.
Dollar-neutralization as a third variant tests whether the mean-centering
of weights (removing any residual net long/short bias from the top-5 /
bottom-5 construction) materially changes the risk profile.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Three portfolio construction variants
---------------------------------------
All three use the same LASSO predictions from walk_forward_backtest() and
rank the 25 portfolios identically — only the weight vectors differ.

  1. Gross top-3 / bottom-3  (w_g):
     The baseline construction used throughout the project.
     Long the top 3 at +1/3 each, short the bottom 3 at −1/3 each.
     Dollar-neutral by design (weights sum to zero).
     Gross leverage = ‖w_g‖₁ = 2.0 (100% long, 100% short).

  2. Long-only top-5  (w_l):
     Long the top 5 predicted portfolios at +1/5 each.
     No short positions.  Not dollar-neutral (net weight = +1.0).
     Gross leverage = ‖w_l‖₁ = 1.0.
     Captures pure signal quality — if this strategy earns positive
     alpha, the LASSO predictions contain genuine long-side information
     independent of any short-side contribution.

  3. Dollar-neutral top-5 / bottom-5  (w_d):
     Long top 5 at +1/5 each, short bottom 5 at −1/5 each — then
     mean-center the weight vector: w_d -= w_d.mean().
     The mean subtraction removes any residual net exposure that arises
     if the top-5 and bottom-5 weights do not exactly cancel (which they
     do here by construction, so the mean-centering has no effect on
     this specific weight vector — but the step is included for
     generality and would matter if position sizes were unequal).
     Gross leverage = ‖w_d‖₁ = 2.0.  Broader rank spread (5 per leg
     vs. 3) than the gross strategy, which may reduce concentration risk
     but also dilutes signal by including less-extreme predictions.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Performance metrics
--------------------
For each strategy, three metrics are computed from the monthly return series:

  Sharpe    (mean × √12) / (std × √12) = mean / std × √12
            Annualized Sharpe assuming i.i.d. monthly returns.

  Ann Ret%  mean monthly return × 12 × 100
            Annualized arithmetic return as a percentage.

  Ann Vol%  std monthly return × √12 × 100
            Annualized volatility as a percentage.

The table layout enables direct reading of the Sharpe decomposition:
  If Long-only Sharpe ≈ Gross Sharpe → the short leg adds little value;
    the strategy's alpha comes entirely from the long-side predictions.
  If Long-only Sharpe << Gross Sharpe → the short leg contributes
    meaningfully, justifying the added operational complexity of shorting.
  If Dollar-neutral Sharpe ≈ Gross Sharpe → the 3-vs-5 leg size and
    mean-centering do not materially change risk-adjusted performance.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Printed output
---------------
A formatted table with one row per strategy:
  Strategy   construction name
  Sharpe     annualized Sharpe ratio (4 decimal places)
  Ann Ret%   annualized return percentage (2 decimal places)
  Ann Vol%   annualized volatility percentage (2 decimal places)

No figure is generated — this is a diagnostic print-only script intended
to be run before or alongside the full plotting pipeline to quickly audit
whether portfolio construction choices are driving the reported results.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Interpretation notes
---------------------
  Ann Ret% is not risk-adjusted and should not be compared across
  strategies with different gross leverage.  w_l has leverage 1.0 while
  w_g and w_d have leverage 2.0 — a fair return comparison would scale
  w_l's return by 2× (or equivalently scale the long-short returns by 0.5×).
  Sharpe, being return/vol, is already leverage-neutral for strategies with
  proportional risk scaling.

  The mean-centering step for w_d is a no-op here (top-5 at +1/5 and
  bottom-5 at −1/5 sum exactly to zero), but is retained in the code as
  a general-purpose dollar-neutralization pattern that would apply when
  leg sizes are asymmetric.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Dependencies
------------
numpy, sys
src.data_loader — load_all_data()
src.backtest    — walk_forward_backtest()
src.solvers     — LassoProximal
"""
import sys, numpy as np
sys.path.insert(0, '.')
from src.data_loader import load_all_data
from src.backtest import walk_forward_backtest
from src.solvers import LassoProximal

X, Y, factor_names, _ = load_all_data()
preds, actuals, dates, _ = walk_forward_backtest(X, Y, LassoProximal, alpha=0.003)

def sharpe(rets):
    r = np.array(rets)
    return r.mean()/r.std()*np.sqrt(12)

gross, lo, dn = [], [], []
for pred, actual in zip(preds, actuals):
    ranks = np.argsort(pred); n = len(pred)
    w_g = np.zeros(n); w_g[ranks[-3:]] = 1/3; w_g[ranks[:3]] = -1/3
    w_l = np.zeros(n); w_l[ranks[-5:]] = 1/5
    w_d = np.zeros(n); w_d[ranks[-5:]] = 1/5; w_d[ranks[:5]] = -1/5
    w_d -= w_d.mean()
    gross.append(w_g @ actual); lo.append(w_l @ actual); dn.append(w_d @ actual)

print(f'{"Strategy":<30} {"Sharpe":>10} {"Ann Ret%":>10} {"Ann Vol%":>10}')
print('='*62)
for name, rets in [('Gross top3/bot3', gross),('Long-only top5', lo),('Dollar-neutral', dn)]:
    r = np.array(rets)
    print(f'{name:<30} {sharpe(r):>10.4f} {r.mean()*12*100:>10.2f} {r.std()*np.sqrt(12)*100:>10.2f}')
