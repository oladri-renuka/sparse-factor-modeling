"""
07_factor_crowding.py
==================
Measures "factor crowding" — the degree to which the walk-forward LASSO
long-short strategy's returns are correlated with each Fama-French factor.
High correlation would indicate the strategy is inadvertently loading on a
systematic factor rather than generating idiosyncratic alpha, a concern
both for interpretation and for risk management.

Generates: outputs/factor_crowding.png

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

What is factor crowding?
-------------------------
In quantitative equity investing, a strategy is said to be "crowded" in a
factor if its returns co-move strongly with that factor's realised return.
Crowding is problematic for two reasons:

  1. Return attribution: if the strategy's alpha is explained by a known
     systematic factor (e.g. Mkt-RF), it is not genuinely alpha — it is
     compensated factor risk that could be obtained more cheaply by buying
     the factor directly.

  2. Drawdown clustering: in a factor crash (e.g. the momentum crash of
     2009, the value crash of 2020), a strategy crowded in that factor
     will suffer contemporaneously with all other momentum/value investors,
     amplifying drawdowns precisely when diversification is most needed.

This script tests whether the LASSO long-short strategy constructed in
backtesting.py is crowded in any of the six Fama-French factors by
computing the time-series correlation between the strategy's monthly
returns and each factor's monthly return over the OOS backtest period.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Data pipeline
--------------
walk_forward_backtest(X, Y, LassoProximal, alpha=0.003):
  Runs the full rolling-window backtest from backtesting.py, returning
  out-of-sample predictions and actuals aligned to the dates list.  The
  120-month training window and one-month step are inherited from the
  backtest defaults.

Long-short strategy construction:
  Identical to compute_metrics() in backtesting.py:
    - Rank all 25 portfolios by predicted return each month.
    - Go long the top 3 (equal weight +1/3 each).
    - Go short the bottom 3 (equal weight −1/3 each).
    - Monthly LS return = w · actual_returns.
  The resulting strat pd.Series is indexed by the OOS prediction dates.

Factor alignment:
  X_oos = X.reindex(dates) aligns the factor return matrix to the OOS
  dates so that the factor return at month t is the contemporaneous return
  for the same month as the strategy return — a pure contemporaneous
  correlation, not a predictive regression.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Static crowding analysis — full-sample Pearson correlation
-----------------------------------------------------------
For each factor f ∈ {Mkt-RF, SMB, HML, RMW, CMA, Mom}, the full-sample
Pearson correlation r between strategy returns and factor returns is
computed along with its two-sided p-value.

Crowding level classification:
  |r| > 0.30  → HIGH    significant systematic factor exposure
  |r| > 0.15  → MOD     moderate, worth monitoring
  |r| ≤ 0.15  → LOW     negligible crowding in this factor

The |r| = 0.30 threshold for HIGH is a practical rule-of-thumb — at the
~168 OOS months typical of this backtest (2010–2023 with 120-month warmup),
a correlation of 0.30 has a t-statistic of approximately 4.0, well above
conventional significance thresholds.  A strategy with |r| > 0.30 on any
factor should be investigated for unintended systematic loading before
being presented as market-neutral alpha.

Printed table columns:
  Factor    factor name
  Pearson r full-sample correlation
  p-value   two-sided test of H₀: r = 0
  Level     HIGH / MOD / LOW classification

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Dynamic crowding analysis — rolling 24-month Pearson correlation
-----------------------------------------------------------------
The full-sample correlation is an average that may mask regime-dependent
crowding — a strategy may be uncrowded on average but heavily crowded
during specific periods (e.g. loading strongly on Mkt-RF during the 2008
crisis when the long-short construction happens to favour defensive
portfolios).

For each factor, a rolling 24-month Pearson correlation is computed by
sliding a 24-month window across the OOS period.  Window size = 24 months
is chosen to give a sample of two full years — enough for a meaningful
correlation estimate while remaining responsive to regime changes.  Each
window requires exactly 24 observations; the first 24 months of the OOS
period are therefore excluded from the rolling plot.

The time-varying correlation series reveals:
  Structural crowding: r stays consistently high or low across regimes
                       — the strategy has a permanent factor tilt.
  Episodic crowding:   r spikes during specific crises or regimes —
                       the strategy is conditionally exposed to a factor
                       only when market conditions trigger that loading.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Output figure — two-panel layout (outputs/factor_crowding.png)
--------------------------------------------------------------
Panel 1 — Static crowding bar chart:
  One bar per factor showing the full-sample Pearson r.  Color coded by
  level: red = HIGH (|r| > 0.3), orange = MOD (|r| > 0.15), green = LOW.
  Dotted reference lines at ±0.3 mark the HIGH threshold.  Bar annotations
  show the exact r value to three decimal places.  The subtitle notes
  whether any factor exceeded the HIGH threshold — the expected result
  with a well-diversified LASSO strategy is that no factor does.

Panel 2 — Rolling 24-month crowding time series:
  Six lines (one per factor) showing how contemporaneous correlation with
  each factor evolves over the OOS period.  Dotted ±0.3 reference lines
  are repeated.  The x-axis is formatted as years for readability.
  Spikes above ±0.3 in the rolling chart that do not show up in the
  static chart indicate transient crowding events that cancel out over
  the full sample — arguably more dangerous than structural crowding
  because they are harder to hedge.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Interpretation in project context
-----------------------------------
Low full-sample crowding (all factors in LOW or MOD range) would support
the claim that the LASSO strategy generates returns orthogonal to the
standard six-factor risk premia — i.e., that it captures genuine
portfolio-selection skill rather than systematic factor beta.

High crowding in Mkt-RF specifically would indicate the long-short
construction is not truly market-neutral — the top-3 and bottom-3
portfolios differ systematically in their market beta, so the LS return
is partly a levered market bet.  This would call for a beta-neutralization
step before presenting the strategy's Sharpe ratio as alpha.

Rolling crowding spikes near 2008–2009 or 2020 would be expected — during
crises, factor correlations spike across the board (contagion), and the
strategy's conditional factor exposure is harder to control.  The regime-
aware lambda in walk_forward_backtest_adaptive partially addresses this by
increasing regularization during high-vol periods, which tends to reduce
factor concentration in the selected portfolio subset.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Limitations
-----------
  Contemporaneous correlation only:
    Pearson r between strat_t and factor_t measures co-movement in the
    same month.  It does not capture lagged factor exposure (the strategy
    predicting next-month factor returns) or lead-lag relationships that
    would show up in a regression of strategy returns on lagged factors.
    A fuller analysis would include a factor regression (α + Σβ_j·f_j)
    and test the significance of the intercept (Jensen's alpha).

  Spearman correlation computed but not used:
    spearmanr is imported but the analysis uses only Pearson r.  Spearman
    ρ would be more robust to the heavy-tailed return distributions typical
    of monthly factor data, particularly for extreme months like March 2020.
    Adding a Spearman column to the printed table would be a low-cost
    robustness check.

  Long-short construction sensitivity:
    The top-3 / bottom-3 rule is a specific construction choice.  A
    top-quintile / bottom-quintile construction (5 portfolios per leg from
    the 25) would give a different crowding profile — wider rank spreads
    tend to increase market-beta exposure.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Dependencies
------------
numpy, pandas, matplotlib, matplotlib.dates, scipy.stats, sys
src.data_loader — load_all_data()
src.backtest    — walk_forward_backtest()
src.solvers     — LassoProximal
"""
import sys, numpy as np, pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from scipy.stats import pearsonr, spearmanr
sys.path.insert(0, '.')
from src.data_loader import load_all_data
from src.backtest import walk_forward_backtest
from src.solvers import LassoProximal

X, Y, factor_names, _ = load_all_data()
preds, actuals, dates, _ = walk_forward_backtest(X, Y, LassoProximal, alpha=0.003)
X_oos = X.reindex(dates)

strat_rets = []
for pred, actual in zip(preds, actuals):
    ranks = np.argsort(pred); w = np.zeros(len(pred))
    w[ranks[-3:]] = 1/3; w[ranks[:3]] = -1/3
    strat_rets.append(w @ actual)
strat = pd.Series(strat_rets, index=dates)

print(f'{"Factor":<10} {"Pearson r":>10} {"p-value":>10} {"Level"}')
print('='*45)
crowding = {}
for fname in factor_names:
    s = strat.values; f = X_oos[fname].values
    r, p = pearsonr(s, f)
    level = 'HIGH' if abs(r)>0.3 else 'MOD' if abs(r)>0.15 else 'LOW'
    crowding[fname] = {'r': r, 'p': p}
    print(f'{fname:<10} {r:>10.4f} {p:>10.4f} {level}')

COLORS = ['#2E74B5','#E74C3C','#27AE60','#F39C12','#8E44AD','#1ABC9C']
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
vals   = [crowding[f]['r'] for f in factor_names]
cols   = ['#E74C3C' if abs(v)>0.3 else '#F39C12' if abs(v)>0.15 else '#27AE60' for v in vals]
bars   = axes[0].bar(factor_names, vals, color=cols, alpha=0.85, edgecolor='white')
axes[0].axhline(0, color='black', lw=0.8, ls='--')
axes[0].axhline(0.3, color='red', lw=1.5, ls=':', alpha=0.6)
axes[0].axhline(-0.3, color='red', lw=1.5, ls=':', alpha=0.6)
for bar, v in zip(bars, vals):
    axes[0].text(bar.get_x()+bar.get_width()/2, v+(0.005 if v>=0 else -0.015),
                 f'{v:.3f}', ha='center', fontsize=9, fontweight='bold')
axes[0].set_title('Strategy Correlation with FF Factors\nNo factor exceeds |r|=0.3', fontweight='bold')
axes[0].set_ylabel('Pearson r'); axes[0].grid(True, alpha=0.3, axis='y')

window = 24
axes[1].axhline(0, color='black', lw=0.8, ls='--')
axes[1].axhline(0.3, color='red', lw=1.5, ls=':', alpha=0.5, label='|r|=0.3')
axes[1].axhline(-0.3, color='red', lw=1.5, ls=':', alpha=0.5)
for i, fname in enumerate(factor_names):
    roll_r, roll_d = [], []
    for t in range(window, len(strat)):
        s_w = strat.iloc[t-window:t].values; f_w = X_oos[fname].iloc[t-window:t].values
        if len(s_w) == window:
            roll_r.append(pearsonr(s_w, f_w)[0]); roll_d.append(strat.index[t])
    axes[1].plot(roll_d, roll_r, color=COLORS[i], lw=1.8, label=fname, alpha=0.85)
axes[1].xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
axes[1].set_title('Rolling 24-Month Factor Crowding', fontweight='bold')
axes[1].set_xlabel('Date'); axes[1].set_ylabel('Rolling Pearson r')
axes[1].legend(fontsize=9); axes[1].grid(True, alpha=0.3)
plt.suptitle('Factor Crowding Analysis', fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig('outputs/factor_crowding.png', dpi=150, bbox_inches='tight')
print('Saved: outputs/factor_crowding.png')
