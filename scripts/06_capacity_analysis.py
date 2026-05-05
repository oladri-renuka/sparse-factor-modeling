"""
06_capacity_analysis.py
====================
Models the relationship between assets under management (AUM) and net
risk-adjusted performance for the walk-forward LASSO long-short strategy.
As AUM grows, market impact costs scale with the dollar value of each
rebalancing trade, eventually eroding the gross Sharpe ratio to the point
where the strategy is no longer attractive.  This script computes and plots
that degradation curve and identifies the breakeven AUM — the capacity
ceiling of the strategy.

Generates: outputs/capacity_analysis.png

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Motivation
----------
Reported Sharpe ratios from backtests are gross of market impact — they
assume trades execute at the closing price with no size effect.  In
practice, larger orders move prices against the trader (market impact),
increasing effective transaction costs above the fixed bps assumed in
compute_metrics_with_costs().  A strategy that achieves Sharpe 1.5 at
$10M AUM may deliver Sharpe 0.8 at $500M and Sharpe 0.0 at $2B, making
capacity analysis essential for understanding the strategy's realistic
commercial potential.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Market impact model — linear permanent impact
----------------------------------------------
The model used is a simplified linear permanent impact framework:

    Impact cost per trade = γ · (trade_size / ADV)

where:
  γ   = 0.1     impact coefficient — fraction of ADV that causes one unit
                of adverse price movement (a standard order-of-magnitude
                assumption for liquid large-cap equity portfolios)
  ADV = $5B     assumed average daily volume of the portfolio constituents
                — calibrated to the Fama-French 25 size×value portfolios,
                which span large- and mid-cap US equities

Total annual impact cost as a fraction of AUM:
    Annual impact drag = γ · (AUM · avg_turnover · 12) / ADV

  AUM · avg_turnover       dollar value traded per month (one-way)
  × 12                     annualized
  / ADV                    fraction of daily volume consumed
  × γ                      converts volume fraction to return drag

This is a linear model in AUM — impact costs grow proportionally with
strategy size.  The quadratic (square-root) impact model used in more
sophisticated capacity analysis would predict faster degradation at large
AUM; the linear model used here is conservative (optimistic) at large AUM.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Base performance inputs
------------------------
The gross strategy statistics are extracted from compute_metrics_with_costs
with bps_cost=0 (zero fixed transaction costs) so that the only cost source
in the capacity model is AUM-scaled market impact:

  base_ann_ret   annualized gross return (Gross_Ann_Ret / 100 to decimal)
  avg_turnover   mean monthly one-way portfolio turnover from the backtest
  ann_vol        implied annualized volatility, back-calculated as
                 base_ann_ret / Gross_Sharpe — this avoids recomputing
                 the full return series and is exact given the Sharpe
                 definition.

These three statistics fully parametrize the capacity curve: the return
degrades linearly with AUM while the volatility is assumed constant
(impact costs are treated as a drag on returns, not as an additional
source of return variance — a standard first-order approximation).

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Net Sharpe curve
-----------------
For each AUM level in a log-spaced grid from $1M to ~$3B (300 points):

    net_return(AUM)  = base_ann_ret − γ · (AUM · avg_turnover · 12) / ADV
    net_sharpe(AUM)  = net_return(AUM) / ann_vol

The net Sharpe curve is monotonically decreasing in AUM — higher AUM means
higher impact, lower net return, lower Sharpe.  The curve starts at the
gross Sharpe (near-zero AUM, negligible impact) and crosses zero when
net_return = 0 (impact cost equals gross alpha).

Breakeven AUM is defined as the AUM where net_sharpe = 1.0 — the point
at which the strategy's risk-adjusted return equals a commonly cited
institutional minimum hurdle rate.  It is found by locating the grid point
where |net_sharpe − 1.0| is minimized.

Printed diagnostics:
  Gross Sharpe            baseline before any impact costs
  Monthly turnover        average one-way turnover from compute_metrics_with_costs
  Net Sharpe at $10M      negligible impact (near-capacity floor)
  Net Sharpe at $100M     small institutional fund size
  Net Sharpe at $500M     mid-size fund
  Net Sharpe at $1B       large fund
  Breakeven AUM           capacity ceiling at the Sharpe=1.0 hurdle

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Output figure (outputs/capacity_analysis.png)
----------------------------------------------
Single panel, semi-log x-axis (AUM in $ millions):

  Blue curve         net Sharpe vs AUM — the primary output
  Gray dotted line   gross Sharpe (AUM-independent ceiling)
  Red dashed line    Sharpe = 1.0 hurdle rate
  Red dotted line    breakeven AUM vertical marker
  Red shading        AUM region where net Sharpe < 1.0 (below hurdle)
  Green shading      AUM region where net Sharpe ≥ 1.0 (above hurdle)
  Annotations        exact net Sharpe values at $10M, $100M, $500M

The semi-log x-axis is essential because the relevant AUM range spans
three orders of magnitude ($1M to $3B); a linear axis would compress the
small-AUM region where most of the interesting variation occurs.

Title includes the breakeven AUM, γ, ADV, and turnover for full
reproducibility — the chart is self-contained without needing to read the
code to understand the model assumptions.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Interpretation in project context
-----------------------------------
The breakeven AUM quantifies the commercial viability of the LASSO
long-short strategy.  A breakeven well above $100M suggests the strategy
is scalable enough to be relevant to institutional allocators; a breakeven
below $50M suggests it is primarily a research result with limited
real-world deployability.

The turnover input (avg_turnover from compute_metrics_with_costs) directly
links to the transaction cost analysis in backtesting.py — strategies with
lower turnover have proportionally lower impact costs at any given AUM,
pushing the breakeven higher.  The adaptive-lambda variant in
walk_forward_backtest_adaptive may produce different turnover (higher
regularization in volatile periods tends to reduce portfolio changes),
and could be substituted to compare capacity across model variants.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Model assumptions and limitations
-----------------------------------
  Linear impact is optimistic at large AUM:
    The square-root impact model (impact ∝ √(trade_size/ADV)) is more
    empirically supported in the microstructure literature and predicts
    faster Sharpe degradation.  The linear model underestimates impact
    costs for AUM well above $500M, making the breakeven estimate
    optimistic at large fund sizes.

  ADV = $5B is a single-market assumption:
    The Fama-French 25 portfolios are value-weighted and span the full
    US equity market cap spectrum.  Small-cap portfolios (P1–P5) have
    much lower ADV than large-cap portfolios; the $5B assumption is
    appropriate for large-cap but overstates liquidity for small-cap
    names, understating impact in those portfolios.

  Constant volatility assumption:
    ann_vol is treated as fixed regardless of AUM.  In practice, larger
    positions are harder to exit during volatile markets, and impact costs
    themselves introduce return variance.  This approximation is standard
    but deteriorates at very large AUM where position unwinding becomes
    a significant second-order effect.

  γ = 0.1 is a single point estimate:
    The impact coefficient varies by stock, time, and market conditions.
    A sensitivity analysis varying γ over [0.05, 0.2] and ADV over
    [$1B, $10B] would bound the uncertainty in the breakeven estimate
    and is a natural extension of this single-scenario analysis.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Dependencies
------------
numpy, matplotlib, sys
src.data_loader — load_all_data()
src.backtest    — walk_forward_backtest(), compute_metrics_with_costs()
src.solvers     — LassoProximal
"""
import sys, numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
sys.path.insert(0, '.')
from src.data_loader import load_all_data
from src.backtest import walk_forward_backtest, compute_metrics_with_costs
from src.solvers import LassoProximal

X, Y, factor_names, _ = load_all_data()
preds, actuals, dates, _ = walk_forward_backtest(X, Y, LassoProximal, alpha=0.003)
m = compute_metrics_with_costs(preds, actuals, bps_cost=0)

base_ann_ret = m['Gross_Ann_Ret'] / 100
avg_turnover = m['Avg_Turnover']
ann_vol      = base_ann_ret / m['Gross_Sharpe']
ADV, gamma   = 5e9, 0.1

aum_range   = np.logspace(6, 9.5, 300)
net_sharpes = [(base_ann_ret - gamma*(a*avg_turnover*12)/ADV)/ann_vol for a in aum_range]
net_sharpes = np.array(net_sharpes)
be_idx      = np.argmin(np.abs(net_sharpes - 1.0))
be_aum      = aum_range[be_idx]

print(f'Gross Sharpe: {m["Gross_Sharpe"]:.4f}')
print(f'Monthly turnover: {avg_turnover:.4f}')
for aum_tgt in [1e7, 1e8, 5e8, 1e9]:
    idx = np.argmin(np.abs(aum_range - aum_tgt))
    print(f'Net Sharpe at ${aum_tgt/1e6:.0f}M: {net_sharpes[idx]:.3f}')
print(f'Breakeven AUM: ${be_aum/1e6:.0f}M')

fig, ax = plt.subplots(figsize=(10, 6))
ax.semilogx(aum_range/1e6, net_sharpes, color='#2E74B5', lw=2.5)
ax.axhline(1.0, color='#E74C3C', lw=2, ls='--', label='Sharpe=1.0')
ax.axhline(m['Gross_Sharpe'], color='gray', lw=1.5, ls=':', label=f'Gross Sharpe={m["Gross_Sharpe"]:.2f}')
ax.axvline(be_aum/1e6, color='#E74C3C', lw=2, ls=':', label=f'Breakeven=${be_aum/1e6:.0f}M')
ax.fill_between(aum_range/1e6, net_sharpes, 1.0, where=net_sharpes<1.0, alpha=0.2, color='red')
ax.fill_between(aum_range/1e6, np.minimum(net_sharpes, m['Gross_Sharpe']+0.5), 1.0,
                where=net_sharpes>=1.0, alpha=0.1, color='green')
for aum_tgt, label in [(10,'$10M'),(100,'$100M'),(500,'$500M')]:
    idx  = np.argmin(np.abs(aum_range/1e6 - aum_tgt))
    ax.annotate(f'{net_sharpes[idx]:.2f}', (aum_tgt, net_sharpes[idx]),
                textcoords='offset points', xytext=(5,6), fontsize=9, color='#2E74B5')
ax.set_xlabel('AUM ($ millions, log scale)', fontsize=12)
ax.set_ylabel('Net Sharpe Ratio', fontsize=12)
ax.set_title(f'Capacity Analysis: Breakeven=${be_aum/1e6:.0f}M\nLinear impact: γ=0.1, ADV=$5B, turnover={avg_turnover:.2f}×/month',
             fontweight='bold')
ax.legend(fontsize=9); ax.grid(True, alpha=0.3); ax.set_ylim(0, m['Gross_Sharpe']+0.5)
plt.tight_layout()
plt.savefig('outputs/capacity_analysis.png', dpi=150, bbox_inches='tight')
print('Saved: outputs/capacity_analysis.png')
