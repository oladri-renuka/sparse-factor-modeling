"""
11_time_series_analysis.py
========================
Diagnostic analysis of the temporal and cross-sectional structure of the
Fama-French 25-portfolio dataset.

Generates: outputs/time_series_structure.png

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Motivation
----------
The backtesting and solver benchmarking pipelines assume:
  (a) Factor returns are stationary — a prerequisite for OLS consistency
      and for rolling-window estimates to remain meaningful.
  (b) OLS residuals are approximately serially uncorrelated — if residuals
      are autocorrelated, standard errors are underestimated and t-stats
      are inflated, calling into question model specification.
  (c) Factor loadings (betas) are approximately stable within rolling
      windows — if they drift substantially, a fixed-window model may
      be misspecified, motivating the online learning and adaptive-lambda
      approaches in online_lasso.py and regime.py.

This script tests all three assumptions directly and visualizes the results.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Diagnostic 1 — Stationarity: Augmented Dickey-Fuller test
-----------------------------------------------------------
For each of the six factors (Mkt-RF, SMB, HML, RMW, CMA, Mom), an ADF
test is run on the raw (unscaled) monthly return series.

  Null hypothesis: the series has a unit root (is non-stationary).
  Alternative: the series is stationary (mean-reverting).
  Rejection at p < 0.05 → series is stationary.

Configuration:
  maxlag=12  — allows up to 12 monthly lags in the ADF regression to
               absorb any seasonal autocorrelation structure.
  autolag='AIC' — lag length selected by minimizing AIC within [0, 12],
               balancing model fit against overfitting the augmentation.

Expected result: all six factor return series should strongly reject the
unit root null (p << 0.05), confirming stationarity.  Monthly returns are
differenced prices and do not exhibit the random walk behavior of price
levels.  Any series failing to reject (p > 0.05) would warrant further
investigation — e.g. checking for structural breaks or level shifts.

Printed output: ADF statistic, p-value, and STATIONARY / NON-STAT label
for each factor.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Diagnostic 2 — Serial correlation: Ljung-Box test and Durbin-Watson
--------------------------------------------------------------------
For a representative subset of six portfolios (P1, P5, P7, P13, P19, P25
— spanning the size/value grid), OLS is fitted with all six factors plus
a constant, and the residuals are tested for serial correlation.

  Ljung-Box test at lag 6:
    H₀: no autocorrelation at any lag up to 6.
    p > 0.05 → cannot reject white noise residuals (good).
    p < 0.05 → evidence of serial correlation in residuals (potential
               model misspecification or omitted variable).

  Durbin-Watson statistic:
    DW ≈ 2.0 → no autocorrelation in residuals.
    DW < 1.5 → positive autocorrelation (residuals cluster).
    DW > 2.5 → negative autocorrelation (residuals alternate).
    DW is computed directly as 2·(1 − ρ₁) where ρ₁ is the lag-1
    autocorrelation of residuals, equivalent to the standard DW formula.

Printed output: LB(6) p-value and DW statistic for each tested portfolio.
Portfolios with significant LB p-values or DW far from 2 would suggest
that the linear factor model is missing time-series structure — supporting
the use of rolling windows, regime-aware regularization, or richer models.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Diagnostic 3 — Time-varying factor loadings: rolling OLS betas
---------------------------------------------------------------
For Portfolio P1 (small-cap growth — the corner portfolio most likely to
exhibit unstable factor loadings), OLS is fitted on a rolling 60-month
window (5 years), stepping one month at a time.

At each step t the model fitted is:
    P1_t = α + β₁·Mkt-RF_t + … + β₆·Mom_t + ε_t
    using months [t-60, t).

The resulting (n − 60) × 6 matrix of rolling betas is stored and plotted.
The 2008–2009 financial crisis window is overlaid as a shaded region to
visualize how factor loadings respond to extreme market stress.

Printed output: min, max, and std of each factor's rolling beta series.
High std (relative to the mean beta) indicates substantial time variation
in that loading — empirical evidence that fixed-coefficient models
are misspecified and that adaptive approaches (regime.py, online_lasso.py)
are well-motivated.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Output figure — three-panel layout (outputs/time_series_structure.png)
-----------------------------------------------------------------------
Panel 1 — 25-portfolio return heatmap (5 × 5 grid):
  Annualized mean return (monthly mean × 12 × 100) for each of the 25
  size × book-to-market portfolios displayed as a color-coded grid.
  Rows = size quintiles (Small at top, Large at bottom); columns = B/M
  quintiles (Growth at left, Value at right).  Green = high return,
  red = low return.  Cell annotations show the numeric return value;
  font color switches to white for extreme values to maintain readability
  against saturated background colors.

  Purpose: establishes the cross-sectional return structure of the
  dependent variable matrix Y, showing the classic size and value premium
  patterns that the factor models are designed to explain.

Panel 2 — Factor autocorrelation functions:
  ACF at lags 1–12 for each of the six factors, computed as the
  Pearson correlation between r_t and r_{t-lag}.  Dashed ±1.96/√n
  confidence bands (approximate 95% bounds under the white-noise null)
  are shown in red.

  Purpose: visualizes the near-IID property of monthly factor returns
  (most autocorrelations should fall inside the confidence bands) while
  identifying any exceptions — the title references mild RMW persistence
  as the notable finding.  Validates the i.i.d. assumption underlying the
  annualization of volatility in regime.py (σ_annual = σ_monthly × √12).

Panel 3 — Rolling 60-month betas for Portfolio P1:
  One line per factor showing the time evolution of its OLS loading.
  The 2008–2009 crisis is shaded in transparent red.  The horizontal
  zero line separates positive from negative loadings.

  Purpose: provides direct visual evidence for time-varying factor
  structure — the key empirical motivation for the rolling-window
  backtesting design, the adaptive-lambda regime model, and the online
  LASSO.  If loadings were constant, a single full-sample OLS fit would
  suffice; the variation shown here justifies the more complex approaches.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Connection to other modules
----------------------------
  Stationarity result (Diagnostic 1)
    → validates the use of raw monthly returns as regression inputs in
      backtesting.py and solvers.py without differencing or detrending.

  Serial correlation result (Diagnostic 2)
    → if residuals are approximately white noise, the i.i.d. assumption
      in compute_metrics() (which computes Sharpe via simple mean/std)
      is approximately satisfied.  Significant autocorrelation would
      require a Newey-West or HAC standard error adjustment.

  Time-varying beta result (Diagnostic 3)
    → directly motivates walk_forward_backtest (rolling window captures
      local loadings), walk_forward_backtest_adaptive (lambda adapts to
      vol regime), and walk_forward_online (coefficients update each month
      rather than being re-estimated from scratch).

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Dependencies
------------
numpy, pandas, matplotlib, matplotlib.dates, scipy.stats
statsmodels.tsa.stattools.adfuller
statsmodels.stats.diagnostic.acorr_ljungbox
statsmodels.regression.linear_model.OLS
statsmodels.tools.add_constant
src.data_loader — load_all_data()

Output
------
outputs/time_series_structure.png  — 18×6 inch, 150 dpi, tight layout
"""
import sys, numpy as np, pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from scipy import stats
from statsmodels.tsa.stattools import adfuller
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.regression.linear_model import OLS
from statsmodels.tools import add_constant
sys.path.insert(0, '.')
from src.data_loader import load_all_data

X, Y, factor_names, _ = load_all_data()
X_vals   = X.values
X_scaled = (X_vals - X_vals.mean(0)) / X_vals.std(0)
n        = len(X)

print('Stationarity (ADF test):')
for fname in factor_names:
    r      = X[fname].values
    result = adfuller(r, maxlag=12, autolag='AIC')
    print(f'  {fname}: ADF={result[0]:.3f}  p={result[1]:.4f}  {"STATIONARY" if result[1]<0.05 else "NON-STAT"}')

print('\nSerial correlation in OLS residuals:')
X_const = add_constant(X_scaled)
for j in [0, 4, 6, 12, 18, 24]:
    y     = Y.iloc[:, j].values
    resid = OLS(y, X_const).fit().resid
    lb    = acorr_ljungbox(resid, lags=[6], return_df=True)
    dw    = 2*(1-np.corrcoef(resid[:-1], resid[1:])[0,1])
    print(f'  P{j+1}: LB(6)p={lb["lb_pvalue"].iloc[0]:.3f}  DW={dw:.3f}')

# Rolling betas
y_p1 = Y.iloc[:, 0].values
window = 60
roll_betas = []
roll_dates = []
for t in range(window, n):
    b = OLS(y_p1[t-window:t], add_constant(X_scaled[t-window:t])).fit().params[1:]
    roll_betas.append(b); roll_dates.append(X.index[t])
roll_betas = np.array(roll_betas)

print('\nRolling betas for P1:')
for i, fname in enumerate(factor_names):
    b = roll_betas[:,i]
    print(f'  {fname}: [{b.min():.3f}, {b.max():.3f}]  std={b.std():.4f}')

# Figure
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
COLORS = ['#2E74B5','#E74C3C','#27AE60','#F39C12','#8E44AD','#1ABC9C']

means_grid = np.array([Y.iloc[:,j].mean()*12*100 for j in range(25)]).reshape(5,5)
im1 = axes[0].imshow(means_grid, cmap='RdYlGn', aspect='auto')
plt.colorbar(im1, ax=axes[0], label='Annual Return (%)')
axes[0].set_xticks(range(5)); axes[0].set_xticklabels(['Growth','2','3','4','Value'], fontsize=9)
axes[0].set_yticks(range(5)); axes[0].set_yticklabels(['Small','2','3','4','Large'], fontsize=9)
for i in range(5):
    for j in range(5):
        axes[0].text(j, i, f'{means_grid[i,j]:.1f}', ha='center', va='center',
                     fontsize=8.5, fontweight='bold',
                     color='white' if means_grid[i,j]>14 or means_grid[i,j]<5 else 'black')
axes[0].set_title('25 Portfolio Annual Returns (%)\nSize × Book-to-Market Sorts (2000-2023)', fontweight='bold')
axes[0].set_xlabel('Book-to-Market (Value →)'); axes[0].set_ylabel('← Size (Small top)')

ci = 1.96/np.sqrt(n)
for i, fname in enumerate(factor_names):
    r     = X[fname].values
    acf_v = [np.corrcoef(r[:-lag], r[lag:])[0,1] for lag in range(1,13)]
    axes[1].plot(range(1,13), acf_v, 'o-', color=COLORS[i], lw=2, ms=5, label=fname)
axes[1].axhline(0, color='black', lw=0.8, ls='--')
axes[1].axhline(ci, color='red', lw=1.5, ls=':', alpha=0.7, label='95% CI')
axes[1].axhline(-ci, color='red', lw=1.5, ls=':', alpha=0.7)
axes[1].set_xlabel('Lag (months)'); axes[1].set_ylabel('Autocorrelation')
axes[1].set_title('Factor Return Autocorrelations\nMost near-IID; RMW mild persistence', fontweight='bold')
axes[1].legend(fontsize=8, ncol=2); axes[1].grid(True, alpha=0.3)

for i, fname in enumerate(factor_names):
    axes[2].plot(roll_dates, roll_betas[:,i], color=COLORS[i], lw=1.8, label=fname, alpha=0.85)
axes[2].axhline(0, color='black', lw=0.8, ls='--')
crisis_s = pd.Timestamp('2008-01-01'); crisis_e = pd.Timestamp('2009-12-31')
axes[2].axvspan(crisis_s, crisis_e, alpha=0.15, color='red', label='2008 crisis')
axes[2].xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
axes[2].set_xlabel('Date'); axes[2].set_ylabel('Rolling 60M Beta')
axes[2].set_title('Rolling Factor Loadings — Portfolio P1\nTime-varying structure motivates online learning', fontweight='bold')
axes[2].legend(fontsize=8, ncol=2); axes[2].grid(True, alpha=0.3)

plt.suptitle('25 Portfolio Dataset Description and Time Series Structure', fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig('outputs/time_series_structure.png', dpi=150, bbox_inches='tight')
print('Saved: outputs/time_series_structure.png')
