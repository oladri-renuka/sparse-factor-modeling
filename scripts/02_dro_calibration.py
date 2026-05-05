"""
02_dro_calibration.py
====================
Generates the DRO sensitivity table (Table 6 in the report) by running the
walk-forward backtest at five epsilon values spanning the range from standard
LASSO (ε = 0) through the data-driven optimal (ε = 0.091) to an
over-regularized regime (ε = 0.15).  Quantifies how much distributional
robustness improves out-of-sample performance and confirms that the
data-driven epsilon selection is near the performance peak.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Background — what ε controls
------------------------------
DROLasso (dro_solvers.py) solves:

    minimize  (1/n) ‖y − Xβ‖₂²  +  ε ‖β‖₂  +  α ‖β‖₁

The Wasserstein ball radius ε scales the L2 norm robustness penalty:
  ε = 0     → standard LASSO, solved by LassoProximal at the same α.
  ε > 0     → heavier shrinkage of non-zero coefficients, providing
               protection against distribution shift between training and
               test windows.
  ε → ∞    → all coefficients shrunk toward zero; model degenerates to
               the zero-return naive forecast.

The data-driven ε = 0.091 is derived from Esfahani & Kuhn (2018)'s
finite-sample bound on the Wasserstein distance between the empirical
training distribution and the true return-generating distribution,
calibrated to the dataset's (n, p) = (120, 6) training window.  See
dro_solvers.py for the theoretical derivation.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Experimental design
--------------------
Five epsilon values are evaluated:
  ε = 0.000   standard LASSO baseline (LassoProximal, no DRO penalty)
  ε = 0.010   light robustness — small perturbation from standard LASSO
  ε = 0.050   moderate robustness
  ε = 0.091   data-driven optimal — Esfahani & Kuhn finite-sample bound
  ε = 0.150   heavy robustness — expected over-regularization

The ε = 0 case uses LassoProximal directly rather than DROLasso(epsilon=0)
to ensure the baseline is exactly the standard LASSO solution from the
main benchmarking pipeline, not an approximation through the DRO code path.
All other cases use DROLasso(alpha=0.003, epsilon=eps).

All five runs use the same walk_forward_backtest() call with identical
training window (120 months), step (1 month), and LassoProximal or
DROLasso model class — ensuring the only difference across rows is the
epsilon value.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Metrics reported (Table 6)
---------------------------
Three metrics from compute_metrics() are shown per epsilon:

  OOS R²    Pooled out-of-sample R² across all 25 portfolios and all
            OOS months.  Measures overall predictive fit relative to
            the zero-return naive forecast.  DRO is expected to improve
            OOS R² by reducing overfitting to the training distribution.

  Sharpe    Annualized Sharpe ratio of the long-short strategy (top-3
            long, bottom-3 short, equal weight).  The primary economic
            performance measure — directly comparable to the results in
            the main backtesting tables.

  ICIR      Information Coefficient Information Ratio — mean monthly
            Spearman rank IC divided by its standard deviation.  Measures
            consistency of the predictive ranking signal; ICIR > 0.5 is
            typically considered strong for a monthly model.

The '<-- data-driven' annotation flags the theoretically motivated epsilon,
making the row easy to identify in the report table.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Expected pattern of results
-----------------------------
  ε = 0.000 → baseline Sharpe and ICIR from the main backtest.

  ε = 0.010 to 0.091 → monotonically improving or plateau in Sharpe
    and ICIR as the robustness penalty protects against the distribution
    shifts that occur when market regimes change between the 120-month
    training window and the one-month test period.

  ε = 0.091 → should be near the peak or within noise of it, validating
    the data-driven calibration.  If a lower ε achieves strictly higher
    Sharpe, it suggests the theoretical bound is conservative (too large)
    for this dataset.

  ε = 0.150 → degrading performance as over-regularization shrinks
    economically meaningful factor loadings alongside noise, reducing
    the model's ability to discriminate between portfolios.

If the performance is non-monotone (e.g. a clear peak at ε = 0.050 with
degradation at ε = 0.091), the data-driven calibration may not be optimal
for this dataset and cross-validated epsilon selection via time_series_cv
would be the appropriate next step.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Relationship to other scripts
-------------------------------
  dro_solvers.py       defines DROLasso and the theoretical justification
                       for the ε = 0.091 calibration.
  backtesting.py       provides walk_forward_backtest and compute_metrics.
  cross_validation.py  time_series_cv could be used to cross-validate ε
                       as an alternative to the data-driven formula.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Limitations
-----------
  Five epsilon values is a coarse grid:
    The sweep uses five hand-chosen values rather than a systematic grid
    search.  The true performance peak may lie between the tested values
    — particularly between ε = 0.050 and ε = 0.091.  A 20-point log-
    spaced grid over [0.01, 0.20] would give a smoother curve and a more
    precise empirical optimum for comparison against the theoretical 0.091.

  No figure generated:
    This script outputs a printed table only — no plot is saved.  Adding a
    line chart of Sharpe and ICIR vs. ε (with the ε = 0.091 vertical
    marker) would make the robustness-performance trade-off visually
    immediate and more suitable for the report than a raw table.

  Computational cost:
    Five full walk-forward backtests are run sequentially, each fitting
    25 models × ~168 OOS months = ~4,200 DROLasso fits (which use CVXPY
    interior-point solves rather than fast iterative methods).  Runtime
    may be several minutes depending on hardware.  Pre-computing and
    caching the backtest results would allow re-running the table with
    different epsilon grids at negligible marginal cost.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Dependencies
------------
sys
src.data_loader  — load_all_data()
src.backtest     — walk_forward_backtest(), compute_metrics()
src.dro_solver   — DROLasso
src.solvers      — LassoProximal
"""
import sys
sys.path.insert(0, '.')
from src.data_loader import load_all_data
from src.backtest import walk_forward_backtest, compute_metrics
from src.dro_solver import DROLasso
from src.solvers import LassoProximal

X, Y, factor_names, _ = load_all_data()

print('DRO with data-driven epsilon = 0.091')
print(f'{"epsilon":>10} {"OOS R2":>10} {"Sharpe":>10} {"ICIR":>10}')
print('='*45)

for eps in [0.0, 0.01, 0.05, 0.091, 0.15]:
    if eps == 0.0:
        preds, actuals, dates, _ = walk_forward_backtest(
            X, Y, LassoProximal, alpha=0.003)
    else:
        preds, actuals, dates, _ = walk_forward_backtest(
            X, Y, DROLasso, alpha=0.003, epsilon=eps)
    m    = compute_metrics(preds, actuals)
    note = '<-- data-driven' if abs(eps-0.091) < 0.001 else ''
    print(f'{eps:>10.3f} {m["OOS_R2"]:>10} {m["Sharpe"]:>10} {m["ICIR"]:>10} {note}')
