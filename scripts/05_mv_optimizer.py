"""
05_mv_optimizer.py
===============
Compares two portfolio construction methods that both use the same LASSO
return predictions as inputs: a naive rank-based long-short strategy and a
mean-variance (MV) optimized portfolio.  The MV optimizer incorporates the
full covariance structure of the 25 portfolios to allocate weights more
efficiently than the equal-weight top-3/bottom-3 rule, at the cost of
solving a quadratic program at each time step.

Generates: outputs/mv_portfolio.png

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Motivation
----------
The naive long-short construction in compute_metrics() (backtesting.py)
allocates equal weight to the top-3 and bottom-3 ranked portfolios,
ignoring two important sources of information:

  1. Prediction magnitude: a portfolio ranked first with predicted return
     0.05 is treated identically to one ranked first with predicted return
     0.001 — the signal strength is discarded after ranking.

  2. Return covariance: if two of the three long portfolios are highly
     correlated (e.g. both small-cap value), holding both provides less
     diversification than holding two uncorrelated portfolios.  The naive
     rule is blind to this.

Mean-variance optimization addresses both: it scales position sizes by
expected return and simultaneously minimizes portfolio variance given the
covariance matrix estimated from historical returns.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Walk-forward structure
-----------------------
The outer loop mirrors walk_forward_backtest() exactly:
  Training window: 120 months (same default as backtesting.py).
  Step: 1 month.
  First prediction date: month 120.

At each step t, both strategies:
  1. Fit 25 independent LASSO models on the training slice to obtain
     preds — a (25,) vector of one-step-ahead predicted returns.
  2. Apply their respective weight construction rules.
  3. Record the realized return w · Y_te at the test observation.

This ensures the comparison is strictly apples-to-apples: same LASSO
predictions, same training window, same test observations — only the
weight construction differs.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Strategy 1 — Naive rank-based long-short
-----------------------------------------
Identical to the construction in compute_metrics() and factor_crowding.py:
  Rank all 25 portfolios by predicted return.
  Long top 3 at +1/3 each, short bottom 3 at −1/3 each.
  Dollar-neutral by construction (weights sum to zero).
  All prediction magnitude information is discarded after ranking.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Strategy 2 — Mean-variance quadratic program
--------------------------------------------
Solves at each time step t:

    minimize    wᵀΣw  −  2λ · μ̂ᵀw
    subject to  Σwᵢ = 0          (dollar-neutral)
                ‖w‖₁ ≤ 2         (gross leverage ≤ 2×)
                −0.15 ≤ wᵢ ≤ 0.15  (individual position cap)

  Σ  (25×25):  sample covariance of the 25 portfolio returns over the
               training window, regularized with 1e-4·I to guarantee
               positive definiteness even when the training window is
               short relative to p = 25.

  μ̂  (25,):   LASSO predicted returns, standardized by their own
               standard deviation (divided by std(preds) + 1e-8) to
               normalize the prediction scale across months.  The scaling
               makes the implicit risk-aversion parameter λ = 2.0
               consistent across periods regardless of the absolute
               magnitude of the predictions.

  Constraints:
    Dollar-neutral (Σwᵢ = 0) ensures the portfolio has no net market
    exposure in expectation — same constraint as the naive strategy.

    Gross leverage ‖w‖₁ ≤ 2 limits total long + total short to at most
    2× AUM.  Without this constraint the QP would concentrate the entire
    position in the single portfolio with the highest predicted
    return-to-variance ratio, producing an undiversified and unstable
    result.

    Individual caps −0.15 ≤ wᵢ ≤ 0.15 prevent any single portfolio from
    receiving more than 15% of AUM on either side.  This is tighter than
    the naive strategy's implicit ±33% cap (one of three equal positions)
    and forces the MV optimizer to distribute risk more broadly when the
    predicted return surface is concentrated.

  Solver: CLARABEL (modern interior-point solver, default in recent CVXPY).
    verbose=False suppresses per-iteration output in the T−120 loop.

  Fallback: if CLARABEL fails to find a feasible solution (w.value is None
    — e.g. infeasible due to numerical issues in Σ), the naive equal-weight
    vector w_n is used instead.  This prevents the MV return series from
    having NaN entries and ensures the backtest completes even if a handful
    of months produce degenerate QP instances.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Performance comparison
-----------------------
Annualized Sharpe ratios are computed for both strategies using:
    Sharpe = (mean monthly return / std monthly return) × √12

The percentage Sharpe improvement of MV over naive is printed directly.
A positive improvement confirms that the covariance information in Σ
and the prediction magnitude in μ̂ add value beyond the rank ordering
used by the naive strategy.

Possible outcomes:
  MV Sharpe > Naive Sharpe:
    The QP's covariance-aware diversification and signal-scaled sizing
    add value.  This is the expected result when portfolio return
    correlations are heterogeneous — the optimizer steers weight away
    from correlated portfolios toward diversifying combinations.

  MV Sharpe ≈ Naive Sharpe:
    The 25 portfolios have approximately equal pairwise correlations
    (the correlation matrix is nearly proportional to the identity),
    so the QP's covariance information adds little.  This is plausible
    for the Fama-French 25 where all portfolios are US equities sharing
    common market beta.

  MV Sharpe < Naive Sharpe:
    Estimation error in Σ dominates — the sample covariance is noisy
    at n=120, p=25, and the QP overfits to historical correlations that
    do not persist OOS.  This would motivate covariance shrinkage (e.g.
    Ledoit-Wolf) as a next step.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Output figure (outputs/mv_portfolio.png)
-----------------------------------------
Single panel: cumulative return (%) vs. date for both strategies.
  Red line   naive rank-based (Sharpe in legend)
  Blue line  MV optimized (Sharpe in legend)

Cumulative returns are computed as np.cumsum of the monthly return series,
which approximates the log-cumulative return for small monthly returns but
does not compound correctly for large positive runs.  For a paper the
exact compounding formula would be np.cumprod(1 + returns) − 1.

The plot subtitle shows the QP objective in mathematical notation,
making the chart self-documenting for a reader unfamiliar with the code.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Computational cost
-------------------
The outer loop runs T − 120 ≈ 168 iterations.  Each iteration:
  25 LASSO fits     — fast at p=6, negligible per iteration.
  1 CVXPY QP solve  — CLARABEL at n=25 variables, 3 constraints is fast
                      (< 5ms per solve), but 168 × 5ms ≈ 1s total.

The bottleneck is the 168 × 25 = 4,200 LASSO fits.  For a production
implementation, the 25 LASSO fits could be replaced with a single
multivariate regression or with the precomputed OOS predictions from
walk_forward_backtest(), avoiding redundant computation.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Limitations
-----------
  Redundant LASSO fits:
    This script re-runs the 168 × 25 LASSO fits from scratch rather than
    reusing the preds from walk_forward_backtest().  This is wasteful and
    makes the script slow.  Refactoring to accept precomputed predictions
    as input would reduce runtime by the cost of all those LASSO fits.

  Sample covariance at n=120, p=25:
    The ratio p/n = 25/120 ≈ 0.21 means the sample covariance matrix is
    invertible but noisy.  Ledoit-Wolf shrinkage (sklearn's
    LedoitWolf estimator) or a factor-model covariance (BARRA-style)
    would reduce estimation error and likely improve OOS performance.

  Approximate cumulative return plot:
    np.cumsum() sums returns linearly rather than compounding them.  The
    correct formula for a buy-and-hold account would be
    np.cumprod(1 + returns) − 1.  For the small monthly returns typical
    of long-short strategies the difference is minor but grows over the
    168-month backtest and could mislead a careful reader.

  Fixed λ = 2.0 risk aversion:
    The objective coefficient −2.0 · μ̂ᵀw implicitly sets the risk
    aversion parameter λ = 2.0.  This is not tuned and is not
    cross-validated — different values of λ produce different weight
    profiles and could yield meaningfully different Sharpe ratios.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Dependencies
------------
numpy, matplotlib, cvxpy (CLARABEL backend), sys
src.data_loader — load_all_data()
src.solvers     — LassoProximal
"""
import sys, numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import cvxpy as cp
sys.path.insert(0, '.')
from src.data_loader import load_all_data
from src.solvers import LassoProximal

X, Y, factor_names, _ = load_all_data()
X_vals   = X.values
X_scaled = (X_vals - X_vals.mean(0)) / X_vals.std(0)
Y_vals   = Y.values
T, n_ports  = len(X), 25
train_window = 120

def sharpe(r): return np.array(r).mean()/np.array(r).std()*np.sqrt(12)

naive_rets, mv_rets, dates = [], [], []
for t in range(train_window, T):
    X_tr = X_scaled[t-train_window:t]; Y_tr = Y_vals[t-train_window:t]
    Y_te = Y_vals[t]
    preds = [LassoProximal(alpha=0.003).fit(X_tr, Y_tr[:,j]).predict(X_scaled[t:t+1])[0]
             for j in range(n_ports)]
    preds = np.array(preds)
    ranks = np.argsort(preds)
    w_n   = np.zeros(n_ports); w_n[ranks[-3:]] = 1/3; w_n[ranks[:3]] = -1/3
    naive_rets.append(w_n @ Y_te)
    mu_hat = preds / (np.std(preds) + 1e-8)
    Sigma  = np.cov(Y_tr.T) + 1e-4*np.eye(n_ports)
    w = cp.Variable(n_ports)
    prob = cp.Problem(cp.Minimize(cp.quad_form(w, Sigma) - 2.0*mu_hat@w),
                      [cp.sum(w)==0, cp.norm(w,1)<=2, w>=-0.15, w<=0.15])
    prob.solve(solver=cp.CLARABEL, verbose=False)
    mv_rets.append((w.value if w.value is not None else w_n) @ Y_te)
    dates.append(Y.index[t])

naive = np.array(naive_rets); mv = np.array(mv_rets)
print(f'Naive Sharpe:        {sharpe(naive):.4f}')
print(f'MV Optimizer Sharpe: {sharpe(mv):.4f}')
print(f'Difference:          {(sharpe(mv)-sharpe(naive))/abs(sharpe(naive))*100:+.1f}%')

fig, ax = plt.subplots(figsize=(13, 5))
ax.plot(dates, np.cumsum(naive)*100, color='#E74C3C', lw=2, label=f'Naive (Sharpe={sharpe(naive):.3f})')
ax.plot(dates, np.cumsum(mv)*100,   color='#2E74B5', lw=2, label=f'MV Optimizer (Sharpe={sharpe(mv):.3f})')
ax.set_title('Naive vs Mean-Variance Portfolio\nQP: min wΣw - λμ̂w  s.t. dollar-neutral', fontweight='bold')
ax.set_xlabel('Date'); ax.set_ylabel('Cumulative Return (%)')
ax.legend(fontsize=10); ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('outputs/mv_portfolio.png', dpi=150, bbox_inches='tight')
print('Saved: outputs/mv_portfolio.png')
