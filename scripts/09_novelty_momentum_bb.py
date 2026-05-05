"""
09_novelty_momentum_bb.py
======================
Empirical evaluation of Novel Contribution 1a: the Momentum-BB LASSO
algorithm, which combines FISTA's momentum extrapolation with Barzilai-
Borwein adaptive step sizing.  This script benchmarks Momentum-BB against
the four established baselines and plots per-iteration convergence curves
on three representative portfolios.

Generates:
  outputs/momentum_bb_convergence.png  — convergence curves (primary output)
  outputs/novelty_momentum_bb.png      — referenced in the script name but
                                         not explicitly saved here; likely
                                         produced by a companion script or
                                         is an alias for the convergence plot

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Algorithm under test — MomentumBBLasso
----------------------------------------
Defined in solvers.py.  At each iteration k:

  1. FISTA momentum extrapolation:
       y_k = β_k + ((t−1)/(t+2)) · (β_k − β_{k−1})
     builds a predicted next point using the recent iterate direction.

  2. Gradient at the momentum point:
       grad_k = −(2/n) Xᵀ(y − X·y_k)
     evaluates the smooth loss gradient at y_k, not at β_k.

  3. BB adaptive step from momentum-point differences:
       s_k = y_k − y_{k−1},   g_k = grad_k − grad_{k−1}
       η_k = (sᵀs)/(sᵀg)  if k even  (BB1, long step)
           = (sᵀg)/(gᵀg)  if k odd   (BB2, short step)
     estimates local curvature along the momentum trajectory.

  4. Proximal soft-threshold step:
       β_{k+1} = sign(y_k − η_k·grad_k) · max(|y_k − η_k·grad_k| − α·η_k, 0)

The novel design choice is applying BB to the sequence of momentum points
(y_k) rather than the iterate sequence (β_k).  This is consistent with
FISTA's philosophy of treating the momentum point as the effective current
position, and gives BB access to the momentum-smoothed curvature signal
rather than the noisier raw iterate differences.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Benchmark setup — iteration and wall-clock comparison
-------------------------------------------------------
Five solvers are benchmarked at α = 0.003 (the CV-optimal value for the
Fama-French dataset):
  LassoProximal     — PGD baseline
  FISTALasso        — vanilla accelerated proximal gradient
  FISTARestart      — FISTA with function-value restart
  BBLasso           — Barzilai-Borwein adaptive step (alternating BB1/BB2)
  MomentumBBLasso   — the novel combination (marked ← NEW in output)

For each solver and each of the 25 portfolio target series:
  • 10 independent timing runs are performed; mean wall-clock ms is recorded.
  • n_iter_ is recorded from the final run (the 10th fit, not an average —
    iteration count is deterministic for a given (X, y, α) triple so the
    final run is representative).
  • If n_iter_ == 0 (non-convergence flag from solvers.py), it is replaced
    with 500 (the max_iter cap) for display purposes.

Mean iters and mean ms are averaged over the 25 portfolios and printed
with the speedup ratio relative to PGD wall-clock time.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Convergence curve experiment
------------------------------
Three portfolios are selected to represent different positions in the
25-portfolio size × value grid:
  P1   (port_idx=0)   small-cap growth  — typically harder to fit,
                      lower R², less stable factor loadings
  P8   (port_idx=7)   mid-grid portfolio — intermediate difficulty
  P19  (port_idx=18)  large-cap value   — typically easier to fit,
                      higher R², more stable loadings

For each portfolio, all five solvers are fitted with max_iter=300 and
their full loss_history_ sequences are extracted.

Objective gap construction:
  opt = min over all methods and all iterations of the observed loss.
  This serves as a proxy for the true optimum — since no ground-truth
  is computed via CVXPY here, the best value observed by any solver
  is treated as the reference.  Each solver's gap series is then:
    gap_k = loss_k − opt + 1e-12
  The 1e-12 floor prevents log(0) when a solver exactly hits opt.

The gap is plotted on a log-scale y-axis (semilogy), so a straight line
indicates geometric (linear) convergence and a curve bending downward
indicates super-linear acceleration.  Iteration count is shown in each
legend entry (the number of iterations the solver actually ran before
converging or hitting max_iter=300).

Visual encoding:
  MomentumBBLasso is plotted with lw=3.0 and zorder=5 to make it the
  most visually prominent line — the new algorithm is the focus.
  All other solvers use lw=1.8 and zorder=3.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

What the convergence plot reveals
-----------------------------------
The three sub-plots allow the following comparisons:

  1. Momentum-BB vs. BB alone (purple vs. orange):
     Does adding FISTA momentum to BB's adaptive step improve convergence
     per iteration, or does the momentum-point BB coupling introduce
     overhead that cancels the benefit?

  2. Momentum-BB vs. FISTA+restart (green vs. orange):
     FISTA+restart is the strongest published baseline at this α.  If
     Momentum-BB converges in fewer iterations or to a lower gap,
     the novel combination provides a measurable improvement over the
     state of the art.

  3. Cross-portfolio consistency:
     If Momentum-BB outperforms on P1 but not P8 or P19, the advantage
     is portfolio-specific rather than a general property of the algorithm.
     Consistent behavior across all three panels is the stronger result.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Theoretical expectation vs. empirical outcome
----------------------------------------------
Hypothesis from solvers.py:
  Combining O(1/t²) momentum direction with BB curvature adaptation
  should converge faster than either method alone — momentum provides
  a better search direction, BB provides a better step size.

Tension:
  BB's curvature estimate is computed from the momentum-point differences
  (y_k, y_{k-1}) rather than the iterate differences (β_k, β_{k-1}).
  The momentum points are extrapolated beyond the current iterate, so
  their secant pairs (s, g) may have different spectral properties than
  those of the raw iterates, potentially making the BB estimate noisier
  or less representative of the true local curvature.  Whether this
  coupling is beneficial or detrimental is an open empirical question
  that this script directly answers.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Known issues and limitations
------------------------------
  opt proxy:
    Using the best observed loss as the optimum reference overstates
    convergence quality for the solver that happens to find the lowest
    value — that solver's gap curve is anchored at its own best point,
    making it appear to converge to zero while other solvers' gaps are
    measured against a tighter reference.  A CVXPY ground truth (as used
    in benchmark_timing.py) would give a fairer comparison; lasso_cvxpy
    is imported but not called in this script.

  n_iter_ timing mismatch:
    The iteration count is read from the final (10th) timing run but ms
    is averaged over all 10 runs.  Since n_iter_ is deterministic given
    the same (X, y, α, init), the 10th run's n_iter_ is representative.
    However, if any solver has non-deterministic n_iter_ (e.g. due to
    floating-point non-reproducibility across runs), the reported iters
    and ms could correspond to slightly different convergence trajectories.

  Second output file:
    The script name and docstring reference outputs/novelty_momentum_bb.png
    but only outputs/momentum_bb_convergence.png is explicitly saved.
    The first filename is either saved by a companion script, is a
    leftover reference from an earlier version, or is intended to be
    added as a second figure (e.g. the benchmark bar chart).

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Dependencies
------------
numpy, matplotlib, time, sys
src.data_loader   — load_all_data()
src.solvers       — LassoProximal, FISTALasso, FISTARestart, BBLasso,
                    MomentumBBLasso
src.cvxpy_solvers — lasso_cvxpy (imported but not called)
"""
import sys, numpy as np, time
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
sys.path.insert(0, '.')
from src.data_loader import load_all_data
from src.solvers import (LassoProximal, FISTALasso, FISTARestart,
                         BBLasso, MomentumBBLasso)
from src.cvxpy_solvers import lasso_cvxpy

X, Y, factor_names, _ = load_all_data()
X_vals   = X.values
X_scaled = (X_vals - X_vals.mean(0)) / X_vals.std(0)
alpha    = 0.003

methods = [
    ('Proximal GD',        LassoProximal,   {}, '#E74C3C'),
    ('FISTA vanilla',      FISTALasso,      {}, '#2E74B5'),
    ('FISTA+fn restart',   FISTARestart,    {'restart':'function'}, '#27AE60'),
    ('BB LASSO (alt)',     BBLasso,         {}, '#8E44AD'),
    ('Momentum-BB (novel)',MomentumBBLasso, {}, '#FF6B00'),
]

print(f'{"Method":<26} {"Mean Iters":>11} {"ms":>8} {"vs PGD"}')
print('='*55)
pgd_ms = None; results = []
for name, cls, kwargs, color in methods:
    iters_all, ms_all = [], []
    for j in range(25):
        yj   = Y.iloc[:, j].values
        runs = []
        for _ in range(10):
            t0 = time.perf_counter()
            m  = cls(alpha=alpha, max_iter=500, **kwargs).fit(X_scaled, yj)
            runs.append((time.perf_counter()-t0)*1000)
        iters_all.append(m.n_iter_ if m.n_iter_ > 0 else 500)
        ms_all.append(np.mean(runs))
    mi = np.mean(iters_all); mt = np.mean(ms_all)
    results.append({'name':name,'iters':mi,'ms':mt,'color':color,'cls':cls,'kwargs':kwargs})
    if name == 'Proximal GD': pgd_ms = mt
    ratio = f'{pgd_ms/mt:.2f}x' if pgd_ms else '—'
    marker = '  ← NEW' if 'novel' in name else ''
    print(f'{name:<26} {mi:>11.1f} {mt:>8.3f} {ratio}{marker}')

# Convergence plot
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
COLORS_MAP = {r['name']: r['color'] for r in results}
for ax_idx, port_idx in enumerate([0, 7, 18]):
    y_p = Y.iloc[:, port_idx].values
    hists = {}; opt = np.inf
    for name, cls, kwargs, _ in methods:
        m = cls(alpha=alpha, max_iter=300, **kwargs).fit(X_scaled, y_p)
        hists[name] = m.loss_history_; opt = min(opt, min(m.loss_history_))
    ax = axes[ax_idx]
    for name, h in hists.items():
        gap = np.array(h) - opt + 1e-12
        lw  = 3.0 if 'novel' in name else 1.8
        ax.semilogy(gap, color=COLORS_MAP[name], lw=lw, label=f'{name} ({len(h)})',
                    zorder=5 if 'novel' in name else 3)
    ax.set_title(f'Portfolio P{port_idx+1}', fontweight='bold')
    ax.set_xlabel('Iteration'); ax.set_ylabel('Objective Gap (log)')
    ax.legend(fontsize=7); ax.grid(True, alpha=0.3)
plt.suptitle('Novel: Momentum-BB LASSO vs All Methods', fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig('outputs/momentum_bb_convergence.png', dpi=150, bbox_inches='tight')
print('Saved: outputs/momentum_bb_convergence.png')
