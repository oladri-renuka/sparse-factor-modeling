"""
12_highdim_oap.py
==============
High-dimensional benchmarking experiment using the Open Asset Pricing (OAP)
dataset — a real-world predictor panel with far more features than the
six Fama-French factors used elsewhere in this project.  Tests whether the
algorithm rankings established at p=6 hold up when p scales to hundreds of
firm characteristics, and demonstrates LASSO's variable selection behavior
in a genuinely high-dimensional setting.

Generates: outputs/highdim_oap.png

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Data source — Open Asset Pricing (PredictorLSretWide.csv)
----------------------------------------------------------
The OAP dataset (Chen & Zimmermann, 2022) contains monthly long-short
portfolio returns for a large cross-section of published return predictors
(momentum variants, value signals, profitability measures, accruals, etc.).
The wide format has one column per predictor and one row per month.

Preprocessing steps:
  1. Filter to post-2000 (year >= 2000) to match the Fama-French sample
     window used in the rest of the project.
  2. Drop any column with any NaN value (axis=1 dropna) — this is a strict
     complete-case filter that retains only predictors with uninterrupted
     monthly coverage across the entire post-2000 period.  The number of
     surviving columns (p) depends on data availability and is printed at
     runtime.
  3. Construct X and y with a one-month forward shift:
       X = rows 0…T-2  (predictor values at month t)
       y = row-means of rows 1…T-1  (average cross-predictor return at t+1)
     The y construction — averaging all predictor returns at t+1 — creates
     a synthetic "consensus signal" target representing the mean return of
     the full predictor universe, which LASSO then tries to decompose back
     into a sparse linear combination of individual predictors.
  4. Column-standardize X (zero mean, unit variance with 1e-8 epsilon
     guard against zero-std columns).

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Alpha selection — α = 0.20
----------------------------
A higher alpha than the Fama-French experiments (where α = 0.003) is used
because p is much larger.  At low alpha, LASSO would select hundreds of
predictors in a p>>6 setting, defeating the sparsity demonstration.
α = 0.20 is chosen to produce a manageable sparse solution (O(10–50)
selected predictors out of p) that can be meaningfully displayed in the
coefficient bar chart.  The Ridge comparison uses α = 0.1 independently
set to a scale appropriate for dense L2 regularization at high p.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

CVXPY reference solution
--------------------------
lasso_cvxpy(X, y, alpha) is called once to provide a ground-truth
coefficient vector.  Wall-clock time is printed to illustrate the cost
of interior-point methods at high p — contrasting with the first-order
solvers benchmarked below.  The reference solution is not directly used
in the plots; the iterative solver outputs are the primary analysis
objects.  If CVXPY times out or fails, the script continues since ref is
not referenced after computation.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Algorithm benchmark — 5 timing repetitions
--------------------------------------------
Six methods are benchmarked:
  LassoProximal       — proximal gradient descent (PGD baseline)
  FISTALasso          — accelerated proximal gradient (vanilla)
  FISTARestart        — FISTA with function-value restart
  BBLasso             — Barzilai-Borwein adaptive step sizes
  CoordinateDescent   — cyclic coordinate descent
  RidgeScratch        — closed-form Ridge (α=0.1), for sparsity contrast

All LASSO solvers use max_iter=3000 to allow convergence at high p where
more iterations may be needed.  Ridge is passed no alpha or max_iter
from the outer dict (its __init__ takes alpha only); the conditional
in the solver instantiation handles this special case.

Each solver is timed over 5 independent runs; mean wall-clock ms is
reported.  The speedup ratio vs. PGD is computed as pgd_ms / method_ms
(> 1.0 means faster than PGD).  Ridge is excluded from the speedup
column ('—') since it solves a different objective.

Printed table columns:
  Method    solver name
  Iters     n_iter_ (1 for Ridge closed-form; capped at max_iter if
            non-convergence; n_iter_=0 fallback → displayed as 1)
  ms        mean wall-clock milliseconds over 5 runs
  Nonzero   number of coefficients with |β_j| > 1e-4
  vs PGD    wall-clock speedup ratio relative to Proximal GD

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Sparsity analysis — top selected predictors
--------------------------------------------
After benchmarking, the PGD (LassoProximal) coefficient vector is used
to identify selected predictors (|β_j| > 1e-4).  The top 10 by absolute
coefficient magnitude are printed with their predictor names and signed
coefficient values.  This answers the research question: given hundreds
of candidate return predictors, which ones does LASSO identify as the
most important components of the consensus signal?

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Output figure — three-panel layout (outputs/highdim_oap.png)
-------------------------------------------------------------
Panel 1 — Iteration and wall-clock comparison (bar chart):
  Five LASSO solvers (Ridge excluded) shown side by side.
  Bar height = iteration count; annotation above each bar shows
  iterations and mean ms.  Highlights whether algorithm rankings from
  the p=6 benchmark_timing.py experiment persist at high p.

Panel 2 — Sparsity: LASSO vs. Ridge (bar chart):
  Three bars: LASSO selected (non-zero), LASSO dropped (zero), Ridge
  non-zero (always p, since Ridge never exactly zeros coefficients).
  Visually demonstrates the core difference between L1 and L2
  regularization in the high-dimensional setting — LASSO performs
  automatic variable selection while Ridge retains all predictors.

Panel 3 — Top LASSO-selected predictor coefficients (horizontal bar):
  Up to 12 predictors with largest |β_j|, sorted by magnitude, with
  green bars for positive coefficients and red for negative.  Provides
  economic interpretability: which predictor long-short strategies have
  the largest positive or negative weights in explaining the consensus
  return signal.

Figure title includes n and p at runtime so the plot is self-describing
regardless of which rows/columns survived the completeness filter.
Saved at 150 dpi to outputs/highdim_oap.png.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Key research questions answered
---------------------------------
  1. Do algorithm rankings generalize to high p?
     If CoordinateDescent or BBLasso retain their p=6 advantages at
     p >> 6, the benchmark conclusions are robust.  If rankings flip,
     it suggests the p=6 results are specific to small problems.

  2. How sparse is LASSO at α = 0.20 on real high-dimensional data?
     The selected/dropped counts in Panel 2 quantify how aggressively
     LASSO prunes the predictor space relative to Ridge's dense solution.

  3. Which firm characteristics dominate?
     The signed coefficient magnitudes in Panel 3 give a data-driven
     answer to which published return predictors contain the most unique
     information about future cross-sectional returns in the post-2000
     sample.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Known limitations
------------------
  Synthetic target y:
    Averaging all predictor returns to form y is a methodological
    simplification — it creates a target that is by construction
    correlated with every predictor column, making LASSO's selection task
    easier than it would be for an independent return series.  Results
    should be interpreted as a stress test of solver scalability rather
    than a realistic return prediction exercise.

  Complete-case filter:
    Dropping all columns with any NaN is conservative and may
    disproportionately remove newer or more specialized predictors,
    biasing the surviving panel toward well-established, long-history
    signals.

  Ridge instantiation workaround:
    The conditional kwargs construction for Ridge (`if not
    name.startswith('Ridge')`) is fragile — it would silently pass
    wrong arguments if a non-Ridge solver happened to be named
    'Ridge...' or if Ridge's __init__ signature changed.  A cleaner
    approach would be separate method specs for Ridge vs. LASSO solvers.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Dependencies
------------
numpy, pandas, matplotlib, time, sys
src.solvers      — LassoProximal, FISTALasso, FISTARestart, BBLasso,
                   CoordinateDescent, RidgeScratch
src.cvxpy_solvers — lasso_cvxpy

Input file
----------
data/PredictorLSretWide.csv — Open Asset Pricing wide-format predictor
                               return panel (Chen & Zimmermann, 2022)

Reference
---------
Chen, A. Y. & Zimmermann, T. (2022). Open Source Cross-Sectional Asset
Pricing. Critical Finance Review, 11(2), 207–264.
"""
import sys, numpy as np, pandas as pd, time
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
sys.path.insert(0, '.')
from src.solvers import (LassoProximal, FISTALasso, FISTARestart,
                         BBLasso, CoordinateDescent, RidgeScratch)
from src.cvxpy_solvers import lasso_cvxpy

df = pd.read_csv("data/PredictorLSretWide.csv")
df['date'] = pd.to_datetime(df['date'])
df['year'] = df['date'].dt.year
recent_df  = df[df['year']>=2000].drop(columns=['year'])
recent_df  = recent_df.set_index('date').sort_index()
clean_df   = recent_df.dropna(axis=1)
print(f"Clean: {clean_df.shape}")

X_raw = clean_df.values[:-1]
y_raw = clean_df.values[1:].mean(axis=1)
X     = (X_raw - X_raw.mean(0)) / (X_raw.std(0) + 1e-8)
y     = y_raw
n, p  = X.shape
alpha = 0.20

print(f"CVXPY reference...")
t0 = time.perf_counter(); ref = lasso_cvxpy(X, y, alpha)
print(f"  {(time.perf_counter()-t0)*1000:.0f}ms")

methods = [
    ('Proximal GD',       LassoProximal,  {}),
    ('FISTA vanilla',     FISTALasso,     {}),
    ('FISTA+fn restart',  FISTARestart,   {'restart':'function'}),
    ('BB LASSO (alt)',    BBLasso,        {}),
    ('Coord Descent',     CoordinateDescent, {}),
    ('Ridge',             RidgeScratch,   {'alpha':0.1}),
]

print(f'\n{"Method":<26} {"Iters":>7} {"ms":>8} {"Nonzero":>9} {"vs PGD"}')
print('='*60)
pgd_ms = None; results = []
for name, cls, kwargs in methods:
    runs = []
    for _ in range(5):
        t0 = time.perf_counter()
        m  = cls(**({'alpha':alpha,'max_iter':3000} if not name.startswith('Ridge') else {}),
                 **kwargs).fit(X, y)
        runs.append((time.perf_counter()-t0)*1000)
    ms    = np.mean(runs)
    iters = m.n_iter_ if hasattr(m,'n_iter_') and m.n_iter_>0 else 1
    nz    = int(np.sum(np.abs(m.coef_)>1e-4))
    if name == 'Proximal GD': pgd_ms = ms
    ratio = f'{pgd_ms/ms:.2f}x' if pgd_ms and not name.startswith('Ridge') else '—'
    results.append({'name':name,'iters':iters,'ms':ms,'nz':nz,'coef':m.coef_.copy()})
    print(f'{name:<26} {iters:>7} {ms:>8.1f} {nz:>9} {ratio}')

lasso_coefs = results[0]['coef']
sel_idx     = np.where(np.abs(lasso_coefs)>1e-4)[0]
print(f'\nLASSO: {len(sel_idx)}/{p} selected')
top_idx = sel_idx[np.argsort(np.abs(lasso_coefs[sel_idx]))[::-1]]
for i in top_idx[:10]:
    print(f'  {clean_df.columns[i]:<28} {lasso_coefs[i]:+.5f}')

fig, axes = plt.subplots(1, 3, figsize=(18, 6))
COLORS = ['#E74C3C','#2E74B5','#27AE60','#8E44AD','#1ABC9C']
names_s = ['PGD','FISTA\nvanilla','FISTA\n+fn','BB\nLASSO','Coord\nDescent']
iters_v = [r['iters'] for r in results[:-1]]
ms_v    = [r['ms']    for r in results[:-1]]
bars    = axes[0].bar(names_s, iters_v, color=COLORS, alpha=0.85, edgecolor='white')
for bar, it, ms in zip(bars, iters_v, ms_v):
    axes[0].text(bar.get_x()+bar.get_width()/2, bar.get_height()+max(iters_v)*0.02,
                 f'{it}\n({ms:.0f}ms)', ha='center', fontsize=8.5, fontweight='bold')
axes[0].set_title(f'Iterations at p={p}\nWith wall-clock ms', fontweight='bold')
axes[0].set_ylabel('Iterations'); axes[0].grid(True, alpha=0.3, axis='y')

nz_l = results[0]['nz']; nz_r = results[-1]['nz']
axes[1].bar(['LASSO\nselected','LASSO\ndropped','Ridge\nnonzero'],
            [nz_l, p-nz_l, nz_r], color=['#27AE60','#E74C3C','#2E74B5'], alpha=0.85, edgecolor='white')
axes[1].set_title(f'Sparsity: LASSO selects {nz_l}/{p}\nRidge retains all {nz_r}', fontweight='bold')
axes[1].set_ylabel('Predictors')
for i, v in enumerate([nz_l, p-nz_l, nz_r]):
    axes[1].text(i, v+1, str(v), ha='center', fontsize=12, fontweight='bold')
axes[1].grid(True, alpha=0.3, axis='y')

if len(sel_idx) > 0:
    top_n  = min(12, len(sel_idx))
    top_c  = lasso_coefs[top_idx[:top_n]]
    top_nm = [clean_df.columns[i] for i in top_idx[:top_n]]
    cb     = ['#27AE60' if c>0 else '#E74C3C' for c in top_c]
    axes[2].barh(range(top_n), top_c, color=cb, alpha=0.85, edgecolor='white')
    axes[2].set_yticks(range(top_n)); axes[2].set_yticklabels(top_nm, fontsize=9)
    axes[2].axvline(0, color='black', lw=0.8); axes[2].invert_yaxis()
axes[2].set_title('Top LASSO-Selected Predictors', fontweight='bold')
axes[2].set_xlabel('LASSO Coefficient'); axes[2].grid(True, alpha=0.3, axis='x')

plt.suptitle(f'OpenAssetPricing: n={n}, p={p} firm characteristics',
             fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig('outputs/highdim_oap.png', dpi=150, bbox_inches='tight')
print('Saved: outputs/highdim_oap.png')
