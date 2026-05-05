"""
08_kkt_dropout_prediction.py
=============================
Full implementation and evaluation of Novel Contribution 2: using the KKT
optimality conditions of the LASSO problem to predict, without running the
solver, which factors will be zeroed out at a given α.  The partial
correlation structure of X — encoded in the precision matrix C⁻¹ — provides
a closed-form factor importance score that predicts LASSO dropout order and,
with a calibrated threshold, predicts the exact active set.

Generates: outputs/dropout_prediction.png

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Theoretical basis — KKT zero prediction
-----------------------------------------
At the LASSO optimum β★, the KKT stationarity condition for a zeroed
coordinate j (β_j★ = 0) is:

    |(2/n) Xⱼᵀ(y − Xβ★)| ≤ α

When β★ is approximately zero (light regularization relative to the data
signal), the residual y − Xβ★ ≈ y − Xβ_OLS and the condition simplifies
to a statement about the partial correlation of feature j with y after
removing shared factor variation.  Specifically, the partial correlation
score for feature j on portfolio k is:

    pc_jk = [C⁻¹ · c_k]_j

where C = corr(X) is the p×p factor correlation matrix (precision matrix
of the standardized factors) and c_k = [corr(X_j, y_k)]_{j=1..p} is the
vector of raw factor-target correlations.

KKT zero prediction rule:
  Feature j is predicted to be zeroed (inactive) for portfolio k if:

    |pc_jk| < τ · α · √n / n

  where τ is a calibrated threshold.  The α·√n/n scaling makes the
  threshold proportional to the regularization strength and inversely
  proportional to √n (tighter predictions with more data).

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Threshold calibration — train/test split
-----------------------------------------
The threshold τ controls the sensitivity/specificity trade-off:
  Large τ  → predicts more zeros (high recall for inactive factors,
             but more false alarms for active factors)
  Small τ  → predicts fewer zeros (conservative, misses some inactive
             factors but avoids false active-set exclusions)

Calibration procedure:
  200 candidate τ values are evaluated over [0.05, 3.0] using portfolios
  P1–P15 as the training set.  For each τ, the predicted zero set is
  compared against the actual LASSO zero set (|β_j| ≤ 1e-4) for each
  portfolio.  The τ maximizing the count of portfolios where the
  predicted set exactly matches the actual set is selected.

  Exact set match criterion: pred == act (both the predicted zero set
  and the actual zero set must be identical — no partial credit).  This
  is a strict criterion: a single wrong prediction for any factor flips
  a portfolio from correct to incorrect.

Out-of-sample validation:
  The selected τ is then applied to portfolios P16–P25 (held-out test set)
  without any further tuning.  Test accuracy is printed per portfolio with
  ✓/✗ indicators and the predicted vs. actual zero sets shown explicitly.

Train/test split rationale:
  The 15/10 split (60% train / 40% test) is conservative given only 25
  portfolios total.  A leave-one-out cross-validation would give a more
  stable τ estimate; the current split is chosen for simplicity and to
  give a meaningful held-out test set size.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Full evaluation — 25 × 6 prediction matrix
--------------------------------------------
After threshold calibration, predictions are generated for all 25
portfolios × 6 factors, producing:

  pred_matrix   (25 × 6)  binary: 1 = predicted active, 0 = predicted zero
  actual_matrix (25 × 6)  binary: 1 = actually active,  0 = actually zero
  diff matrix   (25 × 6)  pred − actual:
                           0  = correct prediction (TP or TN)
                          +1  = false alarm (predicted active, actually zero)
                          −1  = miss (predicted zero, actually active)

Per-factor confusion matrix counts (TP, TN, FP, FN) accumulated over all
25 portfolios provide factor-level accuracy.  Overall accuracy is the
fraction of all 150 (portfolio × factor) cells predicted correctly.

Printed diagnostics:
  Overall accuracy as a fraction and percentage (target: > 85%).
  Per-factor accuracy percentage — identifies which factors are
  systematically harder to predict (low accuracy) and which are
  reliably predicted (high accuracy).

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Output figure — three-panel layout (outputs/dropout_prediction.png)
--------------------------------------------------------------------
Panel 1 — Per-factor prediction accuracy bar chart:
  Six bars, one per factor, showing the percentage of portfolios (out of
  25) for which that factor's active/zero status was correctly predicted.
  Color coding: green ≥ 80%, orange ≥ 60%, red < 60%.  Reference lines at
  60% and 80% mark qualitative accuracy thresholds.  Bar annotations show
  exact percentage.  A factor near 100% is trivially predictable (always
  active or always zero across portfolios); a factor near 50% is the
  hardest to predict and contributes most to overall error.

Panel 2 — Prediction error heatmap (25 portfolios × 6 factors):
  Color-coded difference matrix (pred − actual).  Green cells (0) = correct
  prediction; red cells (−1) = missed zeros (factor predicted active but
  actually zeroed); yellow/green cells (+1) = false alarms (factor
  predicted zero but actually active).  Annotated with numeric values for
  exact reading.  Reveals systematic error patterns — e.g., if a specific
  factor (column) is consistently mispredicted across many portfolios, or
  if a specific portfolio (row) has many errors, suggesting the KKT
  approximation is poor for that portfolio's data structure.

Panel 3 — Partial correlation magnitude heatmap (25 × 6):
  The raw |pc_jk| = |[C⁻¹c_k]_j| scores that drive the predictions.
  Higher values indicate stronger unique predictive content for that
  factor-portfolio pair.  The threshold τ·α·√n/n partitions each row
  into predicted active (above threshold) and predicted zero (below).
  This panel provides interpretability: readers can see which factors
  have genuinely high partial correlations with which portfolios,
  independent of the binary threshold decision.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Practical significance
-----------------------
A reliable KKT-based zero predictor has two direct applications:

  1. Warm-start initialization:
     Before running the solver, zero out predicted inactive coordinates.
     This effectively solves a lower-dimensional problem from the start,
     reducing iterations.  High prediction accuracy (> 85%) ensures the
     warm start rarely excludes a truly active factor.

  2. Interpretable factor screening:
     The partial correlation score |C⁻¹c_k|_j provides a continuous
     importance measure for each (factor, portfolio) pair that can be
     computed in microseconds — useful for real-time factor relevance
     monitoring without re-running the full LASSO path.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Limitations and caveats
------------------------
  Threshold overfitting risk:
    τ is selected by exhaustive search over 200 candidates on 15 portfolios
    with p = 6 features per portfolio (90 binary predictions).  The
    optimization landscape is coarse — many τ values give identical training
    accuracy — so the "best" τ may be one of many near-equivalent solutions.
    Reporting training accuracy alongside test accuracy is essential to
    distinguish genuine generalization from lucky threshold selection.

  Full-sample look-ahead in C and c_k:
    Both the precision matrix C⁻¹ and the factor-target correlations c_k
    are computed from the full 2000–2023 sample.  In a strict no-look-ahead
    setting, these should be recomputed from the training window at each
    step of the rolling backtest.  The full-sample versions are used here
    for the static threshold calibration only.

  KKT approximation quality:
    The derivation approximates y − Xβ★ ≈ y, dropping the fitted residual
    term.  This approximation improves as α → ∞ (more regularization, β★
    closer to zero) and deteriorates as α → 0 (dense solution, β★ far from
    zero).  At the CV-optimal α = 0.003 the approximation quality is
    moderate — accuracy results should be interpreted in this context.

  p = 6 small-factor caveat:
    With only 6 factors, the chance that a random binary predictor achieves
    high exact set-match accuracy is non-trivial.  For p = 6, there are
    only 2⁶ = 64 possible active sets, and the actual LASSO solutions at
    α = 0.003 cluster around a small subset of them.  The prediction
    accuracy should be compared to an informed baseline (e.g., always
    predicting the most common active set) not just to 50% random guessing.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Dependencies
------------
numpy, matplotlib, seaborn, sys
src.data_loader — load_all_data()
src.solvers     — LassoProximal
"""
import sys, numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
sys.path.insert(0, '.')
from src.data_loader import load_all_data
from src.solvers import LassoProximal

X, Y, factor_names, _ = load_all_data()
X_vals   = X.values
X_scaled = (X_vals - X_vals.mean(0)) / X_vals.std(0)
n, p     = X_scaled.shape
alpha    = 0.003

C    = np.corrcoef(X_scaled.T)
Cinv = np.linalg.inv(C)

# Tune threshold on portfolios 1-15
best_thresh, best_acc = None, -1
for tau in np.linspace(0.05, 3.0, 200):
    correct = 0
    for k in range(15):
        y_k  = Y.iloc[:, k].values
        c_k  = np.array([np.corrcoef(X_scaled[:,j], y_k)[0,1] for j in range(p)])
        pc   = Cinv @ c_k
        pred = set(j for j in range(p) if abs(pc[j]) < tau*alpha*np.sqrt(n)/n)
        m    = LassoProximal(alpha=alpha).fit(X_scaled, y_k)
        act  = set(j for j in range(p) if abs(m.coef_[j]) <= 1e-4)
        if pred == act: correct += 1
    if correct > best_acc: best_acc = correct; best_thresh = tau

print(f'Best threshold: {best_thresh:.3f}  Training accuracy: {best_acc}/15')

# Verify on held-out portfolios 16-25
correct_test = 0
for k in range(15, 25):
    y_k  = Y.iloc[:, k].values
    c_k  = np.array([np.corrcoef(X_scaled[:,j], y_k)[0,1] for j in range(p)])
    pc   = Cinv @ c_k
    tau  = best_thresh*alpha*np.sqrt(n)/n
    pred = set(j for j in range(p) if abs(pc[j]) < tau)
    m    = LassoProximal(alpha=alpha).fit(X_scaled, y_k)
    act  = set(j for j in range(p) if abs(m.coef_[j]) <= 1e-4)
    match = pred == act
    if match: correct_test += 1
    pn = [factor_names[j] for j in sorted(pred)]
    an = [factor_names[j] for j in sorted(act)]
    print(f'P{k+1}: pred={pn}  actual={an}  {"✓" if match else "✗"}')
print(f'Test accuracy: {correct_test}/10')

# Full analysis
tp = np.zeros(p); tn = np.zeros(p); fp = np.zeros(p); fn = np.zeros(p)
pred_matrix   = np.zeros((25, p))
actual_matrix = np.zeros((25, p))
for k in range(25):
    y_k   = Y.iloc[:, k].values
    c_k   = np.array([np.corrcoef(X_scaled[:,j], y_k)[0,1] for j in range(p)])
    pc    = Cinv @ c_k
    tau   = best_thresh*alpha*np.sqrt(n)/n
    m     = LassoProximal(alpha=alpha).fit(X_scaled, y_k)
    for j in range(p):
        pz = abs(pc[j]) < tau
        az = abs(m.coef_[j]) <= 1e-4
        pred_matrix[k,j]   = 0 if pz else 1
        actual_matrix[k,j] = 0 if az else 1
        if pz and az:     tn[j] += 1
        if not pz and not az: tp[j] += 1
        if pz and not az: fn[j] += 1
        if not pz and az: fp[j] += 1

print(f'\nOverall: {int(np.sum(pred_matrix==actual_matrix))}/150 = {np.mean(pred_matrix==actual_matrix)*100:.1f}%')
for j in range(p):
    print(f'  {factor_names[j]}: {(tp[j]+tn[j])/25*100:.0f}%')

# Plot
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
accs  = [(tp[j]+tn[j])/25*100 for j in range(p)]
cols  = ['#27AE60' if a>=80 else '#F39C12' if a>=60 else '#E74C3C' for a in accs]
bars  = axes[0].bar(factor_names, accs, color=cols, alpha=0.85, edgecolor='white')
axes[0].axhline(80, color='#27AE60', lw=2, ls='--', label='80%')
axes[0].axhline(60, color='#F39C12', lw=2, ls='--', label='60%')
for bar, acc in zip(bars, accs):
    axes[0].text(bar.get_x()+bar.get_width()/2, bar.get_height()+1,
                 f'{acc:.0f}%', ha='center', fontsize=10, fontweight='bold')
axes[0].set_title('KKT Dropout Prediction Accuracy', fontweight='bold')
axes[0].set_ylabel('Accuracy (%)'); axes[0].legend(fontsize=9)
axes[0].grid(True, alpha=0.3, axis='y'); axes[0].set_ylim(0, 115)

diff = pred_matrix - actual_matrix
sns.heatmap(diff, xticklabels=factor_names,
            yticklabels=[f'P{i+1}' for i in range(25)],
            cmap='RdYlGn', center=0, vmin=-1, vmax=1,
            annot=True, fmt='.0f', ax=axes[1], linewidths=0.5,
            cbar_kws={'label': '-1=Miss  0=Correct  +1=FalseAlarm'})
axes[1].set_title('Prediction Error Map', fontweight='bold')

pc_matrix = np.zeros((25, p))
for k in range(25):
    y_k = Y.iloc[:, k].values
    c_k = np.array([np.corrcoef(X_scaled[:,j], y_k)[0,1] for j in range(p)])
    pc_matrix[k] = Cinv @ c_k
sns.heatmap(np.abs(pc_matrix), xticklabels=factor_names,
            yticklabels=[f'P{i+1}' for i in range(25)],
            cmap='RdYlGn', annot=True, fmt='.2f', ax=axes[2], linewidths=0.5,
            cbar_kws={'label': '|Partial Correlation|'})
axes[2].set_title(f'Partial Correlations |[C⁻¹c_k]_j|', fontweight='bold')

plt.suptitle('KKT Factor Dropout Prediction — Novel Contribution',
             fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig('outputs/dropout_prediction.png', dpi=150, bbox_inches='tight')
print('Saved: outputs/dropout_prediction.png')
