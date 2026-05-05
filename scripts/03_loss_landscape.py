"""
03_loss_landscape.py
=================
Visualizes the objective function landscape of Ridge and LASSO regression
in the two-dimensional subspace spanned by β_HML and β_CMA, providing a
geometric explanation for why LASSO zeros out CMA (investment factor) in
the majority of the 25 portfolios while Ridge retains it with a small
non-zero coefficient.

Generates: outputs/loss_landscape.png

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Motivation — why HML and CMA?
-------------------------------
HML (book-to-market / value) and CMA (investment / conservative-minus-
aggressive) are the most correlated factor pair in the Fama-French five-
factor model, with a Pearson correlation of approximately 0.632 over the
2000–2023 sample.  This high correlation creates a near-flat loss surface
along the HML-CMA diagonal — the MSE objective is almost equally minimized
by many (β_HML, β_CMA) combinations that trade off one factor against the
other.  This degeneracy is precisely the setting where the geometry of the
regularization penalty — elliptical for Ridge, diamond-shaped for LASSO —
determines which solution is selected, making HML-CMA the most instructive
pair for illustrating the qualitative difference between the two penalties.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Landscape construction — 2D slice through coefficient space
-------------------------------------------------------------
Rather than plotting the full 6-dimensional objective, the landscape is
computed over a 2D grid by holding all four non-displayed coefficients
fixed at the LASSO optimum and varying only β_HML and β_CMA:

    b_fixed = lasso_opt.copy()
    b[HML_idx] = HH[i,j]
    b[CMA_idx] = CC[i,j]

This "2D slice" approach gives a faithful picture of the objective
geometry in the HML-CMA plane around the optimum, while keeping the
influence of the other four factors constant.  The slice passes exactly
through the LASSO optimum by construction (since b_fixed is initialized
to lasso_opt), so the LASSO star marker sits at the true minimum of the
displayed LASSO surface.

The Ridge star marker, however, is the Ridge optimum over all six
dimensions — it does not generally sit at the minimum of the displayed
Ridge surface (which uses b_fixed from lasso_opt for the four frozen
coefficients, not from ridge_opt).  This is a known approximation: the
displayed Ridge contours reflect Ridge's penalty geometry but the Ridge
optimum shown is the true 6D Ridge solution projected onto the grid.
For a fully consistent Ridge landscape, b_fixed should be set to
ridge_opt before constructing the Ridge grid.

Grid: 100×100 points over:
  β_HML ∈ [−0.04,  0.015]   spans the typical LASSO and Ridge HML values
  β_CMA ∈ [−0.025, 0.010]   spans zero and both optimum values

For each grid point (i, j), the objective values are:
  Ridge: MSE(b) + 0.1 · ‖b‖₂²
  LASSO: MSE(b) + α · ‖b‖₁   where α = 0.003

The Ridge α = 0.1 and LASSO α = 0.003 are the same values used
throughout the project — the landscape reflects the actual regularization
strength applied in the backtesting pipeline.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Geometric interpretation — ellipse vs. diamond
------------------------------------------------
Ridge penalty ‖β‖₂²:
  The Ridge penalty adds a bowl-shaped quadratic surface centered at
  the origin.  Adding this to the MSE loss shifts the combined objective's
  minimum toward the origin but preserves the elliptical contour shape of
  the MSE.  The combined contours are still ellipses, so the optimum is
  always in the interior of the (β_HML, β_CMA) plane — never exactly on
  an axis.  Both β_HML and β_CMA are non-zero at the Ridge optimum.

LASSO penalty α‖β‖₁:
  The LASSO penalty adds a pyramid (diamond in 2D) with sharp corners on
  the coordinate axes.  When the MSE contours are elongated along the
  HML-CMA diagonal (due to their high correlation), the innermost MSE
  contour that is tangent to the LASSO diamond touches a corner of the
  diamond — specifically the corner at (β_HML ≠ 0, β_CMA = 0) where the
  CMA axis intersects.  This geometric corner-tangency forces β_CMA = 0
  exactly at the LASSO optimum, even though the MSE surface alone would
  prefer a non-zero CMA coefficient.

The L1 ball plotted on the LASSO panel:
  The diamond outline at radius α = 0.003 shows the boundary of the L1
  penalty ball in the (β_HML, β_CMA) subspace.  In practice the true
  L1 ball lives in all 6 dimensions and the 2D projection is a slice
  through it, but the visual correctly illustrates why the corner on
  the β_CMA = 0 axis is the likely contact point when the two factors
  are highly correlated.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Connection to empirical results
---------------------------------
The title annotation "Explains 14/25 CMA dropout" refers to the result
from novel2_kkt_factor_ranking.py and novel2_dropout_prediction.py:
CMA is one of the most frequently zeroed factors across the 25 portfolios
at α = 0.003.  This visualization provides the geometric mechanism behind
that empirical finding — it is not a coincidence that CMA drops out, but
a direct consequence of:
  (a) the high HML-CMA correlation making the MSE surface flat along
      that diagonal, and
  (b) LASSO's diamond constraint having corners exactly on the axes.

For Ridge (with its smooth elliptical penalty), no such corner exists and
CMA retains a small non-zero loading in all portfolios.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Output figure — two-panel layout (outputs/loss_landscape.png)
--------------------------------------------------------------
Panel 1 — Ridge objective landscape (Blues colormap):
  Filled contours (contourf) with 30 levels show the full objective
  surface.  Unfilled contours (contour) at 15 levels add iso-value lines.
  Red star marks the Ridge optimum; dashed zero lines at β_HML=0 and
  β_CMA=0 show the axes.  Elliptical contours curving away from both axes
  confirm that the Ridge optimum lies in the interior.

Panel 2 — LASSO objective landscape (Reds colormap):
  Same structure as Panel 1 but for the LASSO objective.  Blue star marks
  the LASSO optimum, annotated with the exact β_CMA value (expected ≈ 0).
  Thicker zero-axis lines emphasize the coordinate axes where sparsity
  occurs.  The L1 ball boundary (blue diamond) is overlaid to show the
  constraint geometry.  The contours are less elliptical than Ridge due
  to the L1 penalty's kinks along the axes.

Supertitle includes the HML-CMA correlation (0.632) as the key data
point motivating the choice of this factor pair.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Limitations and caveats
------------------------
  Ridge landscape inconsistency:
    b_fixed is initialized to lasso_opt, so the Ridge contours show the
    objective in the HML-CMA plane holding the other four coefficients at
    their LASSO values, not their Ridge values.  The Ridge star (true 6D
    Ridge optimum projected onto the grid) may not sit at the minimum of
    the displayed Ridge surface.  To fix: compute ridge_landscape with
    b_fixed = ridge_opt and lasso_landscape with b_fixed = lasso_opt
    independently.

  2D projection is a slice, not the full picture:
    The displayed surfaces are 2D cross-sections of 6D objective functions.
    The actual LASSO sparsity geometry depends on all six coordinates
    simultaneously; the 2D slice correctly illustrates the mechanism but
    cannot show how the other four coefficients interact with the HML-CMA
    trade-off.

  L1 ball radius uses α = 0.003 in 2D:
    The plotted diamond has radius α = 0.003, which is the full 6D L1
    penalty budget.  In the 2D slice, the effective budget allocated to
    HML and CMA together is α minus the penalty already consumed by the
    other four coefficients.  The plotted diamond is therefore larger than
    the true 2D projection of the constraint set.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Dependencies
------------
numpy, matplotlib, sys
src.data_loader — load_all_data()
src.solvers     — LassoProximal, RidgeScratch
"""
import sys, numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
sys.path.insert(0, '.')
from src.data_loader import load_all_data
from src.solvers import LassoProximal, RidgeScratch

X, Y, factor_names, _ = load_all_data()
X_vals   = X.values
X_scaled = (X_vals - X_vals.mean(0)) / X_vals.std(0)
y_vals   = Y.iloc[:, 0].values
alpha    = 0.003
HML_idx, CMA_idx = 2, 4

lasso_opt = LassoProximal(alpha=alpha).fit(X_scaled, y_vals).coef_
ridge_opt = RidgeScratch(alpha=0.1).fit(X_scaled, y_vals).coef_

print(f'LASSO: beta_HML={lasso_opt[HML_idx]:.5f}, beta_CMA={lasso_opt[CMA_idx]:.5f}')
print(f'Ridge: beta_HML={ridge_opt[HML_idx]:.5f}, beta_CMA={ridge_opt[CMA_idx]:.5f}')

b_fixed = lasso_opt.copy()
grid    = 100
hml_g   = np.linspace(-0.04, 0.015, grid)
cma_g   = np.linspace(-0.025, 0.01,  grid)
HH, CC  = np.meshgrid(hml_g, cma_g)

ridge_obj = np.zeros((grid, grid))
lasso_obj = np.zeros((grid, grid))
for i in range(grid):
    for j in range(grid):
        b = b_fixed.copy()
        b[HML_idx] = HH[i,j]; b[CMA_idx] = CC[i,j]
        mse = np.mean((y_vals - X_scaled@b)**2)
        ridge_obj[i,j] = mse + 0.1*np.sum(b**2)
        lasso_obj[i,j] = mse + alpha*np.sum(np.abs(b))

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
c1 = axes[0].contourf(HH, CC, ridge_obj, levels=30, cmap='Blues')
axes[0].contour(HH, CC, ridge_obj, levels=15, colors='navy', alpha=0.4, lw=0.8)
plt.colorbar(c1, ax=axes[0])
axes[0].plot(ridge_opt[HML_idx], ridge_opt[CMA_idx], 'r*', ms=15, zorder=5, label='Ridge optimum')
axes[0].axhline(0, color='k', lw=0.8, ls='--', alpha=0.5)
axes[0].axvline(0, color='k', lw=0.8, ls='--', alpha=0.5)
axes[0].set_xlabel('β_HML'); axes[0].set_ylabel('β_CMA')
axes[0].set_title('Ridge: Elliptical contours\nInterior solution — both nonzero', fontweight='bold')
axes[0].legend()

c2 = axes[1].contourf(HH, CC, lasso_obj, levels=30, cmap='Reds')
axes[1].contour(HH, CC, lasso_obj, levels=15, colors='darkred', alpha=0.4, lw=0.8)
plt.colorbar(c2, ax=axes[1])
axes[1].plot(lasso_opt[HML_idx], lasso_opt[CMA_idx], 'b*', ms=15, zorder=5,
             label=f'LASSO: β_CMA={lasso_opt[CMA_idx]:.4f}')
axes[1].axhline(0, color='k', lw=1.5, alpha=0.7)
axes[1].axvline(0, color='k', lw=1.5, alpha=0.7)
r = alpha
axes[1].plot([r,0,-r,0,r],[0,r,0,-r,0],'b-',lw=2,alpha=0.5,label='L1 ball')
axes[1].set_xlabel('β_HML'); axes[1].set_ylabel('β_CMA')
axes[1].set_title('LASSO: L1 corner forces β_CMA=0\nExplains 14/25 CMA dropout', fontweight='bold')
axes[1].legend()

plt.suptitle('Loss Landscape: Ridge vs LASSO over (β_HML, β_CMA)\nHML-CMA correlation=0.632', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('outputs/loss_landscape.png', dpi=150, bbox_inches='tight')
print('Saved: outputs/loss_landscape.png')
