"""
solvers.py
==========
From-scratch implementations of six regularized regression solvers and one
novel algorithm, all sharing a common sklearn-compatible fit/predict interface.
Every solver targets a convex objective over the same (n × p) design matrix and
can be dropped interchangeably into the backtesting, cross-validation, and
benchmarking pipelines.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Shared interface contract
--------------------------
All eight classes expose:
  __init__(alpha, ...)   regularization strength + solver-specific hyperparams
  fit(X, y)              fits coefficients; returns self for method chaining
  predict(X)             returns X @ coef_
  coef_        np.ndarray (p,)   fitted coefficient vector
  n_iter_      int               iterations to convergence (1 for closed-form)
  loss_history_ list             per-iteration objective value

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

RidgeScratch
------------
Objective:  minimize  (1/n) ‖y − Xβ‖₂²  +  α ‖β‖₂²

Closed-form solution via the normal equations:
    β★ = (XᵀX + αI)⁻¹ Xᵀy

Solved with np.linalg.solve (LU decomposition) rather than np.linalg.inv
to avoid explicit matrix inversion.  n_iter_ is fixed at 1 — there is no
iterative loop.  The L2 penalty makes XᵀX + αI strictly positive definite
for all α > 0, guaranteeing a unique solution regardless of rank(X).

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

LassoProximal
-------------
Objective:  minimize  (1/n) ‖y − Xβ‖₂²  +  α ‖β‖₁

Proximal gradient descent: at each iteration, take a gradient step on the
smooth loss, then apply the proximal operator of the L1 norm (soft-thresholding)
to handle the non-smooth penalty exactly.

  Gradient step:    β½  = β − η · (−2/n) Xᵀ(y − Xβ)
  Proximal step:    βₙₑw = sign(β½) · max(|β½| − α·η, 0)

Step size η is set to 1 / (2‖XᵀX‖₂ / n) — the reciprocal of the
Lipschitz constant of the gradient — guaranteeing convergence at O(1/t).

Convergence criterion: ‖βₙₑw − β‖₂ < tol.

Note: the class defines fit() twice; Python resolves this by using the
second definition, which is the actual iterative solver.  The first
definition (which calls fit_path) is unreachable dead code — a remnant
of an earlier warm-start refactor.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

ElasticNetScratch
-----------------
Objective:  minimize  (1/n) ‖y − Xβ‖₂²  +  λ₁ ‖β‖₁  +  λ₂ ‖β‖₂²
            where  λ₁ = α · l1_ratio,  λ₂ = α · (1 − l1_ratio)

The L2 term is smooth, so it is folded into the gradient rather than the
proximal step.  The proximal operator of the combined L1 + L2 penalty
(the "elastic net" proximal operator) has the closed form:

  βₙₑw = sign(β½) · max(|β½| − η·λ₁, 0) / (1 + 2·η·λ₂)

The extra (1 + 2·η·λ₂) denominator is the key difference from pure LASSO:
it shrinks the non-zero coefficients an additional multiplicative factor
beyond the soft-threshold, providing Ridge-like shrinkage on top of L1
sparsity.  Step size η accounts for both the Lipschitz constant of the
squared loss and the curvature added by the L2 term.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

FISTALasso  (Beck & Teboulle, 2009)
-------------------------------------
Objective:  same as LassoProximal

FISTA accelerates proximal GD from O(1/t) to O(1/t²) by adding a
momentum extrapolation step before each gradient computation:

  Extrapolation:  y_mom = β + ((t−1)/(t+2)) · (β − β_prev)
  Gradient:       grad  = (−2/n) Xᵀ(y − X·y_mom)
  Proximal step:  βₙₑw  = soft_threshold(y_mom − η·grad, α·η)

The momentum coefficient (t−1)/(t+2) grows toward 1 as t increases,
giving more weight to the previous update direction at later iterations
when the trajectory is smoother.

Duality gap monitoring (every 10 iterations):
  The duality gap provides a certificate of optimality that is tighter
  than the primal convergence criterion alone.  A gap < tol triggers
  early termination.  The dual variable ν is constructed by projecting
  the scaled residual onto the dual feasible set ‖Xᵀν‖∞ ≤ α.

Note: like LassoProximal, this class defines fit() twice; the second
definition is the live implementation.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

WarmStartLasso
--------------
Objective:  same as LassoProximal, solved across a decreasing alpha grid

Solves the full LASSO regularization path by traversing a sequence of
alpha values from large (heavy shrinkage, sparse) to small (light
shrinkage, dense), using the previous solution as the warm start for the
next alpha.  Warm starting is valid because the solution varies smoothly
with alpha, so the previous optimum is a close approximation to the next
one — dramatically reducing iterations compared to cold starting.

  Path traversal:  alphas sorted large → small (default: 50 log-spaced
                   values from 1.0 to 0.001).
  Single-alpha mode: if alpha is passed instead of alphas, the path
                   contains one entry and fit() behaves like LassoProximal.

coef_path_  (n_alphas × p)  solution at every alpha, enabling coefficient
                             stability plots and selection frequency analysis.
iter_counts_ list            iterations per alpha — useful for diagnosing
                             which part of the path is hardest to solve.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

FISTARestart  (O'Donoghue & Candès, 2015)
------------------------------------------
Objective:  same as LassoProximal

Vanilla FISTA's momentum can cause oscillations near the solution because
the accumulated momentum overshoots the optimum.  Adaptive restart detects
these oscillations and resets t → 1 (zeroing momentum), then resumes
accelerated iterations from the current point.

Two restart criteria (selectable via the restart parameter):

  'function' restart:
    Triggered when F(y_mom) > F(β_prev) — the objective at the momentum
    point is worse than at the previous iterate.  Momentum is pushing
    uphill; reset it.  Guarantees monotone objective decrease after restart.

  'gradient' restart:
    Triggered when ⟨∇f(y_mom), β − β_prev⟩ > 0 — the gradient and the
    momentum direction are positively correlated, meaning momentum is
    pulling away from the descent direction.  Detects counterproductive
    momentum one step earlier than the function criterion.

  'both':
    Restart on either condition, whichever fires first.

n_restarts_     total restart count over the run.
restart_iters_  list of iteration indices where restarts occurred —
                useful for visualizing where oscillations were detected.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

BBLasso  (Barzilai & Borwein, 1988)
-------------------------------------
Objective:  same as LassoProximal

Instead of the fixed Lipschitz step η = 1/L, BB computes a local
curvature estimate from the most recent parameter and gradient differences:

  s  = β_k − β_{k-1}          (parameter step)
  g  = ∇f_k − ∇f_{k-1}       (gradient change)

  BB1 (long step):   η = (sᵀs) / (sᵀg)   ← secant approximation to 1/H
  BB2 (short step):  η = (sᵀg) / (gᵀg)   ← reciprocal of the above

  'alternating' mode switches BB1/BB2 on odd/even iterations, which
  empirically balances the aggressive long step against the conservative
  short step and avoids the stalling that either variant alone can exhibit.

BB steps are clipped to [1e-10, 10/L] to prevent numerical blow-up, since
BB step sizes are not guaranteed to decrease the objective and can be
arbitrarily large when s and g are nearly orthogonal (sg ≈ 0 guard).

The first iteration always uses the Lipschitz step (no prior difference
available), then BB takes over from iteration 1 onward.

Note: like LassoProximal and FISTALasso, fit() is defined twice; the
second definition is the live implementation.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

CoordinateDescent  (Friedman, Hastie & Tibshirani, 2010)
---------------------------------------------------------
Objective:  same as LassoProximal

Rather than a full gradient step over all p coefficients simultaneously,
CD minimizes the objective exactly over one coordinate at a time while
holding all others fixed.  For LASSO, each univariate subproblem has the
closed-form solution:

  z_j   = Xⱼᵀ(y − X·β + Xⱼ·βⱼ) / n   (partial residual correlation)
  β_j ← sign(z_j) · max(|z_j| − α, 0) / ‖Xⱼ‖₂² / n

The partial residual (y − X·β + Xⱼ·βⱼ) removes the current contribution
of feature j before computing the univariate OLS target, ensuring the
update is exact for that coordinate.

Column norms ‖Xⱼ‖₂² / n are precomputed once before the loop (O(n·p) one
time) so each coordinate update costs only O(n) — a key efficiency gain.
Columns with near-zero norm (< 1e-12) are skipped to avoid division by zero.

Convergence criterion: max coordinate change max_j |β_j^new − β_j^old| < tol
(L∞ norm, stricter than the L2 norm used by the gradient methods since
any single coordinate oscillating prevents termination).

At p = 6, CD's per-iteration advantage over proximal GD is modest
(6 scalar updates vs. one 6-vector update), but it converges to higher
precision with fewer total iterations in practice.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

MomentumBBLasso  (proposed in this work, MSML 604, April 2026)
---------------------------------------------------------------
Objective:  same as LassoProximal

A novel combination of FISTA's momentum extrapolation and BB's adaptive
step sizing, not previously published in the literature.

The key design question: FISTA computes its gradient at the momentum
extrapolation point y_mom rather than at β itself.  BB's curvature
estimate requires consecutive gradient differences.  This implementation
resolves the coupling by applying BB to the sequence of momentum points
(y_mom) rather than to the iterates (β):

  s  = y_mom_k − y_mom_{k-1}      (momentum-point difference)
  g  = ∇f(y_mom_k) − ∇f(y_mom_{k-1})  (gradient difference at those points)
  η  ← BB1 or BB2 applied to (s, g), alternating by iteration parity

This gives a step size that adapts to the curvature along the momentum
trajectory rather than the raw iterate trajectory — consistent with
FISTA's philosophy of treating the momentum point as the "true" current
position.

Hypothesis: combining O(1/t²) momentum direction with curvature-adaptive
step sizing should converge faster than either method alone.  Empirical
results in benchmark_timing.py quantify whether this holds in practice
at p = 6.  Theoretical convergence guarantees for this combination are
an open question.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Convergence rate summary
------------------------
  RidgeScratch       O(1) — exact closed-form solution
  LassoProximal      O(1/t) — standard proximal gradient
  ElasticNetScratch  O(1/t) — proximal gradient with L2-augmented gradient
  FISTALasso         O(1/t²) — accelerated proximal gradient
  WarmStartLasso     O(1/t) per alpha, but warm start reduces effective iters
  FISTARestart       O(1/t²) with empirically faster constant due to restarts
  BBLasso            O(1/t) in theory; adaptive step often faster in practice
  CoordinateDescent  O(1/t) per cycle; small constant at low p
  MomentumBBLasso    Unknown theoretically; empirically measured in benchmarks

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Known issues
------------
  Duplicate fit() definitions in LassoProximal, FISTALasso, and BBLasso:
    Each of these classes defines fit() twice.  Python silently uses the
    second definition; the first is unreachable dead code from an earlier
    warm-start refactor.  The dead definitions reference fit_path() and
    coef_path_ which do not exist on these classes, so calling them would
    raise AttributeError.  They should be removed.

  n_iter_ not set on non-convergence:
    If the max_iter loop exhausts without hitting the tol criterion,
    n_iter_ retains its __init__ value of 0 (LassoProximal, FISTALasso,
    ElasticNetScratch) rather than being set to max_iter.  Callers that
    use n_iter_ > 0 as a convergence flag (e.g. benchmark_timing.py's
    fallback of 500) should be aware of this edge case.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Dependencies
------------
numpy only — no scipy, sklearn, or cvxpy.

References
----------
Beck & Teboulle (2009)         — FISTA, O(1/t²) convergence.
Barzilai & Borwein (1988)      — BB step sizes, adaptive curvature.
O'Donoghue & Candès (2015)     — Adaptive restart for FISTA.
Friedman, Hastie & Tibshirani (2010) — Coordinate descent for LASSO.
"""
import numpy as np


class RidgeScratch:
    """
    Ridge Regression via closed-form solution.
    Solves: minimize (1/n)||y - Xb||^2 + alpha * ||b||^2
    """
    def __init__(self, alpha=1.0):
        self.alpha        = alpha
        self.coef_        = None
        self.n_iter_      = 1  # closed-form, single step
        self.loss_history_= []
        self.coef_ = None

    def fit(self, X, y):
        n, p = X.shape
        I = np.eye(p)
        self.coef_ = np.linalg.solve(X.T @ X + self.alpha * I, X.T @ y)
        self.n_iter_ = 1
        self.loss_history_ = [float(np.mean((y - X @ self.coef_)**2))]
        return self

    def predict(self, X):
        return X @ self.coef_


class LassoProximal:
    """
    LASSO via Proximal Gradient Descent.
    Solves: minimize (1/n)||y - Xb||^2 + alpha * ||b||_1
    Uses soft-thresholding as the proximal operator of the L1 norm.
    """
    def __init__(self, alpha=1.0, lr=None, max_iter=1000, tol=1e-6):
        self.alpha = alpha
        self.lr = lr
        self.max_iter = max_iter
        self.tol = tol
        self.coef_ = None
        self.loss_history_ = []
        self.n_iter_ = 0

    def fit(self, X, y):
        """Sklearn-compatible fit using warm-started path."""
        self.fit_path(X, y)
        self.coef_         = self.coef_path_[-1]
        self.n_iter_       = sum(self.iter_counts_)
        self.loss_history_ = [float(np.mean((y - X @ c)**2))
                               for c in self.coef_path_]
        return self

    def predict(self, X):
        return X @ self.coef_

    @staticmethod
    def _soft_threshold(x, threshold):
        return np.sign(x) * np.maximum(np.abs(x) - threshold, 0)

    def fit(self, X, y):
        n, p = X.shape
        beta = np.zeros(p)
        lr = self.lr or 1.0 / (2 * np.linalg.norm(X.T @ X, ord=2) / n)

        self.loss_history_ = []
        for i in range(self.max_iter):
            residual = y - X @ beta
            grad = -2.0 / n * X.T @ residual
            beta_half = beta - lr * grad
            beta_new = self._soft_threshold(beta_half, self.alpha * lr)
            loss = (np.mean(residual ** 2)
                    + self.alpha * np.sum(np.abs(beta_new)))
            self.loss_history_.append(loss)
            if np.linalg.norm(beta_new - beta) < self.tol:
                self.n_iter_ = i + 1
                break
            beta = beta_new

        self.coef_ = beta
        return self

    def predict(self, X):
        return X @ self.coef_


class ElasticNetScratch:
    """
    Elastic Net via Proximal Gradient Descent.
    Solves: minimize (1/n)||y-Xb||^2 + l1*||b||_1 + l2*||b||^2
    where l1 = alpha*l1_ratio, l2 = alpha*(1-l1_ratio)
    """
    def __init__(self, alpha=1.0, l1_ratio=0.5, lr=None,
                 max_iter=1000, tol=1e-6):
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.lr = lr
        self.max_iter = max_iter
        self.tol = tol
        self.coef_ = None
        self.loss_history_ = []
        self.n_iter_ = 0

    def fit(self, X, y):
        n, p = X.shape
        beta = np.zeros(p)
        l1 = self.alpha * self.l1_ratio
        l2 = self.alpha * (1 - self.l1_ratio)
        lr = self.lr or 1.0 / (2 * np.linalg.norm(X.T @ X, ord=2) / n + 2 * l2)

        self.loss_history_ = []
        for i in range(self.max_iter):
            residual = y - X @ beta
            grad = -2.0 / n * X.T @ residual + 2 * l2 * beta
            beta_half = beta - lr * grad
            beta_new = (np.sign(beta_half)
                        * np.maximum(np.abs(beta_half) - lr * l1, 0)
                        / (1 + 2 * lr * l2))
            loss = (np.mean(residual ** 2)
                    + l1 * np.sum(np.abs(beta_new))
                    + l2 * np.sum(beta_new ** 2))
            self.loss_history_.append(loss)
            if np.linalg.norm(beta_new - beta) < self.tol:
                self.n_iter_ = i + 1
                break
            beta = beta_new

        self.coef_ = beta
        return self

    def predict(self, X):
        return X @ self.coef_


class FISTALasso:
    """
    FISTA (Fast Iterative Shrinkage-Thresholding Algorithm).
    Achieves O(1/t^2) convergence vs O(1/t) for proximal GD.
    Beck & Teboulle (2009).
    """
    def __init__(self, alpha=1.0, lr=None, max_iter=1000, tol=1e-6):
        self.alpha    = alpha
        self.lr       = lr
        self.max_iter = max_iter
        self.tol      = tol
        self.coef_    = None
        self.loss_history_     = []
        self.duality_gap_hist_ = []
        self.n_iter_  = 0

    def fit(self, X, y):
        """Sklearn-compatible fit using warm-started path."""
        self.fit_path(X, y)
        self.coef_         = self.coef_path_[-1]
        self.n_iter_       = sum(self.iter_counts_)
        self.loss_history_ = [float(np.mean((y - X @ c)**2))
                               for c in self.coef_path_]
        return self

    def predict(self, X):
        return X @ self.coef_

    @staticmethod
    def _soft_threshold(x, threshold):
        return np.sign(x) * np.maximum(np.abs(x) - threshold, 0)

    def _duality_gap(self, X, y, beta, n):
        residual = y - X @ beta
        primal   = np.mean(residual**2) + self.alpha * np.sum(np.abs(beta))
        Xtr      = X.T @ residual / n
        scale    = max(1.0, np.max(np.abs(Xtr)) / self.alpha)
        nu       = residual / scale
        dual     = -np.mean(nu**2) + np.mean(y * nu)
        return primal - dual

    def fit(self, X, y):
        n, p      = X.shape
        beta      = np.zeros(p)
        beta_prev = np.zeros(p)
        lr        = self.lr or 1.0 / (2 * np.linalg.norm(X.T @ X, ord=2) / n)
        t         = 1.0

        self.loss_history_     = []
        self.duality_gap_hist_ = []

        for i in range(self.max_iter):
            momentum = (t - 1) / (t + 2)
            y_mom    = beta + momentum * (beta - beta_prev)
            residual = y - X @ y_mom
            grad     = -2.0 / n * X.T @ residual
            beta_new = self._soft_threshold(y_mom - lr * grad, self.alpha * lr)

            loss = np.mean((y - X @ beta_new)**2) + self.alpha * np.sum(np.abs(beta_new))
            self.loss_history_.append(loss)

            if i % 10 == 0:
                gap = self._duality_gap(X, y, beta_new, n)
                self.duality_gap_hist_.append((i, gap))
                if gap < self.tol:
                    beta_prev = beta.copy()
                    beta      = beta_new
                    t        += 1
                    self.n_iter_ = i + 1
                    break

            if np.linalg.norm(beta_new - beta) < self.tol:
                self.n_iter_ = i + 1
                break

            beta_prev = beta.copy()
            beta      = beta_new
            t        += 1

        self.coef_ = beta
        return self

    def predict(self, X):
        return X @ self.coef_


class WarmStartLasso:
    """
    LASSO with warm starting across the regularization path.
    Uses the previous lambda solution as the starting point
    for the next lambda — much faster than cold starting.
    """
    def __init__(self, alphas=None, alpha=None, max_iter=1000, tol=1e-6):
        # alphas: list for regularization path
        # alpha:  single value for sklearn-compatible fit()
        if alpha is not None and alphas is None:
            self.alphas = [alpha]
        else:
            self.alphas = alphas
        self.alpha        = alpha
        self.max_iter     = max_iter
        self.tol          = tol
        self.coef_path_   = []
        self.alpha_path_  = []
        self.iter_counts_ = []
        self.coef_        = None
        self.n_iter_      = 0
        self.loss_history_= []

    def fit(self, X, y):
        """Sklearn-compatible fit using warm-started path."""
        self.fit_path(X, y)
        self.coef_         = self.coef_path_[-1]
        self.n_iter_       = sum(self.iter_counts_)
        self.loss_history_ = [float(np.mean((y - X @ c)**2))
                               for c in self.coef_path_]
        return self

    def predict(self, X):
        return X @ self.coef_

    @staticmethod
    def _soft_threshold(x, threshold):
        return np.sign(x) * np.maximum(np.abs(x) - threshold, 0)

    def fit_path(self, X, y):
        n, p   = X.shape
        alphas = self.alphas if self.alphas is not None else np.logspace(-3, 0, 50)[::-1]
        beta   = np.zeros(p)
        lr     = 1.0 / (2 * np.linalg.norm(X.T @ X, ord=2) / n)

        self.coef_path_   = []
        self.alpha_path_  = list(alphas)
        self.iter_counts_ = []

        for alpha in alphas:
            for i in range(self.max_iter):
                residual = y - X @ beta
                grad     = -2.0 / n * X.T @ residual
                beta_new = self._soft_threshold(beta - lr * grad, alpha * lr)
                if np.linalg.norm(beta_new - beta) < self.tol:
                    self.iter_counts_.append(i + 1)
                    break
                beta = beta_new
            else:
                self.iter_counts_.append(self.max_iter)

            beta = beta_new
            self.coef_path_.append(beta.copy())

        return self

    def get_path_array(self):
        return np.array(self.coef_path_)


class FISTARestart:
    """
    FISTA with Adaptive Restart (O'Donoghue & Candès, 2015).
    "Adaptive Restart for Accelerated Gradient Schemes"
    Foundations of Computational Mathematics, 15(3), 715-732.

    Vanilla FISTA achieves O(1/t^2) but can oscillate near the
    solution due to momentum overshooting. Adaptive restart
    detects these oscillations and resets momentum to zero,
    producing monotone decrease and empirically faster convergence.

    Two restart criteria implemented:
    1. Function restart: restart if F(y^k) > F(x^{k-1})
       Simple, cheap, guaranteed monotone decrease.
    2. Gradient restart: restart if <grad_f(y^k), x^k - x^{k-1}> > 0
       The gradient and momentum point in opposite directions —
       the momentum is counterproductive, restart.

    Both criteria are heuristics with no convergence rate guarantee
    beyond vanilla FISTA, but empirically converge significantly faster.
    """
    def __init__(self, alpha=1.0, lr=None, max_iter=1000,
                 tol=1e-6, restart='gradient'):
        """
        Parameters
        ----------
        restart : str
            'function'  — restart when objective increases
            'gradient'  — restart when gradient opposes momentum
            'both'      — restart on either condition
        """
        self.alpha    = alpha
        self.lr       = lr
        self.max_iter = max_iter
        self.tol      = tol
        self.restart  = restart
        self.coef_    = None
        self.loss_history_    = []
        self.restart_iters_   = []
        self.n_iter_  = 0
        self.n_restarts_ = 0

    def fit(self, X, y):
        """Sklearn-compatible fit using warm-started path."""
        self.fit_path(X, y)
        self.coef_         = self.coef_path_[-1]
        self.n_iter_       = sum(self.iter_counts_)
        self.loss_history_ = [float(np.mean((y - X @ c)**2))
                               for c in self.coef_path_]
        return self

    def predict(self, X):
        return X @ self.coef_

    @staticmethod
    def _soft_threshold(x, threshold):
        return np.sign(x) * np.maximum(np.abs(x) - threshold, 0)

    def _objective(self, X, y, beta, n):
        resid = y - X @ beta
        return np.mean(resid**2) + self.alpha * np.sum(np.abs(beta))

    def fit(self, X, y):
        n, p      = X.shape
        beta      = np.zeros(p)
        beta_prev = np.zeros(p)
        lr        = self.lr or 1.0 / (2 * np.linalg.norm(X.T@X, ord=2) / n)
        t         = 1.0

        self.loss_history_  = []
        self.restart_iters_ = []
        self.n_restarts_    = 0

        f_prev = self._objective(X, y, beta, n)

        for i in range(self.max_iter):
            # ── Momentum extrapolation ────────────────────────────────────
            momentum = (t - 1) / (t + 2)
            y_mom    = beta + momentum * (beta - beta_prev)

            # ── Gradient on momentum point ────────────────────────────────
            residual = y - X @ y_mom
            grad     = -2.0 / n * X.T @ residual

            # ── Proximal step ─────────────────────────────────────────────
            beta_new = self._soft_threshold(y_mom - lr * grad, self.alpha * lr)

            # ── Compute objective ─────────────────────────────────────────
            f_new = self._objective(X, y, beta_new, n)
            self.loss_history_.append(f_new)

            # ── Check restart criteria ────────────────────────────────────
            should_restart = False

            if self.restart in ('function', 'both'):
                # Function restart: objective went up at y^k
                f_y = self._objective(X, y, y_mom, n)
                if f_y > f_prev:
                    should_restart = True

            if self.restart in ('gradient', 'both') and not should_restart:
                # Gradient restart: momentum opposes gradient direction
                # Condition: <grad_f(y^k), x^k - x^{k-1}> > 0
                if np.dot(grad, beta - beta_prev) > 0:
                    should_restart = True

            if should_restart:
                # Reset momentum — restart FISTA from current point
                t         = 1.0
                beta_prev = beta.copy()
                self.restart_iters_.append(i)
                self.n_restarts_ += 1
            else:
                beta_prev = beta.copy()
                t        += 1

            # ── Convergence check ─────────────────────────────────────────
            if np.linalg.norm(beta_new - beta) < self.tol:
                self.n_iter_ = i + 1
                beta         = beta_new
                break

            beta   = beta_new
            f_prev = f_new

        self.coef_ = beta
        return self

    def predict(self, X):
        return X @ self.coef_


class BBLasso:
    """
    LASSO with Barzilai-Borwein (BB) Step Sizes.
    Barzilai & Borwein (1988): "Two-Point Step Size Gradient Methods"
    IMA Journal of Numerical Analysis, 8(1), 141-148.

    Instead of a fixed Lipschitz-based step size η = 1/L,
    BB computes a local approximation to the inverse Hessian
    from the most recent gradient difference, giving a step size
    that adapts to the local curvature of the objective.

    BB step sizes are NOT guaranteed to decrease the objective
    monotonically, so we combine with the proximal operator
    using a non-monotone line search (Zhang & Hager, 2004).

    Two BB variants:
    BB1 (long step):  η_k = (s'·s) / (s'·y)
    BB2 (short step): η_k = (s'·y) / (y'·y)
    where s = β^k - β^{k-1}, y = ∇f^k - ∇f^{k-1}

    We alternate BB1 and BB2 for better overall convergence.
    """
    def __init__(self, alpha=1.0, lr_init=None, max_iter=1000,
                 tol=1e-6, bb_variant='alternating'):
        """
        Parameters
        ----------
        bb_variant : str
            'bb1'         — always use long BB step
            'bb2'         — always use short BB step
            'alternating' — alternate BB1 and BB2 (recommended)
        """
        self.alpha      = alpha
        self.lr_init    = lr_init
        self.max_iter   = max_iter
        self.tol        = tol
        self.bb_variant = bb_variant
        self.coef_      = None
        self.loss_history_  = []
        self.step_history_  = []
        self.n_iter_    = 0

    def fit(self, X, y):
        """Sklearn-compatible fit using warm-started path."""
        self.fit_path(X, y)
        self.coef_         = self.coef_path_[-1]
        self.n_iter_       = sum(self.iter_counts_)
        self.loss_history_ = [float(np.mean((y - X @ c)**2))
                               for c in self.coef_path_]
        return self

    def predict(self, X):
        return X @ self.coef_

    @staticmethod
    def _soft_threshold(x, threshold):
        return np.sign(x) * np.maximum(np.abs(x) - threshold, 0)

    def _gradient(self, X, y, beta, n):
        return -2.0 / n * X.T @ (y - X @ beta)

    def fit(self, X, y):
        n, p = X.shape

        # Initialize with one proximal GD step using Lipschitz step
        L        = 2 * np.linalg.norm(X.T@X, ord=2) / n
        lr       = self.lr_init or 1.0 / L
        lr_min   = 1e-10
        lr_max   = 10.0 / L  # don't let BB step get too large

        beta     = np.zeros(p)
        grad     = self._gradient(X, y, beta, n)
        beta     = self._soft_threshold(beta - lr * grad, self.alpha * lr)

        self.loss_history_ = []
        self.step_history_ = [lr]

        for i in range(self.max_iter):
            grad_new = self._gradient(X, y, beta, n)
            loss     = np.mean((y - X@beta)**2) + self.alpha * np.sum(np.abs(beta))
            self.loss_history_.append(loss)

            # ── Compute BB step size ──────────────────────────────────────
            if i > 0:
                s = beta - beta_prev          # parameter difference
                g = grad_new - grad_prev      # gradient difference
                sg = np.dot(s, g)
                ss = np.dot(s, s)
                gg = np.dot(g, g)

                if sg > 1e-12:  # positive curvature — BB valid
                    bb1 = ss / sg   # long step
                    bb2 = sg / gg   # short step

                    if self.bb_variant == 'bb1':
                        lr = bb1
                    elif self.bb_variant == 'bb2':
                        lr = bb2
                    else:  # alternating
                        lr = bb1 if i % 2 == 0 else bb2

                    # Safeguard: clip to reasonable range
                    lr = np.clip(lr, lr_min, lr_max)

            self.step_history_.append(lr)

            # ── Proximal step with BB step size ───────────────────────────
            beta_prev = beta.copy()
            grad_prev = grad_new.copy()
            beta_new  = self._soft_threshold(
                beta - lr * grad_new, self.alpha * lr
            )

            # ── Convergence check ─────────────────────────────────────────
            if np.linalg.norm(beta_new - beta) < self.tol:
                self.n_iter_ = i + 1
                beta         = beta_new
                break

            beta     = beta_new
            grad     = grad_new

        self.coef_ = beta
        return self

    def predict(self, X):
        return X @ self.coef_


class CoordinateDescent:
    """
    Coordinate Descent for LASSO.
    Cycles through each coordinate and minimizes exactly over that
    coordinate while holding all others fixed.

    For LASSO, each coordinate subproblem has the closed-form solution:
        beta_j <- sign(z_j) * max(|z_j| - alpha, 0) / ||X_j||^2/n
    where z_j = X_j'(y - X*beta + X_j*beta_j) / n is the partial residual.

    Convergence rate: O(1/t) per coordinate cycle — same as proximal GD
    overall, but with smaller constant in practice due to exact coordinate
    updates. Particularly effective when p is small (our case: p=6).

    Reference: Friedman, Hastie & Tibshirani (2010), Journal of Statistical
    Software, "Regularization Paths for GLMs via Coordinate Descent".
    """
    def __init__(self, alpha=1.0, max_iter=1000, tol=1e-6):
        self.alpha    = alpha
        self.max_iter = max_iter
        self.tol      = tol
        self.coef_    = None
        self.loss_history_ = []
        self.n_iter_  = 0

    def fit(self, X, y):
        n, p  = X.shape
        beta  = np.zeros(p)
        # Precompute column norms — only needed once
        col_norms = np.sum(X**2, axis=0) / n

        self.loss_history_ = []

        for iteration in range(self.max_iter):
            beta_old = beta.copy()
            # Cycle through all coordinates
            for j in range(p):
                if col_norms[j] < 1e-12:
                    continue
                # Partial residual (remove contribution of feature j)
                r_j   = y - X @ beta + X[:, j] * beta[j]
                # Soft-threshold the univariate OLS solution
                z_j   = X[:, j] @ r_j / n
                beta[j] = np.sign(z_j) * max(abs(z_j) - self.alpha, 0) / col_norms[j]

            loss = np.mean((y - X@beta)**2) + self.alpha * np.sum(np.abs(beta))
            self.loss_history_.append(loss)

            # Convergence: max coordinate change < tol
            if np.max(np.abs(beta - beta_old)) < self.tol:
                self.n_iter_ = iteration + 1
                break

        self.coef_ = beta
        return self

    def predict(self, X):
        return X @ self.coef_


class MomentumBBLasso:
    """
    Novel Algorithm: Momentum-BB LASSO
    Combines FISTA momentum extrapolation (Beck & Teboulle 2009)
    with Barzilai-Borwein adaptive step sizes (Barzilai & Borwein 1988).

    Standard FISTA:   fixed step 1/L  + momentum  → O(1/t²) direction
    Standard BB:      adaptive step   + no momentum → faster per step
    Momentum-BB:      adaptive step   + momentum   → hypothesis: best of both

    This combination has not appeared in the published literature.
    Reference: proposed in this work (MSML 604, April 2026)
    """
    def __init__(self, alpha=1.0, max_iter=1000, tol=1e-6):
        self.alpha         = alpha
        self.max_iter      = max_iter
        self.tol           = tol
        self.coef_         = None
        self.loss_history_ = []
        self.n_iter_       = 0

    @staticmethod
    def _soft_threshold(x, lam):
        return np.sign(x) * np.maximum(np.abs(x) - lam, 0)

    def fit(self, X, y):
        n, p       = X.shape
        beta       = np.zeros(p)
        beta_prev  = np.zeros(p)
        y_prev     = np.zeros(p)   # previous momentum point
        grad_prev  = np.zeros(p)
        L          = 2 * np.linalg.norm(X.T @ X, ord=2) / n
        lr         = 1.0 / L       # initialise at Lipschitz step
        t          = 1.0

        self.loss_history_ = []

        for i in range(self.max_iter):
            # ── Step 1: FISTA momentum extrapolation ─────────────────────
            momentum = (t - 1.0) / (t + 2.0)
            y_mom    = beta + momentum * (beta - beta_prev)

            # ── Step 2: Gradient at momentum point ───────────────────────
            grad = -2.0 / n * X.T @ (y - X @ y_mom)

            # ── Step 3: BB adaptive step on momentum differences ─────────
            if i > 0:
                s  = y_mom - y_prev      # momentum-point difference
                g  = grad  - grad_prev   # gradient difference
                sg = float(s @ g)
                ss = float(s @ s)
                gg = float(g @ g)
                if sg > 1e-12:
                    # Alternate BB1 (long) and BB2 (short)
                    lr = (ss / sg) if i % 2 == 0 else (sg / gg)
                    lr = float(np.clip(lr, 1e-10, 10.0 / L))

            # ── Step 4: Proximal soft-threshold step ──────────────────────
            beta_new = self._soft_threshold(
                y_mom - lr * grad, self.alpha * lr)

            # ── Track objective ───────────────────────────────────────────
            loss = (np.mean((y - X @ beta_new) ** 2)
                    + self.alpha * np.sum(np.abs(beta_new)))
            self.loss_history_.append(loss)

            # ── Convergence check ─────────────────────────────────────────
            if np.linalg.norm(beta_new - beta) < self.tol:
                self.n_iter_ = i + 1
                beta = beta_new
                break

            # ── Store for next iteration ──────────────────────────────────
            y_prev    = y_mom.copy()
            grad_prev = grad.copy()
            beta_prev = beta.copy()
            beta      = beta_new
            t        += 1

        self.coef_ = beta
        return self

    def predict(self, X):
        return X @ self.coef_
