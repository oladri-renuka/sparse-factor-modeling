"""
online_solver.py
===============
Incremental proximal gradient descent for LASSO factor models.  Rather
than retraining on a full rolling window each month, coefficients are
updated using only the single new observation that just arrived —
reducing per-step computational cost from O(n·p) to O(p) and mirroring
how production quantitative trading systems actually operate.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Motivation: batch vs. online
-----------------------------
The batch walk-forward engine in backtesting.py discards all prior
computation at each step and refits from scratch on the trailing
train_window observations.  For a 120-month window and p=6 factors this
is fast, but the pattern does not scale:

  Batch:   O(n · p) per time step — cost grows with window length.
  Online:  O(p) per time step     — cost is constant regardless of
                                    how much history has accumulated.

More importantly, the online update preserves the coefficient state from
the previous step as its starting point.  This is conceptually closer to
how a live system works: the model is never "retrained", it is
continuously adapted as the market evolves.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

OnlineLasso
-----------
Solves the cumulative LASSO objective incrementally:

    minimize  Σₜ (yₜ − xₜ'β)²  +  α ‖β‖₁

At each new observation (xₜ, yₜ) one proximal gradient step is taken
rather than solving the full problem from scratch.

  State:
    coef_         np.ndarray (p,)  current coefficient vector; persists
                                   across calls to update().
    t_            int              observation counter, used by the
                                   learning rate schedule.
    loss_history_ list             per-update LASSO objective value,
                                   useful for diagnosing convergence.

  Learning rate schedule — inverse decay:
    lr_t = lr_init / (1 + decay × t)

    The rate shrinks monotonically, guaranteeing that early updates
    (during warm-up, when estimates are noisy) take larger steps while
    later updates (once the model has seen ≥120 observations) make finer
    corrections.  The default lr_init=0.01 and decay=0.01 give roughly
    half the initial step size after 100 observations.

  Inner loop (max_iter per observation):
    Multiple proximal gradient mini-iterations are applied per new
    data point to extract more signal from each observation before
    moving on.  Each iteration:

      1. Gradient on the single new residual:
            grad = −2 (yₜ − xₜ'β) · xₜ
         This is the stochastic gradient — an unbiased estimate of the
         full-batch gradient computed from one sample only.

      2. Gradient step:
            β½ = β − lr · grad

      3. Proximal (soft-threshold) step for L1 penalty:
            β ← sign(β½) · max(|β½| − α·lr, 0)
         Coefficients whose gradient-updated magnitude falls below the
         threshold α·lr are zeroed, maintaining sparsity.

    Default max_iter=10 balances per-update accuracy against the O(p)
    cost target.  Setting max_iter=1 recovers pure stochastic proximal
    gradient descent.

  _soft_threshold(x, threshold):
    Static helper implementing the proximal operator of the L1 norm.
    Identical in form to the operator used in LassoProximal and
    FISTALasso, enabling direct comparison of the shrinkage mechanics.

  initialize(p):
    Resets coefficients to zero and clears the time counter.  Called
    automatically on the first update() if the model has not been
    explicitly initialized — useful for warm-starting with a prior
    coefficient vector by setting coef_ directly before the first call.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

walk_forward_online(X, Y, alpha, train_window=120, step=1)
----------------------------------------------------------
Online analogue of walk_forward_backtest() with an identical return
signature, enabling direct metric comparison between batch and online
strategies via compute_metrics() and compute_metrics_with_costs().

  Preprocessing:
    Full-sample standardization of X (same pragmatic simplification as
    backtesting.py) — mean and std are computed once before the loop.

  Warm-up phase  (t = 0 … train_window − 1):
    The first train_window observations are fed to each model sequentially
    via update() to build an initial coefficient estimate before
    predictions are recorded.  No predictions or actuals are stored during
    warm-up — the output arrays begin at t = train_window, matching the
    batch engine's first prediction index exactly.

  One model per portfolio:
    Twenty-five independent OnlineLasso instances are maintained in a
    list — one for each target column in Y.  Each model evolves its own
    coefficient vector independently, so factor loadings can differ
    across portfolios as in the batch case.

  Prediction-then-update ordering (critical for no look-ahead):
    At each live step t:
      1. Predict:  preds[j] = models[j].predict(X_scaled[t])
      2. Record:   store prediction, actual, date, coefficients
      3. Update:   models[j].update(X_scaled[t], Y_vals[t, j])

    The update step uses the observation at time t, which is the same
    observation used for prediction.  This is consistent with the batch
    engine (which also trains up to and including t−1 and predicts t) —
    the current-period return is observed after the prediction is locked
    in and is then used to update the model for the next period.

  Returns:
    predictions     (N × 25)       out-of-sample predictions
    actuals         (N × 25)       realised returns
    dates           (N,)           prediction dates
    coefs_over_time (N × 25 × 6)  coefficient snapshots at each step,
                                   enabling coefficient drift analysis
                                   analogous to the batch engine output

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Convergence and limitations
----------------------------
  Stochastic gradient noise:
    With n=1 sample per gradient step, the gradient estimate is high-
    variance.  The inner max_iter loop partially mitigates this by taking
    multiple passes on the same observation, but the online model will
    generally not converge to the same solution as the batch LASSO on the
    same window — it tracks a moving target rather than solving a static
    problem.

  Learning rate sensitivity:
    The inverse-decay schedule does not adapt to the curvature of the
    loss surface.  If factor return volatility changes regime, the fixed
    lr_init / decay pair may be too aggressive (overshooting) or too
    conservative (slow adaptation).  An adaptive schedule such as
    AdaGrad or RMSProp would be more robust but adds state per coordinate.

  Sparsity stability:
    Because the threshold α·lr decreases over time (as lr decays), the
    effective sparsity level changes — coefficients that were zeroed early
    may become non-zero later as the threshold shrinks.  If a fixed
    sparsity pattern is desired, the threshold should be held constant
    at α·lr_init.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Dependencies
------------
numpy, time (imported but unused — safe to remove)

Inputs expected
---------------
X : pd.DataFrame (T × 6)   factor returns, DateTime-indexed
Y : pd.DataFrame (T × 25)  portfolio returns, DateTime-indexed
"""
import numpy as np
import time


class OnlineLasso:
    """
    Online LASSO via incremental proximal gradient updates.
    Processes one observation at a time using a decaying
    learning rate schedule.

    Solves: minimize sum_t (y_t - x_t'b)^2 + alpha * ||b||_1
    """
    def __init__(self, alpha=0.003, lr_init=0.01,
                 decay=0.01, max_iter=10):
        self.alpha    = alpha
        self.lr_init  = lr_init
        self.decay    = decay
        self.max_iter = max_iter
        self.coef_    = None
        self.t_       = 0
        self.loss_history_ = []

    @staticmethod
    def _soft_threshold(x, threshold):
        return np.sign(x) * np.maximum(np.abs(x) - threshold, 0)

    def initialize(self, p):
        """Initialize coefficients to zero."""
        self.coef_ = np.zeros(p)
        self.t_    = 0
        return self

    def update(self, x, y):
        """
        Process one new observation (x, y).
        Updates coefficients using proximal gradient step.
        """
        if self.coef_ is None:
            self.initialize(len(x))

        self.t_ += 1
        # Decaying learning rate
        lr       = self.lr_init / (1 + self.decay * self.t_)

        for _ in range(self.max_iter):
            # Gradient on single observation
            residual  = y - np.dot(x, self.coef_)
            grad      = -2 * residual * x
            beta_half = self.coef_ - lr * grad
            # Proximal step
            self.coef_ = self._soft_threshold(
                beta_half, self.alpha * lr
            )

        loss = (y - np.dot(x, self.coef_))**2 + \
               self.alpha * np.sum(np.abs(self.coef_))
        self.loss_history_.append(loss)
        return self

    def predict(self, x):
        return np.dot(x, self.coef_)


def walk_forward_online(X, Y, alpha=0.003,
                         train_window=120, step=1):
    """
    Online walk-forward backtest.
    Warms up on first train_window observations,
    then updates incrementally.

    Returns same format as walk_forward_backtest
    for direct comparison.
    """
    T        = len(X)
    X_vals   = X.values
    X_mean   = X_vals.mean(axis=0)
    X_std    = X_vals.std(axis=0)
    X_scaled = (X_vals - X_mean) / X_std
    Y_vals   = Y.values

    predictions, actuals, dates = [], [], []
    coefs_over_time              = []

    # One online model per portfolio
    models = [OnlineLasso(alpha=alpha)
              for _ in range(Y.shape[1])]

    # Warm up on first train_window observations
    for t in range(train_window):
        for j in range(Y.shape[1]):
            models[j].update(X_scaled[t], Y_vals[t, j])

    # Online updates
    for t in range(train_window, T, step):
        X_test = X_scaled[t:t + 1]
        Y_test = Y_vals[t:t + 1]

        preds, coefs = [], []
        for j in range(Y.shape[1]):
            pred = models[j].predict(X_scaled[t])
            preds.append(pred)
            coefs.append(models[j].coef_.copy())
            # Update with new observation
            models[j].update(X_scaled[t], Y_vals[t, j])

        predictions.append(preds)
        actuals.append(Y_test[0])
        dates.append(Y.index[t])
        coefs_over_time.append(coefs)

    return (np.array(predictions), np.array(actuals),
            dates, np.array(coefs_over_time))
