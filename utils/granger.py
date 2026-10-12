"""
utils/granger.py
=================
Implements the Granger-Causality-Test-based Causal Effect Encoder (CEE)
statistics described in Section 4.2 (Eqs. 5-11).

For every ordered channel pair (i, j) we:
  1. Fit a *restricted* lag-p autoregression of channel j on its own past
     (Eq. 7/8)                                  -> residual variance sigma_Y^2
  2. Fit an *unrestricted* lag-p VAR of channel j on its own past AND the
     past of channel i (Eq. 5/6)                -> residual variance sigma_YX^2
  3. M_GC(i,j) = ln( sigma_Y^2 / sigma_YX^2 )                    (Eq. 9)
  4. E(i,j) = M_GC(i,j) if M_GC(i,j) > 1 else 0                  (Eq. 10)

X^i is built from the Falling-stage segment of channel i,
Y^j is built from the Impact-stage segment of channel j (per Eq. 5/6 wording:
"X^c_t -> Y^c_t" i.e. Falling stage -> Impact stage). We pool the segments
of ALL training samples of a dataset for a single, dataset-level VAR fit
(this is the "precomputed offline, not on the embedded device" strategy
described in Section 5.4.1).

This produces the (C x C) heat-map matrix E used to visualize Fig. 8 and
the normalized per-channel weight vector w used inside the CEE module
(models/cee.py), following Eq. (11):

    CEE(X^i) = [ sum_j E(i,j) ] / [ sum_j sum_k E(j,k) ]  *  W_Scale * X^i

--------------------------------------------------------------------------
[FIX] Cross-window contamination in the pooled AR fit
--------------------------------------------------------------------------
An earlier version of this file pooled all N per-sample windows into ONE
long per-channel series via `reshape(C, N*T)` and then slid a single lag-p
AR window across that whole concatenated series. Because the AR loop looks
`lag` steps back with no awareness of where one sample's window ends and
the next begins, at every one of the N-1 window boundaries the "history"
used to predict y_t was actually the tail of a COMPLETELY DIFFERENT trial
-- a spurious discontinuity injected into the regression at ~N points.
This is not something the paper specifies either way (Eqs. 5-11 describe
the per-channel VAR abstractly), so pooling across samples for a single
dataset-level fit is a legitimate implementation choice (Section 5.4.1
explicitly says CEE is precomputed "across different fall event samples"
offline) -- but the pooled rows must never mix two different samples'
history. `_build_ar_rows` below fixes this: it builds lag-features
independently within each of the N windows (fully vectorized, no Python
loop over samples) and only pools the resulting ROWS afterwards, so the
regression never crosses a trial boundary.
"""
import numpy as np


def _build_ar_rows(y_windows: np.ndarray, x_windows: np.ndarray = None, lag: int = 1):
    """
    Build lag-p AR design rows WITHOUT crossing sample boundaries.

    y_windows: [N, T] -- N independent per-sample windows of the target
        channel (each row is one sample's own, self-contained time axis).
    x_windows: optional [N, T] -- the "other" channel's windows, same N, T,
        for the unrestricted (Eq. 5/6) model.
    lag: number of past steps used as features.

    Returns (rows [N*(T-lag), lag or 2*lag], targets [N*(T-lag)]).
    Vectorized: for each lag offset l in 1..lag we take a shifted [N, T-lag]
    slice of the window (this can only ever read within the same row/sample,
    since numpy slicing never reaches across the N axis), then stack.
    """
    N, T = y_windows.shape
    if T <= lag:
        n_feat = lag if x_windows is None else 2 * lag
        return np.zeros((0, n_feat), dtype=np.float64), np.zeros((0,), dtype=np.float64)

    y_feats = np.stack(
        [y_windows[:, lag - l: T - l] for l in range(1, lag + 1)], axis=-1
    )  # [N, T-lag, lag] -- y_feats[n, t, l-1] = y_windows[n, t + lag - l] = y_{t'-l} for t' = t+lag
    feats = y_feats
    if x_windows is not None:
        x_feats = np.stack(
            [x_windows[:, lag - l: T - l] for l in range(1, lag + 1)], axis=-1
        )  # [N, T-lag, lag]
        feats = np.concatenate([y_feats, x_feats], axis=-1)  # [N, T-lag, 2*lag]

    targets = y_windows[:, lag:T]  # [N, T-lag]
    rows = feats.reshape(-1, feats.shape[-1])
    y_flat = targets.reshape(-1)
    return rows, y_flat


def _ridge_residual_var(X_feats: np.ndarray, Y: np.ndarray, ridge: float = 1e-3) -> float:
    """
    Fit Y = X_feats @ beta (+ bias) via RIDGE-regularized least squares
    (Tikhonov) on already-built, boundary-safe rows, return Var(residual).

    Plain OLS (np.linalg.lstsq) is numerically unstable here: real (and even
    synthetic) IMU channels are frequently near-collinear or near-constant
    over short windows, which makes the raw normal-equations matrix close to
    singular and produces huge/overflowing coefficients (seen in practice as
    "divide by zero"/"overflow" RuntimeWarnings and a channel-weight matrix
    that silently collapses to uniform). A small ridge penalty keeps the
    solve well-conditioned without materially changing the VAR fit, which is
    standard practice for causality estimation on correlated sensor data.
    """
    if X_feats.shape[0] <= X_feats.shape[1] + 2:
        return float(np.var(Y) + 1e-8) if Y.size else 1e-8

    X_design = np.concatenate([X_feats, np.ones((X_feats.shape[0], 1))], axis=1)  # add bias
    # Sanitize any NaN/Inf that slipped through upstream cleaning (real IMU
    # data can contain sensor glitches even after outlier clipping/resampling
    # artifacts) BEFORE computing statistics on it, since a single NaN/Inf
    # here poisons every downstream matmul.
    X_design = np.nan_to_num(X_design, nan=0.0, posinf=0.0, neginf=0.0)
    Y = np.nan_to_num(Y, nan=0.0, posinf=0.0, neginf=0.0)

    # standardize columns before solving so the ridge penalty is scale-fair
    # across channels with very different physical units/ranges (accel vs.
    # gyro vs. magnetometer), then solve in standardized space.
    col_std = X_design.std(axis=0)
    # NOTE: `col_std < 1e-8` is FALSE for NaN (IEEE comparison semantics),
    # so a NaN std would silently slip through this guard undetected -- use
    # an explicit "not finite OR too small" check instead.
    bad = ~np.isfinite(col_std) | (col_std < 1e-8)
    col_std[bad] = 1.0
    Xs = X_design / col_std
    Xs = np.nan_to_num(Xs, nan=0.0, posinf=0.0, neginf=0.0)

    n_features = Xs.shape[1]
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        XtX = Xs.T @ Xs
        XtX_reg = XtX + ridge * np.eye(n_features)
        XtY = Xs.T @ Y
    XtX_reg = np.nan_to_num(XtX_reg, nan=0.0, posinf=1e12, neginf=-1e12)
    XtY = np.nan_to_num(XtY, nan=0.0, posinf=1e12, neginf=-1e12)
    try:
        coef_std = np.linalg.solve(XtX_reg, XtY)
    except np.linalg.LinAlgError:
        return float(np.var(Y) + 1e-8)

    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        resid = Y - (Xs @ coef_std)
    resid = resid[np.isfinite(resid)]
    if resid.size == 0:
        return float(np.var(Y) + 1e-8)
    var = np.var(resid)
    if not np.isfinite(var):
        return float(np.var(Y) + 1e-8)
    return float(max(var, 1e-8))


def _fit_ar_residual_var_windows(y_windows: np.ndarray, x_windows: np.ndarray = None,
                                  lag: int = 1, ridge: float = 1e-3) -> float:
    """Boundary-safe replacement for the old flat-series `_fit_ar_residual_var`."""
    rows, y_flat = _build_ar_rows(y_windows, x_windows, lag=lag)
    if rows.shape[0] == 0:
        return float(np.var(y_windows) + 1e-8)
    return _ridge_residual_var(rows, y_flat, ridge=ridge)


def _select_edges(M_raw: np.ndarray, edge_selection: str = "absolute",
                   abs_threshold: float = 1.0, percentile: float = 75.0,
                   topk: int = None) -> np.ndarray:
    """
    Turn the raw (unthresholded) M_GC matrix into the sparse effect matrix E
    used by Eq. (11) / the CEE, using one of three edge-selection rules.

    [FIX] Root-cause of "0/81 edges every run regardless of lag/ridge":
    smooth, physically continuous IMU signals are already highly
    self-predictable from their own recent history (momentum/inertia), so
    the *restricted* (own-history-only) residual variance is tiny before a
    second channel is even considered -- there's very little residual left
    for cross-channel history to explain, so the raw log-ratio M_GC stays
    small basically by construction, independent of lag order or ridge
    strength (that's exactly the flat 0.000-0.003 pattern seen across the
    full stage1 sweep). Eq. (10)'s fixed "M_GC > 1" bar assumes this raw
    log-ratio is expected to comfortably clear 1 nat (i.e. the unrestricted
    model needs > 63% less residual variance than the restricted one) for a
    real effect -- a bar tuned for noisier/more-independent series, not
    smooth continuous sensor waveforms.
    -
    "percentile"/"topk" sidestep this by thresholding RELATIVE to this
    dataset's own M_GC distribution instead of an absolute constant, so a
    causal *ranking* can still be recovered even when every raw value is
    small in absolute terms. "absolute" keeps the original Eq. (10)-literal
    behaviour for comparison/ablation.
    """
    C = M_raw.shape[0]
    E = np.zeros((C, C), dtype=np.float64)

    if edge_selection == "absolute":
        E = np.where(M_raw > abs_threshold, M_raw, 0.0)

    elif edge_selection == "percentile":
        positive = M_raw[M_raw > 0]
        if positive.size == 0:
            return E  # genuinely no channel has ANY positive causal signal
        thresh = np.percentile(positive, percentile)
        E = np.where(M_raw >= thresh, M_raw, 0.0)

    elif edge_selection == "topk":
        k = topk if topk is not None else C  # default: keep ~C strongest pairs
        k = min(k, C * C)
        flat = M_raw.ravel()
        order = np.argsort(flat)[::-1][:k]
        for idx in order:
            if flat[idx] > 0:
                i, j = np.unravel_index(idx, M_raw.shape)
                E[i, j] = flat[idx]
    else:
        raise ValueError(f"Unknown edge_selection={edge_selection!r}; "
                          "expected 'absolute', 'percentile', or 'topk'.")
    return E


def granger_causality_matrix(falling_segments: np.ndarray,
                              impact_segments: np.ndarray,
                              lag: int = 1,
                              ridge: float = 1e-3,
                              diff: bool = False,
                              edge_selection: str = "absolute",
                              abs_threshold: float = 1.0,
                              percentile: float = 75.0,
                              topk: int = None,
                              return_raw: bool = False):
    """
    falling_segments: [N, C, delta] Falling-stage windows (X)
    impact_segments:  [N, C, delta] Impact-stage windows  (Y)

    diff: if True, first-difference each window along time before fitting
    the VAR (i.e. fit on "jerk" instead of raw accel/gyro). Raw physical
    sensor traces are dominated by autocorrelation/momentum, which inflates
    own-history predictability and suppresses the *relative* contribution
    any other channel can add; differencing removes most of that trivial
    predictability so genuine cross-channel structure is a bigger share of
    what's left to explain. Off by default so `absolute` mode reproduces
    the original, paper-literal behaviour exactly when everything else is
    left at its defaults too.

    edge_selection: 'absolute' (original Eq. 10 behaviour: M_GC > abs_threshold),
    'percentile' (keep pairs at/above the given percentile of this dataset's
    own positive M_GC values), or 'topk' (keep the k strongest pairs
    dataset-wide). See `_select_edges` for the reasoning.

    Returns E: [C, C] matrix, E[i, j] = causal effect strength of
    falling-channel i -> impact-channel j (0 where not selected as an edge).
    If return_raw=True, also returns the raw (unthresholded) M_GC matrix --
    useful for diagnosing whether values cluster just under the threshold
    (borderline / threshold too strict) or far below it (regularization too
    aggressive, or genuinely little differential causal structure at this
    lag order).
    """
    N, C, T = falling_segments.shape
    assert impact_segments.shape == falling_segments.shape

    if diff:
        # np.diff shortens the time axis by 1; both stages are shortened
        # identically so the falling/impact pairing stays aligned.
        falling_segments = np.diff(falling_segments, axis=-1)
        impact_segments = np.diff(impact_segments, axis=-1)
        T = T - 1

    # Keep the per-sample window structure [C, N, T] -- do NOT flatten to
    # [C, N*T] (see module docstring: that used to let the AR lag-window
    # cross from one trial into another at every window boundary).
    X_all = falling_segments.transpose(1, 0, 2)  # [C, N, T]
    Y_all = impact_segments.transpose(1, 0, 2)   # [C, N, T]
    # Defense-in-depth: sanitize here too, in case NaN/Inf reached this point
    # despite upstream outlier clipping (e.g. resampling edge artifacts).
    X_all = np.nan_to_num(X_all, nan=0.0, posinf=0.0, neginf=0.0)
    Y_all = np.nan_to_num(Y_all, nan=0.0, posinf=0.0, neginf=0.0)

    M_raw = np.zeros((C, C), dtype=np.float64)
    for j in range(C):  # target / outcome channel (impact stage)
        y_windows = Y_all[j]  # [N, T]
        var_restricted = _fit_ar_residual_var_windows(y_windows, x_windows=None,
                                                        lag=lag, ridge=ridge)
        for i in range(C):  # source / cause channel (falling stage)
            x_windows = X_all[i]  # [N, T]
            var_unrestricted = _fit_ar_residual_var_windows(y_windows, x_windows=x_windows,
                                                              lag=lag, ridge=ridge)
            m_gc = np.log((var_restricted + 1e-12) / (var_unrestricted + 1e-12))
            if not np.isfinite(m_gc):
                m_gc = 0.0
            M_raw[i, j] = m_gc

    E = _select_edges(M_raw, edge_selection=edge_selection, abs_threshold=abs_threshold,
                       percentile=percentile, topk=topk)
    if return_raw:
        return E, M_raw
    return E


def channel_weights_from_E(E: np.ndarray) -> np.ndarray:
    """
    Eq. (11) numerator/denominator: per-(falling)-channel normalized causal
    strength, used as the scalar weight applied to that channel's raw signal.
    Returns w: [C] with sum(w) == 1 (if E is not all-zero).
    """
    row_sums = E.sum(axis=1)            # sum_j E(i, j) for each falling channel i
    total = E.sum()
    if total <= 0:
        C = E.shape[0]
        return np.ones(C, dtype=np.float64) / C
    return row_sums / total
