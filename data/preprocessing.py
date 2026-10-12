"""
data/preprocessing.py
======================
Shared signal-processing pipeline used for every raw trial, regardless of
dataset (SisFall / KFall / SlowFall), implementing:

  * two-stage outlier / burr cleaning (Section 5.1, SlowFall paragraph)
  * resampling to a common 25 Hz (Section 5.2.3)
  * SMV-based impact-time detection t_impact = argmax_t SMV(t)  (Eq. 1)
  * extraction of the delta-length Falling-stage input window
        X = { x^c_1 ... x^c_delta } ending `theta` samples before t_impact
    and the delta-length Impact-stage window used only to fit the offline
    Granger-causality statistics (Section 4.2).
"""
import numpy as np
from scipy.signal import medfilt, resample_poly
from scipy.ndimage import median_filter


def clean_outliers(signal: np.ndarray, clip_value: float, kernel: int) -> np.ndarray:
    """
    Two-stage cleaning:
      Stage 1: hard-clip values whose magnitude indicates a sensor glitch
               (paper reports outliers "exceeding 1e10"; we use a much
               tighter but still generous physical bound, see config.py).
      Stage 2: median filter to remove short "burr" spikes.
    signal: [T, C]
    """
    signal = np.nan_to_num(signal, nan=0.0, posinf=clip_value, neginf=-clip_value)
    signal = np.clip(signal, -clip_value, clip_value)
    if kernel > 1:
        # median filter along time axis independently per channel
        signal = median_filter(signal, size=(kernel, 1), mode="nearest")
    return signal


def resample_signal(signal: np.ndarray, orig_fs: float, target_fs: float,
                    method: str = "decimate") -> np.ndarray:
    """
    signal: [T, C] -> resampled [T', C] at target_fs.

    method='decimate': take every (orig/target)-th sample — matches MCTN
    (same lab as CausalFall): `data[:, :, ::step]` in utils.GetPthData.
    method='poly': anti-aliased resample_poly (scipy).
    """
    if abs(orig_fs - target_fs) < 1e-6:
        return signal
    if method == "decimate":
        step = int(round(orig_fs / target_fs))
        if step < 1:
            step = 1
        return signal[::step]
    from math import gcd
    g = gcd(int(round(orig_fs)), int(round(target_fs)))
    up = int(round(target_fs)) // g
    down = int(round(orig_fs)) // g
    return resample_poly(signal, up, down, axis=0)


def compute_smv(accel_xyz: np.ndarray) -> np.ndarray:
    """
    Eq. (1): SMV_t = sqrt(Ax^2 + Ay^2 + Az^2)
    accel_xyz: [T, 3] -> returns [T]
    """
    return np.sqrt(np.sum(accel_xyz.astype(np.float64) ** 2, axis=1))


def smooth_signal_1d(x: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return x
    kernel = np.ones(window) / window
    pad = window // 2
    x_padded = np.pad(x, (pad, pad), mode="edge")
    return np.convolve(x_padded, kernel, mode="valid")[: len(x)]


def find_t_impact(accel_xyz: np.ndarray, smooth_window: int = 1) -> int:
    """Phase 1: t_impact = argmax_t SMV(t), optionally smoothed."""
    smv = compute_smv(accel_xyz)
    smv = smooth_signal_1d(smv, smooth_window)
    return int(np.argmax(smv))


def extract_falling_window(signal: np.ndarray, t_impact: int, theta: int, delta: int,
                           allow_pad: bool = False):
    """
    Phase 2: falling = [t_impact - theta - delta, t_impact - theta)

    MCTN (same lab) DROPS trials where the window does not fully fit
    (no zero-padding). We follow that by default (`allow_pad=False` → return None).
    """
    end = t_impact - theta
    start = end - delta
    T, C = signal.shape
    if start < 0 or end > T or end <= 0:
        if not allow_pad:
            return None
        end = max(min(end, T), 0)
        start = end - delta
        if start < 0:
            pad = np.zeros((-start, C), dtype=signal.dtype)
            window = np.concatenate([pad, signal[0:end]], axis=0)
        else:
            window = signal[start:end]
        if window.shape[0] < delta:
            pad = np.zeros((delta - window.shape[0], C), dtype=signal.dtype)
            window = np.concatenate([pad, window], axis=0)
        return window[-delta:]
    return signal[start:end]


def extract_impact_window(signal: np.ndarray, t_impact: int, delta: int):
    """
    Impact-stage window (post t_impact), of the SAME length delta, used only
    to fit the offline Granger-causality matrix (Section 4.2): it represents
    L^c_impact, i.e. the segment "for the body from pose changes to impact
    the ground" on the post side of t_impact.
    """
    T, C = signal.shape
    start = t_impact
    end = start + delta
    if end > T:
        window = signal[start:T]
        pad = np.zeros((delta - window.shape[0], C), dtype=signal.dtype)
        window = np.concatenate([window, pad], axis=0)
    else:
        window = signal[start:end]
    return window[:delta]


def build_sample(signal: np.ndarray, orig_fs: float, cfg) -> dict:
    """
    Full per-trial pipeline: clean -> resample -> locate impact -> extract
    the two windows needed downstream (input window X, and the impact-stage
    reference window used for Granger fitting).

    signal: raw [T, C] array where the FIRST 3 columns are assumed to be the
            tri-axial accelerometer (used for SMV / impact detection), as is
            standard for every IMU dataset used in this paper.
    """
    signal = clean_outliers(signal, cfg.outlier_clip_value, cfg.median_filter_kernel)
    method = getattr(cfg, "resample_method", "decimate")
    signal = resample_signal(signal, orig_fs, cfg.fs, method=method)
    smooth = int(getattr(cfg, "smv_smooth_window", 1))
    # SisFall has two accelerometers (cols 0-2 and 6-8). Take the stronger
    # SMV peak so a noisy ADXL spike cannot steal t_impact from MMA8451Q.
    if signal.shape[1] >= 9 and getattr(cfg, "dual_accel_smv", False):
        smv1 = smooth_signal_1d(compute_smv(signal[:, 0:3]), smooth)
        smv2 = smooth_signal_1d(compute_smv(signal[:, 6:9]), smooth)
        t_impact = int(np.argmax(np.maximum(smv1, smv2)))
    else:
        t_impact = find_t_impact(signal[:, 0:3], smooth_window=smooth)
    allow_pad = bool(getattr(cfg, "allow_window_pad", False))
    x_window = extract_falling_window(
        signal, t_impact, cfg.theta, cfg.delta, allow_pad=allow_pad
    )
    if x_window is None:
        return None
    y_window = extract_impact_window(signal, t_impact, cfg.delta)
    return {"X": x_window.astype(np.float32).T,       # [C, delta]
            "Y_impact": y_window.astype(np.float32).T,  # [C, delta]
            "t_impact": t_impact}
