"""
data/synthetic.py
===================
NOT part of the paper. A small synthetic-signal generator so the entire
training/evaluation pipeline can be smoke-tested end-to-end before you have
downloaded SisFall / KFall. Falls are simulated as a slow drift followed by
a sharp transient (mimicking the Falling -> Impact stages); ADLs are
stationary noise. Use --dataset synthetic to run a fast sanity check.
"""
import numpy as np


def make_synthetic_samples(n_samples: int, cfg, n_types: int = 6, seed: int = 0):
    rng = np.random.default_rng(seed)
    C, T = cfg.num_channels, cfg.delta
    samples = []
    for i in range(n_samples):
        is_fall = int(rng.random() < 0.4)
        event_type = f"{'F' if is_fall else 'D'}{rng.integers(1, n_types + 1):02d}"
        t = np.linspace(0, 1, T)
        base = rng.normal(0, 0.3, size=(C, T))
        if is_fall:
            onset = rng.integers(T // 3, T // 2)
            drift = np.zeros(T)
            drift[onset:] = np.linspace(0, 1, T - onset) ** 2 * rng.uniform(3, 6)
            transient = np.exp(-((np.arange(T) - (T - 5)) ** 2) / 8.0) * rng.uniform(4, 8)
            signal = base + drift[None, :] * rng.normal(1, 0.2, (C, 1)) + transient[None, :]
        else:
            signal = base + 0.1 * np.sin(2 * np.pi * 3 * t)[None, :]
        samples.append({
            "X": signal.astype(np.float32),
            "Y_impact": (signal + rng.normal(0, 0.1, size=(C, T))).astype(np.float32),
            "label": is_fall,
            "event_type": event_type,
            "subject": f"SYN{i % 10:02d}",
        })
    return samples
