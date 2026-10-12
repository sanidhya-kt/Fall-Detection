"""
data/dataset.py
=================
Ties together the dataset-specific loaders (SisFall / KFall / synthetic),
performs the random train/val/test split described in Section 5.3
("Table 4 shows results... in a manner that randomly divides datasets"),
z-score normalizes channels using train-set statistics, and precomputes the
dataset-level Granger-causality matrix E (Section 4.2) needed by the
Causal Effect Encoder.
"""
import numpy as np
import torch
from torch.utils.data import Dataset

from utils.granger import granger_causality_matrix, channel_weights_from_E


class FallDataset(Dataset):
    def __init__(self, samples, mean=None, std=None):
        self.samples = samples
        X = np.stack([s["X"] for s in samples])  # [N, C, T]
        if mean is None:
            mean = X.mean(axis=(0, 2), keepdims=True)
            std = X.std(axis=(0, 2), keepdims=True) + 1e-6
        self.mean, self.std = mean, std

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        x = (s["X"] - self.mean[0]) / self.std[0]
        y = s["label"]
        return torch.from_numpy(x.astype(np.float32)), torch.tensor(y, dtype=torch.long)


def split_samples(samples, cfg):
    """Label-stratified random split (paper: randomly divides; we keep class rates)."""
    rng = np.random.default_rng(cfg.seed)
    labels = np.array([s["label"] for s in samples])
    idx = np.arange(len(samples))
    train_idx, val_idx, test_idx = [], [], []
    for c in (0, 1):
        c_idx = idx[labels == c]
        rng.shuffle(c_idx)
        n = len(c_idx)
        n_train = int(n * cfg.train_ratio)
        n_val = int(n * cfg.val_ratio)
        train_idx.append(c_idx[:n_train])
        val_idx.append(c_idx[n_train:n_train + n_val])
        test_idx.append(c_idx[n_train + n_val:])
    train_idx = np.concatenate(train_idx)
    val_idx = np.concatenate(val_idx)
    test_idx = np.concatenate(test_idx)
    rng.shuffle(train_idx)
    rng.shuffle(val_idx)
    rng.shuffle(test_idx)
    train = [samples[i] for i in train_idx]
    val = [samples[i] for i in val_idx]
    test = [samples[i] for i in test_idx]
    return train, val, test


def build_datasets(samples, cfg):
    train_s, val_s, test_s = split_samples(samples, cfg)
    train_ds = FallDataset(train_s)
    val_ds = FallDataset(val_s, mean=train_ds.mean, std=train_ds.std)
    test_ds = FallDataset(test_s, mean=train_ds.mean, std=train_ds.std)
    return train_ds, val_ds, test_ds


def precompute_granger(train_samples, cfg, verbose: bool = True):
    """
    Fit the dataset-level Granger-causality matrix E (Fig. 8 heat map) once,
    on the FALL samples of the training split only, since Section 4.2
    defines the causal process specifically over the Falling -> Impact
    stages of a fall event.
    """
    fall_samples = [s for s in train_samples if s["label"] == 1]
    if len(fall_samples) < 2:
        # not enough falls to fit VAR reliably -> uniform fallback weights
        C = cfg.num_channels
        return np.zeros((C, C)), np.ones(C) / C
    X = np.stack([s["X"] for s in fall_samples])         # Falling-stage windows
    Y = np.stack([s["Y_impact"] for s in fall_samples])   # Impact-stage windows
    edge_selection = getattr(cfg, "gc_edge_selection", "absolute")
    E, M_raw = granger_causality_matrix(
        X, Y, lag=cfg.gc_max_lag,
        ridge=getattr(cfg, "gc_ridge", 1e-3),
        diff=getattr(cfg, "gc_diff", False),
        edge_selection=edge_selection,
        abs_threshold=getattr(cfg, "gc_abs_threshold", 1.0),
        percentile=getattr(cfg, "gc_percentile", 75.0),
        topk=getattr(cfg, "gc_topk", None),
        return_raw=True,
    )
    w = channel_weights_from_E(E)
    if verbose:
        n_edges = int((E > 0).sum())
        total = E.size
        off_diag_mask = ~np.eye(M_raw.shape[0], dtype=bool)
        print(f"  Granger raw M_GC stats (off-diagonal, diff={getattr(cfg, 'gc_diff', False)}): "
              f"min={M_raw[off_diag_mask].min():.3f} "
              f"max={M_raw[off_diag_mask].max():.3f} "
              f"mean={M_raw[off_diag_mask].mean():.3f}")
        if edge_selection == "absolute":
            print(f"  edge_selection=absolute (M_GC > {getattr(cfg, 'gc_abs_threshold', 1.0)})")
        elif edge_selection == "percentile":
            print(f"  edge_selection=percentile (top {100 - getattr(cfg, 'gc_percentile', 75.0):.0f}% "
                  f"of positive pairs)")
        else:
            print(f"  edge_selection=topk (k={getattr(cfg, 'gc_topk', None) or cfg.num_channels})")
        print(f"  Edges selected: {n_edges}/{total}"
              + ("  -> STILL falling back to UNIFORM weights (no channel had "
                 "ANY positive M_GC at all -- this is a data/lag issue, not "
                 "a threshold issue)" if n_edges == 0 else ""))
    return E, w

