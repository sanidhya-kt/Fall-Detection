"""Train CausalFall over several seeds; report mean±std (paper: random splits)."""
import argparse
import copy
import json
import os
import time

import numpy as np
import torch

from config import CFG
from data.dataset import build_datasets, precompute_granger
from data.kfall_dataset import load_kfall_samples
from data.sisfall_dataset import load_sisfall_samples
from main import resolve_kfall_paths
from train import train_causalfall
from utils.device import resolve_device


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", choices=["kfall", "sisfall"], required=True)
    p.add_argument("--seeds", type=str, default="42,0,1,7,123")
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--device", type=str, default="mps")
    p.add_argument("--out_dir", type=str, default="results/multiseed")
    args = p.parse_args()

    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    os.makedirs(args.out_dir, exist_ok=True)
    cfg0 = copy.deepcopy(CFG)
    cfg0.epochs = args.epochs
    cfg0.device = resolve_device(args.device)
    print(f"device={cfg0.device} epochs={cfg0.epochs} seeds={seeds}")

    class A:
        pass
    a = A()
    a.kfall_root = None
    a.kfall_label_dir = None
    a.sisfall_root = None

    t0 = time.time()
    if args.dataset == "kfall":
        root, label_dir = resolve_kfall_paths(a, cfg0)
        samples = load_kfall_samples(root, cfg0, label_dir=label_dir)
    else:
        samples = load_sisfall_samples(cfg0.sisfall_root, cfg0)
    print(f"loaded {len(samples)} in {time.time()-t0:.1f}s")

    all_metrics = []
    for seed in seeds:
        cfg = copy.deepcopy(cfg0)
        cfg.seed = seed
        torch.manual_seed(seed)
        np.random.seed(seed)
        print(f"\n===== seed {seed} =====")
        train_ds, val_ds, test_ds = build_datasets(samples, cfg)
        _, w = precompute_granger(train_ds.samples, cfg)
        print(f"weights={np.round(w, 4)}")
        _, metrics, _ = train_causalfall(
            train_ds, val_ds, test_ds, cfg,
            channel_weights=w, use_cee=True, use_counterfactual=True,
            intervention_mode="zero",
        )
        metrics["seed"] = seed
        all_metrics.append(metrics)
        print(f"seed {seed} -> acc={metrics['accuracy']:.2f} f1={metrics['f1']:.2f}")

    keys = ["accuracy", "precision", "recall", "f1"]
    summary = {"runs": all_metrics, "mean": {}, "std": {}}
    for k in keys:
        vals = [m[k] for m in all_metrics]
        summary["mean"][k] = float(np.mean(vals))
        summary["std"][k] = float(np.std(vals))
    print("\n===== MEAN ± STD =====")
    for k in keys:
        print(f"{k}: {summary['mean'][k]:.2f} ± {summary['std'][k]:.2f}")

    out = os.path.join(args.out_dir, f"{args.dataset}_multiseed.json")
    with open(out, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
