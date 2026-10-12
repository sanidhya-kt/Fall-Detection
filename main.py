"""
main.py
========
Single entry point that reproduces:

  Table 4  - CausalFall vs. every baseline, on one dataset
  Table 5  - each baseline with & without the CEE module
  Table 6  - each baseline with & without counterfactual intervention
  Table 7  - CausalFall with No-Intervention / Random-Mask / Mean-Mask / Zero-Mask

Usage examples
--------------
Smoke test (no real data needed, ~1 minute on CPU/GPU):
    python main.py --dataset synthetic --table 4 --epochs 5

Full SisFall run (Table 4) on an A100:
    python main.py --dataset sisfall --sisfall_root /data/SisFall --table 4

Full KFall run, all four tables:
    python main.py --dataset kfall --kfall_root /data/KFall --table all
"""
import argparse
import copy
import json
import os
import time

import numpy as np
import torch

from config import CFG
from data.dataset import build_datasets, precompute_granger
from data.sisfall_dataset import load_sisfall_samples
from data.kfall_dataset import load_kfall_samples
from data.synthetic import make_synthetic_samples
from models.baselines import BASELINE_REGISTRY
from models.wrappers import CEEAugmentedBaseline, CounterfactualAugmentedBaseline
from train import train_causalfall, train_baseline, train_causal_style
from utils.device import resolve_device


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", choices=["sisfall", "kfall", "synthetic"], required=True)
    p.add_argument("--sisfall_root", type=str, default=None,
                    help="Defaults to local Mac SisFall path in config.py")
    p.add_argument("--kfall_root", type=str, default=None,
                    help="Defaults to local Mac KFall path in config.py")
    p.add_argument("--kfall_label_dir", type=str, default=None)
    p.add_argument("--table", choices=["4", "5", "6", "7", "all", "cf_only"], default="cf_only")
    p.add_argument("--epochs", type=int, default=None, help="override config.epochs")
    p.add_argument("--batch_size", type=int, default=None)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--n_synthetic", type=int, default=800)
    p.add_argument("--out_dir", type=str, default="results")
    p.add_argument("--models", type=str, default=None,
                    help="comma-separated subset of "
                         "{cnn,lstm,bilstm,convlstm,gcnlstm,transformer} "
                         "(default: all six)")
    return p.parse_args()


def resolve_kfall_paths(args, cfg):
    root = args.kfall_root or cfg.kfall_root
    label_dir = args.kfall_label_dir
    sensor = os.path.join(root, "sensor_data")
    labels = os.path.join(root, "label_data")
    if os.path.isdir(sensor) and os.path.isdir(labels):
        return root, labels if label_dir is None else label_dir
    # Fallback: flat sensor tree + separate label dir
    fb_sensor = cfg.kfall_sensor_fallback
    fb_label = label_dir or cfg.kfall_label_fallback
    if os.path.isdir(fb_sensor) and os.path.isdir(fb_label):
        return fb_sensor, fb_label
    return root, label_dir


def load_samples(args, cfg):
    if args.dataset == "sisfall":
        root = args.sisfall_root or cfg.sisfall_root
        assert os.path.isdir(root), f"SisFall root not found: {root}"
        print(f"SisFall root: {root}")
        return load_sisfall_samples(root, cfg)
    elif args.dataset == "kfall":
        root, label_dir = resolve_kfall_paths(args, cfg)
        assert os.path.isdir(root), f"KFall root not found: {root}"
        print(f"KFall root: {root}")
        print(f"KFall labels: {label_dir}")
        return load_kfall_samples(root, cfg, label_dir=label_dir)
    else:
        return make_synthetic_samples(args.n_synthetic, cfg)


def print_table(title, rows, header=("Method", "Accuracy", "Precision", "Recall", "F1-Score")):
    print(f"\n===== {title} =====")
    col_w = [max(len(str(r[i])) for r in ([header] + rows)) + 2 for i in range(len(header))]
    line = "".join(f"{h:<{col_w[i]}}" for i, h in enumerate(header))
    print(line)
    print("-" * len(line))
    for r in rows:
        print("".join(f"{str(v):<{col_w[i]}}" for i, v in enumerate(r)))


def fmt_row(name, m):
    return (name, f"{m['accuracy']:.2f}", f"{m['precision']:.2f}",
            f"{m['recall']:.2f}", f"{m['f1']:.2f}")


def get_model_list(args):
    if args.models:
        names = [m.strip() for m in args.models.split(",")]
        for n in names:
            assert n in BASELINE_REGISTRY, f"Unknown baseline '{n}'"
        return names
    return list(BASELINE_REGISTRY.keys())


def main():
    args = parse_args()
    cfg = copy.deepcopy(CFG)
    if args.epochs is not None:
        cfg.epochs = args.epochs
    if args.batch_size is not None:
        cfg.batch_size = args.batch_size
    if args.device is not None:
        cfg.device = args.device
    cfg.device = resolve_device(cfg.device)
    print(f"Using device: {cfg.device}")

    os.makedirs(args.out_dir, exist_ok=True)
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    print(f"Loading & preprocessing '{args.dataset}' ...")
    t0 = time.time()
    samples = load_samples(args, cfg)
    print(f"Loaded {len(samples)} samples "
          f"({sum(s['label'] for s in samples)} falls / "
          f"{len(samples) - sum(s['label'] for s in samples)} ADLs) "
          f"in {time.time() - t0:.1f}s")

    train_ds, val_ds, test_ds = build_datasets(samples, cfg)
    print(f"Split: train={len(train_ds)} val={len(val_ds)} test={len(test_ds)}")

    print("Precomputing offline Granger-causality matrix (Section 4.2) ...")
    train_raw = train_ds.samples  # the raw sample dicts backing the training split
    E, channel_weights = precompute_granger(train_raw, cfg)
    print(f"Channel causal weights: {np.round(channel_weights, 4)}")

    results_summary = {}
    model_list = get_model_list(args)

    # ================= CausalFall only (strict reproduction target) ==========
    if args.table in ("cf_only",):
        print("\n--- Training CausalFall (full model) ---")
        _, cf_metrics, _ = train_causalfall(
            train_ds, val_ds, test_ds, cfg,
            channel_weights=channel_weights,
            use_cee=True, use_counterfactual=True,
            intervention_mode="zero",
        )
        rows = [fmt_row("CausalFall (Ours)", cf_metrics)]
        results_summary["table4_causalfall"] = cf_metrics
        print_table(f"CausalFall only - {args.dataset}", rows)

    # ================= Table 4: CausalFall vs. all baselines =================
    if args.table in ("4", "all"):
        rows = []
        for name in model_list:
            print(f"\n--- Training baseline: {name} ---")
            _, metrics, _ = train_baseline(name, train_ds, val_ds, test_ds, cfg)
            rows.append(fmt_row(name, metrics))
            results_summary[f"table4_{name}"] = metrics

        print("\n--- Training CausalFall (full model) ---")
        _, cf_metrics, _ = train_causalfall(train_ds, val_ds, test_ds, cfg,
                                             channel_weights=channel_weights,
                                             use_cee=True, use_counterfactual=True,
                                             intervention_mode="zero")
        rows.append(fmt_row("CausalFall (Ours)", cf_metrics))
        results_summary["table4_causalfall"] = cf_metrics
        print_table(f"Table 4 - {args.dataset}", rows)

    # ================= Table 5: baselines with / without CEE =================
    if args.table in ("5", "all"):
        rows = []
        for name in model_list:
            print(f"\n--- {name} without CEE ---")
            _, m_wo, _ = train_baseline(name, train_ds, val_ds, test_ds, cfg)
            print(f"\n--- {name} with CEE ---")
            model = CEEAugmentedBaseline(cfg, BASELINE_REGISTRY[name],
                                          channel_weights=channel_weights)
            _, m_w, _ = train_baseline(name, train_ds, val_ds, test_ds, cfg, model=model)
            rows.append((name,
                         f"{m_wo['accuracy']:.2f} | {m_w['accuracy']:.2f}",
                         f"{m_wo['precision']:.2f} | {m_w['precision']:.2f}",
                         f"{m_wo['recall']:.2f} | {m_w['recall']:.2f}",
                         f"{m_wo['f1']:.2f} | {m_w['f1']:.2f}"))
            results_summary[f"table5_{name}_wo_cee"] = m_wo
            results_summary[f"table5_{name}_w_cee"] = m_w
        print_table(f"Table 5 (w/o | w/ CEE) - {args.dataset}", rows)

    # ========= Table 6: baselines with / without counterfactual intervention =========
    if args.table in ("6", "all"):
        rows = []
        for name in model_list:
            print(f"\n--- {name} without counterfactual intervention ---")
            _, m_wo, _ = train_baseline(name, train_ds, val_ds, test_ds, cfg)
            print(f"\n--- {name} with counterfactual intervention ---")
            model = CounterfactualAugmentedBaseline(cfg, BASELINE_REGISTRY[name],
                                                     intervention_mode="zero")
            _, m_w, _ = train_causal_style(model, train_ds, val_ds, test_ds, cfg,
                                            tag=f"{name}+CF")
            rows.append((name,
                         f"{m_wo['accuracy']:.2f} | {m_w['accuracy']:.2f}",
                         f"{m_wo['precision']:.2f} | {m_w['precision']:.2f}",
                         f"{m_wo['recall']:.2f} | {m_w['recall']:.2f}",
                         f"{m_wo['f1']:.2f} | {m_w['f1']:.2f}"))
            results_summary[f"table6_{name}_wo_cf"] = m_wo
            results_summary[f"table6_{name}_w_cf"] = m_w
        # Also report CausalFall itself with/without counterfactual intervention:
        print(f"\n--- CausalFall without counterfactual intervention ---")
        _, cf_wo, _ = train_causalfall(train_ds, val_ds, test_ds, cfg,
                                        channel_weights=channel_weights,
                                        use_cee=True, use_counterfactual=False)
        print(f"\n--- CausalFall with counterfactual intervention ---")
        _, cf_w, _ = train_causalfall(train_ds, val_ds, test_ds, cfg,
                                       channel_weights=channel_weights,
                                       use_cee=True, use_counterfactual=True)
        rows.append(("CausalFall (Ours)",
                     f"{cf_wo['accuracy']:.2f} | {cf_w['accuracy']:.2f}",
                     f"{cf_wo['precision']:.2f} | {cf_w['precision']:.2f}",
                     f"{cf_wo['recall']:.2f} | {cf_w['recall']:.2f}",
                     f"{cf_wo['f1']:.2f} | {cf_w['f1']:.2f}"))
        results_summary["table6_causalfall_wo_cf"] = cf_wo
        results_summary["table6_causalfall_w_cf"] = cf_w
        print_table(f"Table 6 (w/o | w/ Counterfactual Intervention) - {args.dataset}", rows)

    # ================= Table 7: intervention-type ablation on CausalFall =================
    if args.table in ("7", "all"):
        rows = []
        for mode, label in [("none", "No-Intervention"), ("random", "Random-Mask"),
                             ("mean", "Mean-Mask"), ("zero", "Zero-Mask")]:
            print(f"\n--- CausalFall intervention mode: {label} ---")
            use_cf = mode != "none"
            _, m, _ = train_causalfall(train_ds, val_ds, test_ds, cfg,
                                        channel_weights=channel_weights,
                                        use_cee=True, use_counterfactual=use_cf,
                                        intervention_mode=mode if use_cf else "zero")
            rows.append(fmt_row(label, m))
            results_summary[f"table7_{mode}"] = m
        print_table(f"Table 7 - {args.dataset}", rows)

    out_path = os.path.join(args.out_dir, f"{args.dataset}_results.json")
    with open(out_path, "w") as f:
        json.dump(results_summary, f, indent=2)
    print(f"\nSaved all metrics to {out_path}")


if __name__ == "__main__":
    main()
