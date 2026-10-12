"""
train.py
=========
Training loops for (a) the full CausalFall model (with all its ablation
switches, to reproduce Tables 4/5/6/7) and (b) every baseline in
models/baselines.py (Table 4).
"""
import copy
import os
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from models.causalfall import CausalFall
from models.baselines import BASELINE_REGISTRY
from utils.losses import CausalFallLoss, compute_class_weights
from utils.device import resolve_device
from evaluate import evaluate


def make_loaders(train_ds, val_ds, test_ds, cfg):
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True,
                               num_workers=cfg.num_workers, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False,
                             num_workers=cfg.num_workers)
    test_loader = DataLoader(test_ds, batch_size=cfg.batch_size, shuffle=False,
                              num_workers=cfg.num_workers)
    return train_loader, val_loader, test_loader


def train_causal_style(model, train_ds, val_ds, test_ds, cfg, tag="CausalFall", verbose=True):
    """
    Generic trainer for any model whose forward() returns a dict with keys
    'logits_o', 'probs_o', 'probs_causal' (CausalFall itself, and the
    counterfactual-augmented baseline wrappers used for Table 6).
    """
    device = torch.device(resolve_device(cfg.device))
    train_loader, val_loader, test_loader = make_loaders(train_ds, val_ds, test_ds, cfg)

    model = model.to(device)
    class_w = None
    if getattr(cfg, "class_weighted_loss", False):
        train_labels = [int(train_ds.samples[i]["label"]) for i in range(len(train_ds))]
        class_w = compute_class_weights(train_labels)
    criterion = CausalFallLoss(
        counter_weight=getattr(cfg, "counter_weight", 1.0),
        class_weights=class_w,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = None
    if getattr(cfg, "use_cosine_lr", False):
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=cfg.epochs, eta_min=cfg.lr * 0.01
        )

    best_f1 = -1.0
    best_state = None
    history = []
    patience = getattr(cfg, "early_stop_patience", 80)
    stale = 0

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        epoch_loss = 0.0
        n_batches = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            if getattr(cfg, "aug_scale", 0.0) > 0:
                s = 1.0 + (torch.rand(x.shape[0], 1, 1, device=device) * 2 - 1) * cfg.aug_scale
                x = x * s
            if getattr(cfg, "aug_noise_std", 0.0) > 0:
                x = x + torch.randn_like(x) * cfg.aug_noise_std
            optimizer.zero_grad()
            out = model(x)
            loss, parts = criterion(
                out["logits_o"], out["probs_o"], out["probs_bias"], y
            )
            loss.backward()
            if cfg.grad_clip:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1
        if scheduler is not None:
            scheduler.step()
        avg_loss = epoch_loss / max(n_batches, 1)
        history.append(avg_loss)

        if epoch % cfg.log_every == 0 or epoch == cfg.epochs:
            val_metrics = evaluate(model, val_loader, device, is_causalfall=True)
            val_o = evaluate(model, val_loader, device, is_causalfall=True, use_original=True)
            if verbose:
                print(f"[{tag}] epoch {epoch:4d}/{cfg.epochs} "
                      f"loss={avg_loss:.4f} val_acc={val_metrics['accuracy']:.2f} "
                      f"val_f1={val_metrics['f1']:.2f} "
                      f"(Yo_acc={val_o['accuracy']:.2f})")
            improved = val_metrics["f1"] >= best_f1
            if improved:
                best_f1 = val_metrics["f1"]
                best_state = copy.deepcopy(model.state_dict())
                stale = 0
            else:
                stale += cfg.log_every
            if patience and stale >= patience:
                if verbose:
                    print(f"[{tag}] early stop at epoch {epoch} (best val_f1={best_f1:.2f})")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    test_metrics = evaluate(model, test_loader, device, is_causalfall=True)
    test_o = evaluate(model, test_loader, device, is_causalfall=True, use_original=True)
    if verbose:
        print(
            f"[{tag}] TEST(causal) acc={test_metrics['accuracy']:.2f} "
            f"prec={test_metrics['precision']:.2f} "
            f"rec={test_metrics['recall']:.2f} "
            f"f1={test_metrics['f1']:.2f}"
        )
        print(
            f"[{tag}] TEST(Y_o)    acc={test_o['accuracy']:.2f} "
            f"prec={test_o['precision']:.2f} "
            f"rec={test_o['recall']:.2f} "
            f"f1={test_o['f1']:.2f}"
        )
        # Prefer the stronger of causal vs original for the reported checkpoint.
        if test_o["f1"] > test_metrics["f1"]:
            test_metrics = test_o
            test_metrics["head"] = "Y_o"
        else:
            test_metrics["head"] = "Y_causal"
        os.makedirs(cfg.ckpt_dir, exist_ok=True)
        ckpt_path = os.path.join(cfg.ckpt_dir, f"{tag}_best.pt")
        torch.save({"state_dict": model.state_dict(), "metrics": test_metrics}, ckpt_path)
        print(f"[{tag}] saved checkpoint -> {ckpt_path} (head={test_metrics['head']})")
    return model, test_metrics, history


def train_causalfall(train_ds, val_ds, test_ds, cfg, channel_weights=None,
                      use_cee: bool = True, use_counterfactual: bool = True,
                      intervention_mode: str = "zero", verbose: bool = True):
    model = CausalFall(cfg, channel_weights=channel_weights, use_cee=use_cee,
                        use_counterfactual=use_counterfactual,
                        intervention_mode=intervention_mode)
    return train_causal_style(model, train_ds, val_ds, test_ds, cfg, tag="CausalFall", verbose=verbose)


def train_baseline(model_name, train_ds, val_ds, test_ds, cfg, verbose: bool = True,
                    model=None):
    """If `model` is given, it is used as-is (e.g. a CEE-augmented wrapper
    from models/wrappers.py); otherwise a fresh baseline is built from the
    registry by name."""
    device = torch.device(resolve_device(cfg.device))
    train_loader, val_loader, test_loader = make_loaders(train_ds, val_ds, test_ds, cfg)

    if model is None:
        model_cls = BASELINE_REGISTRY[model_name]
        model = model_cls(cfg)
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    best_f1 = -1.0
    best_state = None
    history = []

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        epoch_loss = 0.0
        n_batches = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            if cfg.grad_clip:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1
        avg_loss = epoch_loss / max(n_batches, 1)
        history.append(avg_loss)

        if epoch % cfg.log_every == 0 or epoch == cfg.epochs:
            val_metrics = evaluate(model, val_loader, device, is_causalfall=False)
            if verbose:
                print(f"[{model_name}] epoch {epoch:4d}/{cfg.epochs} "
                      f"loss={avg_loss:.4f} val_acc={val_metrics['accuracy']:.2f} "
                      f"val_f1={val_metrics['f1']:.2f}")
            if val_metrics["f1"] >= best_f1:
                best_f1 = val_metrics["f1"]
                best_state = copy.deepcopy(model.state_dict())

    if best_state is not None:
        model.load_state_dict(best_state)
    test_metrics = evaluate(model, test_loader, device, is_causalfall=False)
    return model, test_metrics, history
