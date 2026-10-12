"""
evaluate.py
============
Shared evaluation loop. `is_causalfall=True` models return a dict with
`probs_causal` (the final, debiased prediction -- see models/causalfall.py
docstring and Fig. 3's "Finally, the debiased fall prediction ... is
obtained through State Decoder"); plain baselines just return logits.
"""
import torch
import numpy as np

from utils.metrics import compute_metrics


@torch.no_grad()
def evaluate(model, loader, device, is_causalfall: bool = False, use_original: bool = False):
    model.eval()
    all_true, all_pred = [], []
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        if is_causalfall:
            out = model(x)
            key = "probs_o" if use_original else "probs_causal"
            pred = out[key].argmax(dim=-1)
        else:
            logits = model(x)
            pred = logits.argmax(dim=-1)
        all_true.append(y.cpu().numpy())
        all_pred.append(pred.cpu().numpy())
    y_true = np.concatenate(all_true)
    y_pred = np.concatenate(all_pred)
    return compute_metrics(y_true, y_pred)
