"""Phase A: evaluate last SisFall ckpt on train/val/test (no training)."""
import copy, pickle, numpy as np, torch
from config import CFG
from data.dataset import build_datasets, precompute_granger
from models.causalfall import CausalFall
from utils.losses import CausalFallLoss
from evaluate import evaluate
from torch.utils.data import DataLoader

cfg = copy.deepcopy(CFG)
samples = pickle.load(open("results/cache_sisfall_d75.pkl", "rb"))
tr, va, te = build_datasets(samples, cfg)
E, w = precompute_granger(tr.samples, cfg, verbose=False)
print("CEE weights", np.round(w, 4), "uniform=", round(1 / 9, 4))
print("E matrix nonzero edges:", int((E > 0).sum()))
dev = torch.device("cpu")
ck = torch.load("checkpoints/CausalFall_best.pt", map_location="cpu")
m = CausalFall(cfg, channel_weights=w)
m.load_state_dict(ck["state_dict"])
m.eval()
print("ckpt metrics", ck["metrics"])
crit = CausalFallLoss(cfg.counter_weight)
for name, ds in (("train", tr), ("val", va), ("test", te)):
    ld = DataLoader(ds, batch_size=256)
    r = evaluate(m, ld, dev, True)
    ro = evaluate(m, ld, dev, True, use_original=True)
    tot = []; parts = []
    with torch.no_grad():
        for x, y in ld:
            o = m(x)
            l, p = crit(o["logits_o"], o["probs_o"], o["probs_bias"], y)
            tot.append(l.item()); parts.append(p)
    print(name, "causal acc %.2f f1 %.2f | Y_o acc %.2f" % (r["accuracy"], r["f1"], ro["accuracy"]),
          "loss(eval) %.4f dec %.4f cnt %.4f" % (np.mean(tot), np.mean([p["L_Decoder"] for p in parts]),
                                                  np.mean([p["L_Counter"] for p in parts])))
# input stats / saturation
X = np.stack([s["X"] for s in tr.samples])
print("raw channel mean", np.round(X.mean(axis=(0, 2)), 2))
print("raw channel std ", np.round(X.std(axis=(0, 2)), 2))
# per-class per-channel separability in normalized space
y = np.array([s["label"] for s in tr.samples])
Xn = (X - tr.mean) / tr.std
for c in range(9):
    f = Xn[y == 1, c]; a = Xn[y == 0, c]
    print("ch", c, "fall std %.2f adl std %.2f  mean diff %.2f" % (f.std(), a.std(), f.mean() - a.mean()))
# probs_bias magnitude
with torch.no_grad():
    o = m(torch.from_numpy(((X[:512] - tr.mean) / tr.std).astype(np.float32)))
print("mean probs_bias", o["probs_bias"].mean(0).numpy(), "mean probs_o", o["probs_o"].mean(0).numpy())
# ADL errors by type on val+test
for name, ds in (("test", te),):
    ld = DataLoader(ds, batch_size=256)
    preds = []
    with torch.no_grad():
        for x, yy in ld:
            preds.append(m(x)["probs_causal"].argmax(-1).numpy())
    preds = np.concatenate(preds)
    import collections
    err = collections.Counter(); tot = collections.Counter()
    for s, p in zip(ds.samples, preds):
        tot[s["event_type"]] += 1
        if p != s["label"]:
            err[s["event_type"]] += 1
    print("test errors by type:", {k: f"{err[k]}/{tot[k]}" for k in sorted(tot) if err[k]})
