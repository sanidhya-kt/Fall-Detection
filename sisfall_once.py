"""SisFall-only single run (see results/sisfall_diagnosis.md). Does not touch KFall defaults."""
import copy, json, os, shutil, time
import numpy as np, torch
from torch.utils.data import DataLoader
from config import CFG
from data.dataset import build_datasets, precompute_granger
from data.sisfall_dataset import load_sisfall_samples
from train import train_causalfall
from evaluate import evaluate
from utils.device import resolve_device

OUT = "results/sisfall_once"
cfg = copy.deepcopy(CFG)
# ---- SisFall-specific override (CFG / KFall untouched) ----
cfg.theta = 2                  # 15 raw samples @200Hz ~= 2 samples @25Hz (was 15 @25Hz = 0.6 s blind gap)
cfg.delta = 75
cfg.allow_window_pad = True    # keep all 4505 trials (paper: 1798 falls / 2707 ADLs)
cfg.dual_accel_smv = True
cfg.smv_smooth_window = 5
cfg.dropout = 0.2              # train acc was 100% vs val 93.9%
cfg.aug_noise_std = 0.05
cfg.aug_scale = 0.1
cfg.epochs = 200
cfg.log_every = 5
cfg.seed = 0
cfg.ckpt_dir = OUT
cfg.device = resolve_device("mps")
os.makedirs(OUT, exist_ok=True)
torch.manual_seed(cfg.seed); np.random.seed(cfg.seed)
print("device", cfg.device, flush=True)

samples = load_sisfall_samples(cfg.sisfall_root, cfg)
nf = sum(s["label"] for s in samples)
print(f"samples={len(samples)} falls={nf} adls={len(samples)-nf}", flush=True)
train_ds, val_ds, test_ds = build_datasets(samples, cfg)
print(f"split {len(train_ds)}/{len(val_ds)}/{len(test_ds)}", flush=True)
E, w = precompute_granger(train_ds.samples, cfg)
print("CEE weights", np.round(w, 4), flush=True)

t0 = time.time()
model, m_reported, _ = train_causalfall(train_ds, val_ds, test_ds, cfg, channel_weights=w)
dev = torch.device(cfg.device)
res = {}
for name, ds in (("train", train_ds), ("val", val_ds), ("test", test_ds)):
    ld = DataLoader(ds, batch_size=256)
    res[name] = {"causal": evaluate(model, ld, dev, True), "Y_o": evaluate(model, ld, dev, True, use_original=True)}
res["elapsed_s"] = time.time() - t0
res["paper"] = {"accuracy": 97.73, "precision": 96.85, "recall": 96.85, "f1": 96.73}
res["overrides"] = {k: getattr(cfg, k) for k in ("theta", "delta", "allow_window_pad", "dropout", "aug_noise_std", "aug_scale", "epochs", "seed")}
json.dump(res, open(f"{OUT}/sisfall_results.json", "w"), indent=1)
shutil.copy(f"{OUT}/CausalFall_best.pt", "checkpoints/SisFall_once_best.pt")
for k in ("train", "val", "test"):
    c = res[k]["causal"]
    print(f"FINAL {k}: acc={c['accuracy']:.2f} prec={c['precision']:.2f} rec={c['recall']:.2f} f1={c['f1']:.2f}", flush=True)
