"""Quick SisFall sweep helper: caches preprocessed samples, runs CausalFall with cfg overrides."""
import argparse, copy, json, os, pickle, sys, time
import numpy as np, torch
from config import CFG
from data.dataset import build_datasets, precompute_granger
from data.sisfall_dataset import load_sisfall_samples
from train import train_causalfall
from utils.device import resolve_device

def get_samples(cfg, tag):
    path = f"results/cache_sisfall_{tag}.pkl"
    if os.path.exists(path):
        return pickle.load(open(path, "rb"))
    s = load_sisfall_samples(cfg.sisfall_root, cfg)
    pickle.dump(s, open(path, "wb"))
    return s

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--out_dir", default="results/sisfall_sweep")
    p.add_argument("--name", default="run")
    p.add_argument("--set", nargs="*", default=[], help="key=value cfg overrides")
    p.add_argument("--cache_tag", default="d75")
    a = p.parse_args()
    cfg = copy.deepcopy(CFG)
    cfg.epochs = a.epochs; cfg.seed = a.seed; cfg.device = resolve_device("mps")
    cfg.ckpt_dir = os.path.join(a.out_dir, "ckpt")
    for kv in a.set:
        k, v = kv.split("=")
        old = getattr(cfg, k)
        setattr(cfg, k, type(old)(v) if not isinstance(old, bool) else v.lower() in ("1", "true"))
    torch.manual_seed(cfg.seed); np.random.seed(cfg.seed)
    samples = get_samples(cfg, a.cache_tag)
    train_ds, val_ds, test_ds = build_datasets(samples, cfg)
    print(f"[{a.name}] split {len(train_ds)}/{len(val_ds)}/{len(test_ds)} overrides={a.set}", flush=True)
    E, w = precompute_granger(train_ds.samples, cfg, verbose=False)
    t0 = time.time()
    _, m, _ = train_causalfall(train_ds, val_ds, test_ds, cfg, channel_weights=w)
    print(f"[{a.name}] DONE {time.time()-t0:.0f}s test acc={m['accuracy']:.2f} f1={m['f1']:.2f}", flush=True)
    os.makedirs(a.out_dir, exist_ok=True)
    json.dump({"name": a.name, "seed": a.seed, "set": a.set, "metrics": m}, open(f"{a.out_dir}/{a.name}.json", "w"), indent=1)
