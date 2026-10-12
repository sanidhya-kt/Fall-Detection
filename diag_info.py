"""Phase A: information content of the window vs theta / padding / decimate vs poly (sklearn GBM, no DL)."""
import pickle, os, numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from scipy.signal import resample_poly
from config import CFG
from data.sisfall_dataset import scan_sisfall, _parse_file
from data.preprocessing import clean_outliers, compute_smv, smooth_signal_1d, extract_falling_window

P = "results/sisfall_raw25_cache.pkl"
if os.path.exists(P):
    data = pickle.load(open(P, "rb"))
else:
    data = []
    for it in scan_sisfall(CFG.sisfall_root):
        raw = clean_outliers(_parse_file(it["path"]), 1e4, 1)
        data.append(dict(dec=raw[::8].astype(np.float32), poly=resample_poly(raw, 1, 8, axis=0).astype(np.float32),
                         label=it["label"], code=it["event_type"], subj=it["subject"]))
    pickle.dump(data, open(P, "wb"))


def tim(sig):
    s1 = smooth_signal_1d(compute_smv(sig[:, 0:3]), 5)
    s2 = smooth_signal_1d(compute_smv(sig[:, 6:9]), 5)
    return int(np.argmax(np.maximum(s1, s2)))


def feats(w):
    # w [T, 9] -> stats over full window and last third
    f = []
    for seg in (w, w[-25:]):
        f += [seg.mean(0), seg.std(0), seg.min(0), seg.max(0)]
    smv = compute_smv(w[:, 0:3])
    f += [[smv.max(), smv.min(), smv.std(), smv[-25:].min(), smv[-25:].max(), smv[-10:].mean()]]
    return np.concatenate([np.ravel(a) for a in f])


rng = np.random.default_rng(0)
def run(theta, pad, key="dec", delta=75):
    X, y, subj = [], [], []
    for d in data:
        sig = d[key]
        t = tim(sig)
        w = extract_falling_window(sig, t, theta, delta, allow_pad=pad)
        if w is None:
            continue
        X.append(feats(w)); y.append(d["label"]); subj.append(d["subj"])
    X = np.array(X); y = np.array(y)
    idx = rng.permutation(len(y)); n = int(0.85 * len(y))
    clf = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.08, random_state=0)
    clf.fit(X[idx[:n]], y[idx[:n]])
    acc = (clf.predict(X[idx[n:]]) == y[idx[n:]]).mean() * 100
    print(f"theta={theta:2d} pad={pad!s:5} {key:4} delta={delta} n={len(y)} falls={y.sum()} acc={acc:.2f}", flush=True)

for theta in (15, 8, 3, 0):
    run(theta, True)
run(15, False)
run(15, True, "poly")
run(15, True, delta=50)
