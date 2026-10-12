"""Phase A diagnostics for SisFall (no training). Writes results/sisfall_diag_raw.json"""
import json, collections, pickle
import numpy as np
from config import CFG
from data.sisfall_dataset import scan_sisfall, _parse_file, SISFALL_FS
from data.preprocessing import (clean_outliers, resample_signal, compute_smv,
                                smooth_signal_1d)

cfg = CFG
items = scan_sisfall(cfg.sisfall_root)
rows = []
for it in items:
    raw = _parse_file(it["path"])
    sig = resample_signal(clean_outliers(raw, 1e4, 1), 200.0, 25, "decimate")
    T = len(sig)
    smv1 = smooth_signal_1d(compute_smv(sig[:, 0:3]), 5)
    smv2 = smooth_signal_1d(compute_smv(sig[:, 6:9]), 5)
    smvd = np.maximum(smv1, smv2)
    t_dual = int(np.argmax(smvd)); t_adxl = int(np.argmax(smv1)); t_mma = int(np.argmax(smv2))
    rows.append(dict(label=it["label"], code=it["event_type"], subj=it["subject"], T=T,
                     t_dual=t_dual, t_adxl=t_adxl, t_mma=t_mma,
                     pk_dual=float(smvd[t_dual]), pk_adxl=float(smv1.max()), pk_mma=float(smv2.max()),
                     med=float(np.median(smvd)),
                     # gyro peak
                     gpk=float(np.abs(sig[:, 3:6]).max())))
pickle.dump(rows, open("results/sisfall_diag_rows.pkl", "wb"))

need = cfg.delta + cfg.theta
out = {}
lab = np.array([r["label"] for r in rows])
tdu = np.array([r["t_dual"] for r in rows])
Ts = np.array([r["T"] for r in rows])
keep = tdu >= need
out["total"] = {"fall": int((lab == 1).sum()), "adl": int((lab == 0).sum())}
out["kept"] = {"fall": int((keep & (lab == 1)).sum()), "adl": int((keep & (lab == 0)).sum())}
out["dropped"] = {"fall": int((~keep & (lab == 1)).sum()), "adl": int((~keep & (lab == 0)).sum())}
# dropped by activity
drop_by = collections.Counter(r["code"] for r, k in zip(rows, keep) if not k)
tot_by = collections.Counter(r["code"] for r in rows)
out["dropped_by_code"] = {c: f"{drop_by[c]}/{tot_by[c]}" for c in sorted(tot_by) if drop_by[c]}
# impact time stats
for name, m in (("fall", lab == 1), ("adl", lab == 0)):
    out[f"t_dual_s_{name}"] = dict(zip(["p5", "p25", "p50", "p75", "p95"],
                                         np.round(np.percentile(tdu[m] / 25, [5, 25, 50, 75, 95]), 2).tolist()))
    out[f"peak_g_{name}"] = dict(zip(["p5", "p25", "p50", "p75", "p95"],
                                       np.round(np.percentile([r["pk_dual"] for r, l in zip(rows, lab) if l == (name == "fall")], [5, 25, 50, 75, 95]), 2).tolist()))
out["frac_t_impact_at_edge(last 5 samples)"] = float(np.mean(tdu >= Ts - 5))
# sensor disagreement
dis = np.array([abs(r["t_adxl"] - r["t_mma"]) for r in rows]) / 25
out["adxl_vs_mma_t_impact_disagree>0.5s"] = float((dis > 0.5).mean())
# ADL: how many ADLs have SMV peak that looks like fall (>2.5g)?
pk = np.array([r["pk_dual"] for r in rows])
out["adl_peak>2.5g"] = float(((pk > 2.5) & (lab == 0)).sum() / (lab == 0).sum())
out["fall_peak<1.8g"] = float(((pk < 1.8) & (lab == 1)).sum() / (lab == 1).sum())
# subjects with falls
out["n_subjects"] = len(set(r["subj"] for r in rows))
json.dump(out, open("results/sisfall_diag_raw.json", "w"), indent=1)
print(json.dumps(out, indent=1))
