# CausalFall — Reproduction

Re-implementation of:

> Liao, Xie, Li, Liao, Wang, Liang, Liu. **"CausalFall: Fall prediction wearing
> motion sensors from a causal perspective."** Expert Systems With
> Applications 300 (2026) 130218.

This is a full PyTorch codebase covering the model (Spatio-Temporal Attention
Encoder + Granger-causality Causal Effect Encoder + Counterfactual
Intervention + State Decoder), the six baselines, the data pipeline for
**SisFall** and **KFall**, and driver scripts that reproduce **Tables 4, 5,
6 and 7** of the paper.

## 1. What maps to what

| Paper element | File |
|---|---|
| Eq. 1-2, Fig. 2 (SMV, t_impact, window extraction) | `data/preprocessing.py` |
| Eq. 2 Positional Embedding | `models/positional_encoding.py` |
| Section 4.1 Spatio-Temporal Attention Encoder (Eq. 3-4) | `models/stae.py` |
| Section 4.2 Causal Effect Encoder / Granger test (Eq. 5-11) | `utils/granger.py`, `models/cee.py` |
| Section 4.3.1 State Decoder (Eq. 12-16) | `models/state_decoder.py` |
| Section 4.3.2 Counterfactual Intervention (Eq. 17-19) | `models/counterfactual.py` |
| Eq. 20-23 losses | `utils/losses.py` |
| Full CausalFall model (Fig. 3) | `models/causalfall.py` |
| Table 4 baselines | `models/baselines.py` |
| Table 5 / 6 "baseline + CEE" / "baseline + counterfactual" ablations | `models/wrappers.py` |
| SisFall / KFall parsing | `data/sisfall_dataset.py`, `data/kfall_dataset.py` |
| Metrics (Accuracy/Precision/Recall/F1) | `utils/metrics.py` |
| Training loops | `train.py` |
| CLI / experiment driver | `main.py` |

## 2. Assumptions made where the paper is silent

The paper is precise about the causal *mechanism* (Granger test, counterfactual
masking, loss terms) but does not publish a few implementation details. Every
one of these is called out with an `[ASSUMED]` tag and a one-line rationale
directly in the code (mainly in `config.py`, `models/state_decoder.py`,
`models/stae.py`). In summary:

- **Channel count `C=9`**: each dataset's sensor suite (A+G+O or A+G+M) expands
  to 9 scalar channels (3 sensors x 3 axes). This is what makes `H=9` heads
  line up 1-to-1 with channels in the STAE.
- **Train/val/test split = 70/10/20, random, seed=42**: the paper only says
  "randomly divides datasets" (Section 5.3).
- **Gamma (State-Decoder bottleneck width) = 5**: the paper only states
  `Gamma < C`.
- **Global Average Pooling in the State Decoder pools only the time axis**
  (keeping the Gamma channel dimension), not both axes down to a scalar —
  documented in `models/state_decoder.py`. Pooling literally everything away
  would leave a single number per sample feeding the final classifier, which
  cannot be what was intended operationally.
- **Granger-causality lag = 1**: paper says lags "generally...not exceed
  vartheta (=15)"; lag-1 is the smallest, most stable choice within that bound.
- **L_Counter weight = 1.0** in `L_Global = L_Decoder + L_Counter`: the paper
  writes a generic "(+)" with no scalar.
- **Outlier clipping threshold**: paper only mentions SlowFall outliers
  "exceeding 1e10"; we clip at a much tighter, physically-motivated bound
  (see `config.py`).
- **Baseline architectures** (CNN/LSTM/Bi-LSTM/ConvLSTM/GCN-LSTM/Transformer):
  the paper cites the original works but the CausalFall paper itself gives no
  layer-by-layer spec for its *own* re-implementation of them. Standard,
  similarly-sized versions are provided in `models/baselines.py`.

None of these choices are unreasonable defaults, and all training
hyperparameters that **are** specified in the paper (`epochs=500,
batch_size=128, Adam lr=1e-3, H=9, 3 MHA layers, dim_feedforward=16,
delta=72, theta=15, fs=25Hz`) are hard-coded exactly as given, in
`config.py`.

## 3. Installation

```bash
pip install -r requirements.txt
```

Requires PyTorch with CUDA for A100 use (`pip install torch` on a CUDA host
will pull the right wheel automatically).

## 4. Get the datasets

**SisFall**: download & extract from
http://sistemic.udea.edu.co/en/research/projects/english-sisfall/
so that you have:
```
<SISFALL_ROOT>/SA01/D01_SA01_R01.txt, F01_SA01_R01.txt, ...
<SISFALL_ROOT>/SA02/...
<SISFALL_ROOT>/SE01/...
```

**KFall**: download & extract from
https://sites.google.com/view/kfalldataset
so that you have:
```
<KFALL_ROOT>/sensor_data/SA06/S06T01R01.csv, ...
<KFALL_ROOT>/label_data/SA06_label.xlsx, ...
```
(If your release names columns slightly differently, adjust the alias lists
at the top of `data/kfall_dataset.py` — they're matched case-insensitively
against a handful of known variants already.)

## 5. Quick smoke test (no dataset download needed)

```bash
python main.py --dataset synthetic --table 4 --epochs 5
```
This exercises the entire pipeline (data generation -> Granger precompute ->
training -> evaluation -> results table) in well under a minute and is the
fastest way to confirm your environment is set up correctly before spending
GPU time on the real datasets.

## 6. Full reproduction on an A100

```bash
# Table 4: CausalFall vs. all 6 baselines
python main.py --dataset sisfall --sisfall_root /data/SisFall --table 4
python main.py --dataset kfall   --kfall_root   /data/KFall   --table 4

# All four tables in one run
python main.py --dataset sisfall --sisfall_root /data/SisFall --table all
python main.py --dataset kfall   --kfall_root   /data/KFall   --table all
```
Results (accuracy/precision/recall/F1 for every method) are printed to
stdout as formatted tables (matching the paper's Table 4/5/6/7 layout) and
saved as JSON to `results/<dataset>_results.json`.

Runtime on a single A100 80GB: each `(model, dataset)` pair trains for the
paper's full 500 epochs at batch size 128; with `delta=72, C=9`, every model
here is small (a few hundred K to ~1M parameters), so a full 500-epoch run
typically finishes in a few minutes per model even on SisFall's ~4500
samples. Table 4 (7 models) + Table 5 (12 runs) + Table 6 (14 runs) +
Table 7 (4 runs) end-to-end is on the order of an hour per dataset.

## 7. What we could not fully verify in-house

We do not have redistribution rights to SisFall/KFall, so this codebase was
authored, unit-tested on synthetic data, and manually shape-checked, but
**not** run end-to-end against the real datasets before delivery — you will
be the first to run it on real SisFall/KFall data. Please run the
`--dataset synthetic` smoke test first; if that passes (it exercises every
line of model/training code) the only remaining risk is dataset-specific
column-naming quirks in your particular KFall/SisFall release, which
`data/kfall_dataset.py` and `data/sisfall_dataset.py` are written to
surface as clear, actionable error messages rather than silent failures.

## 8. Reproducing SlowFall (private dataset)

The paper's third dataset, SlowFall, is the authors' own unreleased data
(802 samples, 12 elderly-mimicking volunteers). It cannot be reproduced
since it was never made public (the paper itself says "we will make
SlowFall publicly available to the community" once complete). If you collect
an equivalent dataset yourself (accelerometer+gyroscope+magnetometer at the
waist, 25 Hz), point a new loader at it following the same pattern as
`data/sisfall_dataset.py` / `data/kfall_dataset.py` — the rest of the
pipeline (preprocessing, model, training) is dataset-agnostic.
