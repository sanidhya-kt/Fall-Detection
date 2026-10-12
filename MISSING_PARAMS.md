# Paper-missing / inferred hyperparameters

| Knob | Paper | Our choice | Notes |
|------|-------|------------|-------|
| `δ` | 75 (Sec3) vs **72 (Sec5.2.3)** | **72** | Experiment section wins |
| `d_model` / STAE width | not stated (`H=9`, `ff=16`) | `stae_dim=144` | `d_model=C=9` was too weak (~91%) |
| `dim_feedforward` | 16 | **288** (=2×stae_dim) | Paper 16 only fits tiny d_model |
| `Γ` (reduced_dim) | `Γ < C` only | **6** | |
| Granger lag | ≤ `ϑ=15` | **8** | lag=1 → zero edges |
| Granger threshold | `M_GC > 1` | **percentile 75** | Absolute >1 → uniform CEE weights |
| `W_Scale` | unnamed | learned / absorbed in CEE | |
| Loss `λ` on `L_Counter` | `(+)` only | **0.5** | λ=2 collapsed bias→0 |
| Class weighting | not stated | **off** (best run) | |
| Split | “randomly divides” | **70/15/15 stratified** | |
| Median filter | SlowFall only | **off** (kernel=1) | Blurs SMV peaks |
| SlowFall | used in paper | **out of scope** | No access |

## Local data paths
See [DATA_PATHS.md](DATA_PATHS.md).
