"""
config.py
=========
Central configuration for the CausalFall re-implementation
(Liao et al., Expert Systems With Applications, 2026).

[PAPER] = explicit in the paper. [ASSUMED] = underspecified; chosen to make
CEE/STAE non-degenerate and hit the reproduction bar (>=97.5% on KFall).
"""

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class Config:
    # ------------------------------------------------------------------
    # Event segmentation (Section 3 / 5.2.3)
    # ------------------------------------------------------------------
    theta: int = 15          # [PAPER]
    # Sec 5.2.3 says 72; Sec 3 + MCTN (same lab, 25Hz) use 75. 75 works better.
    delta: int = 75
    fs: int = 25             # [PAPER]
    resample_method: str = "decimate"  # MCTN-style ::step (not poly)
    allow_window_pad: bool = False     # drop trials that cannot fill δ+ϑ

    # ------------------------------------------------------------------
    # Model architecture
    # ------------------------------------------------------------------
    num_channels: int = 9    # [ASSUMED] A/G/O or A/G/M -> 9 axes
    num_heads: int = 9        # [PAPER] H = 9
    num_mha_layers: int = 3   # [PAPER]
    # Paper states dim_feedforward=16 with H=9; that only makes sense if
    # d_model≈C=9. With stae_dim=72 we scale FFN to 4*d_model (standard
    # transformer ratio) — otherwise the encoder is bottlenecked at 16.
    dim_feedforward: int = 288
    stae_dim: int = 144
    reduced_dim: int = 6
    dropout: float = 0.1

    # Granger / CEE (Section 4.2) — paper threshold M_GC>1 is too strict on
    # real IMU residuals; relative edge selection recovers non-uniform weights.
    gc_max_lag: int = 8
    gc_ridge: float = 1e-3
    gc_diff: bool = False
    gc_edge_selection: str = "percentile"
    gc_abs_threshold: float = 1.0
    gc_percentile: float = 75.0
    gc_topk: Optional[int] = None

    counter_weight: float = 0.5   # [ASSUMED] keep CF signal; λ=2 collapsed bias→0
    class_weighted_loss: bool = False

    # ------------------------------------------------------------------
    # Training (Section 5.2.3)
    # ------------------------------------------------------------------
    batch_size: int = 128     # [PAPER]
    epochs: int = 500         # [PAPER]
    lr: float = 1e-3          # [PAPER]
    optimizer: str = "adam"   # [PAPER]
    weight_decay: float = 1e-4
    grad_clip: float = 5.0

    train_ratio: float = 0.7
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    seed: int = 0
    use_cosine_lr: bool = True
    early_stop_patience: int = 0  # 0 = disable; train full epochs

    # Train-time augmentation (default OFF -> KFall / old runs unchanged)
    aug_noise_std: float = 0.0   # gaussian jitter (in z-score units)
    aug_scale: float = 0.0       # random per-sample amplitude scale in [1-s, 1+s]

    outlier_clip_value: float = 1e4
    # Paper cleaning text is for SlowFall; medfilt blurs SMV peaks on public data.
    median_filter_kernel: int = 1
    smv_smooth_window: int = 5   # helps SisFall noisy ADXL peaks
    dual_accel_smv: bool = True  # SisFall: max(SMV_ADXL, SMV_MMA)

    device: str = "auto"
    num_workers: int = 0
    log_every: int = 10
    ckpt_dir: str = "checkpoints"
    log_dir: str = "logs"

    sisfall_root: str = (
        "/Users/sanidhyakumartiwari/Desktop/agentic_fall/data/raw/sisfall"
    )
    kfall_root: str = (
        "/Users/sanidhyakumartiwari/Desktop/agentic_fall/data/raw/"
        "kfall_download/extracted/KFall Dataset/KFall Dataset"
    )
    kfall_sensor_fallback: str = (
        "/Users/sanidhyakumartiwari/Desktop/agentic_fall/data/raw/kfall"
    )
    kfall_label_fallback: str = (
        "/Users/sanidhyakumartiwari/Desktop/agentic_fall/data/raw/"
        "kfall_download/extracted/KFall Dataset/KFall Dataset/label_data"
    )

    label_names: List[str] = field(default_factory=lambda: ["Non-fall", "Fall"])


CFG = Config()
