"""
models/state_decoder.py
=========================
State Decoder (SD), Section 4.3.1, Eqs. (12)-(16).

F_Co = ST (+) C (+) X                         (Eq. 12, concat along channel dim)
F_L1 = W_L1 x F_Co + B_L1                     (Eq. 13, channel reduction C*3 -> Gamma)
F_GAP = GlobalAveragePool_time(F_L1)          (Eq. 14)
F_L2 = W_L2 x F_GAP + B_L2                    (Eq. 15, Gamma -> 2 classes)
Y    = Softmax(F_L2)                          (Eq. 16)

NOTE on Eq. (14): the paper's formula, read literally, globally averages
BOTH the channel and time axes down to a single scalar per sample before
the final classification linear layer. We instead pool only the temporal
axis (keeping the Gamma reduced channels), which is the standard
"Global Average Pooling" usage in CNN classifiers that the paper's own
wording ("without additional parameters to enhance ... generalization
ability") is evidently drawing on -- pooling away the last remaining
feature dimension entirely would leave the classifier with a single number
per sample, discarding all channel information. This is flagged here as an
explicit, documented deviation from the literal formula in the interest of
a functioning, sensibly-sized classifier; everything else follows the
formulas exactly.
"""
import torch
import torch.nn as nn


class StateDecoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        C, Gamma = cfg.num_channels, cfg.reduced_dim
        in_ch = 3 * C  # ST (+) Causal (+) raw X, concatenated along channel dim
        self.reduce = nn.Conv1d(in_ch, Gamma, kernel_size=1)     # Eq. 13 (W_L1, B_L1)
        self.gap = nn.AdaptiveAvgPool1d(1)                        # Eq. 14 (time-axis GAP)
        self.classify = nn.Linear(Gamma, 2)                       # Eq. 15 (W_L2, B_L2)

    def forward(self, st_feat, causal_feat, x_raw):
        """
        st_feat, causal_feat, x_raw: each [B, C, T]
        Returns:
            logits: [B, 2]           (pre-softmax, used for CrossEntropy)
            probs:  [B, 2]            (post-softmax, Eq. 16, used for L_Counter / inference)
        """
        f_co = torch.cat([st_feat, causal_feat, x_raw], dim=1)   # Eq. 12 -> [B, 3C, T]
        f_l1 = self.reduce(f_co)                                  # [B, Gamma, T]
        f_gap = self.gap(f_l1).squeeze(-1)                        # [B, Gamma]
        logits = self.classify(f_gap)                             # [B, 2]
        probs = torch.softmax(logits, dim=-1)                     # Eq. 16
        return logits, probs
