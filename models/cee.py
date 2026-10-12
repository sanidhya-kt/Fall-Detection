"""
models/cee.py
==============
Causal Effect Encoder (CEE), Section 4.2, Eq. (11):

    CEE(X^i) = [ sum_j E(L^i_falling, L^j_impact) ] /
               [ sum_j sum_k E(L^j_falling, L^k_impact) ]  *  W_Scale * X^i

The (C x C) Granger-causality heat-map E is precomputed OFFLINE once per
dataset (see utils/granger.py + data/dataset.py::precompute_granger),
exactly as Section 5.4.1 describes: "the CEE algorithm is designed to run
in offline computer environments... precomputes the causal effects of all
fall event samples... supplementary information for the model training
process... does not affect the online or real-time performance."

At training/inference time this module therefore just applies the fixed,
precomputed per-channel weight vector `w` (row-normalized causal strength)
together with ONE learnable scalar-per-channel W_Scale, directly on the raw
input signal X, producing the global causal feature C (shape preserved:
[B, C, delta]) that is later concatenated with the ST feature and the raw
signal in the State Decoder (Eq. 12).
"""
import torch
import torch.nn as nn


class CausalEffectEncoder(nn.Module):
    def __init__(self, cfg, channel_weights=None):
        """
        channel_weights: optional precomputed [C] numpy/torch array from
        utils.granger.channel_weights_from_E(E). If None, falls back to a
        uniform 1/C vector (equivalent to "no causal preference yet"),
        useful before the offline Granger pass has been run.
        """
        super().__init__()
        C = cfg.num_channels
        if channel_weights is None:
            channel_weights = torch.ones(C) / C
        elif not torch.is_tensor(channel_weights):
            channel_weights = torch.as_tensor(channel_weights, dtype=torch.float32)
        self.register_buffer("channel_weights", channel_weights.view(1, C, 1))
        # W_Scale: one learnable scale per channel (Eq. 11)
        self.w_scale = nn.Parameter(torch.ones(1, C, 1))

    def set_channel_weights(self, channel_weights):
        C = self.channel_weights.shape[1]
        w = torch.as_tensor(channel_weights, dtype=torch.float32).view(1, C, 1)
        self.channel_weights.copy_(w)

    def forward(self, x):
        """x: [B, C, T] -> causal feature C: [B, C, T]."""
        return x * self.channel_weights * self.w_scale
