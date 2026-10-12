"""
models/positional_encoding.py
===============================
Implements Eq. (2):

    PE(x_t) = sin(w_k * t)   if t = 2k
            = cos(w_k * t)   if t = 2k+1
    w_k = 1 / T^(2k/delta),  k = 0, ..., floor(delta/2)

Standard sinusoidal transformer positional embedding, added to every
channel's time series identically (the same PE(t) vector is broadcast
across channels, since Fig. 4 shows PE applied uniformly before the Q/K/V
projections of each sensor stream).
"""
import math
import torch
import torch.nn as nn


class SinusoidalPositionalEmbedding(nn.Module):
    def __init__(self, delta: int, d_model: int):
        super().__init__()
        self.delta = delta
        self.d_model = d_model
        pe = torch.zeros(delta, d_model)
        position = torch.arange(0, delta, dtype=torch.float32).unsqueeze(1)  # [T,1]
        k = torch.arange(0, d_model, dtype=torch.float32)                    # [D]
        # w_k = 1 / T^(2k/delta)  -- T here denotes `delta`, per Eq. 2's own definition
        omega = 1.0 / (delta ** (2 * k / delta))
        angle = position * omega  # [T, D]
        pe[:, 0::2] = torch.sin(angle[:, 0::2])
        pe[:, 1::2] = torch.cos(angle[:, 1::2])
        self.register_buffer("pe", pe.unsqueeze(0))  # [1, T, D]

    def forward(self, x):
        """x: [B, T, D] -> returns x + PE broadcast over batch."""
        return x + self.pe[:, : x.size(1), :]
