"""
models/stae.py
================
Spatio-Temporal Attention Encoder (STAE), Section 4.1.

Input:  X  [B, C, delta]   (raw / normalized IMU signal, C channels)
Output: ST [B, C, delta]   (same shape, so it can later be concatenated
                            with the raw signal and the causal feature,
                            Eq. 12)

Design (Fig. 4): each channel's length-delta series first receives the
sinusoidal Positional Embedding (Eq. 2), after being projected from the raw
C channels up into a wider working dimension `stae_dim` (see config.py for
why -- keeping d_model=C starved the encoder of capacity relative to the
baselines). We then treat each of the delta time steps as a token whose
feature vector is the (projected) reading across all sensor channels at
that instant -- this is what lets a single multi-head self-attention block
capture BOTH intra-channel (temporal) and inter-channel (spatial)
dependencies simultaneously, exactly as described ("The attention mechanism
... capture rich ST dependencies intra-channel and inter-channels").

We stack `num_mha_layers` (=3, [PAPER]) standard pre-norm transformer
encoder blocks: MultiHeadAttention(H=9) -> residual -> LayerNorm ->
FeedForward(dim_feedforward=16) -> residual -> LayerNorm, all operating in
the wider `stae_dim` space, then project back down to C channels before
returning (so nothing downstream needs to change).
"""
import torch
import torch.nn as nn

from models.positional_encoding import SinusoidalPositionalEmbedding


class STAELayer(nn.Module):
    def __init__(self, d_model: int, num_heads: int, dim_feedforward: int, dropout: float):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Linear(dim_feedforward, d_model),
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: [B, T, C]
        attn_out, _ = self.attn(x, x, x, need_weights=False)
        x = self.norm1(x + self.dropout(attn_out))
        ff_out = self.ff(x)
        x = self.norm2(x + self.dropout(ff_out))
        return x


class SpatioTemporalAttentionEncoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        C, T, H, D = cfg.num_channels, cfg.delta, cfg.num_heads, cfg.stae_dim
        assert D % H == 0, (
            f"stae_dim ({D}) must be divisible by num_heads ({H})."
        )
        # Project the raw C channels up into a wider working dimension D
        # before attention (see config.py::stae_dim for why this is needed),
        # and back down to C afterward so the rest of the pipeline (CEE,
        # State Decoder's channel-wise concat in Eq. 12) is unaffected.
        self.input_proj = nn.Linear(C, D)
        self.output_proj = nn.Linear(D, C)
        self.pe = SinusoidalPositionalEmbedding(T, D)
        self.layers = nn.ModuleList([
            STAELayer(D, H, cfg.dim_feedforward, cfg.dropout)
            for _ in range(cfg.num_mha_layers)
        ])

    def forward(self, x):
        """x: [B, C, T] -> returns ST feature [B, C, T]."""
        h = x.transpose(1, 2)          # [B, T, C]
        h = self.input_proj(h)         # [B, T, D]
        h = self.pe(h)
        for layer in self.layers:
            h = layer(h)
        h = self.output_proj(h)        # [B, T, C]
        return h.transpose(1, 2)       # [B, C, T]
