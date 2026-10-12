"""
models/baselines.py
=====================
Standard re-implementations of the six comparison methods used in Table 4
(Section 5.2.1): CNN (Zhang & Zhu, 2018), LSTM (Musci et al., 2020),
Bi-LSTM (Mubibya et al., 2023), ConvLSTM (Yu et al., 2020),
GCN-LSTM (Wu et al., 2021), Transformer (Liu et al., 2023b).

None of these architectures are specified in per-layer detail in the
CausalFall paper (it only cites the original works) -- the versions below
are standard, reasonably-sized implementations for a [B, C, T] IMU
classification task, matched in overall capacity to the CausalFall model
(similar hidden width / depth) so that Table 4-style comparisons are fair.
All of them output raw logits of shape [B, 2].
"""
import torch
import torch.nn as nn


class CNNBaseline(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        C = cfg.num_channels
        self.net = nn.Sequential(
            nn.Conv1d(C, 32, kernel_size=5, padding=2), nn.ReLU(), nn.BatchNorm1d(32),
            nn.Conv1d(32, 64, kernel_size=5, padding=2), nn.ReLU(), nn.BatchNorm1d(64),
            nn.AdaptiveAvgPool1d(1),
        )
        self.fc = nn.Linear(64, 2)

    def forward(self, x):
        h = self.net(x).squeeze(-1)
        return self.fc(h)


class LSTMBaseline(nn.Module):
    def __init__(self, cfg, hidden=64, bidirectional=False):
        super().__init__()
        C = cfg.num_channels
        self.lstm = nn.LSTM(C, hidden, num_layers=2, batch_first=True,
                             bidirectional=bidirectional, dropout=0.1)
        out_dim = hidden * (2 if bidirectional else 1)
        self.fc = nn.Linear(out_dim, 2)

    def forward(self, x):
        h = x.transpose(1, 2)  # [B, T, C]
        out, _ = self.lstm(h)
        last = out[:, -1, :]
        return self.fc(last)


class BiLSTMBaseline(LSTMBaseline):
    def __init__(self, cfg, hidden=64):
        super().__init__(cfg, hidden=hidden, bidirectional=True)


class ConvLSTMBaseline(nn.Module):
    """CNN feature extractor followed by an LSTM temporal head (Yu et al., 2020)."""
    def __init__(self, cfg, hidden=64):
        super().__init__()
        C = cfg.num_channels
        self.conv = nn.Sequential(
            nn.Conv1d(C, 32, kernel_size=5, padding=2), nn.ReLU(), nn.BatchNorm1d(32),
        )
        self.lstm = nn.LSTM(32, hidden, num_layers=1, batch_first=True)
        self.fc = nn.Linear(hidden, 2)

    def forward(self, x):
        h = self.conv(x)               # [B, 32, T]
        h = h.transpose(1, 2)          # [B, T, 32]
        out, _ = self.lstm(h)
        return self.fc(out[:, -1, :])


class GCNLSTMBaseline(nn.Module):
    """
    Simplified GCN-LSTM (Wu et al., 2021): a single learned channel-adjacency
    graph convolution (channels as nodes) applied at every time step, followed
    by an LSTM over time.
    """
    def __init__(self, cfg, hidden=64):
        super().__init__()
        C = cfg.num_channels
        self.adj = nn.Parameter(torch.eye(C) + 0.01 * torch.randn(C, C))
        self.gcn_lin = nn.Linear(C, C)
        self.lstm = nn.LSTM(C, hidden, num_layers=1, batch_first=True)
        self.fc = nn.Linear(hidden, 2)

    def forward(self, x):
        # x: [B, C, T] -> treat channels as graph nodes at each time step
        h = x.transpose(1, 2)                      # [B, T, C]
        adj = torch.softmax(self.adj, dim=-1)       # row-normalized adjacency
        h = torch.einsum("btc,cd->btd", h, adj)     # graph propagation over channels
        h = torch.relu(self.gcn_lin(h))
        out, _ = self.lstm(h)
        return self.fc(out[:, -1, :])


class TransformerBaseline(nn.Module):
    """Vanilla Transformer encoder (Liu et al., 2023b, MCTN precursor architecture)."""
    def __init__(self, cfg, d_model=None, num_layers=2, nhead=None):
        super().__init__()
        C, T = cfg.num_channels, cfg.delta
        d_model = d_model or max(C, 16)
        nhead = nhead or cfg.num_heads if (d_model % cfg.num_heads == 0) else 4
        self.input_proj = nn.Linear(C, d_model)
        self.pos_emb = nn.Parameter(torch.randn(1, T, d_model) * 0.02)
        layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward=cfg.dim_feedforward,
                                            batch_first=True, dropout=cfg.dropout)
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.fc = nn.Linear(d_model, 2)

    def forward(self, x):
        h = x.transpose(1, 2)          # [B, T, C]
        h = self.input_proj(h) + self.pos_emb[:, : h.size(1), :]
        h = self.encoder(h)
        h = h.mean(dim=1)
        return self.fc(h)


BASELINE_REGISTRY = {
    "cnn": CNNBaseline,
    "lstm": LSTMBaseline,
    "bilstm": BiLSTMBaseline,
    "convlstm": ConvLSTMBaseline,
    "gcnlstm": GCNLSTMBaseline,
    "transformer": TransformerBaseline,
}
