"""
models/causalfall.py
======================
Full CausalFall pipeline (Fig. 3): STAE + CEE + State Decoder + counterfactual.
"""
import torch
import torch.nn as nn

from models.stae import SpatioTemporalAttentionEncoder
from models.cee import CausalEffectEncoder
from models.counterfactual import CounterfactualIntervention
from models.state_decoder import StateDecoder


class CausalFall(nn.Module):
    def __init__(self, cfg, channel_weights=None, use_cee: bool = True,
                 use_counterfactual: bool = True, intervention_mode: str = "zero"):
        super().__init__()
        self.use_cee = use_cee
        self.use_counterfactual = use_counterfactual
        self.stae = SpatioTemporalAttentionEncoder(cfg)
        self.cee = CausalEffectEncoder(cfg, channel_weights=channel_weights)
        self.counterfactual = CounterfactualIntervention(cfg, mode=intervention_mode)
        self.decoder = StateDecoder(cfg)

    def set_channel_weights(self, channel_weights):
        self.cee.set_channel_weights(channel_weights)

    def forward(self, x):
        """x: [B, C, T] normalized IMU input window."""
        st_feat = self.stae(x)
        causal_feat = self.cee(x) if self.use_cee else torch.zeros_like(x)
        logits_o, probs_o = self.decoder(st_feat, causal_feat, x)

        if not self.use_counterfactual:
            zeros = torch.zeros_like(probs_o)
            return {
                "logits_o": logits_o,
                "probs_o": probs_o,
                "probs_bias": zeros,
                "probs_causal": probs_o,
            }

        x_prime = self.counterfactual(x)
        st_feat_cf = self.stae(x_prime)
        causal_feat_cf = self.cee(x_prime) if self.use_cee else torch.zeros_like(x_prime)
        _, probs_bias = self.decoder(st_feat_cf, causal_feat_cf, x_prime)
        probs_causal = probs_o - probs_bias

        return {
            "logits_o": logits_o,
            "probs_o": probs_o,
            "probs_bias": probs_bias,
            "probs_causal": probs_causal,
        }
