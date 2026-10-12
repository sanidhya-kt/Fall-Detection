"""
models/wrappers.py
====================
Generic wrappers that let us bolt CausalFall's CEE and Counterfactual
modules onto ANY of the six baseline architectures, needed to reproduce:

  * Table 5: "Comparison results of different models with & w/o CEE"
  * Table 6: "Comparison results of different models with & w/o
              counterfactual intervention"

Section 5.4.1 states this is done by "modulariz[ing] the CEE and load[ing]
it as a separate module into the framework of the other comparison
models" -- i.e. the baseline's raw input is re-weighted by the same
Granger-causality-derived channel weights CausalFall uses, before being
fed into the (otherwise unchanged) baseline network.
"""
import torch
import torch.nn as nn

from models.cee import CausalEffectEncoder
from models.counterfactual import CounterfactualIntervention


class CEEAugmentedBaseline(nn.Module):
    """base_model(x) -> base_model(CEE(x) + x)  (residual channel re-weighting)."""
    def __init__(self, cfg, base_model_cls, channel_weights=None):
        super().__init__()
        self.cee = CausalEffectEncoder(cfg, channel_weights=channel_weights)
        self.base = base_model_cls(cfg)

    def forward(self, x):
        x_weighted = x + self.cee(x)  # residual so the base model still sees the raw signal
        return self.base(x_weighted)


class CounterfactualAugmentedBaseline(nn.Module):
    """
    Wraps a baseline so it can be trained with the same L_Decoder + L_Counter
    objective as CausalFall (Eq. 20-23): run the baseline on X and on the
    counterfactual X', softmax both, and debias by subtraction (Eq. 19).
    Returns a dict compatible with train.train_causal_style /
    evaluate.evaluate(..., is_causalfall=True).
    """
    def __init__(self, cfg, base_model_cls, intervention_mode: str = "zero"):
        super().__init__()
        self.base = base_model_cls(cfg)
        self.counterfactual = CounterfactualIntervention(cfg, mode=intervention_mode)

    def forward(self, x):
        logits_o = self.base(x)
        probs_o = torch.softmax(logits_o, dim=-1)

        x_prime = self.counterfactual(x)
        logits_bias = self.base(x_prime)
        probs_bias = torch.softmax(logits_bias, dim=-1)

        probs_causal = probs_o - probs_bias
        return {"logits_o": logits_o, "probs_o": probs_o, "probs_causal": probs_causal}
