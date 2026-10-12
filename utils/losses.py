"""
utils/losses.py
================
Implements Eq. (20)-(23) of the paper:

    L_Counter  = (1/delta) * sum_t (Y_o - Y_causal)^2          [Eq. 20]
    L_Decoder  = CrossEntropy(Y_o, Y_hat_ground_truth)          [Eq. 21-22]
    L_Global   = L_Decoder (+) L_Counter                        [Eq. 23]

Y_o and Y_causal are both softmax probability vectors of shape [B, 2]
produced by the State Decoder (see models/causalfall.py).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def compute_class_weights(labels) -> torch.Tensor:
    """
    Inverse-frequency class weights for [Non-fall, Fall], computed from the
    TRAIN split only. Both SisFall and KFall have noticeably more ADL samples
    than fall samples, and Table 4/etc. recall (fall-detection sensitivity)
    is exactly the metric that suffers when this imbalance is ignored.
    """
    labels = torch.as_tensor(labels, dtype=torch.long)
    counts = torch.bincount(labels, minlength=2).float()
    counts = torch.clamp(counts, min=1.0)
    weights = counts.sum() / (2.0 * counts)
    return weights  # [w_non_fall, w_fall]


class CausalFallLoss(nn.Module):
    def __init__(self, counter_weight: float = 1.0, class_weights=None):
        """
        counter_weight: [ASSUMED] the paper writes L_Global = L_Decoder (+) L_Counter
        with a generic "(+)" (direct-sum) operator and no explicit scalar weight.
        Swept over {0.5, 1.0, 2.0} in sweep.py.
        class_weights: optional tensor [2] passed to CrossEntropy to counter
        class imbalance (Fall vs. Non-fall are rarely balanced) -- targets the
        recall gap observed empirically on SisFall.
        """
        super().__init__()
        self.counter_weight = counter_weight
        self.class_weights = class_weights

    def forward(self, y_o_logits, y_o_prob, probs_bias, y_true):
        # L_Decoder: standard cross entropy on the *original* (pre-debiasing) prediction
        weight = self.class_weights.to(y_o_logits.device) if self.class_weights is not None else None
        l_decoder = F.cross_entropy(y_o_logits, y_true, weight=weight)

        # L_Counter: per Eq. 20, with Y_causal := Y_o - SD(F_Cf) (Eq. 19)
        # substituted in, (Y_o - Y_causal)^2 reduces exactly to SD(F_Cf)^2,
        # i.e. this loss's only real job is to shrink the counterfactual/
        # bias-branch prediction. Computing it directly as probs_bias**2
        # (rather than literally re-deriving probs_causal = probs_o - probs_bias
        # and squaring the difference against probs_o) keeps the SAME loss
        # value, but the gradient flows ONLY through the bias branch instead
        # of also back-propagating into the main branch's own output twice --
        # the double gradient path was pulling probs_o toward a degenerate
        # majority-class collapse in earlier testing.
        l_counter = torch.mean(probs_bias ** 2)

        l_global = l_decoder + self.counter_weight * l_counter
        return l_global, {
            "L_Decoder": l_decoder.item(),
            "L_Counter": l_counter.item(),
            "L_Global": l_global.item(),
        }
