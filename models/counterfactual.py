"""
models/counterfactual.py
==========================
Counterfactual intervention, Section 4.3.2, Eq. (17):

    [X = X']_do = X (dot) M,
    M_t = 0  if t in [t_impact - theta - delta, t_impact - theta - delta/2]
          1  otherwise

Since our extracted input window X already spans exactly
[t_impact - theta - delta, t_impact - theta) (Section 3, Phase 2), the
masked interval corresponds to the FIRST HALF of the window's time axis
(the earliest delta/2 samples), consistent with "the initial stage of
Falling stage occurring in the human body" described in the paper.
"""
import torch
import torch.nn as nn


class CounterfactualIntervention(nn.Module):
    """
    mode:
      'zero'   [PAPER, Eq. 17] the intervention actually proposed and used
               for the headline CausalFall results (Table 4/5/6).
      'random' [Table 7 ablation] replace the masked interval with random noise.
      'mean'   [Table 7 ablation] replace the masked interval with its
               per-channel mean value (computed per-sample, since the paper
               gives no external statistics to use instead).
      'none'   [Table 7 ablation] no intervention at all (identity).
    """
    def __init__(self, cfg, mode: str = "zero"):
        super().__init__()
        assert mode in ("zero", "random", "mean", "none")
        self.mode = mode
        T = cfg.delta
        keep_mask = torch.ones(T)
        keep_mask[: T // 2] = 0.0  # 0 over the intervened interval, 1 elsewhere
        self.register_buffer("keep_mask", keep_mask.view(1, 1, T))

    def forward(self, x):
        """x: [B, C, T] -> counterfactual signal X'."""
        if self.mode == "none":
            return x
        keep = self.keep_mask
        intervene = 1.0 - keep
        if self.mode == "zero":
            replacement = torch.zeros_like(x)
        elif self.mode == "random":
            replacement = torch.randn_like(x)
        elif self.mode == "mean":
            mean_val = x.mean(dim=2, keepdim=True)  # per-sample, per-channel mean
            replacement = mean_val.expand_as(x)
        else:
            raise ValueError(self.mode)
        return x * keep + replacement * intervene
