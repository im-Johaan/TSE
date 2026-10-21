"""SI-SDR and the TSE training objective."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


def si_sdr(est: torch.Tensor, ref: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Scale-invariant SDR in dB, per signal, computed in float32.

    Args:
        est, ref: (..., T) tensors.
    Returns:
        (...,) SI-SDR values.
    """
    est = est.float()
    ref = ref.float()
    est = est - est.mean(dim=-1, keepdim=True)
    ref = ref - ref.mean(dim=-1, keepdim=True)
    alpha = (est * ref).sum(-1, keepdim=True) / (ref.pow(2).sum(-1, keepdim=True) + eps)
    proj = alpha * ref
    noise = est - proj
    return 10 * torch.log10((proj.pow(2).sum(-1) + eps) / (noise.pow(2).sum(-1) + eps))


class TSELoss(nn.Module):
    """``-SI-SDR(est, target) + lambda_ce * CE(logits, label) + lambda_int * -SI-SDR(interf, mix - target)``.

    Terms whose inputs are missing (no logits / no interference estimate) or whose
    weight is zero are skipped. Labels equal to -1 are ignored by the CE term.
    """

    def __init__(self, lambda_ce: float = 0.5, lambda_int: float = 0.1) -> None:
        super().__init__()
        self.lambda_ce = lambda_ce
        self.lambda_int = lambda_int

    def forward(self, out: dict[str, Any], batch: dict[str, Any]) -> dict[str, torch.Tensor]:
        """Return a dict with ``total``, ``si_sdr`` (mean, dB), ``ce`` and ``int_si_sdr``."""
        sdr = si_sdr(out["est"], batch["target"]).mean()
        total = -sdr
        zero = torch.zeros((), device=sdr.device)
        ce = zero
        logits = out.get("spk_logits")
        labels = batch["speaker_label"]
        if self.lambda_ce > 0 and logits is not None and bool((labels >= 0).any()):
            ce = F.cross_entropy(logits.float(), labels, ignore_index=-1)
            total = total + self.lambda_ce * ce
        int_sdr = zero
        if self.lambda_int > 0 and out.get("interference") is not None:
            int_sdr = si_sdr(out["interference"], batch["interference"]).mean()
            total = total - self.lambda_int * int_sdr
        return {"total": total, "si_sdr": sdr.detach(), "ce": ce.detach(),
                "int_si_sdr": int_sdr.detach()}
