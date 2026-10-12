"""Pick the best available torch device (mps / cuda / cpu)."""
import torch


def resolve_device(requested: str = "auto") -> str:
    req = (requested or "auto").lower()
    if req != "auto":
        return req
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"
