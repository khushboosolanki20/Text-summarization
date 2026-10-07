"""Hardware detection for Transformer inference."""

import logging
from functools import lru_cache

logger = logging.getLogger(__name__)


@lru_cache
def cuda_probe() -> tuple[bool, str | None]:
    """
    Check that CUDA actually works, not just that it is reported available.

    ``torch.cuda.is_available()`` only checks that a driver and device exist.
    It returns True even when every real operation fails, e.g. when the
    installed NVIDIA driver is too old for the CUDA version PyTorch was built
    with. So we run one tiny operation on the GPU. Returns (usable, reason).
    """
    try:
        import torch
    except ImportError:
        return False, "PyTorch is not installed"
    if not torch.cuda.is_available():
        return False, "no CUDA-capable GPU detected"
    try:
        (torch.ones(2, device="cuda") * 2).sum().item()
        return True, None
    except Exception as exc:  # e.g. "CUDA-capable device(s) is/are busy or unavailable"
        reason = str(exc).splitlines()[0]
        logger.warning("CUDA is reported available but unusable (%s); falling back to CPU.", reason)
        return False, reason


@lru_cache
def get_device_info() -> dict:
    """
    Report whether PyTorch is installed and whether a CUDA GPU is usable.

    Imported lazily so that the API can start (and serve the extractive
    methods) even before PyTorch has been installed.
    """
    try:
        import torch
    except ImportError:
        return {"torch_installed": False, "cuda_available": False, "device": "cpu", "gpu_name": None, "gpu_problem": None}

    usable, reason = cuda_probe()
    detected = torch.cuda.is_available()
    return {
        "torch_installed": True,
        "cuda_available": usable,
        "device": "cuda" if usable else "cpu",
        "gpu_name": torch.cuda.get_device_name(0) if detected else None,
        # Set when a GPU exists but cannot be used (e.g. outdated driver).
        "gpu_problem": reason if detected and not usable else None,
    }
