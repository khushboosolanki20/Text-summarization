"""Hardware detection for Transformer inference."""

from functools import lru_cache


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
        return {"torch_installed": False, "cuda_available": False, "device": "cpu", "gpu_name": None}

    cuda = torch.cuda.is_available()
    return {
        "torch_installed": True,
        "cuda_available": cuda,
        "device": "cuda" if cuda else "cpu",
        "gpu_name": torch.cuda.get_device_name(0) if cuda else None,
    }
