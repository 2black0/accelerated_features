"""
	"XFeat: Accelerated Features for Lightweight Image Matching, CVPR 2024."
	https://www.verlab.dcc.ufmg.br/descriptors/xfeat_cvpr24/
"""

import warnings

import torch


# Kornia still uses the deprecated torch.cuda.amp decorators; remap them to the
# torch.amp variants ahead of any Kornia import to avoid noisy FutureWarnings.
def _patch_amp_decorators() -> None:
    if not (hasattr(torch, "cuda") and hasattr(torch.cuda, "amp") and hasattr(torch, "amp")):
        return

    def _wrap_amp_decorator(new_decorator):
        def _factory(*args, **kwargs):
            kwargs.setdefault("device_type", "cuda")
            return new_decorator(*args, **kwargs)

        return _factory

    if hasattr(torch.cuda.amp, "custom_fwd") and hasattr(torch.amp, "custom_fwd"):
        torch.cuda.amp.custom_fwd = _wrap_amp_decorator(torch.amp.custom_fwd)  # type: ignore[attr-defined]
    if hasattr(torch.cuda.amp, "custom_bwd") and hasattr(torch.amp, "custom_bwd"):
        torch.cuda.amp.custom_bwd = _wrap_amp_decorator(torch.amp.custom_bwd)  # type: ignore[attr-defined]


warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
    message=r"`torch.cuda.amp.custom_fwd\(args\.\.\.\)` is deprecated",
)
warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
    message=r"`torch.cuda.amp.custom_bwd\(args\.\.\.\)` is deprecated",
)

_patch_amp_decorators()
