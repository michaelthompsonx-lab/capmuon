"""CapMuon: O(d_in) Capped Input-Geometry Preconditioning for Muon."""
from .optimizer import (
    CapMuon,
    CapTracker,
    split_muon_and_adam_params,
    zeropower_via_newtonschulz5,
)
from .diagnostics import inspect_linear_eigenspectra

__all__ = [
    "CapMuon",
    "CapTracker",
    "zeropower_via_newtonschulz5",
    "split_muon_and_adam_params",
    "inspect_linear_eigenspectra",
]
__version__ = "0.1.0"
